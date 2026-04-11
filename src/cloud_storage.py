"""
Cloud persistence layer – AWS S3 backend.

On Streamlit Cloud the filesystem is ephemeral: every deploy / sleep wipes
`data/cache/`.  This module syncs critical JSON files (stored_plans, accuracy_log,
prediction_archive) to an **AWS S3 bucket** so they survive across restarts.

Unlike the previous Gist backend, S3 has no practical file-size limit,
so the prediction archive can grow indefinitely.

Setup (one-time):
  1. Create an S3 bucket (e.g. "gold-predictor-archive-vvrajkumar") in your preferred region.
  2. Create an IAM user/role with s3:GetObject, s3:PutObject on that bucket.
  3. Add to Streamlit secrets (or .env):
       AWS_ACCESS_KEY_ID      = "AKIA..."
       AWS_SECRET_ACCESS_KEY  = "wJal..."
       AWS_S3_BUCKET_NAME     = "gold-predictor-archive-vvrajkumar"
       AWS_S3_REGION          = "ap-south-2"
       AWS_S3_PREFIX          = "gold-predictor/"   (optional, for namespacing)

When the credentials are absent the module silently becomes a no-op,
so local development is unaffected.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

try:
    from botocore.exceptions import ClientError as _BotoCoreClientError
except ImportError:  # botocore not installed yet (e.g. during pip install)
    _BotoCoreClientError = Exception  # type: ignore[assignment,misc]

from .config import CACHE_DIR

# ── Configuration ───────────────────────────────────────────────────

_TRACKED_FILES = (
    "stored_plans.json",
    "accuracy_log.json",
    "latest_prediction.json",
    "residual_corrections.json",
    "prediction_archive.json",
)

# TTL guard: skip cloud sync if last pull was within this many seconds
_SYNC_TTL_SECONDS = 300  # 5 minutes


def _get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# Lazy-init globals
_s3_client = None
_bucket: Optional[str] = None
_prefix: str = ""
_lock = threading.Lock()
_last_sync_time: float = 0.0
# Per-file content hash cache: avoid pushing when content hasn't changed
_file_content_hashes: dict[str, str] = {}


def _init():
    """Lazily initialise the boto3 S3 client and bucket config."""
    global _s3_client, _bucket, _prefix
    if _s3_client is not None:
        return  # already initialised

    _bucket = _get_secret("AWS_S3_BUCKET_NAME")
    if not _bucket:
        _s3_client = False  # sentinel: credentials missing
        return

    aws_key = _get_secret("AWS_ACCESS_KEY_ID")
    aws_secret = _get_secret("AWS_SECRET_ACCESS_KEY")
    region = _get_secret("AWS_S3_REGION", "ap-south-2")
    _prefix = _get_secret("AWS_S3_PREFIX", "gold-predictor/")
    _prefix = (_prefix.rstrip("/") + "/") if _prefix else ""

    try:
        import boto3
        if aws_key and aws_secret:
            _s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=region,
            )
        else:
            # Fall back to default credential chain (IAM role, env, ~/.aws/credentials)
            _s3_client = boto3.client("s3", region_name=region)
        logger.info(f"[cloud_storage] S3 client initialised (bucket={_bucket}, prefix={_prefix})")
    except Exception as e:
        logger.warning(f"[cloud_storage] Failed to initialise S3 client: {e}")
        _s3_client = False  # sentinel


def is_available() -> bool:
    """Return True when S3 credentials are configured and client is ready."""
    _init()
    return _s3_client not in (None, False)


# ── Low-level S3 API ───────────────────────────────────────────────

def _s3_key(filename: str) -> str:
    """Return the full S3 object key for a tracked file."""
    return f"{_prefix}{filename}"


def _pull_all() -> dict[str, str]:
    """Download all tracked file contents from S3.  Returns {filename: content}."""
    result: dict[str, str] = {}
    for filename in _TRACKED_FILES:
        try:
            resp = _s3_client.get_object(Bucket=_bucket, Key=_s3_key(filename))
            content = resp["Body"].read().decode("utf-8")
            result[filename] = content
        except _BotoCoreClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                logger.debug(f"[cloud_storage] {filename} not found in S3")
            else:
                logger.warning(f"[cloud_storage] S3 pull failed for {filename}: {e}")
    return result


def _push_file(filename: str, content: str):
    """Upload a single file to S3."""
    try:
        _s3_client.put_object(
            Bucket=_bucket,
            Key=_s3_key(filename),
            Body=content.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"[cloud_storage] Pushed {filename} to S3 ({len(content)} bytes)")
    except Exception as e:
        logger.warning(f"[cloud_storage] S3 push failed for {filename}: {e}")


# ── Public API ──────────────────────────────────────────────────────

def sync_from_cloud():
    """Pull tracked files from S3 → local cache (runs once on startup).

    Only restores files that don't already exist locally, so an in-progress
    session is never overwritten.  Skips the pull entirely if it was done
    less than 5 minutes ago (TTL guard) to avoid redundant S3 API calls.
    """
    global _last_sync_time
    if not is_available():
        logger.debug("[cloud_storage] S3 credentials not configured – skipping sync")
        return

    # TTL check: skip pull if synced recently
    now = time.monotonic()
    if now - _last_sync_time < _SYNC_TTL_SECONDS:
        logger.debug("[cloud_storage] Sync skipped – within TTL window")
        return

    remote = _pull_all()
    _last_sync_time = time.monotonic()
    restored = 0
    for filename in _TRACKED_FILES:
        local_path = CACHE_DIR / filename
        if local_path.exists():
            logger.debug(f"[cloud_storage] {filename} exists locally – skipping")
            continue  # local is authoritative during a session
        content = remote.get(filename)
        if content:
            local_path.write_text(content, encoding="utf-8")
            restored += 1
            logger.info(f"[cloud_storage] Restored {filename} from S3 ({len(content)} bytes)")
        else:
            logger.warning(f"[cloud_storage] {filename} not found in S3")

    if restored:
        logger.info(f"[cloud_storage] Restored {restored} file(s) from cloud")


def _is_ephemeral_env() -> bool:
    """Detect Streamlit Cloud or other ephemeral environments."""
    return str(CACHE_DIR).replace("\\", "/").startswith("/mount/src/")


def persist(filename: str, data: Any):
    """Write JSON to local file **and** push to S3.

    On ephemeral environments (Streamlit Cloud) the push is synchronous
    so data survives app sleep.  Locally it remains a background push.
    Skips the S3 push when the serialised content hasn't changed since
    the last push (hash-based deduplication).
    """
    content = json.dumps(data, indent=2, default=str)
    local_path = CACHE_DIR / filename
    local_path.write_text(content, encoding="utf-8")

    if not is_available():
        return

    # Hash-based dedup: skip push if content is identical to last push
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    if _file_content_hashes.get(filename) == content_hash:
        logger.debug(f"[cloud_storage] {filename} unchanged – skipping S3 push")
        return
    _file_content_hashes[filename] = content_hash

    if _is_ephemeral_env():
        # Synchronous push – critical on Cloud where daemon threads die on sleep
        _push_file(filename, content)
    else:
        t = threading.Thread(
            target=_push_file, args=(filename, content),
            daemon=True, name=f"s3-push-{filename}",
        )
        t.start()


def load(filename: str) -> Any:
    """Load JSON from local file; if missing, try S3."""
    local_path = CACHE_DIR / filename
    if local_path.exists():
        try:
            return json.loads(local_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not is_available():
        return None

    try:
        resp = _s3_client.get_object(Bucket=_bucket, Key=_s3_key(filename))
        content = resp["Body"].read().decode("utf-8")
        if content:
            local_path.write_text(content, encoding="utf-8")
            try:
                return json.loads(content)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"[cloud_storage] S3 load failed for {filename}: {e}")

    return None
