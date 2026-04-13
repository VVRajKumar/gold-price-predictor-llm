"""
Cloud persistence layer – AWS S3 (primary) with GitHub Gist fallback.

On Streamlit Cloud the filesystem is ephemeral: every deploy / sleep wipes
`data/cache/`.  This module syncs critical JSON files (stored_plans, accuracy_log,
prediction_archive) to a cloud backend so they survive across restarts.

**Priority order:**
  1. AWS S3  – preferred (no file-size limit).  Uses boto3's default credential
     chain (IAM role, instance profile, ~/.aws/credentials) — no explicit
     access keys needed.  Only ``AWS_S3_BUCKET_NAME`` (and optionally
     ``AWS_S3_REGION`` / ``AWS_S3_PREFIX``) are read from Streamlit secrets.
  2. GitHub Gist – fallback when S3 is unavailable or fails.

Setup:
  • S3:  Add AWS_S3_BUCKET_NAME (and optionally AWS_S3_REGION, AWS_S3_PREFIX)
         to Streamlit secrets.  Credentials come from IAM / default chain.
  • Gist: Add GITHUB_GIST_TOKEN, GITHUB_GIST_ID to Streamlit secrets (or .env).

When neither backend is configured the module silently becomes a no-op,
so local development is unaffected.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Optional

import requests
from loguru import logger

try:
    from botocore.exceptions import ClientError as _BotoCoreClientError
except ImportError:  # botocore not installed yet
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


# ── AWS S3 backend ──────────────────────────────────────────────────

_s3_client = None          # None = uninitialised, False = unavailable
_bucket: Optional[str] = None
_prefix: str = ""


def _init_s3():
    """Lazily initialise the boto3 S3 client and bucket config.

    Uses boto3's default credential chain (IAM role, instance profile,
    ~/.aws/credentials, etc.) — no explicit access keys are read.
    Only the bucket name, region, and prefix are configured via secrets.
    """
    global _s3_client, _bucket, _prefix
    if _s3_client is not None:
        return

    _bucket = _get_secret("AWS_S3_BUCKET_NAME")
    if not _bucket:
        _s3_client = False
        return

    region = _get_secret("AWS_S3_REGION", "ap-south-2")
    _prefix = _get_secret("AWS_S3_PREFIX", "gold-predictor/")
    _prefix = (_prefix.rstrip("/") + "/") if _prefix else ""

    try:
        import boto3
        _s3_client = boto3.client("s3", region_name=region)
        logger.info(f"[cloud_storage] S3 client initialised (bucket={_bucket}, prefix={_prefix})")
    except Exception as e:
        logger.warning(f"[cloud_storage] Failed to initialise S3 client: {e}")
        _s3_client = False


def _s3_available() -> bool:
    _init_s3()
    return _s3_client not in (None, False)


def _s3_key(filename: str) -> str:
    return f"{_prefix}{filename}"


def _s3_pull_all() -> dict[str, str]:
    """Download all tracked files from S3."""
    result: dict[str, str] = {}
    for filename in _TRACKED_FILES:
        try:
            resp = _s3_client.get_object(Bucket=_bucket, Key=_s3_key(filename))
            result[filename] = resp["Body"].read().decode("utf-8")
        except _BotoCoreClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                logger.debug(f"[cloud_storage] {filename} not found in S3")
            else:
                logger.warning(f"[cloud_storage] S3 pull failed for {filename}: {e}")
        except Exception as e:
            logger.warning(f"[cloud_storage] S3 pull error for {filename}: {e}")
    return result


def _s3_push_file(filename: str, content: str) -> bool:
    """Upload a single file to S3.  Returns True on success."""
    try:
        _s3_client.put_object(
            Bucket=_bucket,
            Key=_s3_key(filename),
            Body=content.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"[cloud_storage] Pushed {filename} to S3 ({len(content)} bytes)")
        return True
    except Exception as e:
        logger.warning(f"[cloud_storage] S3 push failed for {filename}: {e}")
        return False


def _s3_load_file(filename: str) -> Optional[str]:
    """Download a single file from S3.  Returns content or None."""
    try:
        resp = _s3_client.get_object(Bucket=_bucket, Key=_s3_key(filename))
        return resp["Body"].read().decode("utf-8")
    except Exception as e:
        logger.debug(f"[cloud_storage] S3 load failed for {filename}: {e}")
        return None


# ── GitHub Gist backend (fallback) ─────────────────────────────────

_GIST_API_BASE = "https://api.github.com/gists"
_GIST_TIMEOUT = 15
_gist_token: Optional[str] = None
_gist_id: Optional[str] = None


def _init_gist():
    global _gist_token, _gist_id
    if _gist_token is None:
        _gist_token = _get_secret("GITHUB_GIST_TOKEN")
        _gist_id = _get_secret("GITHUB_GIST_ID")


def _gist_available() -> bool:
    _init_gist()
    return bool(_gist_token and _gist_id)


def _gist_headers() -> dict:
    return {
        "Authorization": f"Bearer {_gist_token}",
        "Accept": "application/vnd.github+json",
    }


def _gist_pull_all() -> dict[str, str]:
    """Download all file contents from the Gist."""
    try:
        resp = requests.get(
            f"{_GIST_API_BASE}/{_gist_id}",
            headers=_gist_headers(),
            timeout=_GIST_TIMEOUT,
        )
        resp.raise_for_status()
        files = resp.json().get("files", {})
        return {name: info.get("content", "") for name, info in files.items()}
    except Exception as e:
        logger.warning(f"[cloud_storage] Gist pull failed: {e}")
        return {}


def _gist_push_file(filename: str, content: str) -> bool:
    """Update a single file in the Gist.  Returns True on success."""
    try:
        resp = requests.patch(
            f"{_GIST_API_BASE}/{_gist_id}",
            headers=_gist_headers(),
            json={"files": {filename: {"content": content}}},
            timeout=_GIST_TIMEOUT,
        )
        resp.raise_for_status()
        logger.info(f"[cloud_storage] Pushed {filename} to Gist ({len(content)} bytes)")
        return True
    except Exception as e:
        logger.warning(f"[cloud_storage] Gist push failed for {filename}: {e}")
        return False


def _gist_load_file(filename: str) -> Optional[str]:
    """Download a single file from the Gist."""
    remote = _gist_pull_all()
    return remote.get(filename) or None


# ── Unified backend selection ──────────────────────────────────────

_lock = threading.Lock()
_last_sync_time: float = 0.0
_file_content_hashes: dict[str, str] = {}


def is_available() -> bool:
    """Return True when at least one cloud backend is configured."""
    return _s3_available() or _gist_available()


def _is_ephemeral_env() -> bool:
    """Detect Streamlit Cloud or other ephemeral environments."""
    return str(CACHE_DIR).replace("\\", "/").startswith("/mount/src/")


# ── Public API ──────────────────────────────────────────────────────

def sync_from_cloud():
    """Pull tracked files from cloud → local cache on startup.

    Tries AWS S3 first; if S3 is unavailable or returns nothing for a file,
    falls back to GitHub Gist.

    On ephemeral environments (Streamlit Cloud) **always overwrites** local
    files with the cloud version so that manual S3 edits are reflected
    immediately on the next cold start.  On local dev, only restores files
    that don't already exist.

    Skips entirely if synced within the TTL window.
    """
    global _last_sync_time
    if not is_available():
        logger.debug("[cloud_storage] No cloud backend configured – skipping sync")
        return

    now = time.monotonic()
    if now - _last_sync_time < _SYNC_TTL_SECONDS:
        logger.debug("[cloud_storage] Sync skipped – within TTL window")
        return

    ephemeral = _is_ephemeral_env()

    # Pull from available backends
    s3_remote: dict[str, str] = {}
    gist_remote: dict[str, str] = {}

    if _s3_available():
        s3_remote = _s3_pull_all()

    # Only pull Gist if S3 didn't cover all files
    need_gist = any(
        (not (CACHE_DIR / f).exists()) and f not in s3_remote
        for f in _TRACKED_FILES
    )
    if need_gist and _gist_available():
        gist_remote = _gist_pull_all()

    _last_sync_time = time.monotonic()
    restored = 0
    for filename in _TRACKED_FILES:
        local_path = CACHE_DIR / filename

        # On ephemeral envs, always overwrite local with cloud data so
        # manual S3 edits are picked up on every cold start.
        if local_path.exists() and not ephemeral:
            logger.debug(f"[cloud_storage] {filename} exists locally – skipping")
            continue

        # Prefer S3 content, fall back to Gist
        content = s3_remote.get(filename) or gist_remote.get(filename)
        if content:
            local_path.write_text(content, encoding="utf-8")
            restored += 1
            action = "Refreshed" if ephemeral else "Restored"
            logger.info(f"[cloud_storage] {action} {filename} from cloud ({len(content)} bytes)")
        elif not local_path.exists():
            logger.warning(f"[cloud_storage] {filename} not found in any cloud backend")

    if restored:
        logger.info(f"[cloud_storage] {restored} file(s) synced from cloud")


def persist(filename: str, data: Any):
    """Write JSON to local file **and** push to the best available cloud backend.

    Tries AWS S3 first; falls back to GitHub Gist on failure.
    On ephemeral environments the push is synchronous; locally it's threaded.
    Hash-based dedup avoids redundant pushes.
    """
    content = json.dumps(data, indent=2, default=str)
    local_path = CACHE_DIR / filename
    local_path.write_text(content, encoding="utf-8")

    if not is_available():
        return

    # Hash-based dedup
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    if _file_content_hashes.get(filename) == content_hash:
        logger.debug(f"[cloud_storage] {filename} unchanged – skipping cloud push")
        return
    _file_content_hashes[filename] = content_hash

    def _do_push():
        pushed = False
        if _s3_available():
            pushed = _s3_push_file(filename, content)
        # Always push to Gist too (when available) to keep it in sync.
        # Previously Gist was only updated on S3 failure, which left stale
        # data in the Gist that could be pulled on subsequent cold starts.
        if _gist_available():
            _gist_push_file(filename, content)

    if _is_ephemeral_env():
        _do_push()
    else:
        t = threading.Thread(
            target=_do_push, daemon=True,
            name=f"cloud-push-{filename}",
        )
        t.start()


def load(filename: str) -> Any:
    """Load JSON from local file; if missing, try S3 then Gist."""
    local_path = CACHE_DIR / filename
    if local_path.exists():
        try:
            return json.loads(local_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not is_available():
        return None

    content: Optional[str] = None

    # Try S3 first
    if _s3_available():
        content = _s3_load_file(filename)

    # Fall back to Gist
    if not content and _gist_available():
        content = _gist_load_file(filename)

    if content:
        local_path.write_text(content, encoding="utf-8")
        try:
            return json.loads(content)
        except Exception:
            pass

    return None
