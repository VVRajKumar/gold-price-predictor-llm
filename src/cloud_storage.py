"""
Cloud persistence layer – GitHub Gist backend.

On Streamlit Cloud the filesystem is ephemeral: every deploy / sleep wipes
`data/cache/`.  This module syncs critical JSON files (stored_plans, accuracy_log)
to a **private GitHub Gist** so they survive across restarts.

Setup (one-time):
  1. Create a GitHub Personal Access Token with *gist* scope.
  2. Create a private Gist containing two empty files:
       stored_plans.json  →  []
       accuracy_log.json  →  []
  3. Add to Streamlit secrets (or .env):
       GITHUB_GIST_TOKEN = "ghp_..."
       GITHUB_GIST_ID    = "abc123..."

When the token/id are absent the module silently becomes a no-op,
so local development is unaffected.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

import requests
from loguru import logger

from .config import CACHE_DIR

# ── Configuration ───────────────────────────────────────────────────

_TRACKED_FILES = ("stored_plans.json", "accuracy_log.json", "latest_prediction.json", "residual_corrections.json")

_API_BASE = "https://api.github.com/gists"
_TIMEOUT = 15  # seconds


def _get_secret(key: str) -> str:
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""


# Lazy-init globals
_gist_token: Optional[str] = None
_gist_id: Optional[str] = None
_lock = threading.Lock()


def _init():
    global _gist_token, _gist_id
    if _gist_token is None:
        _gist_token = _get_secret("GITHUB_GIST_TOKEN")
        _gist_id = _get_secret("GITHUB_GIST_ID")


def is_available() -> bool:
    """Return True when Gist credentials are configured."""
    _init()
    return bool(_gist_token and _gist_id)


# ── Low-level Gist API ─────────────────────────────────────────────

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_gist_token}",
        "Accept": "application/vnd.github+json",
    }


def _pull_gist() -> dict[str, str]:
    """Download all file contents from the Gist.  Returns {filename: content}."""
    try:
        resp = requests.get(f"{_API_BASE}/{_gist_id}", headers=_headers(), timeout=_TIMEOUT)
        resp.raise_for_status()
        files = resp.json().get("files", {})
        return {name: info.get("content", "") for name, info in files.items()}
    except Exception as e:
        logger.warning(f"[cloud_storage] Gist pull failed: {e}")
        return {}


def _push_file(filename: str, content: str):
    """Update a single file in the Gist."""
    try:
        resp = requests.patch(
            f"{_API_BASE}/{_gist_id}",
            headers=_headers(),
            json={"files": {filename: {"content": content}}},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        logger.info(f"[cloud_storage] Pushed {filename} to Gist ({len(content)} bytes)")
    except Exception as e:
        logger.warning(f"[cloud_storage] Gist push failed for {filename}: {e}")


# ── Public API ──────────────────────────────────────────────────────

def sync_from_cloud():
    """Pull tracked files from Gist → local cache (runs once on startup).

    Only restores files that don't already exist locally, so an in-progress
    session is never overwritten.
    """
    if not is_available():
        logger.debug("[cloud_storage] Gist credentials not configured – skipping sync")
        return

    remote = _pull_gist()
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
            logger.info(f"[cloud_storage] Restored {filename} from Gist ({len(content)} bytes)")
        else:
            logger.warning(f"[cloud_storage] {filename} not found in Gist")

    if restored:
        logger.info(f"[cloud_storage] Restored {restored} file(s) from cloud")


def _is_ephemeral_env() -> bool:
    """Detect Streamlit Cloud or other ephemeral environments."""
    return str(CACHE_DIR).replace("\\", "/").startswith("/mount/src/")


def persist(filename: str, data: Any):
    """Write JSON to local file **and** push to Gist.

    On ephemeral environments (Streamlit Cloud) the push is synchronous
    so data survives app sleep.  Locally it remains a background push.
    """
    content = json.dumps(data, indent=2, default=str)
    local_path = CACHE_DIR / filename
    local_path.write_text(content, encoding="utf-8")

    if not is_available():
        return

    if _is_ephemeral_env():
        # Synchronous push – critical on Cloud where daemon threads die on sleep
        _push_file(filename, content)
    else:
        t = threading.Thread(
            target=_push_file, args=(filename, content),
            daemon=True, name=f"gist-push-{filename}",
        )
        t.start()


def load(filename: str) -> Any:
    """Load JSON from local file; if missing, try Gist."""
    local_path = CACHE_DIR / filename
    if local_path.exists():
        try:
            return json.loads(local_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not is_available():
        return None

    remote = _pull_gist()
    content = remote.get(filename)
    if content:
        local_path.write_text(content, encoding="utf-8")
        try:
            return json.loads(content)
        except Exception:
            pass
    return None
