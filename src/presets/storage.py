"""
User Portfolio Persistence Storage Module with Multi-Tenant Isolation.

Handles saving, loading, updating, deleting, and exporting/importing custom
user-defined portfolios with complete per-user isolation based on user ID or email.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LEGACY_STORAGE_FILE = ROOT_DIR / "user_portfolios.json"
PORTFOLIOS_DIR = ROOT_DIR / "data" / "portfolios"
STORAGE_FILE = LEGACY_STORAGE_FILE


def _sanitize_user_id(user_id: Optional[str]) -> str:
    """Sanitize user ID or email to create a safe filesystem filename."""
    if not user_id or user_id.strip() in ("", "guest", "default"):
        return "guest"
    clean = user_id.strip().lower()
    clean = clean.replace("@", "_at_")
    clean = re.sub(r"[^a-z0-9_.-]", "_", clean)
    return clean[:80]


def get_storage_path(user_id: Optional[str] = None) -> Path:
    """Return the absolute path to the user's isolated portfolio storage file."""
    PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _sanitize_user_id(user_id)
    target_file = PORTFOLIOS_DIR / f"{slug}.json"
    
    # Auto-migrate legacy file to guest if needed
    if slug == "guest" and not target_file.exists() and LEGACY_STORAGE_FILE.exists():
        try:
            target_file.write_text(LEGACY_STORAGE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    return target_file


def load_saved_portfolios(user_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load all saved custom portfolios for the specified user."""
    storage_path = get_storage_path(user_id)
    if not storage_path.exists():
        return {}
    try:
        with open(storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        return {}


def save_custom_portfolio(
    name: str,
    tickers: List[str],
    weights: Dict[str, float],
    description: str = "",
    portfolio_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Save or update a custom portfolio in the active user's storage.
    
    Returns the unique portfolio ID.
    """
    portfolios = load_saved_portfolios(user_id)
    clean_name = name.strip()
    if not clean_name:
        clean_name = f"Portafolio {len(portfolios) + 1}"

    now_iso = datetime.datetime.now().isoformat()
    
    # Generate slug ID if not provided
    if not portfolio_id:
        slug = clean_name.lower().replace(" ", "_")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        portfolio_id = slug if slug and slug not in portfolios else f"custom_{int(datetime.datetime.now().timestamp())}"

    # Normalize weights to sum to 1.0
    total_w = sum(weights.values())
    norm_weights = {}
    if total_w > 1e-6:
        norm_weights = {k: round(float(v / total_w), 6) for k, v in weights.items()}
    else:
        n = len(tickers)
        norm_weights = {t: round(1.0 / n, 6) for t in tickers}

    clean_tickers = [t.strip().upper() for t in tickers if t.strip()]

    portfolio_data = {
        "id": portfolio_id,
        "name": clean_name,
        "description": description.strip(),
        "tickers": clean_tickers,
        "weights": norm_weights,
        "updated_at": now_iso,
        "created_at": portfolios.get(portfolio_id, {}).get("created_at", now_iso),
    }

    portfolios[portfolio_id] = portfolio_data

    storage_path = get_storage_path(user_id)
    with open(storage_path, "w", encoding="utf-8") as f:
        json.dump(portfolios, f, indent=2, ensure_ascii=False)

    return portfolio_id


def delete_custom_portfolio(portfolio_id: str, user_id: Optional[str] = None) -> bool:
    """Delete a custom portfolio from the active user's storage."""
    portfolios = load_saved_portfolios(user_id)
    if portfolio_id in portfolios:
        del portfolios[portfolio_id]
        storage_path = get_storage_path(user_id)
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(portfolios, f, indent=2, ensure_ascii=False)
        return True
    return False


def export_portfolios_json(user_id: Optional[str] = None) -> str:
    """Export all saved portfolios for the active user as a JSON string."""
    portfolios = load_saved_portfolios(user_id)
    return json.dumps(portfolios, indent=2, ensure_ascii=False)


def import_portfolios_json(json_str: str, user_id: Optional[str] = None) -> int:
    """
    Import portfolios from a JSON string into the active user's storage.
    Returns the number of portfolios successfully imported.
    """
    try:
        new_data = json.loads(json_str)
        if not isinstance(new_data, dict):
            return 0
        existing = load_saved_portfolios(user_id)
        count = 0
        for p_id, p_val in new_data.items():
            if isinstance(p_val, dict) and "tickers" in p_val and "weights" in p_val:
                existing[p_id] = p_val
                count += 1
        storage_path = get_storage_path(user_id)
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        return count
    except Exception:
        return 0
