"""
User Portfolio Persistence Storage Module.

Handles saving, loading, updating, deleting, and exporting/importing custom
user-defined portfolios to persistent local JSON storage.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

STORAGE_FILE = Path(__file__).resolve().parent.parent.parent / "user_portfolios.json"


def get_storage_path() -> Path:
    """Return the absolute path to the user portfolios storage file."""
    return STORAGE_FILE


def load_saved_portfolios() -> Dict[str, Dict[str, Any]]:
    """Load all saved custom portfolios from local JSON storage."""
    if not STORAGE_FILE.exists():
        return {}
    try:
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
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
) -> str:
    """
    Save or update a custom portfolio in local persistent JSON storage.
    
    Returns the unique portfolio ID.
    """
    portfolios = load_saved_portfolios()
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

    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolios, f, indent=2, ensure_ascii=False)

    return portfolio_id


def delete_custom_portfolio(portfolio_id: str) -> bool:
    """Delete a custom portfolio from local JSON storage."""
    portfolios = load_saved_portfolios()
    if portfolio_id in portfolios:
        del portfolios[portfolio_id]
        with open(STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(portfolios, f, indent=2, ensure_ascii=False)
        return True
    return False


def export_portfolios_json() -> str:
    """Export all saved portfolios as a JSON string."""
    portfolios = load_saved_portfolios()
    return json.dumps(portfolios, indent=2, ensure_ascii=False)


def import_portfolios_json(json_str: str) -> int:
    """
    Import portfolios from a JSON string, merging with existing portfolios.
    Returns the number of portfolios successfully imported.
    """
    try:
        new_data = json.loads(json_str)
        if not isinstance(new_data, dict):
            return 0
        existing = load_saved_portfolios()
        count = 0
        for p_id, p_val in new_data.items():
            if isinstance(p_val, dict) and "tickers" in p_val and "weights" in p_val:
                existing[p_id] = p_val
                count += 1
        with open(STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        return count
    except Exception:
        return 0
