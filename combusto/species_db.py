"""Species database loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import Species

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "species.json"


def load_species_db(path: Optional[Path] = None) -> Dict[str, Species]:
    """Load species database from JSON file. Returns {name: Species}."""
    path = path or _DEFAULT_DB_PATH
    with open(path, "r") as f:
        data = json.load(f)

    db: Dict[str, Species] = {}
    for entry in data["species"]:
        sp = Species(
            name=entry["name"],
            formula=entry["formula"],
            phase=entry.get("phase", "gas"),
            category=entry.get("category", "small"),
            hf_298=entry.get("hf_298", 0.0),
            t_mid=entry.get("t_mid", 1000.0),
            t_low=entry.get("t_low", 200.0),
            t_high=entry.get("t_high", 6000.0),
            thermo_low=entry.get("thermo_low", [0.0] * 7),
            thermo_high=entry.get("thermo_high", [0.0] * 7),
        )
        db[sp.name] = sp
    return db


def get_species_by_category(db: Dict[str, Species], category: str) -> List[Species]:
    """Filter species by category (small, pah, lump, metal_oxide)."""
    return [sp for sp in db.values() if sp.category == category]


def get_gas_species(db: Dict[str, Species]) -> List[Species]:
    """Return only gas-phase species."""
    return [sp for sp in db.values() if sp.phase == "gas"]


def get_condensed_species(db: Dict[str, Species]) -> List[Species]:
    """Return only condensed-phase species."""
    return [sp for sp in db.values() if sp.phase == "condensed"]
