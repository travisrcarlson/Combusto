"""Species database loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import Species

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "species.json"


# Hirschfelder-Roseveare / Corner / US Army BLAKE (ARBRL-TR-02411) molar covolumes (cm3/mol)
DEFAULT_MOLAR_COVOLUMES_CM3_MOL: Dict[str, float] = {
    "H2": 15.2,
    "H": 10.5,
    "H2O": 22.1,
    "OH": 18.5,
    "CO": 27.9,
    "N2": 30.5,
    "NO": 25.8,
    "NO2": 32.2,
    "N2O": 34.8,
    "O2": 27.2,
    "O": 14.0,
    "CO2": 36.9,
    "CH4": 37.7,
    "NH3": 24.8,
    "HCN": 34.0,
    "C2H2": 36.5,
    "C2H4": 41.2,
    "HCl": 29.8,
    "HF": 20.4,
    "Cl2": 41.0,
    "SO2": 38.4,
    "H2S": 31.5,
    "COS": 41.0,
    "KOH": 32.0,
    "KCl": 39.0,
    "Ar": 26.2,
}


def load_species_db(path: Optional[Path] = None) -> Dict[str, Species]:
    """Load species database from JSON file. Returns {name: Species}."""
    path = path or _DEFAULT_DB_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    db: Dict[str, Species] = {}
    for entry in data["species"]:
        name = entry["name"]
        phase = entry.get("phase", "gas")
        default_b = DEFAULT_MOLAR_COVOLUMES_CM3_MOL.get(name, 35.0 if phase == "gas" else 0.0)
        sp = Species(
            name=name,
            formula=entry["formula"],
            phase=phase,
            category=entry.get("category", "small"),
            hf_298=entry.get("hf_298", 0.0),
            t_mid=entry.get("t_mid", 1000.0),
            t_low=entry.get("t_low", 200.0),
            t_high=entry.get("t_high", 6000.0),
            thermo_low=entry.get("thermo_low", [0.0] * 7),
            thermo_high=entry.get("thermo_high", [0.0] * 7),
            cas=entry.get("cas", ""),
            source=entry.get("source", "NASA Glenn TP-2002-211556 / Burcat"),
            molar_covolume_cm3_mol=entry.get("molar_covolume_cm3_mol", default_b),
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
