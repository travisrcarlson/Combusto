"""Load preset propellant formulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import Formulation, PropellantComponent

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "propellants.json"


def load_presets(path: Path | None = None) -> List[Formulation]:
    """Load preset formulations from JSON."""
    path = path or _DEFAULT_PATH
    with open(path, "r") as f:
        data = json.load(f)

    formulations = []
    for entry in data["formulations"]:
        components = [
            PropellantComponent(
                name=c["name"],
                formula=c["formula"],
                mass_fraction=c["mass_fraction"],
                hf_298=c.get("hf_298", 0.0),
            )
            for c in entry["components"]
        ]
        formulations.append(
            Formulation(
                name=entry["name"],
                components=components,
                description=entry.get("description", ""),
            )
        )
    return formulations


def get_preset_names(path: Path | None = None) -> List[str]:
    """Return list of preset formulation names."""
    return [f.name for f in load_presets(path)]


def formulation_to_dict(formulation: Formulation) -> dict:
    """Convert a Formulation to a JSON-serializable dict."""
    return {
        "name": formulation.name,
        "description": formulation.description,
        "components": [
            {
                "name": c.name,
                "formula": c.formula,
                "mass_fraction": c.mass_fraction,
                "hf_298": c.hf_298,
            }
            for c in formulation.components
        ],
    }


def formulation_from_dict(data: dict) -> Formulation:
    """Create a Formulation from a dict."""
    components = [
        PropellantComponent(
            name=c["name"],
            formula=c["formula"],
            mass_fraction=c["mass_fraction"],
            hf_298=c.get("hf_298", 0.0),
        )
        for c in data.get("components", [])
    ]
    return Formulation(
        name=data["name"],
        components=components,
        description=data.get("description", ""),
    )


def save_formulations(formulations: List[Formulation], path: Path | str) -> None:
    """Save a list of formulations to a JSON file."""
    data = {
        "formulations": [formulation_to_dict(f) for f in formulations],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_formulations(path: Path | str) -> List[Formulation]:
    """Load formulations from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return [formulation_from_dict(entry) for entry in data["formulations"]]
