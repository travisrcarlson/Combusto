"""Load preset propellant formulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import Formulation, PropellantComponent

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "propellants.json"

STANDARD_EXPLOSIVES = [
    Formulation(
        name="TNT",
        description="Trinitrotoluene standard base",
        components=[PropellantComponent(name="TNT", formula="C7H5N3O6", mass_fraction=1.0, hf_298=-66.9)]
    ),
    Formulation(
        name="RDX",
        description="Cyclotrimethylenetrinitramine",
        components=[PropellantComponent(name="RDX", formula="C3H6N6O6", mass_fraction=1.0, hf_298=92.6)]
    ),
    Formulation(
        name="HMX",
        description="Cyclotetramethylenetetranitramine",
        components=[PropellantComponent(name="HMX", formula="C4H8N8O8", mass_fraction=1.0, hf_298=74.4)]
    ),
    Formulation(
        name="PETN",
        description="Pentaerythritol tetranitrate",
        components=[PropellantComponent(name="PETN", formula="C5H8N4O12", mass_fraction=1.0, hf_298=-538.5)]
    ),
    Formulation(
        name="TATB",
        description="Triaminotrinitrobenzene (insensitive)",
        components=[PropellantComponent(name="TATB", formula="C6H6N6O6", mass_fraction=1.0, hf_298=-154.0)]
    ),
    Formulation(
        name="Composition B",
        description="Standard 60/40 RDX/TNT mixture",
        components=[
            PropellantComponent(name="RDX", formula="C3H6N6O6", mass_fraction=0.60, hf_298=92.6),
            PropellantComponent(name="TNT", formula="C7H5N3O6", mass_fraction=0.40, hf_298=-66.9)
        ]
    ),
    Formulation(
        name="Composition C4",
        description="Plastic explosive (estimated binder via PIB)",
        components=[
            PropellantComponent(name="RDX", formula="C3H6N6O6", mass_fraction=0.91, hf_298=92.6),
            PropellantComponent(name="PIB (Binder)", formula="C4H8", mass_fraction=0.09, hf_298=-17.0)
        ]
    ),
    Formulation(
        name="PBX-9404",
        description="HMX-based plastic bonded explosive",
        components=[
            PropellantComponent(name="HMX", formula="C4H8N8O8", mass_fraction=0.94, hf_298=74.4),
            PropellantComponent(name="Nitrocellulose", formula="C6H7.3N2.7O9.9", mass_fraction=0.03, hf_298=-700.0),
            PropellantComponent(name="CEF (Plasticizer)", formula="C6H12Cl3O4P", mass_fraction=0.03, hf_298=-1100.0)
        ]
    ),
    Formulation(
        name="Octol 70/30",
        description="HMX/TNT melt-cast",
        components=[
            PropellantComponent(name="HMX", formula="C4H8N8O8", mass_fraction=0.70, hf_298=74.4),
            PropellantComponent(name="TNT", formula="C7H5N3O6", mass_fraction=0.30, hf_298=-66.9)
        ]
    ),
    Formulation(
        name="TATP",
        description="Triacetone Triperoxide",
        components=[PropellantComponent(name="TATP", formula="C9H18O6", mass_fraction=1.0, hf_298=151.4)]
    )
]

def get_all_library_templates() -> List[Formulation]:
    """Return all preset propellants and standard explosives as a combined library."""
    return load_presets() + STANDARD_EXPLOSIVES

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
