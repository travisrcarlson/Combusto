"""PAH / soot carbon allocation rules.

Before running thermodynamic equilibrium on small molecules, this module
reserves a fraction of the available carbon for PAH and lumped soot species
based on the equivalence ratio and a user-configurable allocation model.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .models import PAHAllocationConfig, Species, parse_formula


# ---------------------------------------------------------------------------
# Built-in allocation presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, PAHAllocationConfig] = {
    "none": PAHAllocationConfig(
        model_name="none",
        base_fraction=0.0,
        phi_exponent=1.0,
        max_fraction=0.0,
    ),
    "conservative": PAHAllocationConfig(
        model_name="conservative",
        base_fraction=0.02,
        phi_exponent=1.5,
        max_fraction=0.15,
        pah_distribution={
            "C6H6": 0.40, "C7H8": 0.20, "C10H8": 0.20,
            "C14H10": 0.10, "C16H10": 0.07, "C24H12": 0.03,
        },
    ),
    "moderate": PAHAllocationConfig(
        model_name="moderate",
        base_fraction=0.05,
        phi_exponent=2.0,
        max_fraction=0.35,
        pah_distribution={
            "C6H6": 0.30, "C7H8": 0.15, "C10H8": 0.25,
            "C14H10": 0.15, "C16H10": 0.10, "C24H12": 0.05,
        },
    ),
    "aggressive": PAHAllocationConfig(
        model_name="aggressive",
        base_fraction=0.10,
        phi_exponent=2.5,
        max_fraction=0.50,
        pah_distribution={
            "C6H6": 0.15, "C7H8": 0.10, "C10H8": 0.20,
            "C14H10": 0.20, "C16H10": 0.20, "C24H12": 0.15,
        },
    ),
}


def carbon_fraction_to_pahs(phi: float, config: PAHAllocationConfig) -> float:
    """Compute fraction of total carbon allocated to PAH species.

    f_PAH = min(base * (phi - 1)^exponent, max_fraction)   for phi > 1
    f_PAH = 0                                                for phi <= 1
    """
    if phi <= 1.0 or config.max_fraction <= 0:
        return 0.0
    f = config.base_fraction * (phi - 1.0) ** config.phi_exponent
    return min(f, config.max_fraction)


def allocate_pahs(
    total_c_moles: float,
    total_h_moles: float,
    phi: float,
    config: PAHAllocationConfig,
    pah_species: Dict[str, Species],
) -> Tuple[Dict[str, float], float, float]:
    """Allocate carbon (and associated hydrogen) to PAH species.

    Returns:
        pah_moles: {species_name: moles}
        remaining_c: moles of C left for small-molecule equilibrium
        remaining_h: moles of H left for small-molecule equilibrium
    """
    f_c = carbon_fraction_to_pahs(phi, config)
    c_for_pahs = total_c_moles * f_c

    if c_for_pahs <= 0:
        return {}, total_c_moles, total_h_moles

    # Normalize distribution weights
    dist = config.pah_distribution
    total_weight = sum(dist.values())
    if total_weight <= 0:
        return {}, total_c_moles, total_h_moles

    norm_dist = {k: v / total_weight for k, v in dist.items()}

    pah_moles: Dict[str, float] = {}
    c_used = 0.0
    h_used = 0.0

    for formula_key, weight in norm_dist.items():
        # Find species by formula match
        sp = None
        for s in pah_species.values():
            if s.formula == formula_key or s.name == formula_key:
                sp = s
                break
        if sp is None:
            continue

        comp = parse_formula(sp.formula)
        c_per_mol = comp.get("C", 0.0)
        h_per_mol = comp.get("H", 0.0)

        if c_per_mol <= 0:
            continue

        c_alloc = c_for_pahs * weight
        moles = c_alloc / c_per_mol
        h_needed = moles * h_per_mol

        pah_moles[sp.name] = moles
        c_used += c_alloc
        h_used += h_needed

    # Clamp hydrogen usage (can't use more H than available)
    if h_used > total_h_moles * 0.9:
        scale = (total_h_moles * 0.9) / h_used
        for k in pah_moles:
            pah_moles[k] *= scale
        c_used *= scale
        h_used *= scale

    remaining_c = total_c_moles - c_used
    remaining_h = total_h_moles - h_used
    return pah_moles, max(remaining_c, 0.0), max(remaining_h, 0.0)
