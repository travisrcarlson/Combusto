"""Chemistry utilities — element balance, oxygen balance, equivalence ratio."""

from __future__ import annotations

from typing import Dict

from .models import ATOMIC_MASSES, Formulation


def oxygen_balance(formulation: Formulation) -> float:
    """Compute oxygen balance (OB%) for a propellant formulation.

    OB% = -1600 * (2*C + H/2 + 2*S + M_metal_valence/2 - O) / MW_formula
    Simplified version assuming full oxidation to CO2, H2O.
    Returns OB% per gram of propellant (using elemental moles per gram).
    """
    elem = formulation.elemental_composition()
    c = elem.get("C", 0.0)
    h = elem.get("H", 0.0)
    o = elem.get("O", 0.0)
    s = elem.get("S", 0.0)
    al = elem.get("Al", 0.0)
    # Oxygen needed for full oxidation: C->CO2 (2O per C), H->H2O (0.5O per H),
    # S->SO2 (2O per S), Al->Al2O3 (1.5O per Al)
    o_needed = 2.0 * c + 0.5 * h + 2.0 * s + 1.5 * al
    o_excess = o - o_needed  # mol O per gram propellant
    # Convert to mass: o_excess * 16 g/mol, per 1 g propellant => * 100 for %
    ob_pct = o_excess * ATOMIC_MASSES["O"] * 100.0
    return ob_pct


def equivalence_ratio(formulation: Formulation) -> float:
    """Compute fuel-oxidizer equivalence ratio (phi).

    phi = (fuel/oxidizer)_actual / (fuel/oxidizer)_stoichiometric
    For a propellant (self-oxidizing), phi = O_needed / O_available.
    phi > 1 means fuel-rich, phi < 1 means oxygen-rich.
    """
    elem = formulation.elemental_composition()
    c = elem.get("C", 0.0)
    h = elem.get("H", 0.0)
    o = elem.get("O", 0.0)
    s = elem.get("S", 0.0)
    al = elem.get("Al", 0.0)
    o_needed = 2.0 * c + 0.5 * h + 2.0 * s + 1.5 * al
    if o == 0:
        return float("inf")
    return o_needed / o


def element_balance_check(
    formulation: Formulation,
    product_moles: Dict[str, "Species"],
    species_amounts: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Check element conservation between reactants and products.

    Returns dict: {element: {"input": moles_in, "output": moles_out, "error_pct": ...}}
    """
    from .models import parse_formula

    input_elem = formulation.elemental_composition()
    output_elem: Dict[str, float] = {}
    for sp_name, moles in species_amounts.items():
        if sp_name in product_moles:
            sp = product_moles[sp_name]
            for elem, count in sp.composition.items():
                output_elem[elem] = output_elem.get(elem, 0.0) + moles * count

    result = {}
    all_elements = set(input_elem.keys()) | set(output_elem.keys())
    for elem in sorted(all_elements):
        inp = input_elem.get(elem, 0.0)
        out = output_elem.get(elem, 0.0)
        err = abs(inp - out) / max(inp, 1e-30) * 100.0
        result[elem] = {"input": inp, "output": out, "error_pct": err}
    return result
