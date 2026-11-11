"""Chemistry utilities — element balance, oxygen balance, equivalence ratio."""

from __future__ import annotations

from typing import Dict

from .models import ATOMIC_MASSES, Formulation


# Stoichiometric O-atom equivalents per mole of electropositive/fuel element
# (valence / 2, corresponding to full oxidation to CO2, H2O, SO2, Al2O3, MgO, K2O, etc.)
_REDUCING_O_EQUIV: Dict[str, float] = {
    "C": 2.0,
    "H": 0.5,
    "S": 2.0,
    "Al": 1.5,
    "B": 1.5,
    "Li": 0.5,
    "Na": 0.5,
    "K": 0.5,
    "Be": 1.0,
    "Mg": 1.0,
    "Ca": 1.0,
    "Ba": 1.0,
    "Zn": 1.0,
    "Cu": 1.0,
    "Pb": 1.0,
    "Fe": 1.5,
    "Bi": 1.5,
    "Si": 2.0,
    "Ti": 2.0,
    "Sn": 2.0,
    "P": 2.5,
    "W": 3.0,
}

# Oxidizing O-atom equivalents per mole of electronegative/oxidizer element
# O (-2) = 1.0 O-equiv; Cl/F (-1, forming HCl/HF or metal halides) = 0.5 O-equiv
_OXIDIZING_O_EQUIV: Dict[str, float] = {
    "O": 1.0,
    "Cl": 0.5,
    "F": 0.5,
}


def _valence_o_equivalents(formulation: Formulation) -> tuple[float, float]:
    """Return (reducing_o_needed, oxidizing_o_available) in mol-O equivalents per gram."""
    elem = formulation.elemental_composition()
    v_fuel = sum(moles * _REDUCING_O_EQUIV.get(el, 0.0) for el, moles in elem.items())
    v_ox = sum(moles * _OXIDIZING_O_EQUIV.get(el, 0.0) for el, moles in elem.items())
    return v_fuel, v_ox


def oxygen_balance(formulation: Formulation) -> float:
    """Compute stoichiometric Oxygen Balance (OB%) for a propellant formulation.

    Accounts for C->CO2, H->H2O/HX, S->SO2, metals/alkalis->oxides/halides, and
    halogen oxidizers (Cl, F) per Meyer 'Explosives' / ICT / NATO STANAG convention.
    For example, Ammonium Perchlorate (NH4ClO4 -> 0.5 N2 + HCl + 1.5 H2O + 1.25 O2)
    yields exact textbook OB% = +34.04%, and inert oxides (Al2O3, MgO, Fe2O3, CaCO3) yield 0.0%.
    """
    o_needed, o_avail = _valence_o_equivalents(formulation)
    o_excess = o_avail - o_needed  # net mol O equivalents per gram propellant
    return o_excess * ATOMIC_MASSES["O"] * 100.0


def equivalence_ratio(formulation: Formulation) -> float:
    """Compute fuel-oxidizer equivalence ratio (phi = V+ / V-).

    Follows NASA CEA (RP-1311) generalized oxidizing/reducing valence ratio:
    phi > 1 means fuel-rich, phi = 1 stoichiometric, phi < 1 oxidizer-rich.
    """
    v_fuel, v_ox = _valence_o_equivalents(formulation)
    if v_ox <= 0.0:
        return float("inf") if v_fuel > 0.0 else 0.0
    return v_fuel / v_ox


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
