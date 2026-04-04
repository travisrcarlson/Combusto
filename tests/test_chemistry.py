"""Tests for chemistry module — element balance, oxygen balance, equivalence ratio."""

import pytest
from combusto.models import Formulation, PropellantComponent, parse_formula, molecular_weight
from combusto.chemistry import oxygen_balance, equivalence_ratio


def test_parse_formula_simple():
    assert parse_formula("H2O") == {"H": 2.0, "O": 1.0}
    assert parse_formula("CO2") == {"C": 1.0, "O": 2.0}
    assert parse_formula("N2") == {"N": 2.0}


def test_parse_formula_complex():
    result = parse_formula("C6H7N3O9")
    assert result["C"] == 6.0
    assert result["H"] == 7.0
    assert result["N"] == 3.0
    assert result["O"] == 9.0


def test_parse_formula_parenthetical():
    result = parse_formula("Ca(NO3)2")
    assert result["Ca"] == 1.0
    assert result["N"] == 2.0
    assert result["O"] == 6.0


def test_parse_formula_fractional():
    result = parse_formula("C6H7.3N2.7O9.9")
    assert abs(result["H"] - 7.3) < 1e-6
    assert abs(result["N"] - 2.7) < 1e-6


def test_molecular_weight():
    # Water: ~18.015
    mw = molecular_weight("H2O")
    assert abs(mw - 18.015) < 0.01

    # CO2: ~44.01
    mw = molecular_weight("CO2")
    assert abs(mw - 44.009) < 0.01


def test_oxygen_balance_ng():
    """Nitroglycerin (C3H5N3O9) has a positive oxygen balance (+3.5%)."""
    f = Formulation(
        name="NG pure",
        components=[PropellantComponent("NG", "C3H5N3O9", 1.0, -370.0)],
    )
    ob = oxygen_balance(f)
    assert ob > 0, f"NG should be oxygen-positive, got {ob}"


def test_oxygen_balance_nc():
    """NC (C6H7.3N2.7O9.9) with 13% N is moderately oxygen-negative."""
    f = Formulation(
        name="NC",
        components=[PropellantComponent("NC", "C6H7.3N2.7O9.9", 1.0, -700.0)],
    )
    ob = oxygen_balance(f)
    assert ob < 0, f"NC should be oxygen-negative, got {ob}"


def test_equivalence_ratio_stoichiometric():
    """A stoichiometric mixture should have phi close to 1."""
    # CH4 + 2O2 = CO2 + 2H2O
    # Mixing CH4 and O2 at stoichiometric ratio
    mw_ch4 = molecular_weight("CH4")
    mw_o2 = molecular_weight("O2")
    total = mw_ch4 + 2 * mw_o2
    f = Formulation(
        name="stoich",
        components=[
            PropellantComponent("CH4", "CH4", mw_ch4 / total, -74.87),
            PropellantComponent("O2", "O2", 2 * mw_o2 / total, 0.0),
        ],
    )
    phi = equivalence_ratio(f)
    assert abs(phi - 1.0) < 0.01, f"Should be stoichiometric, got phi={phi}"


def test_element_balance_sums():
    """Elemental composition moles * atomic mass should sum to ~1 g."""
    from combusto.models import ATOMIC_MASSES
    f = Formulation(
        name="test",
        components=[
            PropellantComponent("NC", "C6H7.3N2.7O9.9", 0.6, -700.0),
            PropellantComponent("NG", "C3H5N3O9", 0.4, -370.0),
        ],
    )
    elem = f.elemental_composition()
    total_mass = sum(moles * ATOMIC_MASSES[e] for e, moles in elem.items())
    assert abs(total_mass - 1.0) < 0.001, f"Should sum to 1 g, got {total_mass}"
