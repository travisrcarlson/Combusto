"""Tests for the equilibrium solver."""

import pytest
from combusto.models import (
    CombustionConditions,
    Formulation,
    PropellantComponent,
    PAHAllocationConfig,
    molecular_weight,
)
from combusto.equilibrium import solve_equilibrium
from combusto.pah_allocation import PRESETS


def test_stoichiometric_ch4_no_pahs():
    """CH4 + 2O2 at equilibrium with no PAH model should produce mostly CO2 and H2O."""
    mw_ch4 = molecular_weight("CH4")
    mw_o2 = molecular_weight("O2")
    total = mw_ch4 + 2 * mw_o2
    f = Formulation(
        name="CH4+O2",
        components=[
            PropellantComponent("CH4", "CH4", mw_ch4 / total, -74.87),
            PropellantComponent("O2", "O2", 2 * mw_o2 / total, 0.0),
        ],
    )
    conditions = CombustionConditions(pressure_atm=1.0)
    pah_config = PRESETS["none"]

    result = solve_equilibrium(f, conditions, pah_config, temperature_k=2500.0)

    # CO2 and H2O should dominate
    co2_frac = result.species_mole_fractions.get("CO2", 0.0)
    h2o_frac = result.species_mole_fractions.get("H2O", 0.0)
    assert co2_frac > 0.1, f"CO2 should be significant, got {co2_frac}"
    assert h2o_frac > 0.1, f"H2O should be significant, got {h2o_frac}"

    # PAH species should be zero
    for detail in result.pah_details:
        assert detail["mole_fraction"] < 1e-10


def test_fuel_rich_produces_pahs():
    """A fuel-rich mixture with moderate PAH model should allocate some carbon to PAHs."""
    # Excess CH4 (fuel-rich)
    mw_ch4 = molecular_weight("CH4")
    mw_o2 = molecular_weight("O2")
    total = 3 * mw_ch4 + 2 * mw_o2  # 3:2 instead of 1:2
    f = Formulation(
        name="rich_CH4",
        components=[
            PropellantComponent("CH4", "CH4", 3 * mw_ch4 / total, -74.87),
            PropellantComponent("O2", "O2", 2 * mw_o2 / total, 0.0),
        ],
    )
    conditions = CombustionConditions(pressure_atm=1.0)
    pah_config = PRESETS["aggressive"]

    result = solve_equilibrium(f, conditions, pah_config, temperature_k=2000.0)

    # Should have some PAH species
    pah_carbon = result.carbon_distribution.get("PAHs", 0.0)
    assert pah_carbon > 0.0, "Fuel-rich with aggressive PAH model should produce PAHs"


def test_element_balance_conservation():
    """Element balance errors should be small."""
    f = Formulation(
        name="NC/NG",
        components=[
            PropellantComponent("NC", "C6H7.3N2.7O9.9", 0.6, -700.0),
            PropellantComponent("NG", "C3H5N3O9", 0.4, -370.0),
        ],
    )
    conditions = CombustionConditions(pressure_atm=100.0)
    pah_config = PRESETS["none"]  # No PAHs for clean balance check

    result = solve_equilibrium(f, conditions, pah_config, temperature_k=3000.0)

    for elem, err in result.element_balance.items():
        assert err < 5.0, f"Element {elem} balance error {err}% is too large"
