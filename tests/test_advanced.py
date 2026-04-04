"""Tests for advanced features: adiabatic solver, isochoric, sensitivity."""

import pytest
import numpy as np

from combusto.models import (
    CombustionConditions,
    Formulation,
    PropellantComponent,
    molecular_weight,
)
from combusto.equilibrium import solve_equilibrium, solve_equilibrium_adiabatic
from combusto.pah_allocation import PRESETS
from combusto.sensitivity import (
    sweep_equivalence_ratio,
    sweep_temperature,
    sweep_pah_models,
    compare_formulations,
)
from combusto.presets import (
    load_presets,
    formulation_to_dict,
    formulation_from_dict,
    save_formulations,
    load_formulations,
)
from combusto.species_db import load_species_db


@pytest.fixture
def nc_ng():
    return Formulation(
        name="NC/NG",
        components=[
            PropellantComponent("NC", "C6H7.3N2.7O9.9", 0.6, -700.0),
            PropellantComponent("NG", "C3H5N3O9", 0.4, -370.0),
        ],
    )


@pytest.fixture
def species_db():
    return load_species_db()


def test_adiabatic_solver_converges(nc_ng, species_db):
    """Adiabatic solver should converge for a typical propellant."""
    conditions = CombustionConditions(pressure_atm=100.0)
    result = solve_equilibrium_adiabatic(
        nc_ng, conditions, PRESETS["none"], species_db=species_db
    )
    assert result.converged, f"Should converge, got {result.iterations} iterations"
    assert result.adiabatic_flame_temp_k > 1000
    assert result.adiabatic_flame_temp_k < 5000


def test_new_result_fields(nc_ng, species_db):
    """New result fields should be populated."""
    conditions = CombustionConditions(pressure_atm=100.0)
    result = solve_equilibrium(
        nc_ng, conditions, PRESETS["none"],
        species_db=species_db, temperature_k=2500.0,
    )
    assert result.gas_moles_total > 0
    assert result.mean_molecular_weight > 0
    assert result.gamma > 1.0
    assert result.force_constant_j_per_g > 0


def test_isochoric_pressure(nc_ng, species_db):
    """Isochoric mode should compute a finite pressure at reasonable loading density."""
    conditions = CombustionConditions(
        pressure_atm=1.0,
        mode="isochoric",
        loading_density=0.05,  # Low enough that V > n*b
        covolume_m3_per_mol=5e-4,
    )
    result = solve_equilibrium(
        nc_ng, conditions, PRESETS["none"],
        species_db=species_db, temperature_k=3000.0,
    )
    assert result.computed_pressure_atm > 0
    assert result.computed_pressure_atm < 1e6


def test_sweep_temperature(nc_ng, species_db):
    """Temperature sweep should return a DataFrame with expected columns."""
    conditions = CombustionConditions(pressure_atm=100.0)
    df = sweep_temperature(
        nc_ng, conditions, PRESETS["none"], species_db,
        t_range=np.linspace(1500, 3500, 5),
    )
    assert len(df) == 5
    assert "sweep_temperature_k" in df.columns
    assert "CO2_mole_frac" in df.columns


def test_sweep_pah_models(nc_ng, species_db):
    """PAH model sweep should return 4 rows."""
    conditions = CombustionConditions(pressure_atm=100.0)
    df = sweep_pah_models(nc_ng, conditions, species_db, 2500.0)
    assert len(df) == 4
    assert "pah_model" in df.columns


def test_formulation_serialization():
    """Formulations should round-trip through dict serialization."""
    f = Formulation(
        name="Test",
        components=[PropellantComponent("NC", "C6H7.3N2.7O9.9", 1.0, -700.0)],
        description="A test",
    )
    d = formulation_to_dict(f)
    f2 = formulation_from_dict(d)
    assert f2.name == "Test"
    assert len(f2.components) == 1
    assert f2.components[0].formula == "C6H7.3N2.7O9.9"


def test_save_load_formulations(tmp_path):
    """Formulations should save and load from JSON."""
    formulations = load_presets()
    path = tmp_path / "test.json"
    save_formulations(formulations, path)
    loaded = load_formulations(path)
    assert len(loaded) == len(formulations)
    assert loaded[0].name == formulations[0].name
