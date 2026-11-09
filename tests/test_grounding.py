"""Tests for scientific grounding, thermochemical database integrity, and benchmark parity."""

from __future__ import annotations

from combusto.grounding import (
    METHODOLOGY_CITATIONS,
    get_ingredient_provenance_df,
    run_benchmark_validation_suite,
    synthesize_nitrocellulose,
    validate_epa_ap42_emissions,
)
from combusto.models import parse_formula
from combusto.presets import load_ingredients, get_all_library_templates
from combusto.species_db import load_species_db
from combusto.thermodynamics import cp, enthalpy


def test_species_polynomial_thermodynamic_consistency():
    """Every species in species.json must match Hf_298 at 298.15 K and have positive Cp(T)."""
    db = load_species_db()
    assert len(db) >= 55

    for name, sp in db.items():
        h_298_kj = enthalpy(sp, 298.15) / 1000.0
        assert abs(h_298_kj - sp.hf_298) < 0.6, f"{name}: poly H(298)={h_298_kj} vs hf_298={sp.hf_298}"
        for t_check in (300.0, 1000.0, 2500.0):
            if t_check <= sp.t_high:
                assert cp(sp, t_check) > 5.0, f"{name}: non-physical Cp at {t_check} K"


def test_ingredients_and_library_element_coverage():
    """All grounded ingredients and preset templates must parse cleanly and have product element coverage."""
    db = load_species_db()
    covered_elements = set()
    for sp in db.values():
        if sp.category not in ("pah", "lump") and sp.t_high >= 2500.0:
            covered_elements.update(parse_formula(sp.formula).keys())

    ings = load_ingredients()
    assert len(ings) >= 45
    prov_df = get_ingredient_provenance_df()
    assert len(prov_df) == len(ings)

    for ing in ings:
        comp = parse_formula(ing["formula"])
        assert len(comp) > 0
        assert set(comp.keys()).issubset(covered_elements), (
            f"Ingredient {ing['name']} contains element(s) without high-T product coverage: "
            f"{set(comp.keys()) - covered_elements}"
        )

    for tmpl in get_all_library_templates():
        elems = set(tmpl.elemental_composition().keys())
        assert elems.issubset(covered_elements), f"Template {tmpl.name} missing element sink: {elems - covered_elements}"


def test_nitrocellulose_brl_continuum():
    nc_126 = synthesize_nitrocellulose(12.60)
    nc_1315 = synthesize_nitrocellulose(13.15)
    assert abs(nc_126["degree_of_substitution_x"] - 2.45) < 0.02
    assert abs(nc_1315["degree_of_substitution_x"] - 2.635) < 0.02
    # Higher nitration increases enthalpy of formation (less negative) and improves oxygen balance
    assert nc_1315["hf_j_per_g"] > nc_126["hf_j_per_g"]
    assert nc_1315["oxygen_balance_pct"] > nc_126["oxygen_balance_pct"]


def test_brl_ict_nasa_cea_benchmark_parity():
    """Combusto must achieve < 3.5% MAPE on Flame Temp and Impetus across all 8 canonical benchmarks."""
    out = run_benchmark_validation_suite()
    bdf = out["benchmark_df"]
    assert len(bdf) == 8
    assert out["mape_temp_pct"] < 3.5, f"Temperature MAPE too high: {out['mape_temp_pct']}%"
    assert out["mape_force_pct"] < 4.0, f"Force/Impetus MAPE too high: {out['mape_force_pct']}%"
    assert out["mape_qv_pct"] < 5.0, f"Calorimetric Qv MAPE too high: {out['mape_qv_pct']}%"


def test_epa_ap42_wc844_emissions_and_citations():
    epa = validate_epa_ap42_emissions(freeze_temp_k=1650.0, pressure_atm=35.0)
    df = epa["emissions_df"].set_index("Species")
    # CO and trace aromatics (benzene, toluene, naphthalene) should fall inside EPA AP-42 measured envelope
    for sp_key in ("CO", "benzene", "toluene", "naphthalene"):
        assert df.loc[sp_key, "Envelope Match"] == "Within Experimental Range"

    assert len(METHODOLOGY_CITATIONS) >= 8
