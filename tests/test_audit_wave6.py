"""Wave 6 scientific audit, physics regression, and new capability tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from combusto.ballistics import (
    estimate_barrel_erosion,
    grain_form_function,
    grain_u_max,
    simulate_ballistics,
)
from combusto.closed_vessel import (
    calibrate_abel_eos_and_burn_rate,
    simulate_closed_vessel,
)
from combusto.equilibrium import solve_equilibrium, solve_equilibrium_adiabatic
from combusto.export_pdf import export_result_pdf
from combusto.grounding import validate_epa_ap42_emissions
from combusto.models import (
    CombustionConditions,
    Formulation,
    PropellantComponent,
    parse_formula,
)
from combusto.nozzle import analyze_nozzle_expansion
from combusto.optimizer import optimize_formulation
from combusto.pah_allocation import PRESETS
from combusto.presets import get_all_library_templates, load_presets
from combusto.rocket_motor import simulate_solid_rocket_motor
from combusto.sensitivity import _modify_oxygen, compare_formulations, sweep_2d
from combusto.species_db import load_species_db
from combusto.thermodynamics import cp, enthalpy
from combusto.toxicity_smoke import (
    PAH_BA_P_TEF,
    TOXIC_SPECIES_LIMITS,
    analyze_toxicity_and_smoke,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_zero_element_templates_inert_gas_and_empty_guard():
    """Verify C=0/H=0 formulations converge without zero-element blow-up, inert gas works, and empty guards."""
    db = load_species_db()

    # Empty formulation must return graceful unconverged result instead of crashing
    empty = Formulation(name="Empty", components=[])
    cond_base = CombustionConditions(temperature_k=298.15, pressure_atm=68.0, mode="isobaric")
    r_empty = solve_equilibrium(empty, cond_base, PRESETS["none"], db, 2500.0)
    r_empty_ad = solve_equilibrium_adiabatic(empty, cond_base, PRESETS["none"], db)
    assert r_empty.converged is False
    assert r_empty.gas_moles_total == 0.0
    assert r_empty_ad.converged is False
    assert r_empty_ad.gas_moles_total == 0.0

    templates = {t.name: t for t in get_all_library_templates()}
    bp = templates["Black Powder (Standard)"]
    ap_al = Formulation(
        name="AP + Al",
        components=[
            PropellantComponent("AP", "NH4ClO4", 0.70, -295.8),
            PropellantComponent("Al", "Al", 0.30, 0.0),
        ],
    )
    an = Formulation(
        name="Pure AN",
        components=[PropellantComponent("NH4NO3", "NH4NO3", 1.0, -365.6)],
    )
    flash = Formulation(
        name="Al + KClO4",
        components=[
            PropellantComponent("KClO4", "KClO4", 0.70, -432.8),
            PropellantComponent("Al", "Al", 0.30, 0.0),
        ],
    )

    for form in (bp, ap_al, an, flash):
        res = solve_equilibrium_adiabatic(form, cond_base, PRESETS["none"], db)
        assert res.converged, f"{form.name} failed to converge"
        assert max(res.element_balance.values()) < 1.0, f"{form.name} element balance error too high"
        assert 1100.0 < res.adiabatic_flame_temp_k < 5500.0
        assert 50.0 < res.force_constant_j_per_g < 2500.0

    # Black Powder at closed-vessel pressure must form condensed K2S and K2CO3 cleanly
    res_bp_cv = solve_equilibrium_adiabatic(
        bp,
        CombustionConditions(temperature_k=298.15, pressure_atm=100.0, mode="isochoric", loading_density=0.20),
        PRESETS["none"],
        db,
    )
    assert res_bp_cv.converged
    assert max(res_bp_cv.element_balance.values()) < 1.0
    assert res_bp_cv.species_moles.get("K2S", 0.0) > 1e-4 or res_bp_cv.species_moles.get("K2CO3", 0.0) > 1e-4

    # Inert Ar purge lowers both Tp and Tv while keeping Qv physically well-behaved
    m1 = load_presets()[0]
    cv_dry = CombustionConditions(temperature_k=298.15, pressure_atm=150.0, mode="isochoric", loading_density=0.20)
    cv_ar = CombustionConditions(
        temperature_k=298.15, pressure_atm=150.0, mode="isochoric", loading_density=0.20,
        inert_gas="Ar", inert_moles=0.012,
    )
    res_dry = solve_equilibrium_adiabatic(m1, cv_dry, PRESETS["none"], db)
    res_ar = solve_equilibrium_adiabatic(m1, cv_ar, PRESETS["none"], db)
    assert res_ar.isochoric_flame_temp_k < res_dry.isochoric_flame_temp_k - 60.0
    assert res_ar.adiabatic_flame_temp_k < res_dry.adiabatic_flame_temp_k - 60.0
    assert abs(res_ar.heat_of_explosion_j_per_g - res_dry.heat_of_explosion_j_per_g) < 250.0


def test_ballistics_c1_slivers_and_duhamel_bore_erosion():
    """Verify C1 grain form function continuity across u=1 and transient Duhamel bore heating."""
    for geom in ("7_perf", "19_perf"):
        u_max = grain_u_max(geom)
        assert u_max > 1.15
        z_m, dz_m = grain_form_function(1.0 - 1e-6, geom)
        z_p, dz_p = grain_form_function(1.0 + 1e-6, geom)
        assert abs(z_p - z_m) < 1e-4
        assert abs(dz_p - dz_m) < 1e-3, f"dz/du jump at u=1.0 for {geom}: {dz_m} vs {dz_p}"
        z_end, _ = grain_form_function(u_max, geom)
        assert z_end == pytest.approx(1.0, abs=1e-6)

    db = load_species_db()
    presets = {p.name: p for p in load_presets()}
    cool_prop = presets["M1 Single-Base (105mm Cannon)"]
    hot_prop = presets["M9 Double-Base (High-Energy Mortar)"]

    cond = CombustionConditions(pressure_atm=150.0, mode="isochoric", loading_density=0.20)
    res_cool = solve_equilibrium_adiabatic(cool_prop, cond, PRESETS["none"], db)
    res_hot = solve_equilibrium_adiabatic(hot_prop, cond, PRESETS["none"], db)

    def _run_gun(res):
        ib = simulate_ballistics(
            force_constant_j_per_g=res.force_constant_j_per_g,
            gamma=res.gamma,
            charge_mass_g=160.0,
            propellant_density_g_per_cm3=1.60,
            covolume_cm3_per_g=res.covolume_cm3_per_g,
            burn_rate_a=1.50,
            burn_rate_n=0.85,
            web_thickness_mm=0.85,
            grain_geometry="7_perf",
            chamber_volume_cm3=160.0,
            bore_diameter_mm=30.0,
            barrel_length_m=2.40,
            projectile_mass_g=360.0,
        )
        return estimate_barrel_erosion(ib, res, bore_diameter_mm=30.0, charge_mass_g=160.0)

    ero_cool = _run_gun(res_cool)
    ero_hot = _run_gun(res_hot)
    assert ero_cool["peak_wall_temp_k"] > 650.0
    assert ero_hot["peak_wall_temp_k"] > ero_cool["peak_wall_temp_k"] + 150.0
    assert ero_hot["estimated_barrel_life_rounds"] < ero_cool["estimated_barrel_life_rounds"]
    # Duhamel memory keeps wall warm at shot exit instead of dropping back to 298 K
    assert ero_hot["wall_temp_k"][-1] > 450.0


def test_sweep_2d_initial_temp_fix_and_sci_notation_formula():
    """Verify sweep_2d never evaluates equilibrium at ambient 298 K and handles sci notation."""
    db = load_species_db()
    m1 = load_presets()[0]
    cond_ambient = CombustionConditions(temperature_k=298.15, pressure_atm=100.0, mode="isobaric")

    df2d = sweep_2d(
        m1,
        cond_ambient,
        PRESETS["none"],
        db,
        param1="equivalence_ratio",
        range1=np.array([0.9, 1.4]),
        param2="pressure",
        range2=np.array([50.0, 150.0]),
    )
    assert len(df2d) == 4
    assert df2d["adiabatic_flame_temp_k"].min() > 1500.0
    assert df2d["CO_mole_frac"].max() > 0.01

    tiny_o = Formulation(
        name="Extreme Rich",
        components=[PropellantComponent("Hydrocarbon", "C10H10O0.0005", 1.0, -50.0)],
    )
    mod = _modify_oxygen(tiny_o, o_scale=0.0001)
    parsed = parse_formula(mod.components[0].formula)
    assert parsed["C"] == pytest.approx(10.0, rel=1e-6)

    comp_df = compare_formulations([Formulation(name="Empty"), m1], cond_ambient, PRESETS["none"], db)
    assert len(comp_df) == 1


def test_toxicity_bap_tef_soot_dedup_and_pyrene_allocation():
    """Verify benzo_a_pyrene TEF=1.0, H2S/SO2 limits, non-double-counted soot, and non-zero pyrene."""
    assert PAH_BA_P_TEF["benzo_a_pyrene"] == pytest.approx(1.0)
    assert "H2S" in TOXIC_SPECIES_LIMITS
    assert "SO2" in TOXIC_SPECIES_LIMITS

    db = load_species_db()
    rich = Formulation(
        name="Rich Hydrocarbon",
        components=[
            PropellantComponent("CH4", "CH4", 0.55, -74.6),
            PropellantComponent("O2", "O2", 0.45, 0.0),
        ],
    )
    res = solve_equilibrium(
        rich,
        CombustionConditions(temperature_k=1600.0, pressure_atm=10.0, mode="isobaric"),
        PRESETS["aggressive"],
        db,
        1600.0,
    )
    pah_mass_g_per_kg = sum(d["mass_fraction"] * 1000.0 for d in res.pah_details)

    tox = analyze_toxicity_and_smoke(res)
    assert pah_mass_g_per_kg > 5.0
    # Must not double-count soot_C / soot_precursor alongside pah_details
    assert tox["primary_smoke_g_per_kg"] == pytest.approx(pah_mass_g_per_kg, rel=1e-4, abs=0.5)

    epa = validate_epa_ap42_emissions(freeze_temp_k=1650.0, pressure_atm=35.0)
    epa_df = epa["emissions_df"]
    pyrene_rows = epa_df[epa_df["Species"].str.contains("pyrene", case=False)]
    assert len(pyrene_rows) == 1
    assert float(pyrene_rows.iloc[0]["Combusto Predicted (g/kg)"]) > 0.0


def test_closed_vessel_400pt_and_abel_nan_filtering():
    """Verify 400-pt dense closed-vessel solver on fast burn and NaN-safe Abel regression."""
    cv = simulate_closed_vessel(
        force_constant_j_per_g=1050.0,
        covolume_cm3_per_g=1.0,
        propellant_density_g_per_cm3=1.60,
        loading_density_g_per_cm3=0.20,
        burn_rate_a=2.4,
        burn_rate_n=0.88,
        web_thickness_mm=0.4,
        grain_geometry="7_perf",
    )
    assert len(cv["trace_df"]) == 400
    assert cv["p_max_mpa"] > 150.0

    cal = calibrate_abel_eos_and_burn_rate(
        loading_densities_g_cm3=[0.10, 0.15, 0.20, float("nan")],
        peak_pressures_mpa=[118.0, 188.0, 268.0, 100.0],
        burn_pressures_mpa=[80.0, 140.0, 200.0, 90.0],
        burn_rates_mm_s=[90.0, 145.0, 200.0, 110.0],
    )
    assert math.isfinite(cal["fitted_F_j_per_g"]) and cal["fitted_F_j_per_g"] > 500.0
    assert len(cal["abel_df"]) == 3

    with pytest.raises(ValueError):
        calibrate_abel_eos_and_burn_rate(
            loading_densities_g_cm3=[0.10, float("nan")],
            peak_pressures_mpa=[118.0, 100.0],
            burn_pressures_mpa=[80.0, 90.0],
            burn_rates_mm_s=[90.0, 100.0],
        )


def test_species_db_dedup_cl2_new_species_and_nozzle_conditioning():
    """Verify 65 unique species, Cl2 Hf=0, new K/S/AlCl species, and conditioned nozzle flow."""
    raw = json.loads((REPO_ROOT / "data" / "species.json").read_text(encoding="utf-8"))
    names = [s["name"] for s in raw["species"]]
    assert len(names) == 65
    assert len(set(names)) == 65

    db = load_species_db()
    assert db["Cl2"].hf_298 == pytest.approx(0.0, abs=1e-6)
    assert enthalpy(db["Cl2"], 298.15) == pytest.approx(0.0, abs=0.05)
    for sp_name in ("K2S", "K", "S2", "AlCl", "AlCl3"):
        assert sp_name in db
        assert cp(db[sp_name], 1000.0) > 10.0

    db_prop = load_presets()[2]  # M9 Double-Base
    noz_cold = analyze_nozzle_expansion(
        db_prop, pc_atm=68.0, pe_atm=1.0, pah_config=PRESETS["none"], species_db=db,
        n_profile_points=6, real_gas_fugacity=True, grain_temp_k=233.15,
    )
    noz_hot = analyze_nozzle_expansion(
        db_prop, pc_atm=68.0, pe_atm=1.0, pah_config=PRESETS["none"], species_db=db,
        n_profile_points=6, real_gas_fugacity=True, grain_temp_k=336.15,
    )
    assert noz_hot["chamber_temp_k"] > noz_cold["chamber_temp_k"] + 15.0
    assert noz_hot["shifting"]["isp_vacuum_s"] >= noz_hot["frozen"]["isp_vacuum_s"] - 1.0
    assert noz_hot["shifting"]["exit_mach"] > 1.5


def test_pdf_xml_escaping_and_carbon_free_pie_guard():
    """Verify ReportLab XML escaping for &, <, > and carbon-free formulation PDF export."""
    db = load_species_db()
    form = Formulation(
        name="AP & Al <Test> #1",
        components=[
            PropellantComponent("AP", "NH4ClO4", 0.70, -295.8),
            PropellantComponent("Al", "Al", 0.30, 0.0),
        ],
    )
    res = solve_equilibrium_adiabatic(
        form,
        CombustionConditions(pressure_atm=68.0, mode="isobaric"),
        PRESETS["none"],
        db,
    )
    pdf_bytes = export_result_pdf(res, form)
    assert pdf_bytes.startswith(b"%PDF-")


def test_constrained_optimizer_locks_and_bounds():
    """Verify optimizer keeps locked components fixed and enforces custom min/max bounds."""
    db = load_species_db()
    form = Formulation(
        name="3-Comp Lock Test",
        components=[
            PropellantComponent("NC", "C6H7.35N2.65O10.30", 0.65, -682.0),
            PropellantComponent("NG", "C3H5N3O9", 0.25, -370.8),
            PropellantComponent("DPA", "C12H11N", 0.10, 130.0),
        ],
    )
    cond = CombustionConditions(pressure_atm=100.0, mode="isobaric")
    opt = optimize_formulation(
        base_formulation=form,
        conditions=cond,
        pah_config=PRESETS["moderate"],
        species_db=db,
        objective_metric="force_constant",
        max_temp_k=3200.0,
        max_pah_frac=0.08,
        component_bounds=[(0.40, 0.85), (0.05, 0.45), (0.0, 0.20)],
        locked_indices=[2],
        equilibrium_temp_k=2600.0,
        self_consistent_adiabatic=False,
    )
    opt_w = [c.mass_fraction for c in opt["optimized_formulation"].components]
    assert sum(opt_w) == pytest.approx(1.0, abs=1e-5)
    assert opt_w[2] == pytest.approx(0.10, abs=1e-5)
    assert 0.40 - 1e-5 <= opt_w[0] <= 0.85 + 1e-5
    assert 0.05 - 1e-5 <= opt_w[1] <= 0.45 + 1e-5


def test_solid_rocket_motor_bates_end_burner_star():
    """Verify SRM simulator BATES, End-Burner, and Star grains + propellant mass conservation."""
    for geom in ("bates", "end_burner", "star"):
        res = simulate_solid_rocket_motor(
            c_star_m_s=1550.0,
            cf_ambient=1.58,
            cf_vacuum=1.74,
            gamma=1.20,
            grain_geometry=geom,
            throat_diam_mm=10.0 if geom == "end_burner" else 28.0,
            t_max_s=150.0 if geom == "end_burner" else 12.0,
        )
        assert res["max_pressure_mpa"] > 0.5
        assert res["action_time_s"] > 0.1
        assert res["total_impulse_n_s"] > 100.0
        assert res["delivered_isp_s"] > 150.0
        assert res["motor_class"][0].isalpha()

        df = res["trace_df"]
        total_out_kg = float(np.trapezoid(df["mass_flow_kg_s"], df["time_s"]))
        rel_err = abs(total_out_kg - res["propellant_mass_kg"]) / res["propellant_mass_kg"]
        assert rel_err < 0.05, f"SRM {geom} mass balance mismatch: {rel_err:.4f}"
