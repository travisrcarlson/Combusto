"""Unit tests for Wave 4 Refinements:
1. High-pressure BLAKE / Virial real-gas fugacity & compressibility Z
2. Shifting-Equilibrium vs. Frozen-Composition 1D rocket nozzle solver
3. NATO STANAG 2895 climatic conditioning (-40 °C to +63 °C)
4. Stochastic Monte Carlo manufacturing tolerance & ballistic dispersion
5. Full Multi-Suite Engineering Dossier PDF export
6. Zero-mojibake verification & strict Web vs. Desktop Pro admin isolation
"""

from __future__ import annotations

from pathlib import Path

from combusto.ballistics import estimate_barrel_erosion, simulate_ballistics
from combusto.equilibrium import solve_equilibrium_adiabatic
from combusto.export_pdf import export_full_dossier_pdf
from combusto.models import CombustionConditions
from combusto.muzzle_flash import analyze_muzzle_flash
from combusto.nozzle import analyze_nozzle_expansion
from combusto.pah_allocation import PRESETS
from combusto.presets import load_presets
from combusto.species_db import load_species_db
from combusto.stability import analyze_thermal_stability
from combusto.toxicity_smoke import analyze_toxicity_and_smoke
from combusto.uncertainty import run_monte_carlo_ballistics, simulate_climatic_zones
from combusto.workspace_ui import WORKSPACES


def test_real_gas_virial_fugacity_and_compressibility():
    db = load_species_db()
    presets = load_presets()
    prop = next(p for p in presets if "JA2" in p.name or "M9" in p.name or "M1" in p.name)

    cond_ideal = CombustionConditions(
        pressure_atm=2000.0, mode="isobaric", real_gas_fugacity=False
    )
    cond_real = CombustionConditions(
        pressure_atm=2000.0, mode="isobaric", real_gas_fugacity=True
    )

    res_ideal = solve_equilibrium_adiabatic(prop, cond_ideal, PRESETS["none"], species_db=db)
    res_real = solve_equilibrium_adiabatic(prop, cond_real, PRESETS["none"], species_db=db)

    assert res_ideal.converged
    assert res_real.converged
    assert res_ideal.compressibility_z == 1.0
    assert res_real.compressibility_z > 1.15
    assert max(res_real.element_balance.values()) < 0.05
    # High-pressure covolume fugacity shifts CH4/H2O/CO2 equilibrium measurably
    assert abs(res_real.adiabatic_flame_temp_k - res_ideal.adiabatic_flame_temp_k) >= 0.1


def test_shifting_vs_frozen_nozzle_expansion():
    db = load_species_db()
    presets = load_presets()
    prop = next(p for p in presets if "M9" in p.name or "JA2" in p.name)

    out = analyze_nozzle_expansion(
        formulation=prop,
        pc_atm=68.0,
        pe_atm=1.0,
        ambient_p_atm=1.0,
        pah_config=PRESETS["none"],
        species_db=db,
        n_profile_points=6,
    )

    fz = out["frozen"]
    sh = out["shifting"]

    assert fz["c_star_m_s"] > 1250.0
    assert sh["c_star_m_s"] > 1250.0
    assert sh["area_ratio_ae_at"] > 4.0
    # Exothermic recombination during expansion maintains higher exit T and higher Isp
    assert sh["exit_temp_k"] >= fz["exit_temp_k"]
    assert sh["isp_vacuum_s"] >= fz["isp_vacuum_s"]
    assert out["isp_vacuum_gain_s"] >= 0.0
    assert len(out["profile_df"]) == 6


def test_stanag_2895_climatic_conditioning():
    clim = simulate_climatic_zones(
        force_constant_j_per_g=1080.0,
        gamma=1.23,
        charge_mass_g=50.0,
        propellant_density_g_per_cm3=1.60,
        covolume_cm3_per_g=1.02,
        burn_rate_a=1.20,
        burn_rate_n=0.85,
        web_thickness_mm=1.20,
        grain_geometry="7_perf",
        chamber_volume_cm3=65.0,
        bore_diameter_mm=30.0,
        barrel_length_m=1.50,
        projectile_mass_g=300.0,
        sigma_p_per_k=0.0022,
        service_limit_mpa=420.0,
        proof_limit_mpa=480.0,
    )

    assert len(clim["zones_df"]) == 5
    assert clim["cold_pmax_mpa"] < clim["ambient_pmax_mpa"] < clim["hot_pmax_mpa"]
    assert clim["pressure_temp_gradient_mpa_per_k"] > 0.0
    assert set(clim["curves"].keys()) == {"C2", "C0", "AMB", "A2", "A1"}


def test_monte_carlo_ballistics_dispersion():
    mc = run_monte_carlo_ballistics(
        force_constant_j_per_g=1080.0,
        gamma=1.23,
        charge_mass_g=50.0,
        propellant_density_g_per_cm3=1.60,
        covolume_cm3_per_g=1.02,
        burn_rate_a=1.20,
        burn_rate_n=0.85,
        web_thickness_mm=1.20,
        grain_geometry="7_perf",
        chamber_volume_cm3=65.0,
        bore_diameter_mm=30.0,
        barrel_length_m=1.50,
        projectile_mass_g=300.0,
        n_trials=35,
        seed=123,
    )

    assert mc["n_trials"] == 35
    assert len(mc["trials_df"]) == 35
    assert mc["std_p_max_mpa"] > 0.5
    assert mc["std_v_muz_m_s"] > 0.2
    assert mc["p_max_plus_3sigma_mpa"] > mc["mean_p_max_mpa"]
    assert len(mc["sensitivity_correlations"]) == 4


def test_full_multi_suite_dossier_pdf_export():
    db = load_species_db()
    prop = load_presets()[0]
    res = solve_equilibrium_adiabatic(
        prop,
        CombustionConditions(pressure_atm=200.0, mode="isochoric", loading_density=0.20, real_gas_fugacity=True),
        PRESETS["moderate"],
        species_db=db,
    )
    noz = analyze_nozzle_expansion(prop, pc_atm=68.0, pe_atm=1.0, species_db=db, n_profile_points=4)
    ib = simulate_ballistics(
        force_constant_j_per_g=res.force_constant_j_per_g, gamma=res.gamma,
        charge_mass_g=50.0, propellant_density_g_per_cm3=1.60, covolume_cm3_per_g=res.covolume_cm3_per_g,
        burn_rate_a=1.2, burn_rate_n=0.85, web_thickness_mm=1.2, grain_geometry="7_perf",
        chamber_volume_cm3=65.0, bore_diameter_mm=30.0, barrel_length_m=1.5, projectile_mass_g=300.0,
    )
    ero = estimate_barrel_erosion(ib, res, bore_diameter_mm=30.0, charge_mass_g=50.0)
    mf = analyze_muzzle_flash(res, prop)
    tox = analyze_toxicity_and_smoke(res)
    stab = analyze_thermal_stability(prop)

    pdf_bytes = export_full_dossier_pdf(
        res,
        prop,
        suite_data={
            "nozzle_out": noz,
            "ib_out": ib,
            "ero_out": ero,
            "mf_out": mf,
            "tox_out": tox,
            "stab_out": stab,
        },
    )
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2500


def test_zero_mojibake_and_strict_admin_isolation():
    root = Path(__file__).resolve().parent.parent
    assert len(WORKSPACES) == 5

    mojibake_tokens = ["\u00e2\u20ac\u201d", "\u00e2\u2013\u00b6", "\u00c2\u00b3", "\u00f0\u0178", "\u00e2\u20ac\u00a6", "Hf\u00e2\u201a"]
    py_files = [root / "app.py", root / "app_local.py"] + list((root / "combusto").glob("*.py"))
    for p in py_files:
        txt = p.read_text(encoding="utf-8")
        for bad in mojibake_tokens:
            assert bad not in txt, f"Found mojibake token {bad!r} in {p.name}"

    app_web = (root / "app.py").read_text(encoding="utf-8")
    ws_shared = (root / "combusto" / "workspace_ui.py").read_text(encoding="utf-8")
    app_local = (root / "app_local.py").read_text(encoding="utf-8")

    forbidden_in_web = [
        "Database Editor",
        "API Key Manager",
        "query_pubchem",
        "query_nist_enthalpy",
        "fetch_compound_thermochemistry",
    ]
    for token in forbidden_in_web:
        assert token not in app_web, f"Admin/local token {token!r} leaked into app.py!"
        assert token not in ws_shared, f"Admin/local token {token!r} leaked into workspace_ui.py!"

    assert "Database Editor" in app_local
    assert "API Key Manager" in app_local
    assert "query_pubchem" in app_local
