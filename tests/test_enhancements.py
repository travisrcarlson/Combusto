"""Tests for the new Combusto enhancements:
- Multi-perforated interior ballistics, 2D grain cross-section, and barrel erosion
- Muzzle flash and afterburning plume predictor
- Multi-objective formulation optimizer and ternary simplex grid
- 2D sensitivity sweeps and PDF report generation
"""

from __future__ import annotations

import numpy as np

from combusto.ballistics import (
    estimate_barrel_erosion,
    generate_grain_cross_section,
    grain_form_function,
    simulate_ballistics,
)
from combusto.equilibrium import solve_equilibrium
from combusto.export_pdf import export_result_pdf
from combusto.models import (
    CombustionConditions,
    Formulation,
    PropellantComponent,
)
from combusto.muzzle_flash import analyze_muzzle_flash
from combusto.optimizer import compute_ternary_grid, optimize_formulation
from combusto.pah_allocation import PRESETS
from combusto.sensitivity import sweep_2d
from combusto.species_db import load_species_db


def _sample_double_base() -> Formulation:
    return Formulation(
        name="Test Double-Base",
        components=[
            PropellantComponent("Nitrocellulose", "C6H7.55N2.45O9.9", 0.58, -670.0),
            PropellantComponent("Nitroglycerin", "C3H5N3O9", 0.38, -370.0),
            PropellantComponent("Ethyl Centralite", "C17H20N2O", 0.04, -105.0),
        ],
    )


def test_multi_perf_form_functions_and_cross_section():
    # 7-perf should be progressive in Phase I (dz/du increases from u=0.1 to u=0.9)
    z_early, dz_early = grain_form_function(0.1, "7_perf")
    z_late, dz_late = grain_form_function(0.9, "7_perf")
    assert 0.0 < z_early < z_late < 1.0
    assert dz_late > dz_early

    # Sliver phase (1.0 < u < 1.18) should complete the remaining ~15% mass
    z_sliver, dz_sliver = grain_form_function(1.08, "7_perf")
    assert z_late < z_sliver < 1.0
    assert dz_sliver > 0.0

    # Cross-section geometry for 7-perf and 19-perf
    cs7 = generate_grain_cross_section("7_perf", web_thickness_mm=1.4, burn_fraction_u=0.4)
    assert len(cs7["perf_centers"]) == 7
    assert cs7["outer_r_cur"] < cs7["outer_r_init"]
    assert cs7["perf_r_cur"] > cs7["perf_r_init"]

    cs19 = generate_grain_cross_section("19_perf", web_thickness_mm=1.4, burn_fraction_u=0.4)
    assert len(cs19["perf_centers"]) == 19


def test_ballistics_and_barrel_erosion():
    ib = simulate_ballistics(
        force_constant_j_per_g=1080.0,
        gamma=1.23,
        charge_mass_g=55.0,
        propellant_density_g_per_cm3=1.60,
        covolume_cm3_per_g=1.0,
        burn_rate_a=1.3,
        burn_rate_n=0.85,
        web_thickness_mm=1.2,
        grain_geometry="7_perf",
        chamber_volume_cm3=65.0,
        bore_diameter_mm=30.0,
        barrel_length_m=1.5,
        projectile_mass_g=300.0,
        start_pressure_mpa=30.0,
    )
    assert ib["max_pressure_mpa"] > 30.0
    assert ib["muzzle_velocity_m_s"] > 100.0

    ero = estimate_barrel_erosion(ib, result=None, bore_diameter_mm=30.0, charge_mass_g=55.0)
    assert ero["peak_wall_temp_k"] > 300.0
    assert ero["wear_um_per_shot"] > 0.0
    assert ero["estimated_barrel_life_rounds"] >= 10


def test_muzzle_flash_suppression():
    db = load_species_db()
    form = _sample_double_base()
    cond = CombustionConditions(temperature_k=2500.0, pressure_atm=100.0, mode="isobaric")
    res = solve_equilibrium(form, cond, PRESETS["moderate"], db, 2500.0)

    mf_unsuppressed = analyze_muzzle_flash(res, form, expansion_ratio=8.0, muzzle_mach=2.6, suppressant_wt_pct=0.0)
    mf_suppressed = analyze_muzzle_flash(res, form, expansion_ratio=8.0, muzzle_mach=2.6, suppressant_wt_pct=3.5)

    assert mf_suppressed["ignition_threshold_k"] > mf_unsuppressed["ignition_threshold_k"] + 100.0
    assert len(mf_unsuppressed["plume_df"]) > 20


def test_optimizer_and_ternary_grid():
    db = load_species_db()
    form = _sample_double_base()
    cond = CombustionConditions(temperature_k=2500.0, pressure_atm=100.0, mode="isobaric")

    opt = optimize_formulation(
        base_formulation=form,
        conditions=cond,
        pah_config=PRESETS["moderate"],
        species_db=db,
        objective_metric="force_constant",
        max_temp_k=3100.0,
        max_pah_frac=0.08,
    )
    opt_form = opt["optimized_formulation"]
    total_w = sum(c.mass_fraction for c in opt_form.components)
    assert abs(total_w - 1.0) < 1e-5
    assert len(opt["comparison_df"]) == 3

    t_df = compute_ternary_grid(
        form.components[0],
        form.components[1],
        form.components[2],
        cond,
        PRESETS["moderate"],
        db,
        n_steps=3,
    )
    # (3+1)*(3+2)/2 = 10 simplex points
    assert len(t_df) == 10
    assert "T_ad_K" in t_df.columns


def test_sweep_2d_and_pdf_export():
    db = load_species_db()
    form = _sample_double_base()
    cond = CombustionConditions(temperature_k=2500.0, pressure_atm=100.0, mode="isobaric")

    df2d = sweep_2d(
        form,
        cond,
        PRESETS["moderate"],
        db,
        param1="equivalence_ratio",
        range1=np.array([1.0, 1.5]),
        param2="pressure",
        range2=np.array([50.0, 150.0]),
    )
    assert len(df2d) == 4
    assert "force_constant_j_per_g" in df2d.columns

    res = solve_equilibrium(form, cond, PRESETS["moderate"], db, 2500.0)
    pdf_bytes = export_result_pdf(res, form)
    assert len(pdf_bytes) > 50
