"""Wave 5 comprehensive engineering audit tests.

Covers:
1. Kamlet-Jacobs (1968) VoD / P_CJ stoichiometry, unit conversion (kbar -> GPa), and duplicate-kwarg regression.
2. Allowlisted Desktop Pro .zip builder (`build_desktop_zip_bytes`) excluding tunnel binaries/logs/secrets.
3. Stdlib-only TOML secrets manager and nested-JSON-safe Master Database Editor roundtrips (`app_local.py`).
4. Halogen (Cl, F) and multi-metal valence oxygen balance & equivalence ratio (`chemistry.py`).
5. NASA 7-term polynomial C¹ constant-Cp clamping outside [t_low, t_high] (`thermodynamics.py`).
6. Analytical Gibbs-Duhem Jacobian solver speed, inert gas dilution, and grain temperature sensible enthalpy (`equilibrium.py`).
7. Alkali salt (K2SO4) flash suppression vs. primary particulate smoke tradeoff (`muzzle_flash.py` + `toxicity_smoke.py`).
8. Dense 400-point interior ballistics trajectory, STANAG 4582 50% depletion rule, and strict Web vs. Desktop Pro isolation.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import app as web_app
import app_local as desktop_app
from combusto.ballistics import simulate_ballistics
from combusto.chemistry import equivalence_ratio, oxygen_balance
from combusto.detonation import formulation_kj_inputs, kamlet_jacobs, parse_chno_formula
from combusto.equilibrium import solve_equilibrium, solve_equilibrium_adiabatic
from combusto.models import (
    CombustionConditions,
    Formulation,
    PropellantComponent,
)
from combusto.muzzle_flash import analyze_muzzle_flash
from combusto.pah_allocation import PRESETS
from combusto.presets import load_presets
from combusto.species_db import load_species_db
from combusto.stability import analyze_thermal_stability
from combusto.thermodynamics import cp, enthalpy, entropy
from combusto.toxicity_smoke import analyze_toxicity_and_smoke

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_kamlet_jacobs_tnt_rdx_hmx_and_no_duplicate_kwarg():
    """Verify Kamlet-Jacobs (1968) stoichiometry (CO2 + C(s), zero CO), GPa units, and benchmark accuracy."""
    # TNT at crystal density 1.654 g/cm3 (published D ~ 6.93 km/s, P_CJ ~ 21.0 GPa)
    a, b, c, d = parse_chno_formula("C7H5N3O6")
    mw_tnt = a * 12.011 + b * 1.008 + c * 14.007 + d * 15.999
    res_tnt = kamlet_jacobs(a, b, c, d, mw_tnt, -66.9, 1.654)
    assert res_tnt["valid"] is True
    assert res_tnt["n_CO"] == 0.0
    assert res_tnt["n_Cs"] > 0.0
    assert res_tnt["P_GPa"] == pytest.approx(0.1 * res_tnt["P_kbar"], rel=1e-9)
    assert 6.75 <= res_tnt["D_km_s"] <= 7.15
    assert 18.5 <= res_tnt["P_GPa"] <= 22.5

    # RDX at 1.806 g/cm3 (published D ~ 8.75 km/s, P_CJ ~ 34.7 GPa)
    a, b, c, d = parse_chno_formula("C3H6N6O6")
    mw_rdx = a * 12.011 + b * 1.008 + c * 14.007 + d * 15.999
    res_rdx = kamlet_jacobs(a, b, c, d, mw_rdx, 70.3, 1.806)
    assert res_rdx["valid"] is True
    assert 8.60 <= res_rdx["D_km_s"] <= 9.00
    assert 32.5 <= res_rdx["P_GPa"] <= 36.5

    # HMX at 1.905 g/cm3 (published D ~ 9.11 km/s, P_CJ ~ 39.0 GPa)
    a, b, c, d = parse_chno_formula("C4H8N8O8")
    mw_hmx = a * 12.011 + b * 1.008 + c * 14.007 + d * 15.999
    res_hmx = kamlet_jacobs(a, b, c, d, mw_hmx, 75.0, 1.905)
    assert res_hmx["valid"] is True
    assert 8.95 <= res_hmx["D_km_s"] <= 9.30
    assert 37.0 <= res_hmx["P_GPa"] <= 41.5

    # Active formulation extraction
    presets = load_presets()
    fkj, err = formulation_kj_inputs(presets[0])
    assert err is None and fkj is not None
    res_f = kamlet_jacobs(fkj["a"], fkj["b"], fkj["c"], fkj["d"], fkj["mw"], fkj["hf_kJ"], 1.60)
    assert res_f["valid"] is True
    assert 5.0 <= res_f["D_km_s"] <= 9.0


def test_desktop_zip_allowlist_excludes_bloat_and_secrets():
    """Verify Combusto_Pro_Desktop.zip is compact (<500 KB) and excludes binaries, logs, and secrets."""
    zip_bytes = web_app.build_desktop_zip_bytes()
    assert isinstance(zip_bytes, bytes)
    assert 20_000 < len(zip_bytes) < 500_000

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())

    expected_present = {
        "app.py",
        "app_local.py",
        "INSTRUCTIONS.md",
        "README.md",
        "requirements.txt",
        "combusto/equilibrium.py",
        "combusto/detonation.py",
        "combusto/workspace_ui.py",
        "data/ingredients.json",
        "data/propellants.json",
        "data/species.json",
    }
    assert expected_present.issubset(names)

    forbidden_substrings = ("cloudflared", ".log", "secrets.toml", ".env", "__pycache__", "tunnel_url", "url_only")
    for arcname in names:
        assert not any(bad in arcname.lower() for bad in forbidden_substrings), f"Unexpected file in zip: {arcname}"


def test_local_admin_secrets_and_nested_db_editor_roundtrip(tmp_path: Path):
    """Verify stdlib TOML secrets IO and nested-array/dict Database Editor preservation."""
    secrets_path = tmp_path / ".streamlit" / "secrets.toml"
    sample_secrets = {
        "NIST_API_KEY": "nist-12345-abc",
        "MATERIALSPROJECT_API_KEY": 'mp-"quoted"-token',
        "PUBCHEM_API_KEY": "",
    }
    desktop_app.save_local_secrets_toml(secrets_path, sample_secrets)
    loaded_secrets = desktop_app.load_local_secrets_toml(secrets_path)
    assert loaded_secrets == sample_secrets

    # Propellants DB has nested `components` list of dicts; Species DB has 7-element `thermo_low`/`thermo_high`
    for rel_db in ("data/propellants.json", "data/species.json", "data/ingredients.json"):
        raw = json.loads((REPO_ROOT / rel_db).read_text(encoding="utf-8"))
        root_key = list(raw.keys())[-1]
        orig_records = raw[root_key]

        df_disp, nested_cols, col_defaults = desktop_app.prepare_records_for_editor(orig_records)
        # Simulate user adding a blank dynamic row with NaNs in st.data_editor
        blank_row = pd.DataFrame([{col: float("nan") for col in df_disp.columns}])
        df_with_blank = pd.concat([df_disp, blank_row], ignore_index=True)

        restored = desktop_app.restore_records_from_editor(df_with_blank, nested_cols, col_defaults)
        assert len(restored) == len(orig_records)
        # Must serialize cleanly under strict RFC-8259 JSON (no NaN tokens)
        encoded = json.dumps({root_key: restored}, allow_nan=False)
        decoded = json.loads(encoded)[root_key]
        assert decoded[0]["name"] == orig_records[0]["name"]
        for ncol in nested_cols:
            assert decoded[0][ncol] == orig_records[0][ncol]

    # Malformed JSON in a nested column must raise ValueError rather than corrupting disk JSON
    bad_df = pd.DataFrame([{"name": "BrokenProp", "components": "{not valid json"}])
    with pytest.raises(ValueError, match="invalid JSON"):
        desktop_app.restore_records_from_editor(bad_df, {"components"}, {"components": []})


def test_halogen_and_metal_oxygen_balance_and_phi():
    """Verify Cl->HCl, F->HF, and metal oxide valence accounting in OB% and phi."""
    ap = Formulation(
        name="Pure AP",
        components=[PropellantComponent("Ammonium Perchlorate", "NH4ClO4", 1.0, -295.8)],
    )
    # NH4ClO4 -> 0.5 N2 + 1.5 H2O + HCl + 1.25 O2 => +2.5 mol O per 117.489 g => +34.04%
    assert oxygen_balance(ap) == pytest.approx(34.04, abs=0.15)
    assert equivalence_ratio(ap) < 1.0

    kno3 = Formulation(
        name="Pure KNO3",
        components=[PropellantComponent("Potassium Nitrate", "KNO3", 1.0, -494.6)],
    )
    # KNO3 -> 0.5 K2O + 0.5 N2 + 1.25 O2 => +2.5 mol O per 101.103 g => +39.56%
    assert oxygen_balance(kno3) == pytest.approx(39.56, abs=0.15)

    for name, formula in [("Alumina", "Al2O3"), ("Magnesia", "MgO"), ("Hematite", "Fe2O3"), ("Calcite", "CaCO3")]:
        inert_ox = Formulation(name=name, components=[PropellantComponent(name, formula, 1.0, -1000.0)])
        assert oxygen_balance(inert_ox) == pytest.approx(0.0, abs=0.05)


def test_nasa_polynomial_high_temperature_clamping():
    """Verify C¹ constant-Cp clamping above t_high prevents high-T divergence."""
    db = load_species_db()
    n2 = db["N2"]
    cp_hi = cp(n2, n2.t_high)
    cp_ext = cp(n2, 7500.0)
    assert cp_ext == pytest.approx(cp_hi, rel=1e-9)
    assert cp_ext > 25.0
    assert enthalpy(n2, 7500.0) > enthalpy(n2, n2.t_high)
    assert entropy(n2, 7500.0) > entropy(n2, n2.t_high)


def test_equilibrium_analytical_jacobian_speed_inert_gas_and_grain_temp():
    """Verify <0.25s analytical-Jacobian equilibrium solve, inert gas dilution, and grain T0 enthalpy."""
    db = load_species_db()
    form = load_presets()[1]  # Double-base

    t0 = time.perf_counter()
    res_fast = solve_equilibrium(
        form,
        CombustionConditions(pressure_atm=100.0, mode="isobaric"),
        PRESETS["none"],
        db,
        2800.0,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.25
    assert res_fast.gas_moles_total > 0.02

    # Inert Ar dilution must lower self-consistent flame temperature
    res_dry = solve_equilibrium_adiabatic(
        form,
        CombustionConditions(temperature_k=298.15, pressure_atm=100.0),
        PRESETS["none"],
        db,
    )
    res_ar = solve_equilibrium_adiabatic(
        form,
        CombustionConditions(temperature_k=298.15, pressure_atm=100.0, inert_gas="Ar", inert_moles=0.012),
        PRESETS["none"],
        db,
    )
    assert res_ar.adiabatic_flame_temp_k < res_dry.adiabatic_flame_temp_k - 80.0

    # STANAG 2895 Hot Desert (+63 C = 336.15 K) vs Arctic (-40 C = 233.15 K) initial grain enthalpy
    res_arctic = solve_equilibrium_adiabatic(
        form,
        CombustionConditions(temperature_k=233.15, pressure_atm=100.0),
        PRESETS["none"],
        db,
    )
    res_desert = solve_equilibrium_adiabatic(
        form,
        CombustionConditions(temperature_k=336.15, pressure_atm=100.0),
        PRESETS["none"],
        db,
    )
    assert res_desert.adiabatic_flame_temp_k > res_arctic.adiabatic_flame_temp_k + 15.0


def test_alkali_flash_suppressant_increases_primary_smoke():
    """Verify K2SO4 flash suppressant raises radical ignition threshold while increasing primary salt smoke."""
    db = load_species_db()
    base_form = Formulation(
        name="Clean NC/NG",
        components=[
            PropellantComponent("NC (13.15% N)", "C6H7.35N2.65O10.30", 0.78, -682.0),
            PropellantComponent("Nitroglycerin", "C3H5N3O9", 0.22, -370.8),
        ],
    )
    supp_form = Formulation(
        name="Suppressed NC/NG + 3% K2SO4",
        components=[
            PropellantComponent("NC (13.15% N)", "C6H7.35N2.65O10.30", 0.75, -682.0),
            PropellantComponent("Nitroglycerin", "C3H5N3O9", 0.22, -370.8),
            PropellantComponent("Potassium Sulfate", "K2SO4", 0.03, -1437.8),
        ],
    )
    cond = CombustionConditions(pressure_atm=100.0, mode="isochoric", loading_density=0.20)
    res_base = solve_equilibrium_adiabatic(base_form, cond, PRESETS["none"], db)
    res_supp = solve_equilibrium_adiabatic(supp_form, cond, PRESETS["none"], db)

    mf_base = analyze_muzzle_flash(res_base, base_form, suppressant_wt_pct=None)
    mf_supp = analyze_muzzle_flash(res_supp, supp_form, suppressant_wt_pct=None)
    assert mf_supp["ignition_threshold_k"] > mf_base["ignition_threshold_k"] + 150.0

    tox_base = analyze_toxicity_and_smoke(res_base)
    tox_supp = analyze_toxicity_and_smoke(res_supp)
    assert tox_supp["primary_smoke_g_per_kg"] > tox_base["primary_smoke_g_per_kg"] + 15.0


def test_ballistics_dense_grid_stanag_4582_and_web_isolation():
    """Verify 400-pt dense ballistic trajectory, STANAG 4582 50% depletion rule, and web isolation."""
    ib = simulate_ballistics(
        force_constant_j_per_g=1120.0,
        gamma=1.23,
        charge_mass_g=60.0,
        propellant_density_g_per_cm3=1.62,
        covolume_cm3_per_g=1.02,
        burn_rate_a=1.6,
        burn_rate_n=0.88,
        web_thickness_mm=0.9,
        grain_geometry="19_perf",
        chamber_volume_cm3=70.0,
        bore_diameter_mm=30.0,
        barrel_length_m=1.6,
        projectile_mass_g=280.0,
    )
    assert len(ib["time"]) == 400
    assert ib["muzzle_velocity_m_s"] > 500.0
    assert (ib["fraction_burned"][1:] >= ib["fraction_burned"][:-1] - 1e-9).all()

    form = load_presets()[0]
    stab = analyze_thermal_stability(form, storage_temp_c=25.0, stabilizer_wt_pct=2.0)
    assert stab["depletion_df"]["critical_limit_pct"].iloc[0] == pytest.approx(1.0)

    # Strict scope boundary: Web app (`app.py`) and shared `workspace_ui.py` must never expose Desktop Pro admin features
    app_src = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    ws_src = (REPO_ROOT / "combusto" / "workspace_ui.py").read_text(encoding="utf-8")
    for src in (app_src, ws_src):
        assert "combusto.lookup" not in src
        assert "from .lookup" not in src
        assert "Database Editor" not in src
        assert "API Key Manager" not in src
