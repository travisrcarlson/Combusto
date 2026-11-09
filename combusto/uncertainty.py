"""NATO STANAG 2895 Climatic Conditioning & Monte Carlo Lot-Tolerance Dispersion.

Models:
1. Temperature sensitivity of interior ballistics across NATO STANAG 2895 operational
   climatic zones (-40 °C Arctic C2 through +63 °C Hot Desert A1), coupling:
   - Saint-Robert burn-rate temperature sensitivity a(T0) = a_ref * exp[sigma_p * (T0 - T_ref)]
   - Solid grain sensible enthalpy shift on closed-vessel impetus F(T0)
   - Propellant solid thermal expansion on grain density rho_p(T0) and web D0(T0)
2. Stochastic Monte Carlo manufacturing tolerance & lot-to-lot ballistic dispersion
   (charge mass, extrusion web thickness, lot vivacity/burn-rate, and MIL-DTL-244C
   nitrocellulose %N / impetus variation), computing 3-sigma breech pressure margins,
   process capability index Cpk, muzzle velocity SD (sigma_V0), and tornado sensitivity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .ballistics import simulate_ballistics

# NATO STANAG 2895 / AOP-4115 standard conditioning temperatures
STANAG_2895_ZONES: List[Dict[str, Any]] = [
    {"code": "C2", "label": "Extreme Cold / Arctic (C2)", "temp_c": -40.0},
    {"code": "C0", "label": "Cold Operational (C0)", "temp_c": -20.0},
    {"code": "AMB", "label": "Standard NATO Reference", "temp_c": 21.0},
    {"code": "A2", "label": "Warm Operational (A2)", "temp_c": 49.0},
    {"code": "A1", "label": "Extreme Hot Desert / Solar (A1)", "temp_c": 63.0},
]


def simulate_climatic_zones(
    force_constant_j_per_g: float,
    gamma: float,
    charge_mass_g: float,
    propellant_density_g_per_cm3: float,
    covolume_cm3_per_g: float,
    burn_rate_a: float,
    burn_rate_n: float,
    web_thickness_mm: float,
    grain_geometry: str,
    chamber_volume_cm3: float,
    bore_diameter_mm: float,
    barrel_length_m: float,
    projectile_mass_g: float,
    start_pressure_mpa: float = 30.0,
    sigma_p_per_k: float = 0.0020,
    t_ref_c: float = 21.0,
    service_limit_mpa: float = 420.0,
    proof_limit_mpa: float = 480.0,
) -> Dict[str, Any]:
    """Evaluate interior ballistics across NATO STANAG 2895 climatic zones."""
    c_solid_j_g_k = 1.35   # Typical NC/NG/RDX solid heat capacity J/(g*K)
    beta_vol_per_k = 2.4e-4  # Volumetric thermal expansion of propellant grain (1/K)

    # First run reference ambient (+21 °C) to anchor deltas
    amb_ib = simulate_ballistics(
        force_constant_j_per_g=force_constant_j_per_g,
        gamma=gamma,
        charge_mass_g=charge_mass_g,
        propellant_density_g_per_cm3=propellant_density_g_per_cm3,
        covolume_cm3_per_g=covolume_cm3_per_g,
        burn_rate_a=burn_rate_a,
        burn_rate_n=burn_rate_n,
        web_thickness_mm=web_thickness_mm,
        grain_geometry=grain_geometry,
        chamber_volume_cm3=chamber_volume_cm3,
        bore_diameter_mm=bore_diameter_mm,
        barrel_length_m=barrel_length_m,
        projectile_mass_g=projectile_mass_g,
        start_pressure_mpa=start_pressure_mpa,
    )
    p_amb = amb_ib["max_pressure_mpa"]
    v_amb = amb_ib["muzzle_velocity_m_s"]

    zone_rows = []
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    # Effective gas Cv per gram ~ F / ((gamma - 1) * T_v), with F/(Cv*T_v) = gamma - 1
    sensible_impetus_coeff = (max(gamma, 1.05) - 1.0) * c_solid_j_g_k / max(force_constant_j_per_g, 200.0)

    for z in STANAG_2895_ZONES:
        t_c = float(z["temp_c"])
        dt = t_c - t_ref_c

        a_eff = burn_rate_a * math.exp(sigma_p_per_k * dt)
        f_eff = force_constant_j_per_g * max(0.85, 1.0 + sensible_impetus_coeff * dt)
        rho_eff = propellant_density_g_per_cm3 / max(0.90, 1.0 + beta_vol_per_k * dt)
        web_eff = web_thickness_mm * max(0.95, 1.0 + (beta_vol_per_k / 3.0) * dt)

        ib = simulate_ballistics(
            force_constant_j_per_g=f_eff,
            gamma=gamma,
            charge_mass_g=charge_mass_g,
            propellant_density_g_per_cm3=rho_eff,
            covolume_cm3_per_g=covolume_cm3_per_g,
            burn_rate_a=a_eff,
            burn_rate_n=burn_rate_n,
            web_thickness_mm=web_eff,
            grain_geometry=grain_geometry,
            chamber_volume_cm3=chamber_volume_cm3,
            bore_diameter_mm=bore_diameter_mm,
            barrel_length_m=barrel_length_m,
            projectile_mass_g=projectile_mass_g,
            start_pressure_mpa=start_pressure_mpa,
        )

        p_max = ib["max_pressure_mpa"]
        v_muz = ib["muzzle_velocity_m_s"]
        dp_pct = ((p_max - p_amb) / max(p_amb, 1e-6)) * 100.0
        dv_ms = v_muz - v_amb

        if p_max >= proof_limit_mpa:
            status = "UNSAFE (Exceeds Proof Limit)"
        elif p_max >= service_limit_mpa:
            status = "WARNING (Exceeds Service MPOP)"
        else:
            status = "PASS (Within Safe Envelope)"

        zone_rows.append({
            "STANAG Zone": f"{z['code']} — {z['label']}",
            "Code": z["code"],
            "Conditioning Temp (°C)": t_c,
            "Effective Burn Coeff a": round(a_eff, 4),
            "Effective Impetus F (J/g)": round(f_eff, 1),
            "Peak Pressure P_max (MPa)": round(p_max, 1),
            "ΔP vs +21°C (%)": round(dp_pct, 2),
            "Muzzle Velocity (m/s)": round(v_muz, 1),
            "ΔV vs +21°C (m/s)": round(dv_ms, 1),
            "Action Time (ms)": round(ib["burn_time_ms"], 2),
            "Safety Status": status,
        })

        curves[z["code"]] = {
            "label": f"{z['code']} ({t_c:+.0f} °C)",
            "temp_c": t_c,
            "time_ms": ib["time"] * 1000.0,
            "pressure_mpa": ib["pressure_mpa"],
            "velocity_m_s": ib["velocity_m_s"],
            "travel_m": ib["travel_m"],
        }

    zones_df = pd.DataFrame(zone_rows)
    cold_row = zones_df.loc[zones_df["Code"] == "C2"].iloc[0]
    hot_row = zones_df.loc[zones_df["Code"] == "A1"].iloc[0]
    dt_span = float(hot_row["Conditioning Temp (°C)"] - cold_row["Conditioning Temp (°C)"])

    p_grad = float(hot_row["Peak Pressure P_max (MPa)"] - cold_row["Peak Pressure P_max (MPa)"]) / max(dt_span, 1.0)
    v_grad = float(hot_row["Muzzle Velocity (m/s)"] - cold_row["Muzzle Velocity (m/s)"]) / max(dt_span, 1.0)
    hot_pmax = float(hot_row["Peak Pressure P_max (MPa)"])
    cold_pmax = float(cold_row["Peak Pressure P_max (MPa)"])

    return {
        "zones_df": zones_df,
        "curves": curves,
        "ambient_pmax_mpa": float(p_amb),
        "ambient_vmuz_m_s": float(v_amb),
        "cold_pmax_mpa": cold_pmax,
        "hot_pmax_mpa": hot_pmax,
        "pressure_temp_gradient_mpa_per_k": p_grad,
        "velocity_temp_gradient_m_s_per_k": v_grad,
        "hot_safety_margin_to_proof_mpa": float(proof_limit_mpa - hot_pmax),
        "passes_stanag_2895": bool(hot_pmax < proof_limit_mpa),
        "service_limit_mpa": float(service_limit_mpa),
        "proof_limit_mpa": float(proof_limit_mpa),
    }


def run_monte_carlo_ballistics(
    force_constant_j_per_g: float,
    gamma: float,
    charge_mass_g: float,
    propellant_density_g_per_cm3: float,
    covolume_cm3_per_g: float,
    burn_rate_a: float,
    burn_rate_n: float,
    web_thickness_mm: float,
    grain_geometry: str,
    chamber_volume_cm3: float,
    bore_diameter_mm: float,
    barrel_length_m: float,
    projectile_mass_g: float,
    start_pressure_mpa: float = 30.0,
    n_trials: int = 120,
    charge_mass_sigma_pct: float = 0.50,
    web_sigma_pct: float = 1.20,
    burn_rate_sigma_pct: float = 1.50,
    nc_nitration_sigma_pct_n: float = 0.05,
    service_limit_mpa: float = 420.0,
    proof_limit_mpa: float = 480.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run stochastic Monte Carlo lot-to-lot ballistic tolerance simulation."""
    n_runs = max(20, min(500, int(n_trials)))
    rng = np.random.default_rng(seed)

    # Sample Gaussian manufacturing tolerances (clipped at +/- 4 sigma for physical positivity)
    z_m = np.clip(rng.standard_normal(n_runs), -4.0, 4.0)
    z_w = np.clip(rng.standard_normal(n_runs), -4.0, 4.0)
    z_a = np.clip(rng.standard_normal(n_runs), -4.0, 4.0)
    z_n = np.clip(rng.standard_normal(n_runs), -4.0, 4.0)

    mc_samples = charge_mass_g * (1.0 + (charge_mass_sigma_pct / 100.0) * z_m)
    web_samples = web_thickness_mm * (1.0 + (web_sigma_pct / 100.0) * z_w)
    a_samples = burn_rate_a * (1.0 + (burn_rate_sigma_pct / 100.0) * z_a)

    # MIL-DTL-244C %N variation (~38 J/g per 1.0 %N around 13.15% N) + minor ingredient weighing drift
    df_per_pct_n = 38.0
    f_samples = np.maximum(
        150.0,
        force_constant_j_per_g + (nc_nitration_sigma_pct_n * df_per_pct_n) * z_n,
    )

    records = []
    for i in range(n_runs):
        ib = simulate_ballistics(
            force_constant_j_per_g=float(f_samples[i]),
            gamma=gamma,
            charge_mass_g=float(mc_samples[i]),
            propellant_density_g_per_cm3=propellant_density_g_per_cm3,
            covolume_cm3_per_g=covolume_cm3_per_g,
            burn_rate_a=float(a_samples[i]),
            burn_rate_n=burn_rate_n,
            web_thickness_mm=float(web_samples[i]),
            grain_geometry=grain_geometry,
            chamber_volume_cm3=chamber_volume_cm3,
            bore_diameter_mm=bore_diameter_mm,
            barrel_length_m=barrel_length_m,
            projectile_mass_g=projectile_mass_g,
            start_pressure_mpa=start_pressure_mpa,
        )
        records.append({
            "trial": i + 1,
            "charge_mass_g": float(mc_samples[i]),
            "web_thickness_mm": float(web_samples[i]),
            "burn_rate_a": float(a_samples[i]),
            "force_j_per_g": float(f_samples[i]),
            "p_max_mpa": float(ib["max_pressure_mpa"]),
            "muzzle_velocity_m_s": float(ib["muzzle_velocity_m_s"]),
            "burn_time_ms": float(ib["burn_time_ms"]),
        })

    df = pd.DataFrame(records)
    p_arr = df["p_max_mpa"].to_numpy()
    v_arr = df["muzzle_velocity_m_s"].to_numpy()

    mean_p = float(np.mean(p_arr))
    std_p = float(np.std(p_arr, ddof=1)) if len(p_arr) > 1 else 0.0
    p99_p = float(np.percentile(p_arr, 99))
    p_plus_3s = mean_p + 3.0 * std_p

    mean_v = float(np.mean(v_arr))
    std_v = float(np.std(v_arr, ddof=1)) if len(v_arr) > 1 else 0.0
    es_v = float(np.max(v_arr) - np.min(v_arr))

    # Empirical + Gaussian tail exceedance probabilities
    emp_exc_srv = float(np.mean(p_arr >= service_limit_mpa) * 100.0)
    emp_exc_prf = float(np.mean(p_arr >= proof_limit_mpa) * 100.0)
    if std_p > 1e-6:
        z_prf = (proof_limit_mpa - mean_p) / std_p
        gauss_exc_prf = 0.5 * math.erfc(z_prf / math.sqrt(2.0)) * 100.0
        cpk_proof = (proof_limit_mpa - mean_p) / (3.0 * std_p)
    else:
        gauss_exc_prf = 100.0 if mean_p >= proof_limit_mpa else 0.0
        cpk_proof = 9.99 if mean_p < proof_limit_mpa else 0.0

    # Linear correlation sensitivity drivers for Peak Pressure (Tornado chart)
    sens_drivers: Dict[str, float] = {}
    param_labels = {
        "charge_mass_g": "Charge Mass (m_c)",
        "web_thickness_mm": "Web Thickness (D₀)",
        "burn_rate_a": "Burn Rate Coeff (a)",
        "force_j_per_g": "NC %N / Impetus (F)",
    }
    for col, label in param_labels.items():
        x = df[col].to_numpy()
        if np.std(x) > 1e-12 and std_p > 1e-12:
            corr = float(np.corrcoef(x, p_arr)[0, 1])
        else:
            corr = 0.0
        sens_drivers[label] = corr

    return {
        "trials_df": df,
        "n_trials": n_runs,
        "mean_p_max_mpa": mean_p,
        "std_p_max_mpa": std_p,
        "p99_p_max_mpa": p99_p,
        "p_max_plus_3sigma_mpa": p_plus_3s,
        "mean_v_muz_m_s": mean_v,
        "std_v_muz_m_s": std_v,
        "extreme_spread_v_m_s": es_v,
        "empirical_exceed_service_pct": emp_exc_srv,
        "empirical_exceed_proof_pct": emp_exc_prf,
        "gaussian_exceed_proof_pct": gauss_exc_prf,
        "cpk_proof": float(cpk_proof),
        "sensitivity_correlations": sens_drivers,
        "service_limit_mpa": float(service_limit_mpa),
        "proof_limit_mpa": float(proof_limit_mpa),
    }
