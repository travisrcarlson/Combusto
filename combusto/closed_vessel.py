"""Closed-vessel (manometric bomb) vivacity and burn-rate calibration module for Combusto.

Simulates closed-bomb pressure traces P(t), dP/dt vs P, and dynamic vivacity
L = (1 / (P * P_max)) * (dP/dt) across loading densities, and provides linear
regression tools for Noble-Abel (F, eta) and Saint-Robert's burn rate (a, n)
calibration from multi-density test data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .ballistics import grain_form_function


def simulate_closed_vessel(
    force_constant_j_per_g: float = 1050.0,
    covolume_cm3_per_g: float = 1.00,
    propellant_density_g_per_cm3: float = 1.60,
    loading_density_g_per_cm3: float = 0.20,
    burn_rate_a: float = 1.20,       # mm/s/MPa^n
    burn_rate_n: float = 0.85,
    web_thickness_mm: float = 1.20,
    grain_geometry: str = "7_perf",
    ignition_pressure_mpa: float = 3.0,
    heat_loss_fraction: float = 0.03,
    t_end: float = 0.04,
) -> Dict[str, Any]:
    """Simulate a constant-volume closed-vessel (manometric bomb) test.

    Computes P(t), dP/dt, mass fraction burned z(t), and dynamic vivacity
    L(P/P_max) = (1 / (P * P_max)) * dP/dt.
    """
    F_eff = force_constant_j_per_g * (1.0 - max(0.0, min(0.35, heat_loss_fraction))) * 1000.0  # J/kg
    eta = covolume_cm3_per_g * 1e-3             # m3/kg
    rho_p = propellant_density_g_per_cm3 * 1000.0  # kg/m3
    delta = loading_density_g_per_cm3 * 1000.0     # kg/m3
    a_burn = burn_rate_a * 1e-3                 # m/s/MPa^n
    D0 = max(1e-4, web_thickness_mm * 1e-3)     # m
    P_ign = ignition_pressure_mpa * 1e6         # Pa

    u_max = 1.18 if grain_geometry == "7_perf" else (1.15 if grain_geometry == "19_perf" else 1.0)

    def pressure_from_z(z: float) -> float:
        """Noble-Abel closed-vessel pressure (Pa) at burned fraction z."""
        z_c = max(0.0, min(1.0, z))
        # Specific volume per unit chamber volume:
        # 1 - delta*(1-z)/rho_p - delta*z*eta
        free_vol_frac = 1.0 - (delta * (1.0 - z_c) / rho_p) - (delta * z_c * eta)
        free_vol_frac = max(1e-4, free_vol_frac)
        return P_ign + (delta * z_c * F_eff) / free_vol_frac

    def odes(t: float, y: List[float]) -> List[float]:
        x_b = y[0]
        z = max(0.0, min(1.0, y[1]))
        u = 2.0 * x_b / D0
        if u >= u_max or z >= 0.9998:
            return [0.0, 0.0]
        P_pa = pressure_from_z(z)
        P_mpa = max(0.1, P_pa / 1e6)
        r = a_burn * (P_mpa ** burn_rate_n)
        _, dz_du = grain_form_function(u, grain_geometry)
        dz_dt = dz_du * (2.0 * r / D0)
        return [r, dz_dt]

    def burnout_event(t: float, y: List[float]) -> float:
        return 0.9995 - y[1]

    burnout_event.terminal = True
    burnout_event.direction = -1

    t_eval = np.linspace(0.0, t_end, 400)
    sol = solve_ivp(
        odes,
        [0.0, t_end],
        [0.0, 0.001],
        t_eval=t_eval,
        events=burnout_event,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    t_arr = sol.t
    xb_arr = sol.y[0]
    z_arr = np.clip(sol.y[1], 0.0, 1.0)

    p_mpa_arr = np.array([pressure_from_z(float(z)) / 1e6 for z in z_arr])
    p_max_mpa = float(np.max(p_mpa_arr))

    # Numerical derivative dP/dt (MPa/s) and Dynamic Vivacity L (1 / (MPa * s))
    dp_dt_mpa_s = np.gradient(p_mpa_arr, t_arr) if len(t_arr) > 1 else np.zeros_like(p_mpa_arr)
    dp_dt_mpa_s = np.maximum(0.0, dp_dt_mpa_s)

    p_ratio = np.clip(p_mpa_arr / max(p_max_mpa, 1e-6), 1e-4, 1.0)
    vivacity = np.where(
        (p_ratio >= 0.05) & (p_ratio <= 0.95),
        dp_dt_mpa_s / (np.maximum(p_mpa_arr, 0.1) * max(p_max_mpa, 0.1)),
        0.0,
    )

    # Mean vivacity between P/P_max = 0.3 and 0.7 (standard NATO STANAG 4115 window)
    mask_30_70 = (p_ratio >= 0.30) & (p_ratio <= 0.70)
    mean_vivacity_30_70 = float(np.mean(vivacity[mask_30_70])) if np.any(mask_30_70) else float(np.max(vivacity) * 0.6)

    trace_df = pd.DataFrame({
        "time_ms": t_arr * 1000.0,
        "pressure_mpa": p_mpa_arr,
        "p_over_pmax": p_ratio,
        "dp_dt_gpa_s": dp_dt_mpa_s / 1000.0,
        "vivacity_per_mpa_s": vivacity,
        "fraction_burned": z_arr,
        "burnt_web_mm": xb_arr * 1000.0,
    })

    return {
        "p_max_mpa": p_max_mpa,
        "peak_dp_dt_gpa_s": float(np.max(dp_dt_mpa_s) / 1000.0),
        "mean_vivacity_30_70": mean_vivacity_30_70,
        "burnout_time_ms": float(t_arr[-1] * 1000.0),
        "trace_df": trace_df,
    }


def calibrate_abel_eos_and_burn_rate(
    loading_densities_g_cm3: Sequence[float],
    peak_pressures_mpa: Sequence[float],
    burn_pressures_mpa: Sequence[float],
    burn_rates_mm_s: Sequence[float],
) -> Dict[str, Any]:
    """Fit Noble-Abel (F, eta) and Saint-Robert's law (a, n) from test points.

    Uses Abel's linear relation:
        P_max / Delta = F + eta * P_max
    and log-log burn rate regression:
        ln(r) = ln(a) + n * ln(P)
    """
    deltas = np.asarray(loading_densities_g_cm3, dtype=float)
    p_maxs = np.asarray(peak_pressures_mpa, dtype=float)

    # Convert P_max (MPa) / Delta (g/cm3) -> J/g (since 1 MPa * cm3/g = 1 J/g)
    y_abel = p_maxs / np.maximum(deltas, 1e-6)
    # Linear fit y = eta * P_max + F
    slope_eta, intercept_F = np.polyfit(p_maxs, y_abel, 1)

    # R^2 for Abel plot
    y_pred = slope_eta * p_maxs + intercept_F
    ss_res = float(np.sum((y_abel - y_pred) ** 2))
    ss_tot = float(np.sum((y_abel - np.mean(y_abel)) ** 2))
    r2_abel = 1.0 - (ss_res / max(ss_tot, 1e-12))

    # Log-log regression for r = a * P^n
    p_b = np.maximum(np.asarray(burn_pressures_mpa, dtype=float), 1e-3)
    r_b = np.maximum(np.asarray(burn_rates_mm_s, dtype=float), 1e-4)
    ln_p = np.log(p_b)
    ln_r = np.log(r_b)
    n_fit, ln_a_fit = np.polyfit(ln_p, ln_r, 1)
    a_fit = float(math.exp(ln_a_fit))

    ln_r_pred = n_fit * ln_p + ln_a_fit
    ss_res_r = float(np.sum((ln_r - ln_r_pred) ** 2))
    ss_tot_r = float(np.sum((ln_r - np.mean(ln_r)) ** 2))
    r2_burn = 1.0 - (ss_res_r / max(ss_tot_r, 1e-12))

    abel_df = pd.DataFrame({
        "loading_density_g_cm3": deltas,
        "p_max_mpa": p_maxs,
        "p_over_delta_j_g": y_abel,
        "fitted_p_over_delta_j_g": y_pred,
    })

    burn_df = pd.DataFrame({
        "pressure_mpa": p_b,
        "measured_burn_rate_mm_s": r_b,
        "fitted_burn_rate_mm_s": a_fit * (p_b ** n_fit),
    })

    return {
        "fitted_F_j_per_g": float(intercept_F),
        "fitted_covolume_cm3_per_g": float(slope_eta),
        "r2_abel": float(r2_abel),
        "fitted_a_mm_s_mpa_n": a_fit,
        "fitted_n": float(n_fit),
        "r2_burn": float(r2_burn),
        "abel_df": abel_df,
        "burn_df": burn_df,
    }
