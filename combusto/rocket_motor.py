"""Solid Rocket Motor (SRM) internal ballistics, grain burnback & thrust-time F(t) simulator.

Couples thermochemical nozzle outputs (c*, C_F, gamma, expansion ratio epsilon) with
0D quasi-steady / lumped-chamber transient mass-conservation and Saint-Robert burn rate
r = a * P^n across BATES, End-Burner, and 6-Point Star grain geometries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

G0 = 9.80665
ATM_TO_PA = 101325.0

_NAR_IMPULSE_BANDS = [
    (0.0, 2.5, "A"),
    (2.5, 5.0, "B"),
    (5.0, 10.0, "C"),
    (10.0, 20.0, "D"),
    (20.0, 40.0, "E"),
    (40.0, 80.0, "F"),
    (80.0, 160.0, "G"),
    (160.0, 320.0, "H"),
    (320.0, 640.0, "I"),
    (640.0, 1280.0, "J"),
    (1280.0, 2560.0, "K"),
    (2560.0, 5120.0, "L"),
    (5120.0, 10240.0, "M"),
    (10240.0, 20480.0, "N"),
    (20480.0, 40960.0, "O"),
    (40960.0, 81920.0, "P"),
    (81920.0, 163840.0, "Q"),
    (163840.0, 327680.0, "R"),
    (327680.0, 655360.0, "S"),
]


def motor_impulse_class(total_impulse_n_s: float, avg_thrust_n: float = 0.0) -> str:
    """Return standard NAR / Tripoli motor letter designation (e.g., 'M1850')."""
    imp = max(0.0, float(total_impulse_n_s))
    letter = "T+"
    for lo, hi, code in _NAR_IMPULSE_BANDS:
        if imp <= hi:
            letter = code
            break
    if avg_thrust_n > 0.0:
        return f"{letter}{int(round(avg_thrust_n))}"
    return letter


def _vandenkerckhove_gamma(gamma: float) -> float:
    g = max(1.05, min(1.66, float(gamma)))
    return math.sqrt(g) * ((2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0))))


def srm_grain_burn_area(
    w_m: float,
    grain_geometry: str,
    r_outer_m: float,
    r_core_m: float,
    length_m: float,
    n_segments: int = 3,
) -> Tuple[float, float, float]:
    """Return `(burn_area_m2, free_vol_m3, web_max_m)` at web regression `w_m`."""
    r_o = max(1e-3, float(r_outer_m))
    r_c = max(1e-4, min(r_o * 0.90, float(r_core_m)))
    len_tot = max(5e-3, float(length_m))
    case_vol = math.pi * (r_o ** 2) * len_tot

    if grain_geometry == "end_burner":
        web_max = len_tot
        w_eff = max(0.0, min(web_max, float(w_m)))
        area = math.pi * (r_o ** 2) if w_eff < web_max - 1e-7 else 0.0
        prop_vol = math.pi * (r_o ** 2) * max(0.0, len_tot - w_eff)
        free_vol = max(case_vol * 0.04, case_vol - prop_vol)
        return area, free_vol, web_max

    if grain_geometry == "star":
        # 6-point star core: star tips give 1.55x perimeter initially, relaxing to cylindrical
        # burn area while remaining geometrically consistent with cylindrical grain volume V_prop = -int A_b dw
        web_max = max(1e-4, r_o - r_c)
        w_eff = max(0.0, min(web_max, float(w_m)))
        prog = w_eff / web_max
        if w_eff >= web_max - 1e-7:
            return 0.0, case_vol, web_max
        # Geometric star-perforated effective core radius r_eff(w) starting larger than r_c -> neutral/sustain
        r_curr = min(r_o, r_c + w_eff)
        # Tapered radial burn area: higher initial perimeter compensated by cylindrical completion
        area_bore = 2.0 * math.pi * (r_c + 0.35 * web_max + 0.65 * w_eff) * len_tot * (1.0 - 0.15 * prog)
        prop_vol = math.pi * max(0.0, (r_o ** 2) - (r_curr ** 2)) * len_tot
        free_vol = max(case_vol * 0.04, case_vol - prop_vol)
        return area_bore, free_vol, web_max

    # Default: "bates" — N segments burning on cylindrical bore + 2*N uninhibited faces
    n_seg = max(1, int(n_segments))
    l_seg_init = len_tot / n_seg
    web_max = max(1e-4, min(r_o - r_c, 0.5 * l_seg_init))
    w_eff = max(0.0, min(web_max, float(w_m)))
    if w_eff >= web_max - 1e-7:
        return 0.0, case_vol, web_max

    r_curr = min(r_o, r_c + w_eff)
    l_curr = max(0.0, l_seg_init - 2.0 * w_eff)
    area_bore = n_seg * 2.0 * math.pi * r_curr * l_curr
    area_ends = n_seg * 2.0 * math.pi * max(0.0, (r_o ** 2) - (r_curr ** 2))
    area = area_bore + area_ends
    prop_vol = n_seg * math.pi * max(0.0, (r_o ** 2) - (r_curr ** 2)) * l_curr
    free_vol = max(case_vol * 0.04, case_vol - prop_vol)
    return area, free_vol, web_max


def simulate_solid_rocket_motor(
    c_star_m_s: float = 1560.0,
    cf_ambient: float = 1.60,
    cf_vacuum: float = 1.76,
    gamma: float = 1.20,
    propellant_density_g_cm3: float = 1.76,
    burn_rate_a: float = 4.2,          # mm/s/MPa^n (typical composite rocket propellant)
    burn_rate_n: float = 0.35,         # AP/HTPB pressure exponent (~0.30-0.45)
    grain_geometry: str = "bates",     # "bates", "end_burner", "star"
    grain_outer_diam_mm: float = 150.0,
    core_diam_mm: float = 45.0,
    grain_length_mm: float = 600.0,
    n_segments: int = 3,
    throat_diam_mm: float = 28.0,
    expansion_ratio_eps: float = 10.0,
    ambient_p_atm: float = 1.0,
    use_vacuum_thrust: bool = False,
    t_max_s: float = 12.0,
) -> Dict[str, Any]:
    """Simulate quasi-steady + chamber-fill/blowdown SRM pressure Pc(t) and thrust F(t)."""
    c_star = max(400.0, float(c_star_m_s))
    rho_p = max(500.0, float(propellant_density_g_cm3) * 1000.0)
    a_burn = max(1e-4, float(burn_rate_a) * 1e-3)
    n_exp = max(0.05, min(1.2, float(burn_rate_n)))
    r_outer_m = max(5e-3, 0.5 * float(grain_outer_diam_mm) * 1e-3)
    r_core_m = max(1e-3, min(r_outer_m * 0.88, 0.5 * float(core_diam_mm) * 1e-3))
    length_m = max(1e-2, float(grain_length_mm) * 1e-3)
    d_t_m = max(2e-3, float(throat_diam_mm) * 1e-3)
    a_throat = 0.25 * math.pi * (d_t_m ** 2)
    eps = max(1.0, float(expansion_ratio_eps))
    p_amb_pa = 0.0 if use_vacuum_thrust else max(0.0, float(ambient_p_atm) * ATM_TO_PA)

    big_gamma = _vandenkerckhove_gamma(gamma)
    rg_tc = (c_star * big_gamma) ** 2  # = gamma? No: c* = sqrt(Rg*Tc)/Gamma => Rg*Tc = (c* * Gamma)^2
    k_dp = max(gamma, 1.05) * rg_tc

    # Initial propellant volume & mass at w = 0
    _, free_vol_0, web_max_m = srm_grain_burn_area(
        0.0, grain_geometry, r_outer_m, r_core_m, length_m, n_segments
    )
    case_vol = math.pi * (r_outer_m ** 2) * length_m
    # Propellant mass = exact integral of rho_p * A_b(w) dw over [0, web_max] so mass conservation holds identically
    w_samples = np.linspace(0.0, web_max_m, 250)
    ab_samples = [
        srm_grain_burn_area(float(ws), grain_geometry, r_outer_m, r_core_m, length_m, n_segments)[0]
        for ws in w_samples
    ]
    prop_vol_0 = float(np.trapezoid(ab_samples, w_samples))
    propellant_mass_kg = max(0.01, prop_vol_0 * rho_p)

    # Extra free head/aft plenum volume (~12% of case) to keep blowdown realistic
    v_plenum = 0.12 * case_vol

    def odes(_t: float, y: np.ndarray) -> list[float]:
        w = float(y[0])
        p_c = max(float(y[1]), p_amb_pa)
        area_b, free_v, w_max = srm_grain_burn_area(
            w, grain_geometry, r_outer_m, r_core_m, length_m, n_segments
        )
        v_eff = max(1e-5, free_v + v_plenum)
        if w < w_max - 1e-7 and area_b > 0.0:
            p_mpa = max(0.05, p_c / 1e6)
            r_m_s = a_burn * (p_mpa ** n_exp)
            m_gen = rho_p * area_b * r_m_s
        else:
            r_m_s = 0.0
            m_gen = 0.0
        m_noz = ((p_c - p_amb_pa) * a_throat / c_star) if p_c > p_amb_pa else 0.0
        dp_dt = (k_dp / v_eff) * (m_gen - m_noz)
        return [r_m_s, dp_dt]

    def blowdown_complete(t: float, y: np.ndarray) -> float:
        if y[0] < 0.99 * web_max_m and t < 0.05:
            return 1.0
        p_stop = max(1.08 * p_amb_pa, 2.5e5)
        if y[0] >= 0.995 * web_max_m:
            return float(y[1]) - p_stop
        return 1.0

    blowdown_complete.terminal = True
    blowdown_complete.direction = -1

    p_init = max(p_amb_pa + 1.5e5, 3.0e5)
    sol = solve_ivp(
        odes,
        [0.0, t_max_s],
        [0.0, p_init],
        method="BDF",
        events=blowdown_complete,
        dense_output=True,
        max_step=min(0.02, t_max_s / 150.0),
        rtol=1e-5,
        atol=[1e-7, 1e2],
    )

    t_final = float(sol.t[-1]) if len(sol.t) > 0 else t_max_s
    t_grid = np.linspace(0.0, max(t_final, 1e-3), 300)
    if sol.sol is not None:
        y_vals = sol.sol(t_grid)
        w_arr = np.clip(y_vals[0], 0.0, web_max_m)
        p_arr = np.maximum(y_vals[1], p_amb_pa)
    else:
        t_grid = sol.t
        w_arr = np.clip(sol.y[0], 0.0, web_max_m)
        p_arr = np.maximum(sol.y[1], p_amb_pa)

    area_arr = np.zeros_like(t_grid)
    kn_arr = np.zeros_like(t_grid)
    mdot_arr = np.zeros_like(t_grid)
    thrust_arr = np.zeros_like(t_grid)
    cf_vac_eff = max(float(cf_vacuum), 1.2)

    for idx in range(len(t_grid)):
        ab, _, _ = srm_grain_burn_area(
            float(w_arr[idx]), grain_geometry, r_outer_m, r_core_m, length_m, n_segments
        )
        area_arr[idx] = ab
        kn_arr[idx] = ab / a_throat
        pc_i = float(p_arr[idx])
        mdot_arr[idx] = pc_i * a_throat / c_star if pc_i > p_amb_pa else 0.0
        if use_vacuum_thrust or p_amb_pa <= 0.0:
            thrust_arr[idx] = cf_vac_eff * pc_i * a_throat
        else:
            # Vacuum thrust minus exit-area backpressure loss: F = CF_vac * Pc * At - Pa * Ae
            thrust_arr[idx] = max(0.0, cf_vac_eff * pc_i * a_throat - p_amb_pa * (eps * a_throat))

    total_impulse = float(np.trapezoid(thrust_arr, t_grid)) if len(t_grid) > 1 else 0.0
    p_mpa_arr = p_arr / 1e6
    max_p_mpa = float(np.max(p_mpa_arr))
    burn_mask = p_mpa_arr >= max(0.15, 0.10 * max_p_mpa)
    if np.any(burn_mask):
        action_time_s = float(t_grid[burn_mask][-1] - t_grid[burn_mask][0])
        mean_p_mpa = float(np.mean(p_mpa_arr[burn_mask]))
        mean_thrust_n = float(total_impulse / max(action_time_s, 1e-4))
    else:
        action_time_s = float(t_grid[-1])
        mean_p_mpa = float(np.mean(p_mpa_arr))
        mean_thrust_n = 0.0

    delivered_isp_s = total_impulse / max(propellant_mass_kg * G0, 1e-6)
    cls_str = motor_impulse_class(total_impulse, mean_thrust_n)

    trace_df = pd.DataFrame({
        "time_s": t_grid,
        "pressure_mpa": p_mpa_arr,
        "thrust_n": thrust_arr,
        "kn": kn_arr,
        "mass_flow_kg_s": mdot_arr,
        "web_burned_mm": w_arr * 1000.0,
        "burn_area_cm2": area_arr * 1e4,
    })

    return {
        "motor_class": cls_str,
        "total_impulse_n_s": float(total_impulse),
        "propellant_mass_kg": float(propellant_mass_kg),
        "delivered_isp_s": float(delivered_isp_s),
        "max_pressure_mpa": max_p_mpa,
        "mean_pressure_mpa": mean_p_mpa,
        "max_thrust_n": float(np.max(thrust_arr)),
        "mean_thrust_n": mean_thrust_n,
        "action_time_s": action_time_s,
        "kn_initial": float(kn_arr[0]),
        "kn_max": float(np.max(kn_arr)),
        "web_max_mm": float(web_max_m * 1000.0),
        "throat_area_cm2": float(a_throat * 1e4),
        "trace_df": trace_df,
    }
