"""Internal ballistics, multi-perforated grain geometry, and barrel erosion solver for Combusto.

Simulates the transient pressure and projectile dynamics in a gun barrel
using Noble-Abel equation of state, Saint-Robert's burn rate law,
progressive multi-perforated grain form functions (including sliver burnout),
2D grain cross-sectional burnback geometry, and thermal/chemical barrel erosion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
from scipy.integrate import solve_ivp

from .models import CombustionResult


def grain_form_function(u: float, grain_geometry: str) -> Tuple[float, float]:
    """Return (z, dz_du) for normalized web burn fraction *u* = 2 * x_b / D0.

    Supports:
    - ``"spherical"``   : Regressive ball powder (0 <= u <= 1)
    - ``"cord"``        : Regressive cylindrical strand (0 <= u <= 1)
    - ``"single_perf"`` : Neutral single-perforated tube (0 <= u <= 1)
    - ``"7_perf"``      : Progressive 7-perforated cylinder with sliver phase (0 <= u <= 1.18)
    - ``"19_perf"``     : Highly progressive 19-perforated rosette with sliver phase (0 <= u <= 1.15)
    """
    if grain_geometry == "spherical":
        u_c = max(0.0, min(1.0, u))
        if u >= 1.0:
            return 1.0, 0.0
        z = 3.0 * u_c - 3.0 * (u_c ** 2) + (u_c ** 3)
        dz_du = 3.0 - 6.0 * u_c + 3.0 * (u_c ** 2)
        return z, dz_du

    elif grain_geometry == "cord":
        u_c = max(0.0, min(1.0, u))
        if u >= 1.0:
            return 1.0, 0.0
        z = 2.0 * u_c - (u_c ** 2)
        dz_du = 2.0 - 2.0 * u_c
        return z, dz_du

    elif grain_geometry == "7_perf":
        # Phase I (0 <= u <= 1.0): progressive burning up to web breakup (z1 = 0.852)
        # Phase II (1.0 < u <= 1.18): degressive burning of 12 triangular slivers
        u_max = 1.18
        z1 = 0.852
        chi = 0.72
        lam = 0.24
        mu = -0.056
        if u <= 0.0:
            return 0.0, chi
        elif u <= 1.0:
            z = chi * u * (1.0 + lam * u + mu * (u ** 2))
            dz_du = chi * (1.0 + 2.0 * lam * u + 3.0 * mu * (u ** 2))
            return min(z, z1), dz_du
        elif u < u_max:
            s = (u - 1.0) / (u_max - 1.0)
            z = z1 + (1.0 - z1) * (1.0 - (1.0 - s) ** 2)
            dz_du = (1.0 - z1) * 2.0 * (1.0 - s) / (u_max - 1.0)
            return min(z, 1.0), dz_du
        else:
            return 1.0, 0.0

    elif grain_geometry == "19_perf":
        # Phase I (0 <= u <= 1.0): strongly progressive burning up to web breakup (z1 = 0.885)
        # Phase II (1.0 < u <= 1.15): degressive sliver burnout
        u_max = 1.15
        z1 = 0.885
        chi = 0.68
        lam = 0.36
        mu = -0.058
        if u <= 0.0:
            return 0.0, chi
        elif u <= 1.0:
            z = chi * u * (1.0 + lam * u + mu * (u ** 2))
            dz_du = chi * (1.0 + 2.0 * lam * u + 3.0 * mu * (u ** 2))
            return min(z, z1), dz_du
        elif u < u_max:
            s = (u - 1.0) / (u_max - 1.0)
            z = z1 + (1.0 - z1) * (1.0 - (1.0 - s) ** 2)
            dz_du = (1.0 - z1) * 2.0 * (1.0 - s) / (u_max - 1.0)
            return min(z, 1.0), dz_du
        else:
            return 1.0, 0.0

    else:  # "single_perf"
        u_c = max(0.0, min(1.0, u))
        if u >= 1.0:
            return 1.0, 0.0
        return u_c, 1.0


def simulate_ballistics(
    force_constant_j_per_g: float,
    gamma: float,
    charge_mass_g: float,
    propellant_density_g_per_cm3: float,
    covolume_cm3_per_g: float,
    burn_rate_a: float,      # mm/s/MPa^n
    burn_rate_n: float,      # exponent
    web_thickness_mm: float,
    grain_geometry: str,     # "spherical", "cord", "single_perf", "7_perf", "19_perf"
    chamber_volume_cm3: float,
    bore_diameter_mm: float,
    barrel_length_m: float,
    projectile_mass_g: float,
    start_pressure_mpa: float = 30.0,
    resistive_pressure_mpa: float = 5.0,
    t_end: float = 0.05,     # max time in seconds
) -> Dict[str, Any]:
    """Simulate interior ballistics of a gun firing.

    Returns a dictionary of time-series arrays and scalar summary metrics.
    """
    F = force_constant_j_per_g * 1000.0          # J/g -> J/kg
    M_p = charge_mass_g / 1000.0                 # g -> kg
    rho_p = propellant_density_g_per_cm3 * 1000.0  # g/cm3 -> kg/m3
    eta = covolume_cm3_per_g * 1e-3              # cm3/g -> m3/kg

    a_burn = burn_rate_a * 1e-3                  # mm/s -> m/s
    D0 = web_thickness_mm * 1e-3                 # mm -> m
    V_c = chamber_volume_cm3 * 1e-6              # cm3 -> m3
    d_bore = bore_diameter_mm * 1e-3             # mm -> m
    A_bore = math.pi * d_bore**2 / 4.0           # m2
    m_proj = projectile_mass_g / 1000.0          # g -> kg

    P_start = start_pressure_mpa * 1e6           # MPa -> Pa
    P_res = resistive_pressure_mpa * 1e6         # MPa -> Pa

    u_max = 1.18 if grain_geometry == "7_perf" else (1.15 if grain_geometry == "19_perf" else 1.0)

    # State vector: [x, v, x_b, z]
    y0 = [0.0, 0.0, 0.0, 0.001]

    def odes(t: float, y: List[float]) -> List[float]:
        x_travel = y[0]
        velocity = y[1]
        x_burnt = y[2]
        z_burned = max(0.0, min(1.0, y[3]))

        V_prop_remaining = M_p * (1.0 - z_burned) / rho_p
        V_gas = V_c - V_prop_remaining + A_bore * x_travel
        m_gas = M_p * z_burned

        # Account for projectile kinetic energy reducing gas temperature (Resal's energy equation)
        ke_proj = 0.5 * (m_proj + M_p / 3.0) * (velocity ** 2)
        effective_energy = max(0.1 * m_gas * F, m_gas * F - (gamma - 1.0) * ke_proj)

        denom = V_gas - m_gas * eta
        if denom > 1e-9:
            P = effective_energy / denom
        else:
            P = 1e9

        # Purely state-determined shot-start (safe under adaptive RK45 step rejections)
        is_moving = (x_travel > 1e-9) or (velocity > 1e-6) or (P >= P_start)
        if is_moving:
            # Secondary work factor (Lagrange gradient: m_eff = m_proj + M_p / 3)
            force = (P - P_res) * A_bore
            accel = force / (m_proj + M_p / 3.0) if force > 0 else 0.0
        else:
            accel = 0.0
            velocity = 0.0

        P_mpa = max(1e-3, P / 1e6)
        u = 2.0 * x_burnt / D0
        if u < u_max and z_burned < 0.9995:
            r = a_burn * (P_mpa ** burn_rate_n)
            _, dz_du = grain_form_function(u, grain_geometry)
            dz_dt = dz_du * (2.0 * r / D0)
        else:
            r = 0.0
            dz_dt = 0.0

        return [velocity, accel, r, dz_dt]

    def projectile_left(t: float, y: List[float]) -> float:
        return barrel_length_m - y[0]

    projectile_left.terminal = True
    projectile_left.direction = -1

    sol = solve_ivp(
        odes,
        [0.0, t_end],
        y0,
        events=projectile_left,
        dense_output=True,
        method="RK45",
        max_step=5e-5,
        rtol=1e-5,
        atol=1e-8,
    )

    t_final = float(sol.t[-1]) if len(sol.t) > 0 else t_end
    if sol.sol is not None and t_final > 0.0:
        times = np.linspace(0.0, t_final, 400)
        y_interp = sol.sol(times)
        x_arr = np.maximum(0.0, y_interp[0])
        v_arr = np.maximum(0.0, y_interp[1])
        xb_arr = np.maximum(0.0, y_interp[2])
        z_arr = np.clip(y_interp[3], 0.0, 1.0)
    else:
        times = sol.t
        x_arr = sol.y[0]
        v_arr = sol.y[1]
        xb_arr = sol.y[2]
        z_arr = np.clip(sol.y[3], 0.0, 1.0)

    p_arr = []
    for x_t, v_t, z_t in zip(x_arr, v_arr, z_arr):
        V_prop_remaining = M_p * (1.0 - z_t) / rho_p
        V_gas = V_c - V_prop_remaining + A_bore * x_t
        m_gas = M_p * z_t
        ke_proj = 0.5 * (m_proj + M_p / 3.0) * (v_t ** 2)
        effective_energy = max(0.1 * m_gas * F, m_gas * F - (gamma - 1.0) * ke_proj)
        denom = V_gas - m_gas * eta
        P = effective_energy / denom if denom > 1e-9 else 1e3
        p_arr.append(P / 1e6)

    return {
        "time": times,
        "travel_m": x_arr,
        "velocity_m_s": v_arr,
        "burnt_depth_mm": xb_arr * 1000.0,
        "fraction_burned": z_arr,
        "pressure_mpa": np.array(p_arr),
        "muzzle_velocity_m_s": float(v_arr[-1]),
        "max_pressure_mpa": float(np.max(p_arr)),
        "burn_time_ms": float(times[-1] * 1000.0),
        "completed": bool(x_arr[-1] >= barrel_length_m),
    }


def generate_grain_cross_section(
    grain_geometry: str,
    web_thickness_mm: float = 1.2,
    burn_fraction_u: float = 0.0,
) -> Dict[str, Any]:
    """Generate 2D cross-section circles for visualizing grain burnback.

    Returns outer radius, perforation centers, perforation radii, and initial
    reference geometry in mm.
    """
    w = max(0.1, web_thickness_mm)
    xb = 0.5 * w * max(0.0, burn_fraction_u)

    if grain_geometry in ("spherical", "cord"):
        r_outer_init = 0.5 * w
        r_outer_cur = max(0.0, r_outer_init - xb)
        return {
            "outer_r_init": r_outer_init,
            "outer_r_cur": r_outer_cur,
            "perf_centers": [],
            "perf_r_init": 0.0,
            "perf_r_cur": 0.0,
        }
    elif grain_geometry == "single_perf":
        r_perf_init = 0.25 * w
        r_outer_init = r_perf_init + w
        return {
            "outer_r_init": r_outer_init,
            "outer_r_cur": max(0.0, r_outer_init - xb),
            "perf_centers": [(0.0, 0.0)],
            "perf_r_init": r_perf_init,
            "perf_r_cur": r_perf_init + xb,
        }
    elif grain_geometry == "7_perf":
        # 1 center hole + 6 ring holes separated by web w
        r_perf_init = 0.22 * w
        ring_pitch = 2.0 * r_perf_init + w
        r_outer_init = ring_pitch + r_perf_init + w
        centers = [(0.0, 0.0)]
        for k in range(6):
            ang = k * math.pi / 3.0
            centers.append((ring_pitch * math.cos(ang), ring_pitch * math.sin(ang)))
        return {
            "outer_r_init": r_outer_init,
            "outer_r_cur": max(0.0, r_outer_init - xb),
            "perf_centers": centers,
            "perf_r_init": r_perf_init,
            "perf_r_cur": r_perf_init + xb,
        }
    else:  # "19_perf"
        r_perf_init = 0.20 * w
        pitch1 = 2.0 * r_perf_init + w
        pitch2 = 2.0 * pitch1
        r_outer_init = pitch2 + r_perf_init + w
        centers = [(0.0, 0.0)]
        for k in range(6):
            ang = k * math.pi / 3.0
            centers.append((pitch1 * math.cos(ang), pitch1 * math.sin(ang)))
        for k in range(12):
            ang = k * math.pi / 6.0
            r_ring = pitch2 if k % 2 == 0 else pitch1 * math.sqrt(3.0)
            centers.append((r_ring * math.cos(ang), r_ring * math.sin(ang)))
        return {
            "outer_r_init": r_outer_init,
            "outer_r_cur": max(0.0, r_outer_init - xb),
            "perf_centers": centers,
            "perf_r_init": r_perf_init,
            "perf_r_cur": r_perf_init + xb,
        }


def estimate_barrel_erosion(
    ib_out: Dict[str, Any],
    result: Optional[CombustionResult],
    bore_diameter_mm: float = 30.0,
    charge_mass_g: float = 50.0,
    initial_wall_temp_k: float = 300.0,
) -> Dict[str, Any]:
    """Estimate bore wall surface temperature and thermochemical barrel erosion.

    Couples interior ballistics pressure/velocity profiles with gas-phase
    corrosivity (CO/CO2 and H2/H2O) using Lawton/Thornhill wear correlations.
    """
    times = np.asarray(ib_out["time"], dtype=float)
    p_mpa = np.asarray(ib_out["pressure_mpa"], dtype=float)
    v_ms = np.asarray(ib_out["velocity_m_s"], dtype=float)

    t_gas_peak = 2850.0
    x_co = 0.25
    x_co2 = 0.12
    x_h2 = 0.12
    x_h2o = 0.18
    if result is not None and result.adiabatic_flame_temp_k > 300.0:
        t_gas_peak = float(result.adiabatic_flame_temp_k)
        x_co = result.species_mole_fractions.get("CO", 0.20)
        x_co2 = result.species_mole_fractions.get("CO2", 0.10)
        x_h2 = result.species_mole_fractions.get("H2", 0.10)
        x_h2o = result.species_mole_fractions.get("H2O", 0.15)

    # Thermal effusivity of CrMoV gun steel: sqrt(k * rho * c) ~ 13,500 J / (m^2 K s^0.5)
    effusivity = 13500.0
    d_m = max(0.004, bore_diameter_mm * 1e-3)

    # Compute convective heat flux q_conv(t) and semi-infinite wall surface T_wall(t)
    p_max = max(float(np.max(p_mpa)), 1.0)
    t_wall = np.zeros_like(times)
    q_flux_mw_m2 = np.zeros_like(times)

    for i, t_s in enumerate(times):
        # Local gas velocity at origin of rifling (~0.35 * projectile velocity)
        v_local = max(15.0, 0.35 * v_ms[i])
        # Krause-style Nusselt correlation h_c (W / (m^2 K))
        h_c = 180.0 * ((p_mpa[i] / 100.0) ** 0.8) * ((v_local / 100.0) ** 0.6) * ((0.03 / d_m) ** 0.2)
        t_gas_inst = initial_wall_temp_k + (t_gas_peak - initial_wall_temp_k) * ((p_mpa[i] / p_max) ** 0.22)
        dt_eff = max(1e-5, t_s)
        # Transient wall temperature rise with thermal boundary layer damping
        delta_t = (t_gas_inst - initial_wall_temp_k) * (
            1.0 - math.exp(-1.15 * h_c * math.sqrt(dt_eff) / effusivity)
        )
        t_wall[i] = initial_wall_temp_k + delta_t
        q_flux_mw_m2[i] = max(0.0, h_c * (t_gas_inst - t_wall[i]) / 1e6)

    peak_wall_temp_k = float(np.max(t_wall))
    peak_heat_flux_mw_m2 = float(np.max(q_flux_mw_m2))

    # Chemical reactivity factor (CO carburization & H2 embrittlement vs oxidizing CO2/H2O)
    co_ratio = x_co / max(x_co2, 0.01)
    h2_ratio = x_h2 / max(x_h2o, 0.01)
    chem_corrosivity = 0.6 + 0.25 * min(4.0, co_ratio) + 0.20 * min(4.0, h2_ratio)

    # Lawton Arrhenius wear per shot (micrometers per shot at origin of rifling)
    # Activation energy E_a ~ 65 kJ/mol above 850 K steel transformation onset
    if peak_wall_temp_k > 750.0:
        wear_um_per_shot = (
            145.0
            * chem_corrosivity
            * math.exp(-6800.0 / peak_wall_temp_k)
            * ((charge_mass_g / 50.0) ** 0.35)
        )
    else:
        wear_um_per_shot = 0.02 * chem_corrosivity

    wear_um_per_shot = max(0.01, float(wear_um_per_shot))
    # Condemnation limit: typically 1.5% of bore diameter (in um)
    condemn_limit_um = 0.015 * bore_diameter_mm * 1000.0
    estimated_barrel_life_rounds = int(min(50000, max(10, condemn_limit_um / wear_um_per_shot)))

    if peak_wall_temp_k >= 1720.0:
        erosion_regime = "CRITICAL (Bore Surface Melt / Ablation)"
    elif peak_wall_temp_k >= 1000.0:
        erosion_regime = "HIGH (Austenite Phase Transformation & Carburization)"
    elif peak_wall_temp_k >= 800.0:
        erosion_regime = "MODERATE (Thermal Fatigue & Gas Wash)"
    else:
        erosion_regime = "LOW (Benign Thermal Regime)"

    return {
        "wall_temp_k": t_wall,
        "heat_flux_mw_m2": q_flux_mw_m2,
        "peak_wall_temp_k": peak_wall_temp_k,
        "peak_wall_temp_c": peak_wall_temp_k - 273.15,
        "peak_heat_flux_mw_m2": peak_heat_flux_mw_m2,
        "chem_corrosivity_index": float(chem_corrosivity),
        "co_co2_ratio": float(co_ratio),
        "h2_h2o_ratio": float(h2_ratio),
        "wear_um_per_shot": wear_um_per_shot,
        "estimated_barrel_life_rounds": estimated_barrel_life_rounds,
        "erosion_regime": erosion_regime,
    }
