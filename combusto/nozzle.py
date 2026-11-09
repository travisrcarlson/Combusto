"""1D Shifting-Equilibrium vs. Frozen-Composition Rocket Nozzle Performance Solver.

Implements NASA CEA / Gordon & McBride (NASA RP-1311) homogeneous 1D isentropic
expansion from chamber pressure Pc through sonic throat Pt to exit pressure Pe:
  1. Frozen Composition: Product moles n_i fixed at chamber state (Pc, Tc);
     exact NASA 7-coeff polynomial entropy conservation S(n_c, T, P) = S_c.
  2. Shifting Equilibrium: Continuous Gibbs free-energy minimization n_eq(T, P)
     coupled with isentropic root-finding S(n_eq(T, P), T, P) = S_c, capturing
     exothermic radical/CO/H2 recombination during expansion.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from .equilibrium import solve_equilibrium, solve_equilibrium_adiabatic
from .models import (
    CombustionConditions,
    CombustionResult,
    Formulation,
    PAHAllocationConfig,
    Species,
)
from .pah_allocation import PRESETS
from .species_db import load_species_db
from .thermodynamics import R_GAS, cp, enthalpy, entropy

G0 = 9.80665        # Standard gravity (m/s^2)
ATM_TO_PA = 101325.0


def _mixture_enthalpy_entropy_j_per_g(
    species_moles: Dict[str, float],
    species_db: Dict[str, Species],
    T: float,
    P_atm: float,
) -> Tuple[float, float, float, float]:
    """Return (h_j_per_g, s_j_per_g_k, n_gas_mol_per_g, sound_speed_m_s).

    Matches the exact reference state of `thermodynamics.gibbs`:
    G_total = sum(n_i * H_i(T)) - T * [sum_gas n_i*(S0_i(T) - R*ln(x_i*P_atm)) + sum_cond n_i*S0_i(T)].
    """
    n_gas = sum(
        m for name, m in species_moles.items()
        if m > 1e-25 and name in species_db and species_db[name].phase == "gas"
    )
    h_tot = 0.0
    s_tot = 0.0
    cp_tot = 0.0
    for name, moles in species_moles.items():
        if moles <= 1e-25 or name not in species_db:
            continue
        sp = species_db[name]
        h_i = enthalpy(sp, T)
        s0_i = entropy(sp, T)
        cp_i = cp(sp, T)
        h_tot += moles * h_i
        cp_tot += moles * cp_i
        if sp.phase == "gas" and n_gas > 1e-25:
            x_i = moles / n_gas
            p_part = max(x_i * max(P_atm, 1e-12), 1e-30)
            s_tot += moles * (s0_i - R_GAS * math.log(p_part))
        else:
            s_tot += moles * s0_i

    # Frozen mixture sound speed a = sqrt(gamma_eff * n_gas * 1000 * R * T)
    cv_tot = max(cp_tot - n_gas * R_GAS, 1e-6)
    gamma_eff = max(1.05, cp_tot / cv_tot) if cp_tot > 0 else 1.2
    a_sound = math.sqrt(max(1.0, gamma_eff * n_gas * 1000.0 * R_GAS * T))
    return h_tot, s_tot, n_gas, a_sound


def _solve_frozen_state_at_p(
    chamber_moles: Dict[str, float],
    species_db: Dict[str, Species],
    s_c: float,
    h_c: float,
    t_c: float,
    p_atm: float,
) -> Dict[str, Any]:
    """Solve frozen-composition state at pressure `p_atm` with S(n_c, T, P) = s_c."""
    if abs(p_atm) <= 0:
        p_atm = 1e-6

    def s_res(t_val: float) -> float:
        _, s_val, _, _ = _mixture_enthalpy_entropy_j_per_g(
            chamber_moles, species_db, t_val, p_atm
        )
        return s_val - s_c

    t_lo = 150.0
    t_hi = max(t_c + 25.0, 400.0)
    f_lo = s_res(t_lo)
    f_hi = s_res(t_hi)
    if f_lo * f_hi <= 0.0:
        t_p = float(brentq(s_res, t_lo, t_hi, xtol=0.1))
    else:
        opt = minimize_scalar(lambda t: s_res(t) ** 2, bounds=(t_lo, t_hi), method="bounded")
        t_p = float(opt.x)

    h_p, s_p, n_gas, a_sound = _mixture_enthalpy_entropy_j_per_g(
        chamber_moles, species_db, t_p, p_atm
    )
    dh_j_kg = max(0.0, (h_c - h_p) * 1000.0)
    vel = math.sqrt(2.0 * dh_j_kg)
    p_pa = p_atm * ATM_TO_PA
    v_spec_m3_kg = (n_gas * 1000.0 * R_GAS * t_p) / max(p_pa, 1e-6)
    mass_flux = vel / max(v_spec_m3_kg, 1e-12)
    mach = vel / max(a_sound, 1.0)

    total_m = max(sum(chamber_moles.values()), 1e-30)
    mole_fracs = {k: v / total_m for k, v in chamber_moles.items()}

    return {
        "p_atm": p_atm,
        "temp_k": t_p,
        "h_j_per_g": h_p,
        "s_j_per_g_k": s_p,
        "velocity_m_s": vel,
        "sound_speed_m_s": a_sound,
        "mach": mach,
        "mass_flux_kg_m2_s": mass_flux,
        "n_gas_mol_per_g": n_gas,
        "mole_fractions": mole_fracs,
    }


def _solve_shifting_state_at_p(
    formulation: Formulation,
    pah_config: PAHAllocationConfig,
    species_db: Dict[str, Species],
    s_c: float,
    h_c: float,
    t_c: float,
    p_atm: float,
    t_hint: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve shifting-equilibrium state at pressure `p_atm` with S(n_eq(T,P), T, P) = s_c."""
    p_eff = max(p_atm, 1e-5)
    cond = CombustionConditions(temperature_k=298.15, pressure_atm=p_eff, mode="isobaric")
    last_eq: Dict[str, CombustionResult] = {}

    def eval_s(t_val: float) -> float:
        res = solve_equilibrium(
            formulation, cond, pah_config, species_db=species_db, temperature_k=t_val
        )
        last_eq["res"] = res
        _, s_val, _, _ = _mixture_enthalpy_entropy_j_per_g(
            res.species_moles, species_db, t_val, p_eff
        )
        return s_val - s_c

    t_lo = max(380.0, (t_hint - 250.0) if t_hint else 380.0)
    t_hi = min(max(t_c + 20.0, 500.0), (t_hint + 750.0) if t_hint else max(t_c + 20.0, 500.0))
    f_hi = eval_s(t_hi)
    f_lo = eval_s(t_lo)
    if f_hi < 0.0 and t_hi < t_c + 20.0:
        t_hi = max(t_c + 20.0, 500.0)
        f_hi = eval_s(t_hi)
    if f_lo > 0.0:
        t_lo = 380.0
        f_lo = eval_s(t_lo)
    if f_lo > 0.0:
        t_lo = 220.0
        f_lo = eval_s(t_lo)

    if f_lo * f_hi <= 0.0:
        t_p = float(brentq(eval_s, t_lo, t_hi, xtol=0.5, maxiter=35))
    else:
        opt = minimize_scalar(
            lambda t: eval_s(t) ** 2,
            bounds=(t_lo, t_hi),
            method="bounded",
            options={"xatol": 1.0, "maxiter": 25},
        )
        t_p = float(opt.x)

    eq_res = solve_equilibrium(
        formulation, cond, pah_config, species_db=species_db, temperature_k=t_p
    )
    h_p, s_p, n_gas, a_sound = _mixture_enthalpy_entropy_j_per_g(
        eq_res.species_moles, species_db, t_p, p_eff
    )
    dh_j_kg = max(0.0, (h_c - h_p) * 1000.0)
    vel = math.sqrt(2.0 * dh_j_kg)
    p_pa = p_eff * ATM_TO_PA
    v_spec_m3_kg = (n_gas * 1000.0 * R_GAS * t_p) / max(p_pa, 1e-6)
    mass_flux = vel / max(v_spec_m3_kg, 1e-12)
    mach = vel / max(a_sound, 1.0)

    return {
        "p_atm": p_eff,
        "temp_k": t_p,
        "h_j_per_g": h_p,
        "s_j_per_g_k": s_p,
        "velocity_m_s": vel,
        "sound_speed_m_s": a_sound,
        "mach": mach,
        "mass_flux_kg_m2_s": mass_flux,
        "n_gas_mol_per_g": n_gas,
        "mole_fractions": eq_res.species_mole_fractions,
    }


def analyze_nozzle_expansion(
    formulation: Formulation,
    pc_atm: float = 68.0,
    pe_atm: float = 1.0,
    ambient_p_atm: float = 1.0,
    pah_config: Optional[PAHAllocationConfig] = None,
    species_db: Optional[Dict[str, Species]] = None,
    n_profile_points: int = 12,
) -> Dict[str, Any]:
    """Compute Frozen-Composition vs. Shifting-Equilibrium 1D rocket nozzle performance.

    Returns chamber state, sonic throat metrics (C*), exit metrics (Ve, Te, Ae/At, CF,
    Isp_amb, Isp_vac) for both Frozen and Shifting models, plus a station profile DataFrame.
    """
    if species_db is None:
        species_db = load_species_db()
    if pah_config is None:
        pah_config = PRESETS["none"]

    pc_atm = max(float(pc_atm), 1.5)
    pe_atm = min(max(float(pe_atm), 0.001), pc_atm * 0.85)
    ambient_p_atm = max(float(ambient_p_atm), 0.0)

    chamber_cond = CombustionConditions(
        temperature_k=298.15, pressure_atm=pc_atm, mode="isobaric"
    )
    res_c = solve_equilibrium_adiabatic(
        formulation, chamber_cond, pah_config, species_db=species_db, t_guess=2600.0, tol=0.5, max_iter=35
    )
    t_c = res_c.adiabatic_flame_temp_k if res_c.adiabatic_flame_temp_k > 300.0 else res_c.temperature_k
    chamber_moles = res_c.species_moles
    h_c, s_c, n_gas_c, a_c = _mixture_enthalpy_entropy_j_per_g(
        chamber_moles, species_db, t_c, pc_atm
    )

    gamma_c = max(1.12, min(1.66, res_c.gamma if res_c.gamma > 1.0 else 1.22))
    pt_guess_ratio = (2.0 / (gamma_c + 1.0)) ** (gamma_c / (gamma_c - 1.0))

    # --- Sonic Throat maximization of mass flux G(P) = rho * V ---
    opt_t_froz = minimize_scalar(
        lambda pr: -_solve_frozen_state_at_p(
            chamber_moles, species_db, s_c, h_c, t_c, pr * pc_atm
        )["mass_flux_kg_m2_s"],
        bounds=(0.45, 0.68),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 18},
    )
    throat_froz = _solve_frozen_state_at_p(
        chamber_moles, species_db, s_c, h_c, t_c, float(opt_t_froz.x) * pc_atm
    )

    # Shifting throat near pt_guess_ratio (bracketed with frozen throat hint)
    opt_t_shift = minimize_scalar(
        lambda pr: -_solve_shifting_state_at_p(
            formulation, pah_config, species_db, s_c, h_c, t_c, pr * pc_atm, t_hint=throat_froz["temp_k"]
        )["mass_flux_kg_m2_s"],
        bounds=(max(0.46, pt_guess_ratio - 0.06), min(0.66, pt_guess_ratio + 0.06)),
        method="bounded",
        options={"xatol": 3e-3, "maxiter": 10},
    )
    throat_shift = _solve_shifting_state_at_p(
        formulation, pah_config, species_db, s_c, h_c, t_c, float(opt_t_shift.x) * pc_atm, t_hint=throat_froz["temp_k"]
    )

    # --- Exit Plane states at Pe ---
    exit_froz = _solve_frozen_state_at_p(
        chamber_moles, species_db, s_c, h_c, t_c, pe_atm
    )
    exit_shift = _solve_shifting_state_at_p(
        formulation, pah_config, species_db, s_c, h_c, t_c, pe_atm, t_hint=exit_froz["temp_k"]
    )

    pc_pa = pc_atm * ATM_TO_PA
    pe_pa = pe_atm * ATM_TO_PA
    pa_pa = ambient_p_atm * ATM_TO_PA

    def _compile_metrics(throat_st: Dict[str, Any], exit_st: Dict[str, Any]) -> Dict[str, float]:
        g_t = max(throat_st["mass_flux_kg_m2_s"], 1e-9)
        g_e = max(exit_st["mass_flux_kg_m2_s"], 1e-9)
        c_star = pc_pa / g_t
        eps = max(1.0, g_t / g_e)
        v_e = exit_st["velocity_m_s"]

        # Thrust per unit mass flow: F/m_dot = V_e + (P_e - P_a) * (A_e / m_dot) = V_e + (P_e - P_a) / G_e
        v_eff_amb = max(0.0, v_e + (pe_pa - pa_pa) / g_e)
        v_eff_vac = max(0.0, v_e + pe_pa / g_e)

        isp_matched = v_e / G0
        isp_amb = v_eff_amb / G0
        isp_vac = v_eff_vac / G0
        cf_amb = v_eff_amb / max(c_star, 1.0)
        cf_vac = v_eff_vac / max(c_star, 1.0)

        return {
            "c_star_m_s": float(c_star),
            "throat_pressure_atm": float(throat_st["p_atm"]),
            "throat_temp_k": float(throat_st["temp_k"]),
            "exit_temp_k": float(exit_st["temp_k"]),
            "exit_velocity_m_s": float(v_e),
            "exit_mach": float(exit_st["mach"]),
            "area_ratio_ae_at": float(eps),
            "cf_ambient": float(cf_amb),
            "cf_vacuum": float(cf_vac),
            "isp_matched_s": float(isp_matched),
            "isp_ambient_s": float(isp_amb),
            "isp_vacuum_s": float(isp_vac),
        }

    froz_metrics = _compile_metrics(throat_froz, exit_froz)
    shift_metrics = _compile_metrics(throat_shift, exit_shift)

    # --- Expansion Station Profile (Pc -> Pe) ---
    p_grid = np.geomspace(pc_atm, pe_atm, max(4, int(n_profile_points)))
    rows = []
    g_t_f = max(throat_froz["mass_flux_kg_m2_s"], 1e-9)
    g_t_s = max(throat_shift["mass_flux_kg_m2_s"], 1e-9)

    for idx, p_st in enumerate(p_grid):
        if idx == 0:
            total_mc = max(sum(chamber_moles.values()), 1e-30)
            mf_c = {k: v / total_mc for k, v in chamber_moles.items()}
            rows.append({
                "pressure_atm": float(pc_atm),
                "pressure_ratio_pc_p": 1.0,
                "t_frozen_k": float(t_c),
                "t_shifting_k": float(t_c),
                "v_frozen_m_s": 0.0,
                "v_shifting_m_s": 0.0,
                "area_ratio_frozen": np.nan,
                "area_ratio_shifting": np.nan,
                "x_CO_shifting": float(mf_c.get("CO", 0.0)),
                "x_CO2_shifting": float(mf_c.get("CO2", 0.0)),
                "x_H2_shifting": float(mf_c.get("H2", 0.0)),
                "x_H2O_shifting": float(mf_c.get("H2O", 0.0)),
            })
            continue

        sf = _solve_frozen_state_at_p(chamber_moles, species_db, s_c, h_c, t_c, float(p_st))
        ss = _solve_shifting_state_at_p(formulation, pah_config, species_db, s_c, h_c, t_c, float(p_st), t_hint=sf["temp_k"])
        mf_s = ss["mole_fractions"]
        rows.append({
            "pressure_atm": float(p_st),
            "pressure_ratio_pc_p": float(pc_atm / p_st),
            "t_frozen_k": float(sf["temp_k"]),
            "t_shifting_k": float(ss["temp_k"]),
            "v_frozen_m_s": float(sf["velocity_m_s"]),
            "v_shifting_m_s": float(ss["velocity_m_s"]),
            "area_ratio_frozen": float(g_t_f / max(sf["mass_flux_kg_m2_s"], 1e-9)),
            "area_ratio_shifting": float(g_t_s / max(ss["mass_flux_kg_m2_s"], 1e-9)),
            "x_CO_shifting": float(mf_s.get("CO", 0.0)),
            "x_CO2_shifting": float(mf_s.get("CO2", 0.0)),
            "x_H2_shifting": float(mf_s.get("H2", 0.0)),
            "x_H2O_shifting": float(mf_s.get("H2O", 0.0)),
        })

    profile_df = pd.DataFrame(rows)

    return {
        "chamber_result": res_c,
        "chamber_temp_k": float(t_c),
        "pc_atm": float(pc_atm),
        "pe_atm": float(pe_atm),
        "ambient_p_atm": float(ambient_p_atm),
        "frozen": froz_metrics,
        "shifting": shift_metrics,
        "isp_vacuum_gain_s": float(shift_metrics["isp_vacuum_s"] - froz_metrics["isp_vacuum_s"]),
        "isp_ambient_gain_s": float(shift_metrics["isp_ambient_s"] - froz_metrics["isp_ambient_s"]),
        "exit_temp_recombination_delta_k": float(shift_metrics["exit_temp_k"] - froz_metrics["exit_temp_k"]),
        "profile_df": profile_df,
    }
