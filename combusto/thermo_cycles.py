"""Real-gas Noble-Abel vs. Ideal Gas polytropic expansion isentrope analyzer for Combusto.

Computes P-V-T thermodynamic state paths, cumulative reversible expansion work
integral(P dV), acoustic sound speed, and real-gas compressibility factor Z
across volumetric expansion ratios V / V0.
"""

from __future__ import annotations

from typing import Any, Dict
import math
import numpy as np
import pandas as pd

from .models import CombustionResult
from .thermodynamics import R_GAS


def compute_expansion_isentrope(
    result: CombustionResult,
    initial_loading_density_g_cm3: float = 0.25,
    covolume_cm3_per_g: float = 1.00,
    max_expansion_ratio: float = 20.0,
    n_points: int = 100,
) -> Dict[str, Any]:
    """Compute Noble-Abel and Ideal Gas adiabatic expansion paths from V0 to V_max.

    For a Noble-Abel gas with covolume eta (cm3/g) and specific volume v = 1/rho (cm3/g),
    the adiabatic isentrope satisfies:
        P * (v - eta)^gamma = const
        T * (v - eta)^(gamma - 1) = const
    """
    t_init_src = result.isochoric_flame_temp_k if getattr(result, "isochoric_flame_temp_k", 0.0) > 300.0 else result.adiabatic_flame_temp_k
    t0_k = max(300.0, float(t_init_src))
    gamma = max(1.05, float(result.gamma))
    f_j_per_g = max(100.0, float(result.force_constant_j_per_g))
    mw_g_mol = max(2.0, float(result.mean_molecular_weight))

    v0_cm3_g = 1.0 / max(0.02, min(0.85, initial_loading_density_g_cm3))
    eta = max(0.0, min(v0_cm3_g * 0.75, covolume_cm3_per_g))

    # Initial pressures (MPa): since F is in J/g and (v - eta) is in cm3/g,
    # P (MPa) = F (J/g) / (v - eta) (cm3/g)
    p0_noble_mpa = f_j_per_g / max(0.05, v0_cm3_g - eta)
    p0_ideal_mpa = f_j_per_g / v0_cm3_g

    vol_ratios = np.linspace(1.0, max(2.0, max_expansion_ratio), n_points)
    v_arr = v0_cm3_g * vol_ratios

    # Noble-Abel isentrope
    free_vol_ratio = (v_arr - eta) / max(0.05, v0_cm3_g - eta)
    p_noble_mpa = p0_noble_mpa * (free_vol_ratio ** (-gamma))
    t_noble_k = t0_k * (free_vol_ratio ** (-(gamma - 1.0)))

    # Ideal-gas isentrope
    p_ideal_mpa = p0_ideal_mpa * (vol_ratios ** (-gamma))
    t_ideal_k = t0_k * (vol_ratios ** (-(gamma - 1.0)))

    # Cumulative reversible work W(V) = integral_{v0}^{v} P dv (J/g)
    # Analytical integral of P0 * ((v - eta)/(v0 - eta))^(-gamma):
    work_noble_j_g = (f_j_per_g / (gamma - 1.0)) * (1.0 - (free_vol_ratio ** (-(gamma - 1.0))))
    work_ideal_j_g = (f_j_per_g / (gamma - 1.0)) * (1.0 - (vol_ratios ** (-(gamma - 1.0))))

    # Real-gas compressibility factor Z = P * v / (n_gas * R * T) = v / (v - eta)
    z_factor = v_arr / np.maximum(0.05, v_arr - eta)

    # Local acoustic sound speed a (m/s) in Noble-Abel gas:
    # a = (v / (v - eta)) * sqrt(gamma * R_spec * T)
    r_spec_j_kg_k = (R_GAS * 1000.0) / mw_g_mol
    sound_speed_m_s = z_factor * np.sqrt(gamma * r_spec_j_kg_k * t_noble_k)

    isentrope_df = pd.DataFrame({
        "volume_ratio_v_v0": vol_ratios,
        "specific_volume_cm3_g": v_arr,
        "p_noble_abel_mpa": p_noble_mpa,
        "p_ideal_gas_mpa": p_ideal_mpa,
        "t_noble_abel_k": t_noble_k,
        "t_ideal_gas_k": t_ideal_k,
        "cumulative_work_noble_j_g": work_noble_j_g,
        "cumulative_work_ideal_j_g": work_ideal_j_g,
        "compressibility_z": z_factor,
        "sound_speed_m_s": sound_speed_m_s,
    })

    return {
        "p0_noble_abel_mpa": float(p0_noble_mpa),
        "p0_ideal_gas_mpa": float(p0_ideal_mpa),
        "z0_compressibility": float(z_factor[0]),
        "max_work_noble_j_g": float(work_noble_j_g[-1]),
        "final_temp_noble_k": float(t_noble_k[-1]),
        "initial_sound_speed_m_s": float(sound_speed_m_s[0]),
        "isentrope_df": isentrope_df,
    }
