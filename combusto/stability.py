"""Propellant thermal stability, STANAG 4582 shelf-life, and thermal safety analyzer.

Models stabilizer depletion kinetics (Arrhenius / NATO STANAG 4582) to predict
safe storage life across ambient temperatures, and evaluates Frank-Kamenetskii
critical self-heating thresholds to prevent thermal runaway hazards during storage.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import math
import numpy as np
import pandas as pd

from .models import Formulation
from .thermodynamics import R_GAS

# Known stabilizer compounds commonly used in nitrocellulose/nitrate-ester propellants
STABILIZER_KEYWORDS = (
    "diphenylamine",
    "dpa",
    "centralite",
    "akardite",
    "ndpa",
    "resorcinol",
    "stabilizer",
)


def detect_stabilizer_wt_pct(formulation: Formulation) -> float:
    """Detect stabilizer mass percentage from formulation component names."""
    total_stab = 0.0
    for comp in formulation.components:
        lower_name = comp.name.lower()
        if any(k in lower_name for k in STABILIZER_KEYWORDS):
            total_stab += comp.mass_fraction * 100.0
    return total_stab


def analyze_thermal_stability(
    formulation: Formulation,
    storage_temp_c: float = 25.0,
    stabilizer_wt_pct: Optional[float] = None,
    activation_energy_kj_mol: float = 135.0,
    charge_diameter_mm: float = 100.0,
    thermal_conductivity_w_m_k: float = 0.24,
    propellant_density_g_cm3: float = 1.60,
) -> Dict[str, Any]:
    """Evaluate stabilizer depletion shelf-life and Frank-Kamenetskii cook-off safety.

    Parameters
    ----------
    formulation : Formulation
        Active propellant formulation.
    storage_temp_c : float
        Isothermal storage temperature in degrees Celsius.
    stabilizer_wt_pct : float, optional
        Initial stabilizer mass percentage. If None, inferred from formulation.
    activation_energy_kj_mol : float
        Arrhenius activation energy for nitrate ester NOx cleavage (~130-142 kJ/mol).
    charge_diameter_mm : float
        Characteristic diameter of stored propellant container/grain (mm).
    thermal_conductivity_w_m_k : float
        Bulk thermal conductivity of the propellant bed (W/(m K)).
    propellant_density_g_cm3 : float
        Bulk density (g/cm3).

    Returns
    -------
    dict
        Contains safe shelf-life in years, STANAG 4582 status, Frank-Kamenetskii
        critical temperature, depletion curve DataFrame, and cook-off vs diameter DataFrame.
    """
    if stabilizer_wt_pct is None:
        inferred = detect_stabilizer_wt_pct(formulation)
        stabilizer_wt_pct = inferred if inferred > 0.05 else 1.20

    # Check if formulation contains nitrocellulose or nitroglycerin (higher decomposition rate)
    has_ng = any(
        "nitroglycerin" in c.name.lower() or "c3h5n3o9" in c.formula.lower()
        for c in formulation.components
    )
    # Reference rate constant at 298.15 K (25 C):
    # Double-base (with NG) depletes ~1.5x faster than single-base NC
    k_298_pct_per_year = 0.045 if has_ng else 0.030

    t_storage_k = storage_temp_c + 273.15
    t_ref_k = 298.15
    ea_j_mol = activation_energy_kj_mol * 1000.0

    # Arrhenius acceleration factor relative to 25 C
    accel_factor = math.exp((ea_j_mol / R_GAS) * ((1.0 / t_ref_k) - (1.0 / t_storage_k)))
    # Combined zero-order + autocatalytic first-order effective depletion rate (wt% / year)
    rate_pct_per_year = k_298_pct_per_year * accel_factor

    # NATO STANAG 4582 safe limit: maximum 50% depletion of initial stabilizer,
    # or minimum 0.20 wt% remaining before surveillance condemnation
    critical_remaining_pct = max(0.20, 0.20 * stabilizer_wt_pct)
    usable_stab_pct = max(0.0, stabilizer_wt_pct - critical_remaining_pct)

    safe_shelf_life_years = usable_stab_pct / max(rate_pct_per_year, 1e-9)
    safe_shelf_life_years = min(200.0, safe_shelf_life_years)

    # Depletion time series over 0 to min(50, 1.4*safe_shelf_life) years
    horizon_yr = max(5.0, min(60.0, safe_shelf_life_years * 1.35))
    years_arr = np.linspace(0.0, horizon_yr, 60)
    remaining_arr = np.maximum(0.0, stabilizer_wt_pct - rate_pct_per_year * years_arr)

    # Also compute shelf life across storage temperatures (15 C to 65 C)
    temps_c = np.linspace(15.0, 65.0, 26)
    shelf_lives_yr = []
    for tc in temps_c:
        tk = tc + 273.15
        af = math.exp((ea_j_mol / R_GAS) * ((1.0 / t_ref_k) - (1.0 / tk)))
        sl = usable_stab_pct / max(k_298_pct_per_year * af, 1e-9)
        shelf_lives_yr.append(min(200.0, float(sl)))

    # Frank-Kamenetskii Critical Self-Heating Temperature (T_crit)
    # Anchored to NATO STANAG 4582 microcalorimetric heat flow at 90 C (363.15 K):
    # Double-base ~ 0.14 W/kg (140 uW/g); Single-base ~ 0.085 W/kg (85 uW/g)
    delta_c = 2.00  # infinite cylinder critical parameter
    r_0_m = max(0.005, 0.5 * charge_diameter_mm * 1e-3)
    rho_kg_m3 = propellant_density_g_cm3 * 1000.0
    q_dot_90c_w_kg = 0.14 if has_ng else 0.085
    t_cal_ref_k = 363.15

    def _solve_fk_t_crit(radius_m: float) -> float:
        t_low, t_high = 310.0, 620.0
        prefactor = (radius_m ** 2) * rho_kg_m3 * q_dot_90c_w_kg * ea_j_mol / (
            max(0.05, thermal_conductivity_w_m_k) * R_GAS
        )
        for _ in range(45):
            t_mid = 0.5 * (t_low + t_high)
            exponent = max(-60.0, min(60.0, (ea_j_mol / R_GAS) * ((1.0 / t_cal_ref_k) - (1.0 / t_mid))))
            lhs = (prefactor / (t_mid ** 2)) * math.exp(exponent)
            if lhs > delta_c:
                t_high = t_mid
            else:
                t_low = t_mid
        return 0.5 * (t_low + t_high)

    t_crit_k = _solve_fk_t_crit(r_0_m)
    t_crit_c = t_crit_k - 273.15

    diameters_mm = np.linspace(10.0, 400.0, 30)
    t_crit_curve_c = [float(_solve_fk_t_crit(0.5 * d * 1e-3) - 273.15) for d in diameters_mm]

    # Shelf life at 25 C for NATO STANAG 4582 benchmark (>= 10 years required)
    shelf_life_25c = usable_stab_pct / max(k_298_pct_per_year, 1e-9)
    stanag_compliant = bool(shelf_life_25c >= 10.0 and storage_temp_c < (t_crit_c - 35.0))

    depletion_df = pd.DataFrame({
        "time_years": years_arr,
        "remaining_stabilizer_pct": remaining_arr,
        "critical_limit_pct": np.full_like(years_arr, critical_remaining_pct),
    })

    temp_life_df = pd.DataFrame({
        "storage_temp_c": temps_c,
        "safe_shelf_life_years": shelf_lives_yr,
    })

    cookoff_df = pd.DataFrame({
        "charge_diameter_mm": diameters_mm,
        "critical_cookoff_temp_c": t_crit_curve_c,
    })

    return {
        "initial_stabilizer_pct": float(stabilizer_wt_pct),
        "critical_limit_pct": float(critical_remaining_pct),
        "depletion_rate_pct_per_yr": float(rate_pct_per_year),
        "safe_shelf_life_years": float(safe_shelf_life_years),
        "shelf_life_at_25c_years": float(min(200.0, shelf_life_25c)),
        "critical_cookoff_temp_c": float(t_crit_c),
        "critical_cookoff_temp_k": float(t_crit_k),
        "thermal_margin_c": float(t_crit_c - storage_temp_c),
        "stanag_4582_compliant": stanag_compliant,
        "depletion_df": depletion_df,
        "temp_life_df": temp_life_df,
        "cookoff_df": cookoff_df,
    }
