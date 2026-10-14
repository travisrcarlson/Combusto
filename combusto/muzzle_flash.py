"""Muzzle flash and afterburning plume predictor for Combusto.

Models the expansion of fuel-rich gun propellant exhaust at the muzzle,
shock-recovery heating across the Mach disk, turbulent mixing with ambient
air, radical scavenging by alkali flash suppressants (K2SO4 / KNO3), and
secondary plume auto-ignition and flame temperature.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import math
import numpy as np
import pandas as pd

from .models import CombustionResult, Formulation, parse_formula


def analyze_muzzle_flash(
    result: CombustionResult,
    formulation: Formulation,
    expansion_ratio: float = 12.0,
    muzzle_mach: float = 2.4,
    suppressant_wt_pct: Optional[float] = None,
    ambient_temp_k: float = 298.15,
) -> Dict[str, Any]:
    """Predict primary, intermediate, and secondary muzzle flash behavior.

    Parameters
    ----------
    result : CombustionResult
        Equilibrium combustion result inside the gun chamber.
    formulation : Formulation
        Active propellant formulation.
    expansion_ratio : float
        Volumetric expansion ratio V_total / V_chamber at projectile exit.
    muzzle_mach : float
        Mach number of the underexpanded exhaust jet entering the Mach disk.
    suppressant_wt_pct : float, optional
        Mass percentage of chemical flash suppressant (e.g., K2SO4, KNO3).
        If None, automatically inferred from K/Na content in the formulation.
    ambient_temp_k : float
        Ambient air temperature in Kelvin.

    Returns
    -------
    dict
        Contains scalar metrics, flash classification, and ``plume_df`` across
        air-to-exhaust mass mixing ratios.
    """
    t_chamber = max(result.adiabatic_flame_temp_k, 800.0)
    gamma = max(result.gamma, 1.12)
    cp_exhaust_j_g_k = 1.45  # typical specific heat of gun exhaust (J / (g K))
    if result.mean_molecular_weight > 0 and result.mixture_cp > 0:
        cp_exhaust_j_g_k = max(0.9, min(2.5, result.mixture_cp / result.mean_molecular_weight))

    # 1. Isentropic expansion from chamber to muzzle exit plane
    exp_ratio = max(1.2, expansion_ratio)
    t_exit_k = t_chamber / (exp_ratio ** (gamma - 1.0))

    # 2. Shock recovery temperature behind the muzzle Mach disk (intermediate flash zone)
    # Recovery factor r_rec ~ 0.85 for turbulent supersonic jet mixing layer
    r_rec = 0.85
    t_mach_disk_k = t_exit_k * (1.0 + r_rec * 0.5 * (gamma - 1.0) * (muzzle_mach ** 2))
    t_mach_disk_k = min(t_mach_disk_k, t_chamber * 0.96)

    # 3. Combustible gas inventory (moles per gram of propellant)
    moles_co = result.species_moles.get("CO", 0.0)
    moles_h2 = result.species_moles.get("H2", 0.0)
    moles_ch4 = result.species_moles.get("CH4", 0.0)
    moles_c2h2 = result.species_moles.get("C2H2", 0.0)
    moles_c2h4 = result.species_moles.get("C2H4", 0.0)

    # Combustible mole fraction in gas phase
    x_co = result.species_mole_fractions.get("CO", 0.0)
    x_h2 = result.species_mole_fractions.get("H2", 0.0)
    x_ch4 = result.species_mole_fractions.get("CH4", 0.0)
    combustible_mole_pct = (x_co + x_h2 + x_ch4) * 100.0

    # Heat of combustion of unburned exhaust per gram of propellant (J/g)
    # CO + 0.5 O2 -> CO2 (283.0 kJ/mol)
    # H2 + 0.5 O2 -> H2O(g) (241.8 kJ/mol)
    # CH4 + 2 O2 -> CO2 + 2 H2O(g) (802.3 kJ/mol)
    q_afterburn_j_per_g = (
        moles_co * 283000.0
        + moles_h2 * 241800.0
        + moles_ch4 * 802300.0
        + moles_c2h2 * 1255000.0
        + moles_c2h4 * 1323000.0
    )

    # Stoichiometric O2 moles required per gram of exhaust
    o2_req_mol_per_g = 0.5 * moles_co + 0.5 * moles_h2 + 2.0 * moles_ch4 + 2.5 * moles_c2h2 + 3.0 * moles_c2h4
    # Ambient air has 0.21 mol O2 per 28.97 g of air => 0.00725 mol O2 per gram of air
    stoich_air_to_exhaust_mass = o2_req_mol_per_g / 0.00725 if o2_req_mol_per_g > 0 else 0.5

    # 4. Alkali radical scavenger suppression (K / Na salts)
    if suppressant_wt_pct is None:
        # Infer from K or Na in formulation components
        inferred_pct = 0.0
        for comp in formulation.components:
            atoms = parse_formula(comp.formula)
            if "K" in atoms or "Na" in atoms or "k2so4" in comp.name.lower() or "kno3" in comp.name.lower():
                inferred_pct += comp.mass_fraction * 100.0
        suppressant_wt_pct = inferred_pct

    # Base auto-ignition temperature of H2/CO jet (K)
    # More H2 lowers ignition temperature toward ~830 K; pure CO is ~920 K
    h2_ratio = x_h2 / max(x_h2 + x_co, 1e-6)
    base_t_ign_k = 920.0 - 90.0 * h2_ratio
    # Alkali salts scavenge H and OH radicals in the shear layer, raising T_ign
    # Saturation effect above ~3 wt%
    supp_shift_k = 240.0 * (1.0 - math.exp(-0.65 * max(0.0, suppressant_wt_pct)))
    ignition_threshold_k = base_t_ign_k + supp_shift_k

    # 5. Sweep Air-to-Exhaust Mass Ratio (alpha = m_air / m_exhaust)
    alphas = np.linspace(0.0, 8.0, 81)
    cp_air_j_g_k = 1.01
    rows = []

    peak_plume_temp_k = t_mach_disk_k
    ignited_any = False

    for alpha in alphas:
        # Unreacted adiabatic mixing temperature in the Mach-disk shear layer
        t_mix = (cp_exhaust_j_g_k * t_mach_disk_k + alpha * cp_air_j_g_k * ambient_temp_k) / (
            cp_exhaust_j_g_k + alpha * cp_air_j_g_k
        )

        # Fraction of exhaust fuel that can oxidize at mixing ratio alpha
        if stoich_air_to_exhaust_mass > 1e-6:
            ox_frac = min(1.0, alpha / stoich_air_to_exhaust_mass)
        else:
            ox_frac = 0.0

        # Heat released per gram of exhaust at this mixing ratio
        q_rel = q_afterburn_j_per_g * ox_frac
        total_heat_cap = cp_exhaust_j_g_k + alpha * cp_air_j_g_k

        # Potential equilibrium plume temperature if ignited
        t_potential_plume = t_mix + (q_rel / total_heat_cap)

        # Ignition occurs in the shear layer if combustible gas is sufficient (>12%)
        # and local recovery/mixing temperature exceeds the radical ignition threshold
        local_ignited = (
            alpha >= 0.15
            and combustible_mole_pct >= 10.0
            and t_mach_disk_k >= ignition_threshold_k
            and t_mix >= (ignition_threshold_k - 140.0)
        )
        if local_ignited:
            ignited_any = True
            t_actual_plume = t_potential_plume
        else:
            # Partial recombination / non-ignited cooling plume
            t_actual_plume = t_mix + 0.08 * (q_rel / total_heat_cap)

        if t_actual_plume > peak_plume_temp_k:
            peak_plume_temp_k = float(t_actual_plume)

        rows.append({
            "air_to_exhaust_ratio": float(alpha),
            "unreacted_mix_temp_k": float(t_mix),
            "potential_afterburn_temp_k": float(t_potential_plume),
            "actual_plume_temp_k": float(t_actual_plume),
            "ignition_threshold_k": float(ignition_threshold_k),
            "ignited": bool(local_ignited),
        })

    # Soot incandescence index (0..100) based on PAH/soot fraction and exit temperature
    pah_c_frac = result.carbon_distribution.get("PAHs", 0.0)
    incandescence_index = min(100.0, (pah_c_frac * 600.0) * ((t_mach_disk_k / 1200.0) ** 2))

    # Classification
    if ignited_any and peak_plume_temp_k > 1500.0:
        flash_status = "HIGH (Severe Secondary Muzzle Flash)"
    elif t_mach_disk_k >= ignition_threshold_k - 50.0 or incandescence_index > 35.0:
        flash_status = "MODERATE (Intermediate Mach-Disk Glow)"
    else:
        flash_status = "LOW (Suppressed / Primary Glow Only)"

    return {
        "flash_status": flash_status,
        "ignited": ignited_any,
        "chamber_temp_k": float(t_chamber),
        "muzzle_exit_temp_k": float(t_exit_k),
        "mach_disk_temp_k": float(t_mach_disk_k),
        "ignition_threshold_k": float(ignition_threshold_k),
        "peak_plume_temp_k": float(peak_plume_temp_k),
        "combustible_mole_pct": float(combustible_mole_pct),
        "afterburn_energy_j_per_g": float(q_afterburn_j_per_g),
        "stoich_air_ratio": float(stoich_air_to_exhaust_mass),
        "suppressant_wt_pct": float(suppressant_wt_pct),
        "incandescence_index": float(incandescence_index),
        "plume_df": pd.DataFrame(rows),
    }
