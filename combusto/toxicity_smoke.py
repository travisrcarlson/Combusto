"""Exhaust plume condensation/smoke visibility and occupational toxicity (OSHA/NIOSH) analyzer.

Evaluates gaseous and particulate combustion emissions (CO, NOx, HCN, NH3, HCl, HF,
and carcinogenic PAHs weighted by EPA Benzo[a]pyrene Toxic Equivalency Factors)
against OSHA PEL / NIOSH REL workplace exposure limits, and models primary/secondary
exhaust condensation smoke across ambient temperature and relative humidity.
"""

from __future__ import annotations

from typing import Any, Dict, List
import math
import numpy as np
import pandas as pd

from .models import CombustionResult

# EPA Nisbet & LaGoy Benzo[a]pyrene Toxic Equivalency Factors (TEF) for PAHs
PAH_BA_P_TEF: Dict[str, float] = {
    "benzene": 0.0001,
    "toluene": 0.0001,
    "naphthalene": 0.001,
    "acenaphthylene": 0.001,
    "fluorene": 0.001,
    "phenanthrene": 0.001,
    "anthracene": 0.01,
    "fluoranthene": 0.001,
    "pyrene": 0.001,
    "chrysene": 0.01,
    "benzo[a]pyrene": 1.0,
    "coronene": 0.01,
    "soot_precursor": 0.05,
}

# Occupational exposure limits (mg/m3 at 25 C, 1 atm)
# Sources: OSHA PEL (8-hr TWA), NIOSH REL, and NIOSH IDLH
TOXIC_SPECIES_LIMITS: Dict[str, Dict[str, float]] = {
    "CO": {"mw": 28.01, "osha_pel_mg_m3": 55.0, "niosh_rel_mg_m3": 40.0, "idlh_mg_m3": 1375.0},
    "NO": {"mw": 30.01, "osha_pel_mg_m3": 30.0, "niosh_rel_mg_m3": 30.0, "idlh_mg_m3": 123.0},
    "NO2": {"mw": 46.01, "osha_pel_mg_m3": 9.0, "niosh_rel_mg_m3": 1.8, "idlh_mg_m3": 38.0},
    "HCN": {"mw": 27.03, "osha_pel_mg_m3": 11.0, "niosh_rel_mg_m3": 5.0, "idlh_mg_m3": 55.0},
    "NH3": {"mw": 17.03, "osha_pel_mg_m3": 35.0, "niosh_rel_mg_m3": 18.0, "idlh_mg_m3": 210.0},
    "HCl": {"mw": 36.46, "osha_pel_mg_m3": 7.0, "niosh_rel_mg_m3": 7.0, "idlh_mg_m3": 75.0},
    "HF": {"mw": 20.01, "osha_pel_mg_m3": 2.5, "niosh_rel_mg_m3": 2.5, "idlh_mg_m3": 25.0},
}


def analyze_toxicity_and_smoke(
    result: CombustionResult,
    charge_mass_g: float = 50.0,
    rounds_fired: int = 5,
    enclosure_volume_m3: float = 120.0,
    air_changes_per_hour: float = 6.0,
    ambient_temp_c: float = 15.0,
    relative_humidity_pct: float = 65.0,
) -> Dict[str, Any]:
    """Evaluate indoor range occupational toxicity and plume smoke visibility.

    Returns workplace concentrations vs OSHA PEL / NIOSH REL, BaP-TEQ carcinogenic
    PAH yield, required ventilation flow rate, and primary/secondary smoke indices.
    """
    total_prop_mass_g = max(0.1, charge_mass_g * max(1, rounds_fired))
    vol_m3 = max(5.0, enclosure_volume_m3)
    # Effective dilution over a 15-minute short-term exposure window with ventilation ACH
    ach_dilution_factor = math.exp(-0.25 * max(0.0, air_changes_per_hour))
    effective_vol_m3 = vol_m3 / max(0.05, ach_dilution_factor)

    tox_rows: List[Dict[str, Any]] = []
    max_hazard_quotient = 0.0
    limiting_pollutant = "None"
    peak_required_vol_m3 = vol_m3

    for sp_name, info in TOXIC_SPECIES_LIMITS.items():
        mass_frac = result.species_mass_fractions.get(sp_name, 0.0)
        yield_mg_per_g = mass_frac * 1000.0
        total_emitted_mg = yield_mg_per_g * total_prop_mass_g
        conc_mg_m3 = total_emitted_mg / effective_vol_m3

        pel = info["osha_pel_mg_m3"]
        rel = info["niosh_rel_mg_m3"]
        idlh = info["idlh_mg_m3"]
        hq_pel = conc_mg_m3 / pel

        if hq_pel > max_hazard_quotient:
            max_hazard_quotient = hq_pel
            limiting_pollutant = sp_name

        req_vol = total_emitted_mg / pel
        if req_vol > peak_required_vol_m3:
            peak_required_vol_m3 = req_vol

        status = "SAFE (< NIOSH REL)"
        if conc_mg_m3 >= idlh:
            status = "CRITICAL (>= IDLH)"
        elif conc_mg_m3 >= pel:
            status = "EXCEEDS OSHA PEL"
        elif conc_mg_m3 >= rel:
            status = "CAUTION (> NIOSH REL)"

        tox_rows.append({
            "Species": sp_name,
            "Yield (mg/g)": round(yield_mg_per_g, 3),
            "Total Emitted (mg)": round(total_emitted_mg, 1),
            "Range Conc (mg/m³)": round(conc_mg_m3, 3),
            "NIOSH REL (mg/m³)": rel,
            "OSHA PEL (mg/m³)": pel,
            "IDLH (mg/m³)": idlh,
            "Hazard Quotient (Conc/PEL)": round(hq_pel, 3),
            "Safety Status": status,
        })

    # Carcinogenic PAH Benzo[a]pyrene Toxic Equivalence (BaP-TEQ)
    bap_teq_ug_per_g = 0.0
    total_pah_mg_per_g = 0.0
    for pah_item in result.pah_details:
        p_name = str(pah_item.get("name", "")).lower()
        p_mf = float(pah_item.get("mass_fraction", 0.0))
        p_mg_g = p_mf * 1000.0
        total_pah_mg_per_g += p_mg_g
        tef = PAH_BA_P_TEF.get(p_name, 0.005)
        bap_teq_ug_per_g += p_mg_g * 1000.0 * tef  # ug BaP-TEQ per gram propellant

    bap_teq_conc_ug_m3 = (bap_teq_ug_per_g * total_prop_mass_g) / effective_vol_m3

    # Primary & Secondary Smoke Index (NATO STANAG 4400 / AGARD methodology)
    # 1. Primary Smoke: solid particulates (Soot C(s), PAHs, Al2O3, PbO, CuO, Fe2O3)
    solid_particulate_mf = (
        result.species_mass_fractions.get("C(s)", 0.0)
        + result.species_mass_fractions.get("soot_precursor", 0.0)
        + result.species_mass_fractions.get("Al2O3", 0.0)
        + result.species_mass_fractions.get("PbO", 0.0)
        + result.species_mass_fractions.get("CuO", 0.0)
        + result.species_mass_fractions.get("Fe2O3", 0.0)
        + (total_pah_mg_per_g / 1000.0)
    )
    primary_smoke_g_per_kg = solid_particulate_mf * 1000.0

    # 2. Secondary Smoke: H2O + HCl/HF vapor condensation upon cooling into humid air
    x_h2o = result.species_mole_fractions.get("H2O", 0.0)
    x_hcl = result.species_mole_fractions.get("HCl", 0.0)
    x_hf = result.species_mole_fractions.get("HF", 0.0)

    def _secondary_condensation_index(t_c: float, rh_pct: float) -> float:
        # Arden Buck saturation vapor pressure of water (kPa)
        p_sat_kpa = 0.61121 * math.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
        x_sat = p_sat_kpa / 101.325
        # Acid gases (HCl/HF) lower saturation vapor pressure by hygroscopic nucleation
        acid_factor = 1.0 / (1.0 + 8.0 * (x_hcl + x_hf))
        x_sat_eff = x_sat * acid_factor
        ambient_deficit = max(0.0005, x_sat_eff * (1.0 - 0.01 * min(99.5, rh_pct)))
        # Plume supersaturation index at dilution factor ~15
        plume_water = (x_h2o + 2.5 * (x_hcl + x_hf)) / 15.0
        cond_ratio = max(0.0, (plume_water - ambient_deficit) / max(ambient_deficit, 1e-4))
        return min(100.0, cond_ratio * 25.0)

    secondary_smoke_index = _secondary_condensation_index(ambient_temp_c, relative_humidity_pct)

    # NATO AGARD Smoke Classification (AA, A, B, C)
    if primary_smoke_g_per_kg < 5.0 and secondary_smoke_index < 15.0:
        agard_class = "Class AA (Minimum Smoke — Non-Condensing & Low Particulate)"
    elif primary_smoke_g_per_kg < 20.0 and secondary_smoke_index < 45.0:
        agard_class = "Class A (Reduced Smoke — Low Particulate, Mild Secondary Fog)"
    elif primary_smoke_g_per_kg < 60.0:
        agard_class = "Class B (Moderate Smoke)"
    else:
        agard_class = "Class C (High Smoke — Dense Particulate / Soot / Oxide Plume)"

    # 2D Humidity-Temperature Secondary Smoke Matrix for contour visualization
    t_grid_c = np.linspace(-20.0, 40.0, 25)
    rh_grid_pct = np.linspace(10.0, 100.0, 19)
    smoke_z = np.zeros((len(rh_grid_pct), len(t_grid_c)))
    for ir, rh_v in enumerate(rh_grid_pct):
        for it, tc_v in enumerate(t_grid_c):
            smoke_z[ir, it] = _secondary_condensation_index(float(tc_v), float(rh_v))

    # Minimum required fresh air flow rate (m3/min) to clear limiting pollutant within 15 min
    recommended_ventilation_m3_min = max(2.0, (peak_required_vol_m3 - vol_m3 * 0.2) / 15.0)

    return {
        "tox_df": pd.DataFrame(tox_rows),
        "max_hazard_quotient": float(max_hazard_quotient),
        "limiting_pollutant": limiting_pollutant,
        "bap_teq_ug_per_g": float(bap_teq_ug_per_g),
        "bap_teq_conc_ug_m3": float(bap_teq_conc_ug_m3),
        "primary_smoke_g_per_kg": float(primary_smoke_g_per_kg),
        "secondary_smoke_index": float(secondary_smoke_index),
        "agard_smoke_class": agard_class,
        "recommended_ventilation_m3_min": float(recommended_ventilation_m3_min),
        "t_grid_c": t_grid_c,
        "rh_grid_pct": rh_grid_pct,
        "smoke_matrix": smoke_z,
    }
