"""Kamlet-Jacobs (1968) empirical detonation velocity (VoD) and Chapman-Jouguet (CJ) screening.

Reference:
  Kamlet, M. J., & Jacobs, S. J. (1968). Chemistry of Detonations. I. A Simple Method for
  Calculating Detonation Properties of C-H-N-O Explosives. J. Chem. Phys., 48(1), 23-35.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from .models import ATOMIC_MASSES, Formulation, parse_formula

VOD_PRESETS: Dict[str, Optional[Dict[str, Any]]] = {
    "Custom": None,
    "TNT (C₇H₅N₃O₆)": dict(formula="C7H5N3O6", hf_kJ=-66.9, rho_tmd=1.654),
    "RDX (C₃H₆N₆O₆)": dict(formula="C3H6N6O6", hf_kJ=70.3, rho_tmd=1.806),
    "HMX (C₄H₈N₈O₈)": dict(formula="C4H8N8O8", hf_kJ=75.0, rho_tmd=1.905),
    "PETN (C₅H₈N₄O₁₂)": dict(formula="C5H8N4O12", hf_kJ=-538.5, rho_tmd=1.778),
    "TATB (C₆H₆N₆O₆)": dict(formula="C6H6N6O6", hf_kJ=-154.2, rho_tmd=1.937),
    "Tetryl (C₇H₅N₅O₈)": dict(formula="C7H5N5O8", hf_kJ=33.9, rho_tmd=1.730),
    "NQ / Nitroguanidine (CH₄N₄O₂)": dict(formula="CH4N4O2", hf_kJ=-92.4, rho_tmd=1.715),
    "FOX-7 (C₂H₄N₄O₄)": dict(formula="C2H4N4O4", hf_kJ=-133.7, rho_tmd=1.885),
    "CL-20 / HNIW (C₆H₆N₁₂O₁₂)": dict(formula="C6H6N12O12", hf_kJ=377.4, rho_tmd=2.044),
}


def parse_chno_formula(formula: str) -> Tuple[float, float, float, float]:
    """Extract (C, H, N, O) atom counts using full parenthetical/fractional parser."""
    comp = parse_formula(formula)
    return (
        float(comp.get("C", 0.0)),
        float(comp.get("H", 0.0)),
        float(comp.get("N", 0.0)),
        float(comp.get("O", 0.0)),
    )


def formulation_kj_inputs(form: Formulation) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Extract molar CHNO per gram and specific enthalpy of formation (kJ/g) for a Formulation."""
    try:
        elem = form.elemental_composition()
    except Exception:
        return None, "Could not compute elemental composition."

    a = float(elem.get("C", 0.0))
    b = float(elem.get("H", 0.0))
    c = float(elem.get("N", 0.0))
    d = float(elem.get("O", 0.0))
    if a + b + c + d < 1e-9:
        return None, "CHNO elemental composition is zero — check formulation components."

    hf_per_g = 0.0
    for comp in form.components:
        mw_comp = sum(ATOMIC_MASSES.get(el, 0.0) * cnt for el, cnt in comp.composition.items())
        if mw_comp > 0:
            hf_per_g += comp.mass_fraction * comp.hf_298 / mw_comp

    return dict(a=a, b=b, c=c, d=d, mw=1.0, hf_kJ=hf_per_g), None


def kamlet_jacobs(
    a: float,
    b: float,
    c: float,
    d: float,
    mw: float,
    hf_kJ_per_mol: float,
    rho: float,
) -> Dict[str, Any]:
    """Compute Kamlet-Jacobs (1968) CJ detonation velocity D (km/s) and pressure P_CJ (GPa).

    Kamlet & Jacobs (1968) product hierarchy:
      1. N -> 0.5 N2(g)
      2. H -> 0.5 H2O(g) using available O
      3. Remaining O -> 0.5 CO2(g) (ZERO CO formed; excess C stays solid graphite C(s))
      4. Any excess O beyond 2a + b/2 -> 0.5 O2(g)

    Equations (J. Chem. Phys. 48, 23):
      Phi = N * M_avg^0.5 * Q^0.5
      D [km/s]   = 1.01 * Phi^0.5 * (1 + 1.30 * rho_0)
      P_CJ [kbar] = 15.58 * rho_0^2 * Phi   ==>   P_CJ [GPa] = 1.558 * rho_0^2 * Phi
    """
    HF_H2O_CAL = -57_800.0  # cal/mol H2O(g) per Kamlet & Jacobs (1968)
    HF_CO2_CAL = -94_050.0  # cal/mol CO2(g) per Kamlet & Jacobs (1968)

    mw_eff = max(float(mw), 1e-9)
    hf_cal = float(hf_kJ_per_mol) * 1000.0 / 4.184

    n_N2 = 0.5 * c
    n_CO = 0.0  # Kamlet-Jacobs high-pressure Boudouard limit forms CO2 + C(s), never CO
    if d >= 0.5 * b:
        n_H2O = 0.5 * b
        n_H2 = 0.0
        o_rem = d - n_H2O
        n_CO2 = min(a, 0.5 * o_rem)
        n_Cs = max(0.0, a - n_CO2)
        n_O2 = max(0.0, 0.5 * (o_rem - 2.0 * n_CO2))
    else:
        n_H2O = max(0.0, d)
        n_H2 = max(0.0, 0.5 * b - n_H2O)
        n_CO2 = 0.0
        n_Cs = max(0.0, a)
        n_O2 = 0.0

    gas_mol = n_N2 + n_H2O + n_H2 + n_CO2 + n_O2
    N = gas_mol / mw_eff
    M_avg = (
        (28.014 * n_N2 + 18.015 * n_H2O + 2.016 * n_H2 + 44.009 * n_CO2 + 31.998 * n_O2) / gas_mol
        if gas_mol > 0
        else 28.014
    )
    Q = (hf_cal - (n_CO2 * HF_CO2_CAL + n_H2O * HF_H2O_CAL)) / mw_eff

    base_info = dict(
        N=N,
        M_avg=M_avg,
        Q_cal_per_g=Q,
        n_N2=n_N2,
        n_H2O=n_H2O,
        n_H2=n_H2,
        n_CO2=n_CO2,
        n_CO=n_CO,
        n_Cs=n_Cs,
        n_O2=n_O2,
        OB_pct=(d - 2.0 * a - 0.5 * b) / mw_eff * 1599.9,
    )
    if Q <= 0.0 or N <= 0.0:
        return dict(valid=False, **base_info)

    phi = N * (M_avg ** 0.5) * (Q ** 0.5)
    D_km_s = 1.01 * (phi ** 0.5) * (1.0 + 1.30 * rho)
    P_kbar = 15.58 * (rho ** 2) * phi
    P_GPa = 0.1 * P_kbar  # 1 GPa = 10 kbar
    gamma_cj = 3.0  # Classical condensed-phase detonation polytropic exponent

    return dict(
        valid=True,
        D_km_s=D_km_s,
        P_GPa=P_GPa,
        P_kbar=P_kbar,
        phi=phi,
        N_mol_per_g=N,
        M_avg_g_per_mol=M_avg,
        Q_kJ_per_g=Q * 4.184 / 1000.0,
        rho_CJ=rho * (gamma_cj + 1.0) / gamma_cj,
        u_CJ=D_km_s / (gamma_cj + 1.0),
        c_CJ=D_km_s * gamma_cj / (gamma_cj + 1.0),
        T_CJ_K=(Q * 4.184) / max(N * 32.0, 1e-6),
        dD_drho=1.01 * 1.30 * (phi ** 0.5),
        **base_info,
    )
