"""Thermodynamic property evaluation using NASA 7-coefficient polynomials.

NASA polynomial form (dimensionless):
    Cp/R  = a1 + a2*T + a3*T² + a4*T³ + a5*T⁴
    H/RT  = a1 + a2*T/2 + a3*T²/3 + a4*T³/4 + a5*T⁴/5 + a6/T
    S/R   = a1*ln(T) + a2*T + a3*T²/2 + a4*T³/3 + a5*T⁴/4 + a7

where R = 8.314462 J/(mol·K).
"""

from __future__ import annotations

import math
from typing import List, Tuple

from .models import Species

R_GAS = 8.314462  # J/(mol·K)


def _get_coeffs(species: Species, T: float) -> List[float]:
    """Select low or high T coefficients."""
    if T <= species.t_mid:
        return species.thermo_low
    return species.thermo_high


def _poly_cp_over_r(a: List[float], T: float) -> float:
    return a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4


def _poly_h_over_rt(a: List[float], T: float) -> float:
    return a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T


def _poly_s_over_r(a: List[float], T: float) -> float:
    return (a[0]*math.log(T) + a[1]*T + a[2]*T**2/2 +
            a[3]*T**3/3 + a[4]*T**3*T/4 + a[6])


def cp_over_r(species: Species, T: float) -> float:
    """Cp/R at temperature T, with NASA CEA constant-Cp clamping outside [t_low, t_high]."""
    t_clamped = min(max(T, species.t_low), species.t_high)
    a = _get_coeffs(species, t_clamped)
    return _poly_cp_over_r(a, t_clamped)


def h_over_rt(species: Species, T: float) -> float:
    """H/(RT) at temperature T, with C1-continuous linear enthalpy extrapolation outside [t_low, t_high]."""
    t_safe = max(T, 50.0)
    if t_safe > species.t_high:
        t_b = species.t_high
        a_b = _get_coeffs(species, t_b)
        h_b_over_r = _poly_h_over_rt(a_b, t_b) * t_b
        cp_b_over_r = _poly_cp_over_r(a_b, t_b)
        return (h_b_over_r + cp_b_over_r * (t_safe - t_b)) / t_safe
    if t_safe < species.t_low:
        t_b = species.t_low
        a_b = _get_coeffs(species, t_b)
        h_b_over_r = _poly_h_over_rt(a_b, t_b) * t_b
        cp_b_over_r = _poly_cp_over_r(a_b, t_b)
        return (h_b_over_r + cp_b_over_r * (t_safe - t_b)) / t_safe
    a = _get_coeffs(species, t_safe)
    return _poly_h_over_rt(a, t_safe)


def s_over_r(species: Species, T: float) -> float:
    """S/R at temperature T, with C1-continuous logarithmic entropy extrapolation outside [t_low, t_high]."""
    t_safe = max(T, 50.0)
    if t_safe > species.t_high:
        t_b = species.t_high
        a_b = _get_coeffs(species, t_b)
        s_b_over_r = _poly_s_over_r(a_b, t_b)
        cp_b_over_r = _poly_cp_over_r(a_b, t_b)
        return s_b_over_r + cp_b_over_r * math.log(t_safe / t_b)
    if t_safe < species.t_low:
        t_b = species.t_low
        a_b = _get_coeffs(species, t_b)
        s_b_over_r = _poly_s_over_r(a_b, t_b)
        cp_b_over_r = _poly_cp_over_r(a_b, t_b)
        return s_b_over_r + cp_b_over_r * math.log(t_safe / t_b)
    a = _get_coeffs(species, t_safe)
    return _poly_s_over_r(a, t_safe)


def cp(species: Species, T: float) -> float:
    """Heat capacity Cp in J/(mol·K)."""
    return cp_over_r(species, T) * R_GAS


def enthalpy(species: Species, T: float) -> float:
    """Molar enthalpy H in J/mol."""
    return h_over_rt(species, T) * R_GAS * T


def entropy(species: Species, T: float) -> float:
    """Molar entropy S in J/(mol·K)."""
    return s_over_r(species, T) * R_GAS


def gibbs(species: Species, T: float, p_atm: float = 1.0) -> float:
    """Molar Gibbs free energy G in J/mol at given T and partial pressure.

    G = H - T*S + RT*ln(p/p0) for ideal gas (p0 = 1 atm).
    For condensed species, pressure correction is ignored.
    """
    H = enthalpy(species, T)
    S = entropy(species, T)
    G = H - T * S
    if species.phase == "gas" and p_atm > 0:
        G += R_GAS * T * math.log(max(p_atm, 1e-30))
    return G


def mixture_cp_mol(species_list: list, mole_fractions: dict, T: float) -> float:
    """Mixture Cp in J/(mol_mix·K) from mole-fraction-weighted species Cp."""
    cp_mix = 0.0
    for sp in species_list:
        xi = mole_fractions.get(sp.name, 0.0)
        if xi > 0:
            cp_mix += xi * cp(sp, T)
    return cp_mix


def adiabatic_flame_temperature(
    species_list: list,
    mole_fractions: dict,
    h_reactants: float,
    T_guess: float = 2500.0,
    p_atm: float = 1.0,
    tol: float = 0.5,
    max_iter: int = 200,
) -> float:
    """Estimate constant-pressure adiabatic flame temperature T_p (Delta H = 0).

    Note: `h_reactants` must be on the same molar basis as `mole_fractions`
    (i.e. J per mole of product mixture when `mole_fractions` sum to 1.0).
    """
    T = T_guess
    for _ in range(max_iter):
        h_prod = sum(
            mole_fractions.get(sp.name, 0.0) * enthalpy(sp, T)
            for sp in species_list
        )
        cp_mix = mixture_cp_mol(species_list, mole_fractions, T)
        if cp_mix < 1e-10:
            break
        dT = (h_reactants - h_prod) / cp_mix
        T += dT
        T = max(T, 300.0)
        T = min(T, 8000.0)
        if abs(dT) < tol:
            break
    return T


def isochoric_flame_temperature(
    species_list: list,
    mole_fractions: dict,
    u_reactants: float,
    T_guess: float = 2800.0,
    tol: float = 0.5,
    max_iter: int = 200,
) -> float:
    """Estimate constant-volume (closed-vessel) flame temperature T_v (Delta U = 0).

    Used for NATO STANAG 4115 / US Army BRL propellant Impetus (F = n_gas * R * T_v).
    `u_reactants` must be on the same molar basis as `mole_fractions` (J/mol_mix).
    """
    T = T_guess
    for _ in range(max_iter):
        u_prod = 0.0
        cv_mix = 0.0
        for sp in species_list:
            xi = mole_fractions.get(sp.name, 0.0)
            if xi <= 0:
                continue
            hi = enthalpy(sp, T)
            cpi = cp(sp, T)
            if sp.phase == "gas":
                u_prod += xi * (hi - R_GAS * T)
                cv_mix += xi * max(cpi - R_GAS, 1.5 * R_GAS)
            else:
                u_prod += xi * hi
                cv_mix += xi * cpi
        if cv_mix < 1e-10:
            break
        dT = (u_reactants - u_prod) / cv_mix
        T += dT
        T = max(T, 300.0)
        T = min(T, 8000.0)
        if abs(dT) < tol:
            break
    return T
