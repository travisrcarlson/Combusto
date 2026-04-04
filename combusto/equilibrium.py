"""Gibbs free energy minimization equilibrium solver.

Uses a two-phase approach:
1. Find a feasible initial point via least-squares element balance.
2. Minimize Gibbs free energy with element-conservation constraints (SLSQP).

For ideal gases: mu_i = G_i^0(T) + RT*ln(x_i * P/P0)
For condensed species: mu_i = G_i^0(T)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize, lsq_linear

from .models import (
    ATOMIC_MASSES,
    CombustionConditions,
    CombustionResult,
    Formulation,
    PAHAllocationConfig,
    Species,
    parse_formula,
)
from .chemistry import equivalence_ratio, oxygen_balance
from .pah_allocation import allocate_pahs
from .species_db import load_species_db
from .thermodynamics import R_GAS, enthalpy, gibbs, mixture_cp_mol, adiabatic_flame_temperature


def _build_element_matrix(
    species_list: List[Species], elements: List[str]
) -> np.ndarray:
    """Build A matrix where A[j, i] = atoms of element j in species i."""
    A = np.zeros((len(elements), len(species_list)))
    for i, sp in enumerate(species_list):
        comp = parse_formula(sp.formula)
        for j, elem in enumerate(elements):
            A[j, i] = comp.get(elem, 0.0)
    return A


def _find_feasible_start(
    A: np.ndarray, b: np.ndarray, n_species: int
) -> np.ndarray:
    """Find a feasible non-negative starting point using least-squares."""
    result = lsq_linear(A, b, bounds=(1e-15, np.inf))
    n0 = np.maximum(result.x, 1e-15)
    return n0


def _gibbs_objective(
    n_vec: np.ndarray,
    small_species: List[Species],
    T: float,
    P: float,
    scale: float,
) -> float:
    """Compute total Gibbs free energy (scaled)."""
    n = n_vec * scale
    n = np.maximum(n, 1e-30)
    n_total_gas = sum(
        n[i] for i in range(len(small_species)) if small_species[i].phase == "gas"
    )
    if n_total_gas < 1e-30:
        n_total_gas = 1e-30

    G_total = 0.0
    for i, sp in enumerate(small_species):
        ni = n[i]
        if ni < 1e-30:
            continue
        G0 = gibbs(sp, T, 1.0)
        if sp.phase == "gas":
            x_i = ni / n_total_gas
            mu_i = G0 + R_GAS * T * math.log(max(x_i * P, 1e-30))
        else:
            mu_i = G0
        G_total += ni * mu_i
    return G_total


def solve_equilibrium(
    formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Optional[Dict[str, Species]] = None,
    temperature_k: Optional[float] = None,
) -> CombustionResult:
    """Run the full combustion equilibrium calculation."""
    if species_db is None:
        species_db = load_species_db()

    T = temperature_k if temperature_k else 2500.0
    P = conditions.pressure_atm

    if conditions.mode == "bomb_calorimeter":
        T = 298.15

    # --- Step 1: Elemental composition (mol per gram propellant) ---
    elem_per_gram = formulation.elemental_composition()

    # --- Step 2: Equivalence ratio ---
    phi = equivalence_ratio(formulation)
    ob_pct = oxygen_balance(formulation)

    # --- Step 3: PAH allocation ---
    total_c = elem_per_gram.get("C", 0.0)
    total_h = elem_per_gram.get("H", 0.0)

    pah_species = {n: s for n, s in species_db.items() if s.category in ("pah", "lump")}
    pah_moles, remaining_c, remaining_h = allocate_pahs(
        total_c, total_h, phi, pah_config, pah_species
    )

    # --- Step 4: Build equilibrium problem ---
    adjusted_elem = dict(elem_per_gram)
    adjusted_elem["C"] = remaining_c
    adjusted_elem["H"] = remaining_h

    formulation_elements = set(adjusted_elem.keys())

    # Select candidate species (exclude PAHs, species with wrong elements,
    # and condensed species that shouldn't exist at this temperature)
    small_species: List[Species] = []
    for sp in species_db.values():
        if sp.category in ("pah", "lump"):
            continue
        sp_elements = set(parse_formula(sp.formula).keys())
        if not sp_elements.issubset(formulation_elements):
            continue
        # Skip condensed-phase species at high temperatures
        # (e.g., liquid water above 373 K, solid oxides above ~3000 K)
        if sp.phase == "condensed":
            if sp.formula == "H2O" and T > 373.0:
                continue
            if T > sp.t_high:
                continue
        small_species.append(sp)

    if not small_species:
        for name in ["CO2", "CO", "H2O", "H2", "N2"]:
            if name in species_db:
                small_species.append(species_db[name])

    # Elements to balance (only those with non-negligible amounts)
    elements = sorted(e for e in formulation_elements if adjusted_elem.get(e, 0.0) > 1e-20)

    n_sp = len(small_species)
    n_el = len(elements)

    if n_sp == 0 or n_el == 0:
        return CombustionResult()

    A = _build_element_matrix(small_species, elements)
    b = np.array([adjusted_elem.get(e, 0.0) for e in elements])

    # --- Scale factor: work in units where values are O(1) ---
    scale = max(b.max(), 1e-10)
    b_scaled = b / scale

    # --- Find feasible initial point ---
    A_scaled = A  # A is dimensionless (atom counts)
    n0_scaled = _find_feasible_start(A_scaled, b_scaled, n_sp)

    # --- Gibbs minimization ---
    def objective(n_vec):
        return _gibbs_objective(n_vec, small_species, T, P, scale)

    def constraint_fun(n_vec):
        return A @ n_vec - b_scaled

    constraints = {
        "type": "eq",
        "fun": constraint_fun,
        "jac": lambda n: A,
    }

    bounds = [(1e-20 / scale, None)] * n_sp

    # Try SLSQP with the feasible start
    result = minimize(
        objective,
        n0_scaled,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-14},
    )

    n_eq = np.maximum(result.x * scale, 0.0)

    # --- Step 5: Combine results ---
    all_moles: Dict[str, float] = {}
    species_lookup: Dict[str, Species] = {}

    for i, sp in enumerate(small_species):
        if n_eq[i] > 1e-25:
            all_moles[sp.name] = float(n_eq[i])
            species_lookup[sp.name] = sp

    for name, moles in pah_moles.items():
        if moles > 1e-25:
            all_moles[name] = moles
            species_lookup[name] = species_db[name]

    # Compute fractions
    total_moles = sum(all_moles.values())
    total_mass = sum(
        moles * species_lookup[name].mw
        for name, moles in all_moles.items()
    )

    mole_fracs = {
        name: moles / total_moles if total_moles > 0 else 0.0
        for name, moles in all_moles.items()
    }
    mass_fracs = {
        name: (moles * species_lookup[name].mw) / total_mass if total_mass > 0 else 0.0
        for name, moles in all_moles.items()
    }

    # --- Carbon distribution ---
    c_in_co2 = all_moles.get("CO2", 0.0)
    c_in_co = all_moles.get("CO", 0.0)
    c_in_ch4 = all_moles.get("CH4", 0.0)
    c_in_pahs = sum(
        moles * parse_formula(species_lookup[n].formula).get("C", 0.0)
        for n, moles in all_moles.items()
        if species_lookup[n].category in ("pah", "lump")
    )
    c_in_other = total_c - c_in_co2 - c_in_co - c_in_ch4 - c_in_pahs
    c_in_other = max(c_in_other, 0.0)

    carbon_dist = {}
    if total_c > 0:
        carbon_dist = {
            "CO2": c_in_co2 / total_c,
            "CO": c_in_co / total_c,
            "CH4": c_in_ch4 / total_c,
            "PAHs": c_in_pahs / total_c,
            "Other": c_in_other / total_c,
        }

    pah_details = []
    for name, moles in all_moles.items():
        sp = species_lookup[name]
        if sp.category in ("pah", "lump"):
            c_atoms = parse_formula(sp.formula).get("C", 0.0)
            pah_details.append({
                "name": name,
                "formula": sp.formula,
                "mole_fraction": mole_fracs.get(name, 0.0),
                "mass_fraction": mass_fracs.get(name, 0.0),
                "carbon_share": (moles * c_atoms / total_c) if total_c > 0 else 0.0,
            })

    # --- Element balance check ---
    output_elem: Dict[str, float] = {}
    for name, moles in all_moles.items():
        comp = parse_formula(species_lookup[name].formula)
        for elem, count in comp.items():
            output_elem[elem] = output_elem.get(elem, 0.0) + moles * count

    elem_balance = {}
    for elem in set(list(elem_per_gram.keys()) + list(output_elem.keys())):
        inp = elem_per_gram.get(elem, 0.0)
        out = output_elem.get(elem, 0.0)
        err = abs(inp - out) / max(inp, 1e-30) * 100.0
        elem_balance[elem] = err

    # --- Adiabatic flame temperature estimate ---
    h_react = 0.0
    for comp in formulation.components:
        moles_comp = comp.mass_fraction / comp.mw
        h_react += moles_comp * comp.hf_298 * 1000.0  # kJ -> J

    all_species_list = [species_lookup[n] for n in all_moles.keys()]
    t_ad = adiabatic_flame_temperature(
        all_species_list, mole_fracs, h_react, T_guess=2800.0, p_atm=P
    )

    cp_mix = mixture_cp_mol(all_species_list, mole_fracs, t_ad)

    # --- Extended fields ---
    # Total gas moles (per gram propellant)
    gas_moles_total = sum(
        moles for name, moles in all_moles.items()
        if species_lookup[name].phase == "gas"
    )

    # Mean molecular weight of gas mixture
    gas_mass = sum(
        moles * species_lookup[name].mw
        for name, moles in all_moles.items()
        if species_lookup[name].phase == "gas"
    )
    mean_mw = gas_mass / gas_moles_total if gas_moles_total > 0 else 0.0

    # Gamma (Cp/Cv) for gas mixture: Cv = Cp - R for ideal gas
    # cp_mix is per mol of total mixture; compute gas-only Cp
    gas_mole_fracs = {}
    if gas_moles_total > 0:
        gas_mole_fracs = {
            name: moles / gas_moles_total
            for name, moles in all_moles.items()
            if species_lookup[name].phase == "gas"
        }
    gas_species_list = [
        species_lookup[name] for name in gas_mole_fracs
    ]
    cp_gas_mix = mixture_cp_mol(gas_species_list, gas_mole_fracs, t_ad) if gas_species_list else 0.0
    gamma = cp_gas_mix / (cp_gas_mix - R_GAS) if cp_gas_mix > R_GAS else 1.0

    # Force constant: F = n_gas_total * R * T_ad  (J per gram propellant)
    force_constant = gas_moles_total * R_GAS * t_ad

    # --- Isochoric pressure via Noble-Abel EOS ---
    computed_pressure_atm = 0.0
    if conditions.mode == "isochoric" and conditions.loading_density is not None:
        # loading_density in g/cm³ => volume per gram = 1/rho  cm³/g
        # Convert to m³/g: 1 cm³ = 1e-6 m³
        v_per_gram = 1.0 / conditions.loading_density * 1e-6  # m³ per gram
        b = conditions.covolume_m3_per_mol
        # P = n_gas * R * T / (V - n_gas * b)   [per gram basis]
        denominator = v_per_gram - gas_moles_total * b
        if denominator > 0:
            p_pa = gas_moles_total * R_GAS * t_ad / denominator
            computed_pressure_atm = p_pa / 101325.0
        else:
            computed_pressure_atm = float("inf")

    # --- Standard Energetic Metrics ---
    # Gas volume V0 (L/kg at standard 273.15K, 1 atm)
    gas_volume_l_per_kg = gas_moles_total * 1000.0 * 22.414

    # Specific Impulse Isp (vacuum, assuming expansion to 1 atm)
    specific_impulse_s = 0.0
    if gamma > 1.0 and mean_mw > 0:
        pe = 1.0
        pc = max(P, 1.01) # Avoid negative ratio if P <= 1 atm
        g0 = 9.80665
        r_spec = R_GAS / (mean_mw / 1000.0) # J/(kg K)
        term1 = (2 * gamma / (gamma - 1)) * r_spec * t_ad
        term2 = 1.0 - (pe / pc)**((gamma - 1) / gamma)
        if term1 * term2 > 0:
            v_e = math.sqrt(term1 * term2)
            specific_impulse_s = v_e / g0

    # Heat of Explosion (Qv) for Bomb Calorimeter
    # delta_U = delta_H - delta(nRT)
    h_prod_298 = sum(
        moles * enthalpy(species_lookup[name], 298.15)
        for name, moles in all_moles.items()
    )
    delta_h_298 = h_prod_298 - h_react
    
    # If not in bomb mode, we approximate Qv by assuming H2O condenses
    n_gas_298 = gas_moles_total
    if conditions.mode != "bomb_calorimeter" and "H2O" in species_lookup and species_lookup["H2O"].phase == "gas":
        n_gas_298 -= all_moles.get("H2O", 0.0)

    delta_u_298 = delta_h_298 - n_gas_298 * R_GAS * 298.15
    heat_of_explosion_j_per_g = -1.0 * delta_u_298

    return CombustionResult(
        species_moles=all_moles,
        species_mole_fractions=mole_fracs,
        species_mass_fractions=mass_fracs,
        temperature_k=T,
        pressure_atm=P,
        equivalence_ratio=phi,
        oxygen_balance_pct=ob_pct,
        element_balance=elem_balance,
        adiabatic_flame_temp_k=t_ad,
        mixture_cp=cp_mix,
        enthalpy_change_kj=h_react / 1000.0,
        carbon_distribution=carbon_dist,
        pah_details=pah_details,
        computed_pressure_atm=computed_pressure_atm,
        gas_moles_total=gas_moles_total,
        force_constant_j_per_g=force_constant,
        mean_molecular_weight=mean_mw,
        gamma=gamma,
        heat_of_explosion_j_per_g=heat_of_explosion_j_per_g,
        specific_impulse_s=specific_impulse_s,
        gas_volume_l_per_kg=gas_volume_l_per_kg,
    )


def solve_equilibrium_adiabatic(
    formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Optional[Dict[str, Species]] = None,
    t_guess: float = 2500.0,
    tol: float = 1.0,
    max_iter: int = 30,
) -> CombustionResult:
    """Self-consistent adiabatic flame temperature + equilibrium solver.

    Iterates between solving equilibrium at temperature T and recomputing
    the adiabatic flame temperature from the resulting product composition,
    until the temperature converges (|T_new - T_old| < tol).

    Parameters
    ----------
    formulation : Formulation
        Propellant formulation.
    conditions : CombustionConditions
        Operating conditions.
    pah_config : PAHAllocationConfig
        PAH allocation configuration.
    species_db : dict, optional
        Species database (loaded automatically if None).
    t_guess : float
        Initial temperature guess (K).
    tol : float
        Convergence tolerance on temperature (K). Default 1 K.
    max_iter : int
        Maximum number of T-composition iterations. Default 30.

    Returns
    -------
    CombustionResult
        Equilibrium result with converged=True if iteration converged,
        and iterations set to the number of outer loops performed.
    """
    if species_db is None:
        species_db = load_species_db()

    T = t_guess
    converged = False
    result = None
    # Relaxation factor: blend old and new T to damp oscillations.
    # Start with aggressive updates, increase damping if oscillating.
    relax = 0.5

    for iteration in range(1, max_iter + 1):
        # Solve equilibrium at current T
        result = solve_equilibrium(
            formulation, conditions, pah_config,
            species_db=species_db, temperature_k=T,
        )

        # The adiabatic flame temperature computed inside solve_equilibrium
        T_new_raw = result.adiabatic_flame_temp_k

        # Damped update: T_next = T + relax * (T_new_raw - T)
        T_new = T + relax * (T_new_raw - T)
        # Clamp to physical range
        T_new = max(T_new, 300.0)
        T_new = min(T_new, 6000.0)

        if abs(T_new - T) < tol:
            converged = True
            T = T_new
            break

        T = T_new

    # Do one final solve at the converged temperature to ensure consistency
    if converged and abs(T - result.temperature_k) > 0.1:
        result = solve_equilibrium(
            formulation, conditions, pah_config,
            species_db=species_db, temperature_k=T,
        )

    # Annotate convergence info
    result.converged = converged
    result.iterations = iteration  # noqa: F821 — set in loop

    return result
