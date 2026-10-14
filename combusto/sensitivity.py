"""Sensitivity analysis module for Combusto.

Provides parametric sweep functions that vary equivalence ratio, temperature,
PAH model presets, or compare multiple formulations.  Every function returns
a ``pandas.DataFrame`` for easy plotting and downstream analysis.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .models import (
    CombustionConditions,
    CombustionResult,
    Formulation,
    PAHAllocationConfig,
    PropellantComponent,
    Species,
)
from .equilibrium import solve_equilibrium
from .chemistry import equivalence_ratio
from .pah_allocation import PRESETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_to_row(result: CombustionResult) -> Dict[str, float]:
    """Extract key metrics from a CombustionResult into a flat dict."""
    cd = result.carbon_distribution
    return {
        "equivalence_ratio": result.equivalence_ratio,
        "temperature_k": result.temperature_k,
        "adiabatic_flame_temp_k": result.adiabatic_flame_temp_k,
        "pressure_atm": result.pressure_atm,
        "oxygen_balance_pct": result.oxygen_balance_pct,
        "CO2_mole_frac": result.species_mole_fractions.get("CO2", 0.0),
        "CO_mole_frac": result.species_mole_fractions.get("CO", 0.0),
        "H2O_mole_frac": result.species_mole_fractions.get("H2O", 0.0),
        "H2_mole_frac": result.species_mole_fractions.get("H2", 0.0),
        "N2_mole_frac": result.species_mole_fractions.get("N2", 0.0),
        "CH4_mole_frac": result.species_mole_fractions.get("CH4", 0.0),
        "CO2_carbon_frac": cd.get("CO2", 0.0),
        "CO_carbon_frac": cd.get("CO", 0.0),
        "PAH_carbon_frac": cd.get("PAHs", 0.0),
        "mixture_cp": result.mixture_cp,
        "enthalpy_change_kj": result.enthalpy_change_kj,
        "force_constant_j_per_g": result.force_constant_j_per_g,
        "gamma": result.gamma,
        "specific_impulse_s": result.specific_impulse_s,
    }


def _modify_oxygen(formulation: Formulation, o_scale: float) -> Formulation:
    """Return a copy of *formulation* with oxygen content scaled by *o_scale*.

    Oxygen content is adjusted by scaling the O atoms in every component's
    formula.  This is a synthetic manipulation used to explore how the
    equivalence ratio affects product distribution — the resulting formulation
    does not necessarily correspond to a real propellant.

    The mass fractions are renormalised after the adjustment so that they
    still sum to 1.
    """
    new_components: List[PropellantComponent] = []
    for comp in formulation.components:
        # Rebuild the formula with scaled O count
        from .models import parse_formula, ATOMIC_MASSES

        atoms = parse_formula(comp.formula)
        if "O" in atoms:
            atoms["O"] *= o_scale

        # Reconstruct a formula string from the atom dict
        formula_parts: List[str] = []
        for elem in sorted(atoms.keys()):
            count = atoms[elem]
            if count == 0.0:
                continue
            if count == int(count) and count > 0:
                subscript = str(int(count)) if count != 1 else ""
            else:
                subscript = f"{count:.4g}"
            formula_parts.append(f"{elem}{subscript}")
        new_formula = "".join(formula_parts)

        new_components.append(PropellantComponent(
            name=comp.name,
            formula=new_formula,
            mass_fraction=comp.mass_fraction,
            hf_298=comp.hf_298,
        ))

    # Renormalise mass fractions (molecular weights have changed)
    total_mf = sum(c.mass_fraction for c in new_components)
    if total_mf > 0:
        for c in new_components:
            c.mass_fraction /= total_mf

    return Formulation(
        name=f"{formulation.name} (O×{o_scale:.3f})",
        components=new_components,
        description=formulation.description,
    )


# ---------------------------------------------------------------------------
# Public sweep functions
# ---------------------------------------------------------------------------

def sweep_equivalence_ratio(
    formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Dict[str, Species],
    temperature_k: float = 2500.0,
    phi_range: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Sweep across equivalence ratios by artificially varying oxygen content.

    For each target *phi* in *phi_range*, the oxygen atoms in the formulation
    are scaled so that the resulting equivalence ratio is approximately equal
    to the target value.  An equilibrium calculation is then performed and key
    product metrics are collected.

    Parameters
    ----------
    formulation : Formulation
        Base propellant formulation.
    conditions : CombustionConditions
        Operating conditions (pressure, mode, etc.).
    pah_config : PAHAllocationConfig
        PAH allocation settings.
    species_db : dict[str, Species]
        Species thermodynamic database.
    temperature_k : float, optional
        Equilibrium temperature in Kelvin (default 2500).
    phi_range : np.ndarray, optional
        Array of target equivalence ratios.  Defaults to
        ``np.linspace(0.5, 3.0, 20)``.

    Returns
    -------
    pd.DataFrame
        One row per phi value with columns for phi, mole fractions of major
        products, carbon distribution fractions, adiabatic flame temperature,
        and other metrics.
    """
    if phi_range is None:
        phi_range = np.linspace(0.5, 3.0, 20)

    # Determine the baseline phi so we know how to scale O
    phi_base = equivalence_ratio(formulation)
    if phi_base == 0 or not np.isfinite(phi_base):
        raise ValueError(
            "Cannot sweep equivalence ratio: the base formulation has no "
            "oxygen or an undefined equivalence ratio."
        )

    rows: List[Dict[str, float]] = []
    for phi_target in phi_range:
        if phi_target <= 0:
            continue
        # phi = O_needed / O_available.  Scaling O by factor k changes phi by
        # 1/k, so to go from phi_base to phi_target we need k = phi_base / phi_target.
        o_scale = phi_base / phi_target
        mod_form = _modify_oxygen(formulation, o_scale)

        result = solve_equilibrium(
            mod_form, conditions, pah_config, species_db, temperature_k
        )
        row = _result_to_row(result)
        row["phi_target"] = float(phi_target)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder so phi_target is the first column
    cols = ["phi_target"] + [c for c in df.columns if c != "phi_target"]
    return df[cols]


def sweep_temperature(
    formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Dict[str, Species],
    t_range: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run equilibrium at multiple temperatures and track product changes.

    Parameters
    ----------
    formulation : Formulation
        Base propellant formulation.
    conditions : CombustionConditions
        Operating conditions.
    pah_config : PAHAllocationConfig
        PAH allocation settings.
    species_db : dict[str, Species]
        Species thermodynamic database.
    t_range : np.ndarray, optional
        Array of temperatures in Kelvin.  Defaults to
        ``np.linspace(1000, 4000, 20)``.

    Returns
    -------
    pd.DataFrame
        One row per temperature with product distribution columns.
    """
    if t_range is None:
        t_range = np.linspace(1000, 4000, 20)

    rows: List[Dict[str, float]] = []
    for T in t_range:
        result = solve_equilibrium(
            formulation, conditions, pah_config, species_db, float(T)
        )
        row = _result_to_row(result)
        row["sweep_temperature_k"] = float(T)
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["sweep_temperature_k"] + [c for c in df.columns if c != "sweep_temperature_k"]
    return df[cols]


def sweep_pah_models(
    formulation: Formulation,
    conditions: CombustionConditions,
    species_db: Dict[str, Species],
    temperature_k: float = 2500.0,
) -> pd.DataFrame:
    """Run equilibrium with each built-in PAH preset and compare results.

    The four presets are: ``none``, ``conservative``, ``moderate``, and
    ``aggressive``.

    Parameters
    ----------
    formulation : Formulation
        Base propellant formulation.
    conditions : CombustionConditions
        Operating conditions.
    species_db : dict[str, Species]
        Species thermodynamic database.
    temperature_k : float, optional
        Equilibrium temperature in Kelvin (default 2500).

    Returns
    -------
    pd.DataFrame
        One row per PAH preset with columns for the preset name and all
        standard product metrics.
    """
    rows: List[Dict[str, object]] = []
    for preset_name in ("none", "conservative", "moderate", "aggressive"):
        pah_cfg = PRESETS[preset_name]
        result = solve_equilibrium(
            formulation, conditions, pah_cfg, species_db, temperature_k
        )
        row: Dict[str, object] = {"pah_model": preset_name}
        row.update(_result_to_row(result))
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["pah_model"] + [c for c in df.columns if c != "pah_model"]
    return df[cols]


def compare_formulations(
    formulations: Sequence[Formulation],
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Dict[str, Species],
    temperature_k: float = 2500.0,
) -> pd.DataFrame:
    """Run equilibrium for several formulations and return a comparison table.

    Parameters
    ----------
    formulations : sequence of Formulation
        The propellant formulations to compare.
    conditions : CombustionConditions
        Operating conditions (shared across all formulations).
    pah_config : PAHAllocationConfig
        PAH allocation settings (shared across all formulations).
    species_db : dict[str, Species]
        Species thermodynamic database.
    temperature_k : float, optional
        Equilibrium temperature in Kelvin (default 2500).

    Returns
    -------
    pd.DataFrame
        One row per formulation with the formulation name and all standard
        product metrics.
    """
    rows: List[Dict[str, object]] = []
    for form in formulations:
        result = solve_equilibrium(
            form, conditions, pah_config, species_db, temperature_k
        )
        row: Dict[str, object] = {"formulation": form.name}
        row.update(_result_to_row(result))
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["formulation"] + [c for c in df.columns if c != "formulation"]
    return df[cols]


def sweep_2d(
    formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Dict[str, Species],
    param1: str,  # "equivalence_ratio", "temperature", "pressure"
    range1: np.ndarray,
    param2: str,  # "equivalence_ratio", "temperature", "pressure"
    range2: np.ndarray,
) -> pd.DataFrame:
    """Perform a 2D parameter sweep varying two conditions simultaneously.

    Returns a DataFrame with param1 and param2 as columns, along with all results.
    """
    rows: List[Dict[str, float]] = []

    # Get base equivalence ratio to scale if needed
    phi_base = equivalence_ratio(formulation)

    for val1 in range1:
        for val2 in range2:
            # Set up parameters
            temp_k = float(conditions.temperature_k if conditions.temperature_k else 2500.0)
            press_atm = float(conditions.pressure_atm)
            phi_val = phi_base
            o_scale = 1.0

            # Override parameters based on selection
            if param1 == "equivalence_ratio":
                o_scale = phi_base / float(val1) if val1 > 0 else 1.0
                phi_val = float(val1)
            elif param1 == "temperature":
                temp_k = float(val1)
            elif param1 == "pressure":
                press_atm = float(val1)

            if param2 == "equivalence_ratio":
                o_scale = phi_base / float(val2) if val2 > 0 else 1.0
                phi_val = float(val2)
            elif param2 == "temperature":
                temp_k = float(val2)
            elif param2 == "pressure":
                press_atm = float(val2)

            # Apply modified formulation if equivalence ratio was swept
            sweep_form = formulation
            if param1 == "equivalence_ratio" or param2 == "equivalence_ratio":
                sweep_form = _modify_oxygen(formulation, o_scale)

            # Build sweep conditions
            sweep_conds = copy.deepcopy(conditions)
            sweep_conds.pressure_atm = press_atm

            # Run solver
            result = solve_equilibrium(
                sweep_form, sweep_conds, pah_config, species_db, temp_k
            )

            # Log data
            row = _result_to_row(result)
            row[f"param_{param1}"] = float(val1)
            row[f"param_{param2}"] = float(val2)
            rows.append(row)

    df = pd.DataFrame(rows)
    # Put param columns first
    param_cols = [f"param_{param1}", f"param_{param2}"]
    other_cols = [c for c in df.columns if c not in param_cols]
    return df[param_cols + other_cols]

