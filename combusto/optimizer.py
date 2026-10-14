"""Multi-objective formulation optimizer and ternary composition mapper for Combusto.

Provides inverse-design optimization of propellant ingredient mass fractions
under temperature and soot/PAH constraints, as well as 3-component simplex
grid evaluation for ternary contour diagrams.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .chemistry import equivalence_ratio, oxygen_balance
from .equilibrium import solve_equilibrium
from .models import (
    CombustionConditions,
    CombustionResult,
    Formulation,
    PAHAllocationConfig,
    PropellantComponent,
    Species,
)


def optimize_formulation(
    base_formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Dict[str, Species],
    objective_metric: str = "force_constant",
    max_temp_k: Optional[float] = 3000.0,
    max_pah_frac: Optional[float] = 0.05,
    min_force_j_per_g: Optional[float] = None,
    component_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    equilibrium_temp_k: float = 2500.0,
) -> Dict[str, Any]:
    """Optimize component mass fractions of *base_formulation*.

    Parameters
    ----------
    base_formulation : Formulation
        Starting formulation (must have at least 2 components).
    conditions : CombustionConditions
        Operating conditions.
    pah_config : PAHAllocationConfig
        PAH/soot model configuration.
    species_db : dict[str, Species]
        Thermodynamic species database.
    objective_metric : str
        One of:
        - ``"force_constant"`` : Maximize Force Constant F (J/g)
        - ``"specific_impulse"`` : Maximize Specific Impulse Isp (s)
        - ``"min_flame_temp"`` : Minimize Adiabatic Flame Temperature T_ad (K)
        - ``"min_soot"`` : Minimize PAH/Soot Carbon Fraction
    max_temp_k : float, optional
        Soft/hard upper constraint on adiabatic flame temperature (K).
    max_pah_frac : float, optional
        Upper constraint on fraction of carbon allocated to PAHs/soot (0..1).
    min_force_j_per_g : float, optional
        Lower constraint on propellant force constant (J/g).
    component_bounds : sequence of (min_frac, max_frac), optional
        Bounds for each component's mass fraction.

    Returns
    -------
    dict
        Contains ``optimized_formulation``, ``optimized_result``,
        ``baseline_result``, ``converged``, ``message``, and ``comparison_df``.
    """
    n_comp = len(base_formulation.components)
    if n_comp < 2:
        raise ValueError("Formulation must contain at least 2 components to optimize.")

    x0 = np.array([c.mass_fraction for c in base_formulation.components], dtype=float)
    s0 = x0.sum()
    if s0 > 0:
        x0 = x0 / s0
    else:
        x0 = np.ones(n_comp) / n_comp

    if component_bounds is None:
        bounds = [(0.01, 0.98) for _ in range(n_comp)]
    else:
        bounds = list(component_bounds)

    # Evaluate baseline first
    baseline_result = solve_equilibrium(
        base_formulation, conditions, pah_config, species_db, equilibrium_temp_k
    )

    def _build_form(weights: np.ndarray) -> Formulation:
        w = np.maximum(weights, 0.0)
        tot = w.sum()
        if tot > 0:
            w = w / tot
        comps = [
            PropellantComponent(
                name=c.name,
                formula=c.formula,
                mass_fraction=float(w[i]),
                hf_298=c.hf_298,
            )
            for i, c in enumerate(base_formulation.components)
        ]
        return Formulation(
            name=f"{base_formulation.name} (Optimized)",
            components=comps,
            description=f"Optimized for {objective_metric}",
        )

    def _eval(weights: np.ndarray) -> CombustionResult:
        f = _build_form(weights)
        return solve_equilibrium(f, conditions, pah_config, species_db, equilibrium_temp_k)

    def _objective(weights: np.ndarray) -> float:
        res = _eval(weights)
        pah_frac = res.carbon_distribution.get("PAHs", 0.0)

        if objective_metric == "force_constant":
            val = -res.force_constant_j_per_g / 1000.0
        elif objective_metric == "specific_impulse":
            val = -res.specific_impulse_s / 250.0
        elif objective_metric == "min_flame_temp":
            val = res.adiabatic_flame_temp_k / 3000.0
        elif objective_metric == "min_soot":
            val = pah_frac * 10.0
        else:
            val = -res.force_constant_j_per_g / 1000.0

        # Smooth quadratic penalties for constraint violations
        penalty = 0.0
        if max_temp_k is not None and res.adiabatic_flame_temp_k > max_temp_k:
            excess_t = (res.adiabatic_flame_temp_k - max_temp_k) / 100.0
            penalty += 5.0 * (excess_t ** 2)
        if max_pah_frac is not None and pah_frac > max_pah_frac:
            excess_p = (pah_frac - max_pah_frac) / 0.01
            penalty += 5.0 * (excess_p ** 2)
        if min_force_j_per_g is not None and res.force_constant_j_per_g < min_force_j_per_g:
            deficit_f = (min_force_j_per_g - res.force_constant_j_per_g) / 50.0
            penalty += 5.0 * (deficit_f ** 2)

        return float(val + penalty)

    constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}

    opt_res = minimize(
        _objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 60, "ftol": 1e-4, "eps": 1e-2},
    )

    best_w = np.maximum(opt_res.x, 0.0)
    if best_w.sum() > 0:
        best_w = best_w / best_w.sum()

    opt_formulation = _build_form(best_w)
    opt_result = solve_equilibrium(
        opt_formulation, conditions, pah_config, species_db, equilibrium_temp_k
    )

    comp_rows = []
    for i, c in enumerate(base_formulation.components):
        comp_rows.append({
            "Component": c.name,
            "Formula": c.formula,
            "Baseline Mass %": round(c.mass_fraction * 100.0, 2),
            "Optimized Mass %": round(float(best_w[i]) * 100.0, 2),
            "Delta %": round((float(best_w[i]) - c.mass_fraction) * 100.0, 2),
        })

    return {
        "optimized_formulation": opt_formulation,
        "optimized_result": opt_result,
        "baseline_result": baseline_result,
        "converged": bool(opt_res.success),
        "message": str(opt_res.message),
        "comparison_df": pd.DataFrame(comp_rows),
    }


def compute_ternary_grid(
    comp_a: PropellantComponent,
    comp_b: PropellantComponent,
    comp_c: PropellantComponent,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Dict[str, Species],
    n_steps: int = 8,
    equilibrium_temp_k: float = 2500.0,
) -> pd.DataFrame:
    """Evaluate combustion equilibrium across a 3-component simplex grid.

    Returns a DataFrame with columns ``w_A``, ``w_B``, ``w_C`` and key
    thermochemical output metrics suitable for Plotly ternary plots.
    """
    rows: List[Dict[str, float]] = []
    step_vals = np.linspace(0.0, 1.0, n_steps + 1)

    for i in range(n_steps + 1):
        w_a = float(step_vals[i])
        for j in range(n_steps + 1 - i):
            w_b = float(step_vals[j])
            w_c = max(0.0, 1.0 - w_a - w_b)

            # Avoid exact zero-element singularities by nudging tiny epsilons
            wa_eff = max(w_a, 1e-4)
            wb_eff = max(w_b, 1e-4)
            wc_eff = max(w_c, 1e-4)
            tot = wa_eff + wb_eff + wc_eff

            form = Formulation(
                name=f"Ternary ({w_a:.2f}, {w_b:.2f}, {w_c:.2f})",
                components=[
                    PropellantComponent(comp_a.name, comp_a.formula, wa_eff / tot, comp_a.hf_298),
                    PropellantComponent(comp_b.name, comp_b.formula, wb_eff / tot, comp_b.hf_298),
                    PropellantComponent(comp_c.name, comp_c.formula, wc_eff / tot, comp_c.hf_298),
                ],
            )
            res = solve_equilibrium(form, conditions, pah_config, species_db, equilibrium_temp_k)
            rows.append({
                "w_A": w_a,
                "w_B": w_b,
                "w_C": w_c,
                "T_ad_K": res.adiabatic_flame_temp_k,
                "force_J_per_g": res.force_constant_j_per_g,
                "Isp_s": res.specific_impulse_s,
                "PAH_carbon_frac": res.carbon_distribution.get("PAHs", 0.0),
                "OB_pct": res.oxygen_balance_pct,
                "phi": res.equivalence_ratio,
                "gamma": res.gamma,
            })

    return pd.DataFrame(rows)
