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
from .equilibrium import solve_equilibrium, solve_equilibrium_adiabatic
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
    locked_indices: Optional[Sequence[int]] = None,
    equilibrium_temp_k: float = 2500.0,
    self_consistent_adiabatic: bool = True,
) -> Dict[str, Any]:
    """Optimize component mass fractions of *base_formulation*.

    Supports locked minor additives (`locked_indices`) and custom per-component
    bounds (`component_bounds`) while preserving exact mass-fraction sum = 1.0
    and warm-starting adiabatic flame temperature solves for smooth convergence.
    """
    n_comp = len(base_formulation.components)
    if n_comp < 2:
        raise ValueError("Formulation must contain at least 2 components to optimize.")

    locked_set = {int(idx) for idx in (locked_indices or []) if 0 <= int(idx) < n_comp}
    free_indices = [i for i in range(n_comp) if i not in locked_set]
    if len(free_indices) < 1:
        raise ValueError("At least 1 free (unlocked) component is required for optimization.")

    x0 = np.array([max(0.0, float(c.mass_fraction)) for c in base_formulation.components], dtype=float)
    s0 = float(x0.sum())
    if s0 > 0.0:
        x0 = x0 / s0
    else:
        x0 = np.ones(n_comp, dtype=float) / n_comp

    if component_bounds is None:
        all_bounds = [
            (float(x0[i]), float(x0[i])) if i in locked_set else (min(0.005, float(x0[i])), max(0.98, float(x0[i])))
            for i in range(n_comp)
        ]
    else:
        if len(component_bounds) != n_comp:
            raise ValueError("component_bounds length must match number of formulation components.")
        all_bounds = []
        for i in range(n_comp):
            if i in locked_set:
                all_bounds.append((float(x0[i]), float(x0[i])))
            else:
                lo_b, hi_b = float(component_bounds[i][0]), float(component_bounds[i][1])
                lo_b = max(0.0, min(1.0, lo_b))
                hi_b = max(lo_b, min(1.0, hi_b))
                all_bounds.append((lo_b, hi_b))

    locked_sum = sum(float(x0[i]) for i in locked_set)
    free_target_sum = max(0.0, 1.0 - locked_sum)
    if len(free_indices) == 1:
        x0[free_indices[0]] = free_target_sum

    t_warm = [float(equilibrium_temp_k)]

    def _solve_point(f: Formulation, tol: float = 0.35) -> CombustionResult:
        if self_consistent_adiabatic:
            res = solve_equilibrium_adiabatic(
                f,
                conditions,
                pah_config,
                species_db=species_db,
                t_guess=t_warm[0],
                tol=tol,
                max_iter=20,
            )
            t_cur = res.isochoric_flame_temp_k if conditions.mode == "isochoric" else res.adiabatic_flame_temp_k
            if 500.0 <= t_cur <= 5500.0:
                t_warm[0] = t_cur
            return res
        return solve_equilibrium(f, conditions, pah_config, species_db, equilibrium_temp_k)

    # Evaluate baseline first
    baseline_result = _solve_point(base_formulation, tol=0.5)

    def _expand_weights(w_free: np.ndarray) -> np.ndarray:
        w = x0.copy()
        w_f = np.maximum(np.asarray(w_free, dtype=float), 0.0)
        tot_f = float(np.sum(w_f))
        if tot_f > 0.0:
            w_f = w_f * (free_target_sum / tot_f)
        for k, idx in enumerate(free_indices):
            w[idx] = float(w_f[k])
        return w

    def _build_form(weights_full: np.ndarray) -> Formulation:
        w = np.maximum(np.asarray(weights_full, dtype=float), 0.0)
        tot = float(w.sum())
        if tot > 0.0:
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

    def _objective(w_free: np.ndarray) -> float:
        res = _solve_point(_build_form(_expand_weights(w_free)))
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

    if len(free_indices) == 1:
        best_w = _expand_weights(np.array([free_target_sum]))
        opt_success = True
        opt_msg = "Single free component determined by mass balance."
    else:
        w0_free = np.array([
            min(max(float(x0[i]), all_bounds[i][0]), all_bounds[i][1])
            for i in free_indices
        ], dtype=float)
        if np.sum(w0_free) > 0.0:
            w0_free = w0_free * (free_target_sum / np.sum(w0_free))
        free_bounds = [all_bounds[i] for i in free_indices]
        constraints = {
            "type": "eq",
            "fun": lambda wf: float(np.sum(wf) - free_target_sum),
            "jac": lambda wf: np.ones_like(wf, dtype=float),
        }
        opt_res = minimize(
            _objective,
            w0_free,
            method="SLSQP",
            bounds=free_bounds,
            constraints=constraints,
            options={"maxiter": 60, "ftol": 1e-4, "eps": 1e-2},
        )
        best_w = _expand_weights(opt_res.x)
        opt_success = bool(opt_res.success)
        opt_msg = str(opt_res.message)

    opt_formulation = _build_form(best_w)
    opt_result = _solve_point(opt_formulation, tol=0.25)

    comp_rows = []
    for i, c in enumerate(base_formulation.components):
        comp_rows.append({
            "Component": c.name,
            "Formula": c.formula,
            "Locked": i in locked_set,
            "Baseline Mass %": round(c.mass_fraction * 100.0, 2),
            "Optimized Mass %": round(float(best_w[i]) * 100.0, 2),
            "Delta %": round((float(best_w[i]) - c.mass_fraction) * 100.0, 2),
        })

    return {
        "optimized_formulation": opt_formulation,
        "optimized_result": opt_result,
        "baseline_result": baseline_result,
        "converged": opt_success,
        "message": opt_msg,
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
    self_consistent_adiabatic: bool = True,
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
            if self_consistent_adiabatic:
                res = solve_equilibrium_adiabatic(
                    form, conditions, pah_config, species_db=species_db, t_guess=equilibrium_temp_k, tol=2.0, max_iter=20
                )
            else:
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
