"""Deterministic memoization (@st.cache_data) and unified Plotly theme for Combusto."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .equilibrium import solve_equilibrium, solve_equilibrium_adiabatic
from .grounding import run_benchmark_validation_suite
from .models import (
    CombustionConditions,
    CombustionResult,
    Formulation,
    PAHAllocationConfig,
    PropellantComponent,
    Species,
)
from .nozzle import analyze_nozzle_expansion
from .species_db import load_species_db


def apply_combusto_plotly_theme() -> None:
    """Register and activate the unified cyber-scientific dark Plotly template."""
    if "combusto_cyber" not in pio.templates:
        pio.templates["combusto_cyber"] = go.layout.Template(
            layout=go.Layout(
                paper_bgcolor="rgba(14, 18, 30, 0.72)",
                plot_bgcolor="rgba(10, 13, 22, 0.88)",
                font=dict(
                    family="Outfit, -apple-system, BlinkMacSystemFont, sans-serif",
                    color="#e2e8f0",
                    size=12,
                ),
                title=dict(font=dict(size=15, color="#f8fafc")),
                colorway=[
                    "#00f0ff",
                    "#00ff88",
                    "#ffaa00",
                    "#ff5566",
                    "#a78bfa",
                    "#38bdf8",
                    "#f472b6",
                    "#facc15",
                ],
                xaxis=dict(
                    gridcolor="rgba(148, 163, 184, 0.12)",
                    zerolinecolor="rgba(148, 163, 184, 0.22)",
                    linecolor="rgba(148, 163, 184, 0.28)",
                ),
                yaxis=dict(
                    gridcolor="rgba(148, 163, 184, 0.12)",
                    zerolinecolor="rgba(148, 163, 184, 0.22)",
                    linecolor="rgba(148, 163, 184, 0.28)",
                ),
                legend=dict(
                    bgcolor="rgba(14, 18, 30, 0.65)",
                    bordercolor="rgba(0, 240, 255, 0.16)",
                    borderwidth=1,
                ),
                margin=dict(t=48, b=42, l=52, r=28),
            )
        )
    pio.templates.default = "combusto_cyber"


def _formulation_key(form: Formulation) -> Tuple:
    return (
        form.name,
        form.description,
        tuple(
            (c.name, c.formula, round(float(c.mass_fraction), 8), round(float(c.hf_298), 4))
            for c in form.components
        ),
    )


def _conditions_key(cond: CombustionConditions) -> Tuple:
    return (
        round(float(cond.temperature_k), 3),
        round(float(cond.pressure_atm), 5),
        cond.mode,
        round(float(cond.loading_density), 6) if cond.loading_density is not None else None,
        cond.chamber_volume_cm3,
        cond.propellant_mass_g,
        cond.inert_gas,
        round(float(cond.inert_moles), 6),
        round(float(cond.covolume_m3_per_mol), 8),
        bool(cond.real_gas_fugacity),
    )


def _pah_key(pah: PAHAllocationConfig) -> Tuple:
    return (
        pah.model_name,
        round(float(pah.base_fraction), 6),
        round(float(pah.phi_exponent), 4),
        round(float(pah.max_fraction), 6),
        tuple(sorted((k, round(float(v), 6)) for k, v in pah.pah_distribution.items())),
    )


def _species_fingerprint(species_db: Optional[Dict[str, Species]]) -> Tuple:
    if not species_db:
        return (0, 0.0)
    return (
        len(species_db),
        round(sum(float(s.hf_298) + float(s.molar_covolume_cm3_mol) for s in species_db.values()), 3),
    )


def _rebuild_formulation(fkey: Tuple) -> Formulation:
    name, desc, comps = fkey
    return Formulation(
        name=name,
        description=desc,
        components=[PropellantComponent(n, f, mf, hf) for (n, f, mf, hf) in comps],
    )


def _rebuild_conditions(ckey: Tuple) -> CombustionConditions:
    return CombustionConditions(
        temperature_k=ckey[0],
        pressure_atm=ckey[1],
        mode=ckey[2],
        loading_density=ckey[3],
        chamber_volume_cm3=ckey[4],
        propellant_mass_g=ckey[5],
        inert_gas=ckey[6],
        inert_moles=ckey[7],
        covolume_m3_per_mol=ckey[8],
        real_gas_fugacity=ckey[9],
    )


def _rebuild_pah(pkey: Tuple) -> PAHAllocationConfig:
    return PAHAllocationConfig(
        model_name=pkey[0],
        base_fraction=pkey[1],
        phi_exponent=pkey[2],
        max_fraction=pkey[3],
        pah_distribution=dict(pkey[4]),
    )


@st.cache_data(show_spinner=False, max_entries=256)
def _solve_eq_cached_impl(
    fkey: Tuple,
    ckey: Tuple,
    pkey: Tuple,
    sp_fp: Tuple,
    temp_k: Optional[float],
    _species_db: Optional[Dict[str, Species]],
) -> CombustionResult:
    db = _species_db if _species_db is not None else load_species_db()
    return solve_equilibrium(
        _rebuild_formulation(fkey),
        _rebuild_conditions(ckey),
        _rebuild_pah(pkey),
        species_db=db,
        temperature_k=temp_k,
    )


@st.cache_data(show_spinner=False, max_entries=256)
def _solve_eq_ad_cached_impl(
    fkey: Tuple,
    ckey: Tuple,
    pkey: Tuple,
    sp_fp: Tuple,
    t_guess: float,
    _species_db: Optional[Dict[str, Species]],
) -> CombustionResult:
    db = _species_db if _species_db is not None else load_species_db()
    return solve_equilibrium_adiabatic(
        _rebuild_formulation(fkey),
        _rebuild_conditions(ckey),
        _rebuild_pah(pkey),
        species_db=db,
        t_guess=t_guess,
    )


@st.cache_data(show_spinner=False, max_entries=64)
def _nozzle_cached_impl(
    fkey: Tuple,
    pc_atm: float,
    pe_atm: float,
    ambient_p_atm: float,
    pkey: Tuple,
    n_pts: int,
    sp_fp: Tuple,
    _species_db: Optional[Dict[str, Species]],
) -> Dict[str, Any]:
    db = _species_db if _species_db is not None else load_species_db()
    return analyze_nozzle_expansion(
        formulation=_rebuild_formulation(fkey),
        pc_atm=pc_atm,
        pe_atm=pe_atm,
        ambient_p_atm=ambient_p_atm,
        pah_config=_rebuild_pah(pkey),
        species_db=db,
        n_profile_points=n_pts,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _benchmark_cached_impl(
    sp_fp: Tuple,
    _species_db: Optional[Dict[str, Species]],
) -> Dict[str, Any]:
    db = _species_db if _species_db is not None else load_species_db()
    return run_benchmark_validation_suite(db)


def cached_solve_equilibrium(
    formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Optional[Dict[str, Species]] = None,
    temperature_k: Optional[float] = None,
) -> CombustionResult:
    """Memoized wrapper for `solve_equilibrium`."""
    t_k = round(float(temperature_k), 3) if temperature_k is not None else None
    return _solve_eq_cached_impl(
        _formulation_key(formulation),
        _conditions_key(conditions),
        _pah_key(pah_config),
        _species_fingerprint(species_db),
        t_k,
        species_db,
    )


def cached_solve_equilibrium_adiabatic(
    formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    species_db: Optional[Dict[str, Species]] = None,
    t_guess: float = 2500.0,
) -> CombustionResult:
    """Memoized wrapper for `solve_equilibrium_adiabatic`."""
    return _solve_eq_ad_cached_impl(
        _formulation_key(formulation),
        _conditions_key(conditions),
        _pah_key(pah_config),
        _species_fingerprint(species_db),
        round(float(t_guess), 2),
        species_db,
    )


def cached_analyze_nozzle_expansion(
    formulation: Formulation,
    pc_atm: float = 68.0,
    pe_atm: float = 1.0,
    ambient_p_atm: float = 1.0,
    pah_config: Optional[PAHAllocationConfig] = None,
    species_db: Optional[Dict[str, Species]] = None,
    n_profile_points: int = 12,
) -> Dict[str, Any]:
    """Memoized wrapper for `analyze_nozzle_expansion`."""
    from .pah_allocation import PRESETS
    p_cfg = pah_config if pah_config is not None else PRESETS["none"]
    return _nozzle_cached_impl(
        _formulation_key(formulation),
        round(float(pc_atm), 4),
        round(float(pe_atm), 5),
        round(float(ambient_p_atm), 5),
        _pah_key(p_cfg),
        int(n_profile_points),
        _species_fingerprint(species_db),
        species_db,
    )


def cached_benchmark_validation_suite(
    species_db: Optional[Dict[str, Species]] = None,
) -> Dict[str, Any]:
    """Memoized wrapper for `run_benchmark_validation_suite`."""
    return _benchmark_cached_impl(_species_fingerprint(species_db), species_db)
