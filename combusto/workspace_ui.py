"""Unified 5-Workspace Engineering UI for Combusto.

Shared by `app.py` (public web build) and `app_local.py` (Desktop Pro build).
Contains zero admin/database-editing or external-API scraping code; Desktop Pro
injects its local-only PubChem/NIST lookup hook via `extra_editor_hook`.
"""

from __future__ import annotations

import json
import math
import re as _re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from .ballistics import (
    estimate_barrel_erosion,
    generate_grain_cross_section,
    grain_form_function,
    simulate_ballistics,
)
from .cache_helpers import (
    apply_combusto_plotly_theme,
    cached_analyze_nozzle_expansion,
    cached_solve_equilibrium,
    cached_solve_equilibrium_adiabatic,
)
from .chemistry import equivalence_ratio, oxygen_balance
from .export import (
    carbon_distribution_to_dataframe,
    element_balance_to_dataframe,
    pah_details_to_dataframe,
    result_to_csv,
    result_to_dataframe,
    result_to_excel_bytes,
)
from .export_pdf import export_full_dossier_pdf, export_result_pdf
from .grounding import EPA_AP42_CALIBRATED_PAH
from .models import (
    CombustionConditions,
    Formulation,
    PAHAllocationConfig,
    PropellantComponent,
)
from .muzzle_flash import analyze_muzzle_flash
from .optimizer import compute_ternary_grid, optimize_formulation
from .pah_allocation import PRESETS
from .presets import (
    formulation_from_dict,
    formulation_to_dict,
    get_all_library_templates,
    load_ingredients,
    load_presets,
    save_formulations,
)
from .sensitivity import (
    compare_formulations,
    sweep_2d,
    sweep_equivalence_ratio,
    sweep_pah_models,
    sweep_temperature,
)
from .species_db import load_species_db
from .stability import analyze_thermal_stability, detect_stabilizer_wt_pct
from .toxicity_smoke import analyze_toxicity_and_smoke
from .ui_extensions import (
    render_benchmarks_and_grounding_tab,
    render_closed_vessel_tab,
    render_nc_nitration_calculator,
    render_noble_abel_isentrope_section,
    render_smoke_toxicity_tab,
    render_stability_tab,
)
from .uncertainty import run_monte_carlo_ballistics, simulate_climatic_zones


# ---------------------------------------------------------------------------
# Styling & Kamlet-Jacobs VoD Helpers
# ---------------------------------------------------------------------------

def inject_cyber_styles() -> None:
    """Apply Combusto cyber-scientific CSS, tab styling, and dark Plotly template."""
    apply_combusto_plotly_theme()

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: radial-gradient(circle at top left, #141724 0%, #0a0b10 100%) !important;
}

.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2rem !important;
    max-width: 1420px !important;
}

h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(90deg, #00f0ff 0%, #00bbff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

div[data-testid="stCaptionContainer"] {
    color: #a1aabf !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.03em;
    margin-bottom: 0.75rem !important;
}

div[data-testid="stMetric"] {
    background: rgba(20, 24, 39, 0.65);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 240, 255, 0.12);
    border-radius: 12px;
    padding: 0.9rem !important;
    transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border: 1px solid rgba(0, 240, 255, 0.42);
    box-shadow: 0 8px 18px rgba(0, 240, 255, 0.14);
}

div[data-testid="stMetric"] label {
    font-size: 0.73rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6ee6f5 !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #00f0ff 0%, #0077ff 100%) !important;
    border: none !important;
    color: #000 !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.75rem !important;
    border-radius: 8px !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 0 14px rgba(0, 240, 255, 0.35) !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #33f3ff 0%, #3399ff 100%) !important;
    box-shadow: 0 0 22px rgba(0, 240, 255, 0.75) !important;
    transform: scale(1.015);
}

div[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 6px !important;
}

div[data-testid="stExpander"] {
    background: rgba(10, 12, 20, 0.45);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
}

/* Workspace selector pill bar */
div[role="radiogroup"] {
    background: rgba(12, 16, 28, 0.8);
    border: 1px solid rgba(0, 240, 255, 0.18);
    border-radius: 12px;
    padding: 6px 10px;
    gap: 8px !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

    components.html("""
<script>
(function() {
  const TAB_CSS = `
    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        background: rgba(0,0,0,0.25) !important;
        border-radius: 8px !important;
        padding: 6px !important;
        border-bottom: none !important;
    }
    button[data-baseweb="tab"] {
        padding: 0.55rem 1.25rem !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        color: #94a3b8 !important;
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        transition: all .2s !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255,255,255,0.06) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(0, 240, 255, 0.92) !important;
        color: #000000 !important;
        border-bottom: none !important;
        box-shadow: 0 0 14px rgba(0, 240, 255, 0.38) !important;
    }
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
      display: none !important;
      height: 0 !important;
    }
  `;
  function injectStyles(doc) {
    const id = 'combusto-tab-styles-cyber';
    if (doc && !doc.getElementById(id)) {
      const s = doc.createElement('style');
      s.id = id;
      s.textContent = TAB_CSS;
      doc.head.appendChild(s);
    }
  }
  try {
    injectStyles(window.parent.document);
    const obs = new window.parent.MutationObserver(() => injectStyles(window.parent.document));
    obs.observe(window.parent.document.body, { childList: true, subtree: false });
  } catch(e) {
    injectStyles(document);
  }
})();
</script>
""", height=0)


def check_web_password() -> bool:
    """Password gate for Combusto session access."""
    def password_entered() -> None:
        if st.session_state["password"] == "combusto2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Please enter the password to access this app",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Please enter the password to access this app",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("😕 Password incorrect")
        return False
    return True


_VOD_PRESETS: Dict[str, Optional[Dict[str, Any]]] = {
    "Custom": None,
    "TNT  (C₇H₅N₃O₆)": dict(formula="C7H5N3O6", hf_kJ=-66.9, rho_tmd=1.654),
    "RDX  (C₃H₆N₆O₆)": dict(formula="C3H6N6O6", hf_kJ=92.6, rho_tmd=1.806),
    "HMX  (C₄H₈N₈O₈)": dict(formula="C4H8N8O8", hf_kJ=74.4, rho_tmd=1.906),
    "PETN (C₅H₈N₄O₁₂)": dict(formula="C5H8N4O12", hf_kJ=-538.5, rho_tmd=1.778),
    "TATB (C₆H₆N₆O₆)": dict(formula="C6H6N6O6", hf_kJ=-154.0, rho_tmd=1.937),
    "Tetryl (C₇H₅N₅O₈)": dict(formula="C7H5N5O8", hf_kJ=-38.9, rho_tmd=1.730),
    "NQ / Nitroguanidine (CH₄N₄O₂)": dict(formula="CH4N4O2", hf_kJ=-90.4, rho_tmd=1.715),
    "FOX-7 (C₂H₄N₄O₄)": dict(formula="C2H4N4O4", hf_kJ=-133.4, rho_tmd=1.885),
    "DINA  (C₄H₈N₂O₆)": dict(formula="C4H8N2O6", hf_kJ=-392.0, rho_tmd=1.490),
}

_AW = {
    "C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999,
    "S": 32.06, "K": 39.098, "NA": 22.990, "CU": 63.546,
    "FE": 55.845, "AL": 26.982, "PB": 207.2, "CR": 51.996,
    "MN": 54.938, "BA": 137.33, "SR": 87.62, "CA": 40.078,
    "CL": 35.45, "F": 18.998, "B": 10.81, "SI": 28.085,
    "P": 30.974, "ZN": 65.38, "NI": 58.693, "MG": 24.305,
}


def _parse_chno(formula: str):
    counts = {"C": 0.0, "H": 0.0, "N": 0.0, "O": 0.0}
    for m in _re.finditer(r"([CHNO])(\d+\.?\d*)?", formula.upper()):
        el = m.group(1)
        counts[el] += float(m.group(2)) if m.group(2) else 1.0
    return counts["C"], counts["H"], counts["N"], counts["O"]


def _formula_mw_general(formula: str) -> float:
    mw = 0.0
    for m in _re.finditer(r"([A-Z][a-z]?)(\d+\.?\d*)?", formula):
        el = m.group(1).upper()
        cnt = float(m.group(2)) if m.group(2) else 1.0
        mw += _AW.get(el, 0.0) * cnt
    return mw


def _formulation_kj_inputs(form: Formulation):
    try:
        elem = form.elemental_composition()
    except Exception:
        return None, "Could not compute elemental composition."
    a, b, c, d = elem.get("C", 0.0), elem.get("H", 0.0), elem.get("N", 0.0), elem.get("O", 0.0)
    if a + b + c + d < 1e-9:
        return None, "Elemental composition is zero — check formulation components."
    hf_per_g = 0.0
    for comp in form.components:
        mw_comp = _formula_mw_general(comp.formula)
        if mw_comp > 0:
            hf_per_g += comp.mass_fraction * comp.hf_298 / mw_comp
    return dict(a=a, b=b, c=c, d=d, mw=1.0, hf_kJ=hf_per_g), None


def _kamlet_jacobs(a: float, b: float, c: float, d: float, mw: float, hf_kJ_per_mol: float, rho: float) -> Dict[str, Any]:
    HF_H2O, HF_CO2, HF_CO = -57_800.0, -94_050.0, -26_420.0
    hf_cal = hf_kJ_per_mol * 1000.0 / 4.184
    n_N2 = c / 2.0
    n_H2O = b / 2.0
    O_rem = d - n_H2O
    if O_rem >= 2 * a:
        n_CO2, n_CO, n_Cs, n_O2 = a, 0.0, 0.0, (O_rem - 2 * a) / 2.0
    elif O_rem >= a:
        n_CO2, n_CO, n_Cs, n_O2 = O_rem - a, 2 * a - O_rem, 0.0, 0.0
    elif O_rem >= 0:
        n_CO2, n_CO, n_Cs, n_O2 = 0.0, O_rem, a - O_rem, 0.0
    else:
        n_H2O = max(0.0, d)
        n_CO2, n_CO, n_Cs, n_O2 = 0.0, 0.0, a, 0.0

    gas_mol = n_N2 + n_H2O + n_CO2 + n_CO + n_O2
    N = gas_mol / mw
    M_avg = (28 * n_N2 + 18 * n_H2O + 44 * n_CO2 + 28 * n_CO + 32 * n_O2) / gas_mol if gas_mol > 0 else 28.0
    Q = (hf_cal - (n_CO2 * HF_CO2 + n_H2O * HF_H2O + n_CO * HF_CO)) / mw

    base_info = dict(
        N=N, M_avg=M_avg, Q_cal_per_g=Q,
        n_N2=n_N2, n_H2O=n_H2O, n_CO2=n_CO2, n_CO=n_CO, n_Cs=n_Cs, n_O2=n_O2,
        OB_pct=(d - 2 * a - b / 2) / mw * 1600,
    )
    if Q <= 0:
        return dict(valid=False, **base_info)

    phi = N * (M_avg ** 0.5) * (Q ** 0.5)
    D = 1.01 * (phi ** 0.5) * (1 + 1.30 * rho)
    P = 15.58 * (rho ** 2) * phi
    gamma = 3.0
    return dict(
        valid=True,
        D_km_s=D, P_GPa=P, phi=phi,
        N_mol_per_g=N, M_avg_g_per_mol=M_avg,
        Q_cal_per_g=Q, Q_kJ_per_g=Q * 4.184 / 1000.0,
        rho_CJ=rho * (gamma + 1) / gamma,
        u_CJ=D / (gamma + 1),
        c_CJ=D * gamma / (gamma + 1),
        T_CJ_K=(Q * 4.184) / max(N * 30.0, 1e-6),
        dD_drho=1.01 * 1.30 * (phi ** 0.5),
        **base_info,
    )


# ---------------------------------------------------------------------------
# Session Initialization & Shared Formulation Bar
# ---------------------------------------------------------------------------

def _ensure_session_state() -> None:
    if "formulations" not in st.session_state:
        st.session_state.formulations = load_presets()
        st.session_state.formulations.append(Formulation(name="Blank Custom"))
    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = 0
    if "compare_idx" not in st.session_state:
        st.session_state.compare_idx = None
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "species_db" not in st.session_state:
        st.session_state.species_db = load_species_db()
    if "active_conditions" not in st.session_state:
        st.session_state["active_conditions"] = CombustionConditions(
            temperature_k=298.15, pressure_atm=100.0, mode="isobaric"
        )
    if "active_pah_config" not in st.session_state:
        st.session_state["active_pah_config"] = EPA_AP42_CALIBRATED_PAH
    if "active_eq_temp" not in st.session_state:
        st.session_state["active_eq_temp"] = 2500.0
    if "active_mode_label" not in st.session_state:
        st.session_state["active_mode_label"] = "adiabatic (self-consistent)"


def _render_formulation_manager(
    extra_editor_hook: Optional[Callable[[Formulation], None]] = None,
) -> Formulation:
    """Render top-level library selector, comparison controls, and Formulation Editor."""
    formulation_names = [f.name for f in st.session_state.formulations]

    st.markdown("**Standard Military & Aerospace Propellant Library:**")
    lib_c1, lib_c2 = st.columns([3, 1])
    with lib_c1:
        lib_templates = get_all_library_templates()
        lib_names = [e.name for e in lib_templates]
        selected_lib_idx = st.selectbox(
            "Standard Propellant Library",
            range(len(lib_names)),
            format_func=lambda i: f"{lib_templates[i].name} — {lib_templates[i].description}",
            label_visibility="collapsed",
        )
    with lib_c2:
        if st.button("➕ Inject into Session", use_container_width=True):
            src = lib_templates[selected_lib_idx]
            new_f = Formulation(
                name=src.name,
                components=[PropellantComponent(c.name, c.formula, c.mass_fraction, c.hf_298) for c in src.components],
                description=src.description,
            )
            st.session_state.formulations.append(new_f)
            st.session_state.selected_idx = len(st.session_state.formulations) - 1
            st.rerun()

    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        with st.expander("💾 Save & Load JSON Formulations"):
            cs, cl = st.columns(2)
            with cs:
                if st.button("Save Current Session", use_container_width=True):
                    save_formulations(st.session_state.formulations, Path("data/custom_formulations.json"))
                    st.success("Saved formulations to data/custom_formulations.json")
            with cl:
                uploaded = st.file_uploader("Load JSON", type="json", key="load_file", label_visibility="collapsed")
                if uploaded:
                    try:
                        data = json.load(uploaded)
                        loaded = [formulation_from_dict(e) for e in data["formulations"]]
                        st.session_state.formulations.extend(loaded)
                        st.success(f"Loaded {len(loaded)} formulations")
                        st.rerun()
                    except Exception:
                        st.error("Invalid JSON formulation file.")
    with exp_c2:
        with st.expander("📊 View All Loaded Formulations"):
            summary_data = []
            for idx, f in enumerate(st.session_state.formulations):
                try:
                    phi_val = equivalence_ratio(f)
                    ob_val = oxygen_balance(f)
                except Exception:
                    phi_val, ob_val = 0.0, 0.0
                components_str = ", ".join([f"{c.name} ({c.mass_fraction * 100:.1f}%)" for c in f.components])
                summary_data.append({
                    "idx": idx,
                    "Name": f.name,
                    "Equivalence Ratio (φ)": round(phi_val, 3),
                    "Oxygen Balance (%)": round(ob_val, 1),
                    "Components": components_str,
                })
            st.dataframe(pd.DataFrame(summary_data).set_index("idx"), use_container_width=True, height=210)

    top_c1, top_c2, top_c3 = st.columns([1.4, 1.2, 1.4])
    with top_c1:
        st.markdown("**Active Propellant Formulation:**")
        selected_name = st.selectbox(
            "Select Active Formulation",
            formulation_names,
            index=min(st.session_state.selected_idx, len(formulation_names) - 1),
            key="formulation_select",
            label_visibility="collapsed",
        )
        st.session_state.selected_idx = formulation_names.index(selected_name)
    with top_c2:
        st.markdown("**Compare Against:**")
        compare_options = ["None"] + [n for i, n in enumerate(formulation_names) if i != st.session_state.selected_idx]
        compare_choice = st.selectbox("Compare with", compare_options, label_visibility="collapsed")
        st.session_state.compare_idx = formulation_names.index(compare_choice) if compare_choice != "None" else None
    with top_c3:
        st.markdown("**Formulation Actions:**")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Duplicate", use_container_width=True):
                src = st.session_state.formulations[st.session_state.selected_idx]
                new_f = Formulation(
                    name=f"{src.name} (copy)",
                    components=[PropellantComponent(c.name, c.formula, c.mass_fraction, c.hf_298) for c in src.components],
                    description=src.description,
                )
                st.session_state.formulations.append(new_f)
                st.session_state.selected_idx = len(st.session_state.formulations) - 1
                st.rerun()
        with b2:
            if st.button("New Blank", use_container_width=True):
                st.session_state.formulations.append(Formulation(name=f"Custom {len(st.session_state.formulations) + 1}"))
                st.session_state.selected_idx = len(st.session_state.formulations) - 1
                st.rerun()
        with b3:
            if len(st.session_state.formulations) > 1 and st.button("Delete", use_container_width=True):
                st.session_state.formulations.pop(st.session_state.selected_idx)
                st.session_state.selected_idx = max(0, st.session_state.selected_idx - 1)
                st.rerun()

    formulation = st.session_state.formulations[st.session_state.selected_idx]

    with st.expander(f"⚙️ Formulation Editor — {formulation.name}", expanded=False):
        with st.expander("ℹ️  How to use — Formulation Editor", expanded=False):
            st.markdown("""
            **Define your gun or rocket propellant ingredients here.**
            - Add one row per component (`Name`, stoichiometric `Formula`, `Mass Fraction`, `Hf₂₉₈` in kJ/mol).
            - Mass fractions should **sum to 1.0** — verified automatically in the status strip below.
            - Use the **Empirical Ingredient Database Search** (48 grounded ingredients with CAS numbers) or the **Nitrocellulose %N Continuum Synthesizer** (MIL-DTL-244C).
            """)
        col_name, col_desc = st.columns([1, 2])
        with col_name:
            formulation.name = st.text_input(
                "Formulation name", value=formulation.name, key=f"form_name_{st.session_state.selected_idx}"
            )
        with col_desc:
            formulation.description = st.text_input(
                "Description", value=formulation.description, key=f"form_desc_{st.session_state.selected_idx}"
            )
        st.session_state.formulations[st.session_state.selected_idx] = formulation

        st.markdown("**Empirical Ingredient Database Search (48 Grounded Energetics, Binders & Stabilizers)**")
        ing_c1, ing_c2 = st.columns([3, 1])
        raw_ingredients = load_ingredients()
        with ing_c1:
            ing_opts = [
                f"{ing['name']}  ({ing['formula']} | {ing['hf_298']} kJ/mol | CAS {ing.get('cas', 'N/A')})"
                for ing in raw_ingredients
            ]
            selected_ing_idx = st.selectbox(
                "Search and inject standardized ingredients",
                range(len(ing_opts)),
                format_func=lambda i: ing_opts[i],
                label_visibility="collapsed",
                key=f"ing_search_{st.session_state.selected_idx}",
            )
        with ing_c2:
            if st.button("➕ Add Ingredient", use_container_width=True, key=f"add_ing_{st.session_state.selected_idx}"):
                if raw_ingredients:
                    ing = raw_ingredients[selected_ing_idx]
                    formulation.components.append(PropellantComponent(
                        name=ing["name"], formula=ing["formula"], mass_fraction=0.0, hf_298=ing["hf_298"]
                    ))
                    st.session_state.formulations[st.session_state.selected_idx] = formulation
                    st.rerun()

        render_nc_nitration_calculator(formulation)

        if extra_editor_hook is not None:
            extra_editor_hook(formulation)

        st.markdown("**Components** — mass fractions should sum to 1.0")
        comp_data = [
            {"Name": c.name, "Formula": c.formula, "Mass Fraction": c.mass_fraction, "Hf₂₉₈ (kJ/mol)": c.hf_298}
            for c in formulation.components
        ] or [{"Name": "", "Formula": "", "Mass Fraction": 0.0, "Hf₂₉₈ (kJ/mol)": 0.0}]

        edited_df = st.data_editor(
            pd.DataFrame(comp_data),
            num_rows="dynamic",
            use_container_width=True,
            key=f"comp_editor_{st.session_state.selected_idx}",
            column_config={
                "Mass Fraction": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.4f"),
                "Hf₂₉₈ (kJ/mol)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        new_components = []
        for _, row in edited_df.iterrows():
            if row["Name"] and row["Formula"]:
                try:
                    new_components.append(PropellantComponent(
                        name=str(row["Name"]),
                        formula=str(row["Formula"]),
                        mass_fraction=float(row["Mass Fraction"]),
                        hf_298=float(row["Hf₂₉₈ (kJ/mol)"]),
                    ))
                except Exception:
                    pass
        formulation.components = new_components
        st.session_state.formulations[st.session_state.selected_idx] = formulation

        if formulation.components:
            total_frac = sum(c.mass_fraction for c in formulation.components)
            frac_delta = total_frac - 1.0
            m_frac, m_phi, m_ob = st.columns(3)
            m_frac.metric(
                "Total Mass Fraction",
                f"{total_frac:.4f}",
                delta=f"{frac_delta:+.4f}" if abs(frac_delta) > 0.001 else "Normalized",
                delta_color="inverse" if abs(frac_delta) > 0.001 else "off",
            )
            try:
                phi = equivalence_ratio(formulation)
                ob = oxygen_balance(formulation)
                m_phi.metric("Equivalence Ratio (φ)", f"{phi:.3f}", delta="Fuel-rich" if phi > 1 else "O₂-rich")
                m_ob.metric("Oxygen Balance (OB%)", f"{ob:+.2f}%")
            except Exception:
                m_phi.metric("Equivalence Ratio (φ)", "Error")
                m_ob.metric("Oxygen Balance", "Error")

            col_elem, col_mass = st.columns(2)
            with col_elem:
                st.markdown("**Elemental Composition (mol/g)**")
                elem = formulation.elemental_composition()
                st.dataframe(
                    pd.DataFrame([{"Element": k, "mol/g": f"{v:.6f}"} for k, v in sorted(elem.items())]),
                    use_container_width=True, hide_index=True, height=180,
                )
            with col_mass:
                st.markdown("**Elemental Mass Fractions (%)**")
                emf = formulation.elemental_mass_fractions()
                st.dataframe(
                    pd.DataFrame([{"Element": k, "Mass %": f"{v * 100:.2f}"} for k, v in sorted(emf.items())]),
                    use_container_width=True, hide_index=True, height=180,
                )

    return formulation


# ---------------------------------------------------------------------------
# Workspace 1: Passport & Thermochemistry
# ---------------------------------------------------------------------------

def _collect_suite_data() -> Dict[str, Any]:
    return {
        "nozzle_out": st.session_state.get("nozzle_out"),
        "ib_out": st.session_state.get("ib_out"),
        "ero_out": st.session_state.get("ero_out"),
        "climatic_out": st.session_state.get("climatic_out"),
        "mc_out": st.session_state.get("mc_out"),
        "mf_out": st.session_state.get("mf_out"),
        "tox_out": st.session_state.get("tox_out"),
        "stab_out": st.session_state.get("stab_out"),
    }


def _render_workspace_passport_thermo(formulation: Formulation) -> None:
    sub_pass, sub_cond, sub_res, sub_noz = st.tabs([
        "🪪 Executive Propellant Passport",
        "⚙️ Conditions & Equilibrium Solver",
        "📊 Equilibrium Results & Exports",
        "🚀 Nozzle Isp (Shifting vs Frozen) & Calorimetry",
    ])

    conditions: CombustionConditions = st.session_state["active_conditions"]
    pah_config: PAHAllocationConfig = st.session_state["active_pah_config"]
    eq_temp: float = st.session_state["active_eq_temp"]

    # --- Sub-tab 1: Executive Propellant Passport ---
    with sub_pass:
        st.subheader(f"🪪 Executive Propellant Passport — {formulation.name}")
        st.caption(
            "Unified multi-physics engineering scorecard across Gibbs Thermochemistry, Shifting Nozzle Propulsion, "
            "Interior Ballistics & Erosion, NATO STANAG 2895 / Monte Carlo Dispersion, Signature/Hygiene, and STANAG 4582 Surveillance."
        )

        if not formulation.components:
            st.info("Add at least one ingredient in the Formulation Editor above to generate a Propellant Passport.")
        else:
            p_act1, p_act2 = st.columns([1.4, 1.0])
            with p_act1:
                run_all = st.button(
                    "⚡ Run Complete Multi-Suite Passport & Characterization",
                    type="primary",
                    key="btn_passport_all",
                    use_container_width=True,
                )
            if run_all:
                with st.spinner("Executing coupled Gibbs equilibrium, Shifting/Frozen nozzle, Ballistics/Erosion, STANAG 2895/MC, Flash/Smoke/Toxicity & STANAG 4582..."):
                    pass_cond = CombustionConditions(
                        temperature_k=298.15,
                        pressure_atm=max(conditions.pressure_atm, 100.0),
                        mode="isochoric",
                        loading_density=0.20,
                        real_gas_fugacity=True,
                    )
                    res_p = cached_solve_equilibrium_adiabatic(
                        formulation, pass_cond, pah_config, st.session_state.species_db, 2600.0
                    )
                    st.session_state.results[formulation.name] = res_p
                    st.session_state["result"] = res_p

                    noz_p = cached_analyze_nozzle_expansion(
                        formulation, 68.0, 1.0, 1.0, PRESETS["none"], st.session_state.species_db, 8
                    )
                    st.session_state["nozzle_out"] = noz_p

                    f_val = res_p.force_constant_j_per_g if res_p.force_constant_j_per_g > 50 else 1050.0
                    gam_val = res_p.gamma if res_p.gamma > 1.02 else 1.23
                    eta_val = res_p.covolume_cm3_per_g if res_p.covolume_cm3_per_g > 0.3 else 1.00

                    ib_p = simulate_ballistics(
                        force_constant_j_per_g=f_val, gamma=gam_val, charge_mass_g=50.0,
                        propellant_density_g_per_cm3=1.60, covolume_cm3_per_g=eta_val,
                        burn_rate_a=1.20, burn_rate_n=0.85, web_thickness_mm=1.20,
                        grain_geometry="7_perf", chamber_volume_cm3=65.0,
                        bore_diameter_mm=30.0, barrel_length_m=1.50, projectile_mass_g=300.0,
                    )
                    ero_p = estimate_barrel_erosion(ib_p, res_p, bore_diameter_mm=30.0, charge_mass_g=50.0)
                    st.session_state["ib_out"] = ib_p
                    st.session_state["ero_out"] = ero_p

                    clim_p = simulate_climatic_zones(
                        force_constant_j_per_g=f_val, gamma=gam_val, charge_mass_g=50.0,
                        propellant_density_g_per_cm3=1.60, covolume_cm3_per_g=eta_val,
                        burn_rate_a=1.20, burn_rate_n=0.85, web_thickness_mm=1.20,
                        grain_geometry="7_perf", chamber_volume_cm3=65.0,
                        bore_diameter_mm=30.0, barrel_length_m=1.50, projectile_mass_g=300.0,
                    )
                    st.session_state["climatic_out"] = clim_p

                    mc_p = run_monte_carlo_ballistics(
                        force_constant_j_per_g=f_val, gamma=gam_val, charge_mass_g=50.0,
                        propellant_density_g_per_cm3=1.60, covolume_cm3_per_g=eta_val,
                        burn_rate_a=1.20, burn_rate_n=0.85, web_thickness_mm=1.20,
                        grain_geometry="7_perf", chamber_volume_cm3=65.0,
                        bore_diameter_mm=30.0, barrel_length_m=1.50, projectile_mass_g=300.0,
                        n_trials=80,
                    )
                    st.session_state["mc_out"] = mc_p

                    mf_p = analyze_muzzle_flash(res_p, formulation, expansion_ratio=12.0, muzzle_mach=2.4, suppressant_wt_pct=0.5)
                    st.session_state["mf_out"] = mf_p

                    tox_p = analyze_toxicity_and_smoke(res_p, charge_mass_g=45.0, rounds_fired=5, enclosure_volume_m3=120.0)
                    st.session_state["tox_out"] = tox_p

                    stab_wt = detect_stabilizer_wt_pct(formulation)
                    stab_p = analyze_thermal_stability(formulation, storage_temp_c=25.0, stabilizer_wt_pct=max(stab_wt, 1.5))
                    st.session_state["stab_out"] = stab_p

            res_cur = st.session_state.results.get(formulation.name) or st.session_state.get("result")
            with p_act2:
                if res_cur is not None:
                    dossier_bytes = export_full_dossier_pdf(res_cur, formulation, _collect_suite_data())
                    st.download_button(
                        "📄 Download Full Multi-Suite Engineering Dossier (PDF)",
                        data=dossier_bytes,
                        file_name=f"Combusto_{formulation.name.replace(' ', '_')}_Engineering_Dossier.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            if res_cur is None:
                st.info("Click **⚡ Run Complete Multi-Suite Passport & Characterization** above (or run an equilibrium solve) to populate the Executive Passport.")
            else:
                noz = st.session_state.get("nozzle_out")
                ib = st.session_state.get("ib_out")
                ero = st.session_state.get("ero_out")
                clim = st.session_state.get("climatic_out")
                mc = st.session_state.get("mc_out")
                mf = st.session_state.get("mf_out")
                tox = st.session_state.get("tox_out")
                stab = st.session_state.get("stab_out")

                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                r1c1.metric("Closed-Vessel Temp Tv", f"{res_cur.isochoric_flame_temp_k:.0f} K", delta=f"Tp = {res_cur.adiabatic_flame_temp_k:.0f} K", delta_color="off")
                r1c2.metric("Gun Impetus / Force F", f"{res_cur.force_constant_j_per_g:.1f} J/g", delta=f"η = {res_cur.covolume_cm3_per_g:.3f} cm³/g", delta_color="off")
                r1c3.metric("Calorimetric Heat Qv", f"{res_cur.heat_of_explosion_j_per_g:.0f} J/g", delta=f"OB = {res_cur.oxygen_balance_pct:+.1f}%", delta_color="off")
                isp_sh = noz["shifting"]["isp_vacuum_s"] if noz else res_cur.specific_impulse_s
                isp_gain = noz["isp_vacuum_gain_s"] if noz else 0.0
                r1c4.metric("Shifting Vacuum Isp", f"{isp_sh:.1f} s", delta=f"{isp_gain:+.1f} s vs Frozen" if noz else "68→1 atm", delta_color="normal")

                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                r2c1.metric(
                    "Peak Breech P_max / V₀",
                    f"{ib['max_pressure_mpa']:.0f} MPa" if ib else "Not Run",
                    delta=f"V₀ = {ib['muzzle_velocity_m_s']:.0f} m/s" if ib else None,
                    delta_color="off",
                )
                r2c2.metric(
                    "Est. Barrel Life (EFC)",
                    f"{ero['estimated_barrel_life_rounds']:,} rds" if ero else "Not Run",
                    delta=f"{ero['wear_um_per_shot']:.2f} µm/shot" if ero else None,
                    delta_color="off",
                )
                r2c3.metric(
                    "STANAG 2895 Hot (+63 °C)",
                    f"{clim['hot_pmax_mpa']:.0f} MPa" if clim else "Not Run",
                    delta=f"σ(V₀) = {mc['std_v_muz_m_s']:.2f} m/s" if mc else None,
                    delta_color="off",
                )
                r2c4.metric(
                    "STANAG 4582 Life / Cook-Off",
                    f"{stab['shelf_life_at_25c_years']:.1f} yr" if stab else "Not Run",
                    delta=f"T_crit = {stab['critical_cookoff_temp_c']:.0f} °C" if stab else None,
                    delta_color="off",
                )

                pc_left, pc_right = st.columns([1.05, 1.25])
                with pc_left:
                    # 6-Axis Normalized Multi-Physics Radar Profile (0-100 scale)
                    s_impetus = min(100.0, max(10.0, (res_cur.force_constant_j_per_g - 750.0) / 5.5))
                    s_isp = min(100.0, max(10.0, (isp_sh - 180.0) / 1.3))
                    s_barrel = min(100.0, max(5.0, (math.log10(max(ero["estimated_barrel_life_rounds"], 10)) - 1.0) * 28.0)) if ero else 55.0
                    s_flash = (85.0 if not mf["ignited"] else max(15.0, 90.0 - (mf["peak_plume_temp_k"] - 1000.0) / 22.0)) if mf else 60.0
                    s_hygiene = min(100.0, max(10.0, 100.0 - tox["primary_smoke_g_per_kg"] * 1.2 - tox["max_hazard_quotient"] * 18.0)) if tox else 65.0
                    s_stab = min(100.0, max(10.0, stab["shelf_life_at_25c_years"] * 3.5)) if stab else 70.0

                    axes = [
                        "Gun Impetus (F)",
                        "Rocket Isp (Vac)",
                        "Barrel Life (EFC)",
                        "Low Muzzle Flash",
                        "Low Smoke & Hygiene",
                        "Shelf-Life (STANAG 4582)",
                    ]
                    vals = [s_impetus, s_isp, s_barrel, s_flash, s_hygiene, s_stab]
                    fig_rad = go.Figure(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=axes + [axes[0]],
                        fill="toself",
                        fillcolor="rgba(0, 240, 255, 0.18)",
                        line=dict(color="#00f0ff", width=2.5),
                        name=formulation.name,
                    ))
                    fig_rad.update_layout(
                        title="6-Axis Normalized Energetic & Survivability Footprint",
                        polar=dict(
                            bgcolor="rgba(10, 13, 22, 0.85)",
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.18)"),
                            angularaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
                        ),
                        showlegend=False,
                        height=360,
                    )
                    st.plotly_chart(fig_rad, use_container_width=True)

                with pc_right:
                    st.markdown("#### Multi-Suite Qualification Summary")
                    qual_rows = [
                        {
                            "Engineering Domain": "Thermochemistry & EOS",
                            "Key Indicator": f"Tv = {res_cur.isochoric_flame_temp_k:.0f} K | F = {res_cur.force_constant_j_per_g:.1f} J/g | Z = {res_cur.compressibility_z:.3f}",
                            "Qualification Status": "Converged (BLAKE Real-Gas)" if conditions.real_gas_fugacity or res_cur.compressibility_z > 1.001 else "Converged (NASA Glenn)",
                        },
                        {
                            "Engineering Domain": "Nozzle Propulsion",
                            "Key Indicator": f"C* = {noz['shifting']['c_star_m_s']:.0f} m/s | Isp,vac = {noz['shifting']['isp_vacuum_s']:.1f} s" if noz else f"Isp = {res_cur.specific_impulse_s:.1f} s",
                            "Qualification Status": f"Shifting Gain {noz['isp_vacuum_gain_s']:+.1f} s" if noz else "Ideal Polytropic",
                        },
                        {
                            "Engineering Domain": "Ballistics & Bore Wear",
                            "Key Indicator": f"P_max = {ib['max_pressure_mpa']:.1f} MPa | Wall = {ero['peak_wall_temp_k']:.0f} K" if (ib and ero) else "Pending Simulation",
                            "Qualification Status": ero["erosion_regime"] if ero else "Run Ballistics Suite",
                        },
                        {
                            "Engineering Domain": "STANAG 2895 & Lot MC",
                            "Key Indicator": f"-40→+63 °C: {clim['cold_pmax_mpa']:.0f}→{clim['hot_pmax_mpa']:.0f} MPa | Cpk = {mc['cpk_proof']:.2f}" if (clim and mc) else "Pending Simulation",
                            "Qualification Status": ("PASS (< Proof Limit)" if clim["passes_stanag_2895"] else "EXCEEDS PROOF") if clim else "Run Climatic/MC Suite",
                        },
                        {
                            "Engineering Domain": "Signature & Hygiene",
                            "Key Indicator": f"Plume {mf['peak_plume_temp_k']:.0f} K | HQ = {tox['max_hazard_quotient']:.2f}× PEL" if (mf and tox) else "Pending Simulation",
                            "Qualification Status": tox["agard_smoke_class"] if tox else "Run Signature Suite",
                        },
                        {
                            "Engineering Domain": "STANAG 4582 Stability",
                            "Key Indicator": f"25 °C Life = {stab['shelf_life_at_25c_years']:.1f} yr | T_crit = {stab['critical_cookoff_temp_c']:.1f} °C" if stab else "Pending Simulation",
                            "Qualification Status": ("STANAG 4582 Compliant" if stab["stanag_4582_compliant"] else "Surveillance Alert") if stab else "Run Surveillance Suite",
                        },
                    ]
                    st.dataframe(pd.DataFrame(qual_rows), use_container_width=True, hide_index=True, height=280)

    # --- Sub-tab 2: Conditions & Equilibrium Solver ---
    with sub_cond:
        with st.expander("ℹ️  How to use — Conditions & Equilibrium Solver", expanded=False):
            st.markdown("""
            **Configure operating conditions and launch the Gibbs free-energy minimization solver.**
            - **Mode**: *adiabatic (self-consistent)* iteratively solves flame temperature ($T_p$ or $T_v$) and species composition; *isobaric* and *isochoric* solve at fixed equilibrium temperature or loading density $\\Delta$.
            - **BLAKE / Virial Real-Gas Fugacity**: Applies high-pressure covolume chemical potential corrections ($\\ln \\phi_i = b_i P / RT$) per US Army BRL `BLAKE` (`ARBRL-TR-02411`).
            """)
        st.subheader("Combustion & Real-Gas Equilibrium Conditions")

        col1, col2, col3 = st.columns(3)
        with col1:
            mode = st.selectbox(
                "Solver Mode",
                ["adiabatic (self-consistent)", "isobaric", "isochoric"],
                index=["adiabatic (self-consistent)", "isobaric", "isochoric"].index(st.session_state.get("active_mode_label", "adiabatic (self-consistent)")),
                key="cond_mode",
            )
            pressure = st.number_input("Chamber Pressure (atm)", 0.1, 10000.0, float(conditions.pressure_atm), 10.0, key="cond_p")
        with col2:
            eq_temp = st.number_input(
                "Fixed Equilibrium Temp (K)",
                500.0, 6000.0, float(st.session_state.get("active_eq_temp", 2500.0)), 100.0,
                disabled=(mode == "adiabatic (self-consistent)"),
                help="Disabled in self-consistent adiabatic mode — T is solved iteratively.",
                key="cond_eq_t",
            )
            init_temp = st.number_input("Initial Grain Temp (K)", 200.0, 500.0, float(conditions.temperature_k), 1.0, key="cond_init_t")
        with col3:
            real_gas_fug = st.checkbox(
                "Enable BLAKE Real-Gas Fugacity (ln φᵢ = bᵢP/RT)",
                value=bool(conditions.real_gas_fugacity),
                help="Applies truncated 2nd-Virial / Noble-Abel covolume fugacity chemical potential shifts at high chamber pressures.",
                key="cond_rg_fug",
            )
            loading_density = None
            covolume = 1.0e-3
            if mode == "isochoric":
                loading_density = st.number_input(
                    "Loading Density Δ (g/cm³)", 0.01, 2.0, float(conditions.loading_density or 0.20), 0.01, key="cond_ld"
                )
                covolume = st.number_input(
                    "Override Covolume (m³/mol, leave 0.0010 for auto BLAKE Σnᵢbᵢ)",
                    1e-5, 1e-2, float(conditions.covolume_m3_per_mol), 1e-4, format="%.4f", key="cond_cov",
                )

        st.divider()
        st.subheader("PAH / Soot Carbon Allocation Model")
        col_pah1, col_pah2 = st.columns([1, 2])
        with col_pah1:
            ui_pah_presets = dict(PRESETS)
            ui_pah_presets["epa_ap42_calibrated (Trace VOC/PAH)"] = EPA_AP42_CALIBRATED_PAH
            pah_keys = list(ui_pah_presets.keys())
            pah_preset = st.selectbox(
                "Preset",
                pah_keys,
                index=pah_keys.index("epa_ap42_calibrated (Trace VOC/PAH)"),
                key="cond_pah_preset",
            )
            pah_config = ui_pah_presets[pah_preset]
        with col_pah2:
            with st.expander("Customize PAH Partitioning Parameters", expanded=False):
                c1, c2, c3 = st.columns(3)
                custom_base = c1.slider("Base C fraction", 0.0, 0.3, float(pah_config.base_fraction), 0.01)
                custom_exp = c2.slider("φ exponent", 0.5, 5.0, float(pah_config.phi_exponent), 0.1)
                custom_max = c3.slider("Max C fraction", 0.0, 0.8, float(pah_config.max_fraction), 0.05)
                if (custom_base != pah_config.base_fraction or custom_exp != pah_config.phi_exponent or custom_max != pah_config.max_fraction):
                    pah_config = PAHAllocationConfig(
                        model_name="custom",
                        base_fraction=custom_base,
                        phi_exponent=custom_exp,
                        max_fraction=custom_max,
                        pah_distribution=pah_config.pah_distribution,
                    )

        conditions = CombustionConditions(
            temperature_k=init_temp,
            pressure_atm=pressure,
            mode="isochoric" if mode == "isochoric" else "isobaric",
            loading_density=loading_density,
            covolume_m3_per_mol=covolume,
            real_gas_fugacity=real_gas_fug,
        )
        st.session_state["active_conditions"] = conditions
        st.session_state["active_pah_config"] = pah_config
        st.session_state["active_eq_temp"] = eq_temp
        st.session_state["active_mode_label"] = mode

        st.divider()
        if st.button("▶ Run Combustion Equilibrium Calculation", type="primary", key="btn_run_eq"):
            if not formulation.components:
                st.error("Add at least one component to the formulation.")
            else:
                with st.spinner("Solving Gibbs free-energy equilibrium..."):
                    if mode == "adiabatic (self-consistent)":
                        result = cached_solve_equilibrium_adiabatic(
                            formulation, conditions, pah_config, st.session_state.species_db, 2500.0
                        )
                        if result.converged:
                            st.success(
                                f"Converged in {result.iterations} iterations "
                                f"(Tp = {result.adiabatic_flame_temp_k:.0f} K | Tv = {result.isochoric_flame_temp_k:.0f} K | F = {result.force_constant_j_per_g:.1f} J/g)"
                            )
                        else:
                            st.warning(f"Did not fully converge after {result.iterations} iterations (T ≈ {result.temperature_k:.0f} K)")
                    else:
                        result = cached_solve_equilibrium(
                            formulation, conditions, pah_config, st.session_state.species_db, eq_temp
                        )
                        st.success("Equilibrium calculation complete!")

                    st.session_state.results[formulation.name] = result
                    st.session_state["result"] = result

                    if st.session_state.compare_idx is not None:
                        comp_form = st.session_state.formulations[st.session_state.compare_idx]
                        if mode == "adiabatic (self-consistent)":
                            res_b = cached_solve_equilibrium_adiabatic(
                                comp_form, conditions, pah_config, st.session_state.species_db, 2500.0
                            )
                        else:
                            res_b = cached_solve_equilibrium(
                                comp_form, conditions, pah_config, st.session_state.species_db, eq_temp
                            )
                        st.session_state.results[comp_form.name] = res_b

    # --- Sub-tab 3: Equilibrium Results & Exports ---
    with sub_res:
        active_name = formulation.name
        result = st.session_state.results.get(active_name)
        if result is None:
            st.info("Run a calculation in **Conditions & Equilibrium Solver** (or **Executive Propellant Passport**) first.")
        else:
            st.subheader(f"Equilibrium Thermochemistry: {active_name}")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Equivalence Ratio φ", f"{result.equivalence_ratio:.3f}")
            m2.metric("Oxygen Balance", f"{result.oxygen_balance_pct:+.2f}%")
            m3.metric("Tp Isobaric (K)", f"{result.adiabatic_flame_temp_k:.0f}", help="Constant-pressure flame temp (ΔH = 0)")
            m4.metric("Tv Closed-Vessel (K)", f"{result.isochoric_flame_temp_k:.0f}", help="Constant-volume gun chamber flame temp (ΔU = 0)")
            m5.metric("Impetus F (J/g)", f"{result.force_constant_j_per_g:.1f}", help="n_gas · R · Tv (STANAG 4115 / BRL)")

            m6, m7, m8, m9, m10 = st.columns(5)
            m6.metric("Covolume η (cm³/g)", f"{result.covolume_cm3_per_g:.3f}", help="BLAKE composition-derived Noble-Abel covolume Σ nᵢbᵢ")
            m7.metric("Specific Heat Ratio γ", f"{result.gamma:.3f}")
            m8.metric("Mean Gas MW", f"{result.mean_molecular_weight:.2f} g/mol")
            m9.metric("Qv Calorimetric", f"{result.heat_of_explosion_j_per_g:.0f} J/g")
            m10.metric("Compressibility Z", f"{getattr(result, 'compressibility_z', 1.0):.3f}")

            st.markdown("### Equilibrium Product Species Distribution")
            prod_df = result_to_dataframe(result)
            st.dataframe(
                prod_df.style.format({"Moles": "{:.6e}", "Mole Fraction": "{:.6f}", "Mass Fraction": "{:.6f}"}),
                use_container_width=True, hide_index=True,
            )

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                top10 = prod_df.nlargest(10, "Mole Fraction")
                fig = px.bar(
                    top10, x="Species", y="Mole Fraction", color="Mole Fraction",
                    color_continuous_scale="Tealgrn", text_auto=".4f",
                    title="Top-10 Equilibrium Product Mole Fractions",
                )
                fig.update_layout(showlegend=False, height=390, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with col_chart2:
                if result.carbon_distribution:
                    c_df = carbon_distribution_to_dataframe(result)
                    c_df = c_df[c_df["Fraction"] > 0.0005]
                    fig2 = px.pie(
                        c_df, names="Category", values="Fraction", hole=0.38,
                        title="Carbon Element Partitioning",
                    )
                    fig2.update_layout(height=390)
                    st.plotly_chart(fig2, use_container_width=True)

            if result.pah_details:
                st.markdown("### PAH & Soot Precursor Inventory")
                pah_df = pah_details_to_dataframe(result)
                cp1, cp2 = st.columns(2)
                with cp1:
                    st.dataframe(
                        pah_df.style.format({"Mole Fraction": "{:.6e}", "Mass Fraction": "{:.6e}", "Carbon Share": "{:.4f}"}),
                        use_container_width=True, hide_index=True,
                    )
                with cp2:
                    if len(pah_df) > 0:
                        fig_pah = px.bar(pah_df, x="Species", y="Carbon Share", color="Species", text_auto=".4f", title="PAH Species Carbon Share")
                        fig_pah.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig_pah, use_container_width=True)

            with st.expander("Element Conservation Verification"):
                eb_df = element_balance_to_dataframe(result)
                max_err = eb_df["Error %"].max() if len(eb_df) > 0 else 0.0
                if max_err < 1.0:
                    st.success(f"Element conservation verified — max error: {max_err:.5f}%")
                else:
                    st.warning(f"Max element balance error: {max_err:.2f}%")
                st.dataframe(eb_df.style.format({"Error %": "{:.5f}"}), use_container_width=True, hide_index=True)

            if st.session_state.compare_idx is not None:
                comp_name = st.session_state.formulations[st.session_state.compare_idx].name
                result_b = st.session_state.results.get(comp_name)
                if result_b:
                    st.divider()
                    st.subheader(f"Side-by-Side Comparison: {active_name} vs. {comp_name}")
                    metrics_data = {
                        "Metric": ["Equivalence Ratio φ", "OB (%)", "Tp Isobaric (K)", "Tv Closed-Vessel (K)", "Impetus F (J/g)", "Covolume η (cm³/g)", "γ (Cp/Cv)", "Mean Gas MW"],
                        active_name: [
                            f"{result.equivalence_ratio:.3f}", f"{result.oxygen_balance_pct:+.2f}",
                            f"{result.adiabatic_flame_temp_k:.0f}", f"{result.isochoric_flame_temp_k:.0f}",
                            f"{result.force_constant_j_per_g:.1f}", f"{result.covolume_cm3_per_g:.3f}",
                            f"{result.gamma:.3f}", f"{result.mean_molecular_weight:.2f}",
                        ],
                        comp_name: [
                            f"{result_b.equivalence_ratio:.3f}", f"{result_b.oxygen_balance_pct:+.2f}",
                            f"{result_b.adiabatic_flame_temp_k:.0f}", f"{result_b.isochoric_flame_temp_k:.0f}",
                            f"{result_b.force_constant_j_per_g:.1f}", f"{result_b.covolume_cm3_per_g:.3f}",
                            f"{result_b.gamma:.3f}", f"{result_b.mean_molecular_weight:.2f}",
                        ],
                    }
                    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Export Reports & Structured Data")
            ex1, ex2, ex3, ex4, ex5 = st.columns(5)
            ex1.download_button("CSV Table", result_to_csv(result), file_name=f"combusto_{active_name.replace(' ', '_')}.csv", mime="text/csv", use_container_width=True)
            ex2.download_button("Excel Workbook", result_to_excel_bytes(result), file_name=f"combusto_{active_name.replace(' ', '_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            json_str = json.dumps({
                "formulation": formulation_to_dict(formulation),
                "results_summary": {
                    "phi": result.equivalence_ratio, "OB_pct": result.oxygen_balance_pct,
                    "Tp_K": result.adiabatic_flame_temp_k, "Tv_K": result.isochoric_flame_temp_k,
                    "force_J_per_g": result.force_constant_j_per_g, "covolume_cm3_per_g": result.covolume_cm3_per_g,
                    "gamma": result.gamma, "mean_MW": result.mean_molecular_weight,
                },
            }, indent=2)
            ex3.download_button("JSON Bundle", json_str, file_name=f"combusto_{active_name.replace(' ', '_')}.json", mime="application/json", use_container_width=True)
            ex4.download_button("📄 Lab Summary PDF", export_result_pdf(result, formulation), file_name=f"combusto_{active_name.replace(' ', '_')}_lab.pdf", mime="application/pdf", use_container_width=True)
            ex5.download_button("📘 Full Dossier PDF", export_full_dossier_pdf(result, formulation, _collect_suite_data()), file_name=f"combusto_{active_name.replace(' ', '_')}_dossier.pdf", mime="application/pdf", use_container_width=True)

    # --- Sub-tab 4: Nozzle Isp (Shifting vs Frozen) & Calorimetry ---
    with sub_noz:
        st.subheader("🚀 1D Rocket Nozzle Expansion — Shifting Equilibrium vs. Frozen Composition")
        st.caption(
            "Solves NASA CEA-style homogeneous isentropic expansion S(T, P) = S_c through sonic throat Pt to exit pressure Pe, "
            "quantifying exothermic CO/H₂/radical recombination gains in C*, CF, and Vacuum/Ambient Isp."
        )
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            pc_atm = st.number_input("Chamber Pressure Pc (atm)", 5.0, 500.0, 68.0, 2.0, key="noz_pc")
        with nc2:
            pe_atm = st.number_input("Nozzle Exit Pressure Pe (atm)", 0.005, 20.0, 1.0, 0.1, format="%.3f", key="noz_pe")
        with nc3:
            pa_atm = st.number_input("Ambient Backpressure Pa (atm, 0 = Vacuum)", 0.0, 5.0, 1.0, 0.1, format="%.2f", key="noz_pa")

        if st.button("▶ Solve Shifting vs. Frozen Nozzle Performance", type="primary", key="btn_run_nozzle"):
            if not formulation.components:
                st.error("Define a formulation first.")
            else:
                with st.spinner("Solving chamber state, sonic throat flux maximum, and shifting vs. frozen expansion profiles..."):
                    noz_out = cached_analyze_nozzle_expansion(
                        formulation=formulation,
                        pc_atm=pc_atm,
                        pe_atm=pe_atm,
                        ambient_p_atm=pa_atm,
                        pah_config=PRESETS["none"],
                        species_db=st.session_state.species_db,
                        n_profile_points=12,
                    )
                    st.session_state["nozzle_out"] = noz_out
                    st.session_state["res_perf_cached"] = noz_out["chamber_result"]

        if "nozzle_out" in st.session_state:
            noz = st.session_state["nozzle_out"]
            fz, sh = noz["frozen"], noz["shifting"]

            nm1, nm2, nm3, nm4 = st.columns(4)
            nm1.metric("Shifting Vacuum Isp", f"{sh['isp_vacuum_s']:.1f} s", delta=f"{noz['isp_vacuum_gain_s']:+.2f} s vs Frozen")
            nm2.metric("Shifting Ambient Isp", f"{sh['isp_ambient_s']:.1f} s", delta=f"{noz['isp_ambient_gain_s']:+.2f} s vs Frozen")
            nm3.metric("Characteristic Velocity C*", f"{sh['c_star_m_s']:.1f} m/s", delta=f"Tc = {noz['chamber_temp_k']:.0f} K", delta_color="off")
            nm4.metric("Optimum Area Ratio Ae/At", f"{sh['area_ratio_ae_at']:.2f}", delta=f"CF,vac = {sh['cf_vacuum']:.3f}", delta_color="off")

            comp_noz_df = pd.DataFrame([
                {"Parameter": "Characteristic Velocity C* (m/s)", "Frozen Composition": round(fz["c_star_m_s"], 1), "Shifting Equilibrium": round(sh["c_star_m_s"], 1), "Recombination Shift": f"{sh['c_star_m_s'] - fz['c_star_m_s']:+.1f}"},
                {"Parameter": "Sonic Throat Temp Tt (K)", "Frozen Composition": round(fz["throat_temp_k"], 1), "Shifting Equilibrium": round(sh["throat_temp_k"], 1), "Recombination Shift": f"{sh['throat_temp_k'] - fz['throat_temp_k']:+.1f}"},
                {"Parameter": "Exit Plane Temp Te (K)", "Frozen Composition": round(fz["exit_temp_k"], 1), "Shifting Equilibrium": round(sh["exit_temp_k"], 1), "Recombination Shift": f"{sh['exit_temp_k'] - fz['exit_temp_k']:+.1f}"},
                {"Parameter": "Exit Exhaust Velocity Ve (m/s)", "Frozen Composition": round(fz["exit_velocity_m_s"], 1), "Shifting Equilibrium": round(sh["exit_velocity_m_s"], 1), "Recombination Shift": f"{sh['exit_velocity_m_s'] - fz['exit_velocity_m_s']:+.1f}"},
                {"Parameter": "Expansion Area Ratio ε = Ae/At", "Frozen Composition": round(fz["area_ratio_ae_at"], 2), "Shifting Equilibrium": round(sh["area_ratio_ae_at"], 2), "Recombination Shift": f"{sh['area_ratio_ae_at'] - fz['area_ratio_ae_at']:+.2f}"},
                {"Parameter": "Thrust Coefficient CF (Ambient)", "Frozen Composition": round(fz["cf_ambient"], 3), "Shifting Equilibrium": round(sh["cf_ambient"], 3), "Recombination Shift": f"{sh['cf_ambient'] - fz['cf_ambient']:+.3f}"},
                {"Parameter": "Ambient Specific Impulse Isp (s)", "Frozen Composition": round(fz["isp_ambient_s"], 2), "Shifting Equilibrium": round(sh["isp_ambient_s"], 2), "Recombination Shift": f"{sh['isp_ambient_s'] - fz['isp_ambient_s']:+.2f}"},
                {"Parameter": "Vacuum Specific Impulse Isp,vac (s)", "Frozen Composition": round(fz["isp_vacuum_s"], 2), "Shifting Equilibrium": round(sh["isp_vacuum_s"], 2), "Recombination Shift": f"{sh['isp_vacuum_s'] - fz['isp_vacuum_s']:+.2f}"},
            ])
            st.dataframe(comp_noz_df, use_container_width=True, hide_index=True)

            pdf_n = noz["profile_df"]
            ng1, ng2 = st.columns(2)
            with ng1:
                fig_nt = go.Figure()
                fig_nt.add_trace(go.Scatter(x=pdf_n["pressure_ratio_pc_p"], y=pdf_n["t_shifting_k"], name="Shifting Temp T(P)", line=dict(color="#00f0ff", width=2.8)))
                fig_nt.add_trace(go.Scatter(x=pdf_n["pressure_ratio_pc_p"], y=pdf_n["t_frozen_k"], name="Frozen Temp T(P)", line=dict(color="#94a3b8", dash="dash", width=2.2)))
                fig_nt.add_trace(go.Scatter(x=pdf_n["pressure_ratio_pc_p"], y=pdf_n["v_shifting_m_s"], name="Shifting Velocity V (m/s)", yaxis="y2", line=dict(color="#00ff88", width=2.2)))
                fig_nt.update_layout(
                    title="Shifting vs. Frozen Temperature & Velocity Along Expansion (Pc / P)",
                    xaxis=dict(title="Expansion Pressure Ratio Pc / P", type="log"),
                    yaxis=dict(title="Static Temperature (K)"),
                    yaxis2=dict(title="Exhaust Velocity (m/s)", overlaying="y", side="right"),
                    height=390,
                )
                st.plotly_chart(fig_nt, use_container_width=True)
            with ng2:
                fig_nx = go.Figure()
                fig_nx.add_trace(go.Scatter(x=pdf_n["pressure_ratio_pc_p"], y=pdf_n["x_CO_shifting"], name="CO", line=dict(width=2.4)))
                fig_nx.add_trace(go.Scatter(x=pdf_n["pressure_ratio_pc_p"], y=pdf_n["x_CO2_shifting"], name="CO₂", line=dict(width=2.4)))
                fig_nx.add_trace(go.Scatter(x=pdf_n["pressure_ratio_pc_p"], y=pdf_n["x_H2_shifting"], name="H₂", line=dict(width=2.4)))
                fig_nx.add_trace(go.Scatter(x=pdf_n["pressure_ratio_pc_p"], y=pdf_n["x_H2O_shifting"], name="H₂O", line=dict(width=2.4)))
                fig_nx.update_layout(
                    title="Water-Gas Shift & Exothermic Recombination Along Nozzle (Shifting)",
                    xaxis=dict(title="Expansion Pressure Ratio Pc / P", type="log"),
                    yaxis=dict(title="Species Mole Fraction"),
                    height=390,
                )
                st.plotly_chart(fig_nx, use_container_width=True)

        render_noble_abel_isentrope_section(formulation, conditions, pah_config, eq_temp)

        st.divider()
        st.subheader("🧪 Isothermal Bomb Calorimetry (Qv at 298.15 K, Liquid H₂O Standard)")
        if st.button("▶ Run 298.15 K Bomb Calorimeter Simulation", type="secondary", key="btn_cal_298"):
            if formulation.components:
                cal_cond = CombustionConditions(
                    temperature_k=298.15, pressure_atm=1.0, mode="bomb_calorimeter", loading_density=0.10
                )
                res_cal = cached_solve_equilibrium(
                    formulation, cal_cond, PRESETS["moderate"], st.session_state.species_db, 298.15
                )
                st.session_state["res_cal_298"] = res_cal
        if "res_cal_298" in st.session_state:
            rc = st.session_state["res_cal_298"]
            cc1, cc2 = st.columns(2)
            cc1.metric("Calorimetric Heat of Explosion Qv (H₂O liq)", f"{rc.heat_of_explosion_j_per_g:.0f} J/g")
            cc2.metric("Permanent Gas Moles at 298.15 K", f"{rc.gas_moles_total:.5f} mol/g")
            st.dataframe(result_to_dataframe(rc), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Workspace 2: Ballistics & Chamber Tolerance
# ---------------------------------------------------------------------------

def _render_workspace_ballistics_tolerance(formulation: Formulation) -> None:
    sub_ib, sub_tol, sub_cv, sub_vod = st.tabs([
        "🔫 Interior Ballistics & Barrel Erosion",
        "🌡️ STANAG 2895 Climatic & Monte Carlo Dispersion",
        "💣 Closed-Vessel & Vivacity",
        "💥 VoD Detonation Screening",
    ])

    res_cur = st.session_state.results.get(formulation.name) or st.session_state.get("result")
    default_F = float(res_cur.force_constant_j_per_g) if (res_cur and res_cur.force_constant_j_per_g > 0) else 1050.0
    default_gamma = float(res_cur.gamma) if (res_cur and res_cur.gamma > 1.0) else 1.24
    default_eta = float(round(res_cur.covolume_cm3_per_g, 3)) if (res_cur and getattr(res_cur, "covolume_cm3_per_g", 0.0) > 0.3) else 1.00

    # --- Sub-tab 1: Interior Ballistics & Erosion ---
    with sub_ib:
        st.subheader("Transient Gun Interior Ballistics, 7/19-Perf Burnback & Barrel Erosion")
        if not formulation.components:
            st.info("Define a formulation first.")
        else:
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.markdown("**Propellant & Real-Gas EOS**")
                ib_F = st.number_input("Force Constant F (J/g)", 100.0, 2500.0, default_F, 10.0, key="ib_F")
                ib_gamma = st.number_input("Specific Heat Ratio γ", 1.05, 1.67, default_gamma, 0.01, key="ib_gamma")
                ib_charge_g = st.number_input("Charge Mass (g)", 0.5, 50000.0, 50.0, 1.0, key="ib_charge")
                ib_rho_p = st.number_input("Propellant Solid Density (g/cm³)", 0.8, 2.2, 1.60, 0.05, key="ib_rhop")
                ib_covol = st.number_input("Gas Covolume η (cm³/g, BLAKE Σnᵢbᵢ)", 0.4, 2.0, default_eta, 0.02, key="ib_covol")
            with col_b2:
                st.markdown("**Burn Rate & Multi-Perf Geometry**")
                geom_labels = {
                    "7_perf": "7-Perforated Cylinder (Progressive + Slivers)",
                    "19_perf": "19-Perforated Rosette (Highly Progressive + Slivers)",
                    "single_perf": "Single-Perforated Tube (Neutral)",
                    "cord": "Cylindrical Cord (Regressive)",
                    "spherical": "Spherical Ball (Regressive)",
                }
                ib_geom = st.selectbox("Grain Geometry", list(geom_labels.keys()), format_func=lambda g: geom_labels[g], key="ib_geom")
                ib_web_mm = st.number_input("Web Thickness D₀ (mm)", 0.1, 20.0, 1.2, 0.1, key="ib_web")
                ib_a = st.number_input("Burn Rate Coeff a (mm/s/MPaⁿ)", 0.05, 20.0, 1.2, 0.1, key="ib_a")
                ib_n = st.number_input("Pressure Exponent n", 0.3, 1.2, 0.85, 0.02, key="ib_n")
            with col_b3:
                st.markdown("**Chamber & Gun Tube Hardware**")
                ib_vc_cm3 = st.number_input("Chamber Volume Vc (cm³)", 1.0, 100000.0, 65.0, 5.0, key="ib_vc")
                ib_bore_mm = st.number_input("Bore Diameter (mm)", 4.0, 200.0, 30.0, 1.0, key="ib_bore")
                ib_len_m = st.number_input("Barrel Travel Length (m)", 0.1, 10.0, 1.5, 0.1, key="ib_len")
                ib_mproj_g = st.number_input("Projectile Mass (g)", 1.0, 100000.0, 300.0, 10.0, key="ib_mproj")
                ib_pstart_mpa = st.number_input("Shot-Start Pressure (MPa)", 1.0, 150.0, 30.0, 5.0, key="ib_pstart")

            with st.expander("🔬 2D Grain Cross-Section Burnback & Form Function Visualizer", expanded=False):
                u_max_vis = 1.18 if ib_geom == "7_perf" else (1.15 if ib_geom == "19_perf" else 1.0)
                u_vis = st.slider("Inspect Normalized Web Burn Depth u = 2·x_b / D₀", 0.0, float(u_max_vis), 0.35, 0.02, key="ib_u_vis")
                cs = generate_grain_cross_section(ib_geom, ib_web_mm, u_vis)
                z_cur, _ = grain_form_function(u_vis, ib_geom)

                vis_c1, vis_c2 = st.columns(2)
                with vis_c1:
                    theta = np.linspace(0, 2 * np.pi, 120)
                    fig_cs = go.Figure()
                    r_oi, r_oc = cs["outer_r_init"], cs["outer_r_cur"]
                    fig_cs.add_trace(go.Scatter(x=r_oi * np.cos(theta), y=r_oi * np.sin(theta), mode="lines", name="Initial Outer", line=dict(color="#64748b", dash="dot", width=1.5)))
                    fig_cs.add_trace(go.Scatter(x=r_oc * np.cos(theta), y=r_oc * np.sin(theta), mode="lines", name="Active Outer Front", line=dict(color="#00f0ff", width=2.5), fill="toself", fillcolor="rgba(0, 240, 255, 0.10)"))
                    for idx_p, (cx, cy) in enumerate(cs["perf_centers"]):
                        r_pi, r_pc = cs["perf_r_init"], cs["perf_r_cur"]
                        fig_cs.add_trace(go.Scatter(x=cx + r_pi * np.cos(theta), y=cy + r_pi * np.sin(theta), mode="lines", name="Initial Perf" if idx_p == 0 else None, showlegend=(idx_p == 0), line=dict(color="#64748b", dash="dot", width=1.0)))
                        fig_cs.add_trace(go.Scatter(x=cx + r_pc * np.cos(theta), y=cy + r_pc * np.sin(theta), mode="lines", name="Active Perf Front" if idx_p == 0 else None, showlegend=(idx_p == 0), line=dict(color="#ffaa00", width=2.0), fill="toself", fillcolor="rgba(14, 18, 30, 0.90)"))
                    lim = r_oi * 1.15
                    fig_cs.update_layout(title=f"Cross-Section at u={u_vis:.2f} (Mass Burned z={z_cur * 100:.1f}%)", xaxis=dict(title="mm", range=[-lim, lim], scaleanchor="y", scaleratio=1), yaxis=dict(title="mm", range=[-lim, lim]), height=360)
                    st.plotly_chart(fig_cs, use_container_width=True)
                with vis_c2:
                    u_arr = np.linspace(0.0, u_max_vis, 100)
                    z_vals = [grain_form_function(float(uu), ib_geom)[0] * 100.0 for uu in u_arr]
                    dz_vals = [grain_form_function(float(uu), ib_geom)[1] for uu in u_arr]
                    fig_ff = go.Figure()
                    fig_ff.add_trace(go.Scatter(x=u_arr, y=z_vals, name="Mass Burned z (%)", line=dict(color="#00ff88", width=2.5)))
                    fig_ff.add_trace(go.Scatter(x=u_arr, y=dz_vals, name="Relative Surface Area dz/du", yaxis="y2", line=dict(color="#ffaa00", dash="dash", width=2.0)))
                    fig_ff.add_vline(x=u_vis, line_dash="dot", line_color="#00f0ff")
                    fig_ff.update_layout(title="Grain Form Function z(u) & Relative Burning Area dz/du", xaxis_title="Normalized Web Burn Depth u", yaxis=dict(title="Mass Burned z (%)", range=[0, 105]), yaxis2=dict(title="Relative Area dz/du", overlaying="y", side="right"), height=360)
                    st.plotly_chart(fig_ff, use_container_width=True)

            if st.button("▶ Run Interior Ballistics & Barrel Erosion Simulation", type="primary", key="btn_run_ib"):
                with st.spinner("Integrating Lagrange ballistic ODEs and Lawton/Krause bore heat transfer..."):
                    ib_out = simulate_ballistics(
                        force_constant_j_per_g=ib_F, gamma=ib_gamma, charge_mass_g=ib_charge_g,
                        propellant_density_g_per_cm3=ib_rho_p, covolume_cm3_per_g=ib_covol,
                        burn_rate_a=ib_a, burn_rate_n=ib_n, web_thickness_mm=ib_web_mm,
                        grain_geometry=ib_geom, chamber_volume_cm3=ib_vc_cm3,
                        bore_diameter_mm=ib_bore_mm, barrel_length_m=ib_len_m,
                        projectile_mass_g=ib_mproj_g, start_pressure_mpa=ib_pstart_mpa,
                    )
                    ero_out = estimate_barrel_erosion(ib_out, res_cur, bore_diameter_mm=ib_bore_mm, charge_mass_g=ib_charge_g)
                    st.session_state["ib_out"] = ib_out
                    st.session_state["ero_out"] = ero_out

            if "ib_out" in st.session_state:
                ib_out = st.session_state["ib_out"]
                m_ib1, m_ib2, m_ib3, m_ib4 = st.columns(4)
                m_ib1.metric("Peak Chamber Pressure", f"{ib_out['max_pressure_mpa']:.1f} MPa")
                m_ib2.metric("Muzzle Velocity", f"{ib_out['muzzle_velocity_m_s']:.1f} m/s")
                ke_kj = 0.5 * (ib_mproj_g / 1000.0) * (ib_out["muzzle_velocity_m_s"] ** 2) / 1000.0
                m_ib3.metric("Muzzle Kinetic Energy", f"{ke_kj:.1f} kJ")
                m_ib4.metric("Action Time", f"{ib_out['burn_time_ms']:.2f} ms")

                ib_g1, ib_g2 = st.columns(2)
                with ib_g1:
                    fig_ib1 = go.Figure()
                    fig_ib1.add_trace(go.Scatter(x=ib_out["time"] * 1000.0, y=ib_out["pressure_mpa"], name="Pressure (MPa)", line=dict(color="#00f0ff", width=2.5)))
                    fig_ib1.add_trace(go.Scatter(x=ib_out["time"] * 1000.0, y=ib_out["fraction_burned"] * 100.0, name="Burned %", yaxis="y2", line=dict(color="#ffaa00", dash="dot", width=2)))
                    fig_ib1.update_layout(title="Chamber Pressure & Burn Fraction vs Time", xaxis_title="Time (ms)", yaxis=dict(title="Pressure (MPa)"), yaxis2=dict(title="Burned (%)", overlaying="y", side="right", range=[0, 105]), height=390)
                    st.plotly_chart(fig_ib1, use_container_width=True)
                with ib_g2:
                    fig_ib2 = go.Figure()
                    fig_ib2.add_trace(go.Scatter(x=ib_out["travel_m"], y=ib_out["velocity_m_s"], name="Velocity (m/s)", line=dict(color="#00ff88", width=2.5)))
                    fig_ib2.add_trace(go.Scatter(x=ib_out["travel_m"], y=ib_out["pressure_mpa"], name="Pressure (MPa)", yaxis="y2", line=dict(color="#00f0ff", dash="dash", width=2)))
                    fig_ib2.update_layout(title="Projectile Velocity & Pressure vs Barrel Travel", xaxis_title="Travel (m)", yaxis=dict(title="Velocity (m/s)"), yaxis2=dict(title="Pressure (MPa)", overlaying="y", side="right"), height=390)
                    st.plotly_chart(fig_ib2, use_container_width=True)

                if "ero_out" in st.session_state:
                    ero_out = st.session_state["ero_out"]
                    st.divider()
                    st.markdown("### 🔥 Barrel Thermal Erosion & Wear Life Assessment")
                    st.info(f"**Erosion Regime:** {ero_out['erosion_regime']}")
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Peak Bore Wall Temp", f"{ero_out['peak_wall_temp_k']:.0f} K ({ero_out['peak_wall_temp_c']:.0f} °C)")
                    e2.metric("Peak Wall Heat Flux", f"{ero_out['peak_heat_flux_mw_m2']:.1f} MW/m²")
                    e3.metric("Throat Wear Rate", f"{ero_out['wear_um_per_shot']:.2f} µm/shot")
                    e4.metric("Est. Barrel Life (EFC)", f"{ero_out['estimated_barrel_life_rounds']:,} rounds")

                    fig_ero = go.Figure()
                    fig_ero.add_trace(go.Scatter(x=ib_out["time"] * 1000.0, y=ero_out["wall_temp_k"], name="Bore Wall Surface Temp (K)", line=dict(color="#ff5566", width=2.5)))
                    fig_ero.add_hline(y=1000.0, line_dash="dash", line_color="#ffaa00", annotation_text="Steel Austenitization (~1000 K)")
                    fig_ero.add_hline(y=1720.0, line_dash="dot", line_color="#ff2222", annotation_text="Gun Steel Melt (~1720 K)")
                    fig_ero.update_layout(title="Transient Bore Wall Surface Temperature at Origin of Rifling", xaxis_title="Time (ms)", yaxis_title="Wall Temperature (K)", height=360)
                    st.plotly_chart(fig_ero, use_container_width=True)

    # --- Sub-tab 2: STANAG 2895 Climatic & Monte Carlo Dispersion ---
    with sub_tol:
        st.subheader("🌡️ NATO STANAG 2895 Climatic Conditioning & Monte Carlo Lot-Tolerance Dispersion")
        st.caption(
            "Uses the active gun/grain setup from Interior Ballistics to evaluate extreme climatic zones (-40 °C Arctic C2 to +63 °C Hot Desert A1) "
            "and stochastic lot-to-lot manufacturing tolerances against Maximum Permissible (MPOP) and Gun Proof pressure limits."
        )
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            sigma_p = st.number_input("Burn-Rate Temp Sensitivity σ_p (1/K)", 0.0005, 0.0060, 0.0020, 0.0002, format="%.4f", key="clim_sigmap", help="Typical NC/DB propellants: 0.0016–0.0028 K⁻¹ (0.16–0.28 %/K)")
            mc_trials = st.slider("Monte Carlo Lot Sample Size (N shots)", 40, 300, 120, 20, key="mc_ntrials")
        with tc2:
            srv_lim = st.number_input("Max Permissible Service Pressure MPOP (MPa)", 100.0, 900.0, 420.0, 10.0, key="clim_srv")
            prf_lim = st.number_input("Gun Proof / Piezo Design Limit (MPa)", 120.0, 1100.0, 480.0, 10.0, key="clim_prf")
        with tc3:
            tol_m = st.slider("Charge Mass 1σ Tolerance (±%)", 0.1, 2.5, 0.50, 0.1, key="mc_tolm")
            tol_w = st.slider("Web Thickness D₀ 1σ Tolerance (±%)", 0.2, 4.0, 1.20, 0.2, key="mc_tolw")
            tol_a = st.slider("Lot Burn-Rate a 1σ Variation (±%)", 0.2, 5.0, 1.50, 0.2, key="mc_tola")
            tol_n = st.slider("MIL-DTL-244C NC Nitration 1σ (±%N)", 0.01, 0.20, 0.05, 0.01, key="mc_toln")

        b_F = float(st.session_state.get("ib_F", default_F))
        b_gam = float(st.session_state.get("ib_gamma", default_gamma))
        b_chg = float(st.session_state.get("ib_charge", 50.0))
        b_rho = float(st.session_state.get("ib_rhop", 1.60))
        b_eta = float(st.session_state.get("ib_covol", default_eta))
        b_geom = str(st.session_state.get("ib_geom", "7_perf"))
        b_web = float(st.session_state.get("ib_web", 1.20))
        b_a = float(st.session_state.get("ib_a", 1.20))
        b_n = float(st.session_state.get("ib_n", 0.85))
        b_vc = float(st.session_state.get("ib_vc", 65.0))
        b_bore = float(st.session_state.get("ib_bore", 30.0))
        b_len = float(st.session_state.get("ib_len", 1.50))
        b_mp = float(st.session_state.get("ib_mproj", 300.0))
        b_ps = float(st.session_state.get("ib_pstart", 30.0))

        if st.button("▶ Run STANAG 2895 Climatic & Monte Carlo Dispersion Suite", type="primary", key="btn_run_clim_mc"):
            with st.spinner(f"Simulating 5 STANAG 2895 climatic zones and {mc_trials} stochastic Monte Carlo firing trials..."):
                st.session_state["climatic_out"] = simulate_climatic_zones(
                    force_constant_j_per_g=b_F, gamma=b_gam, charge_mass_g=b_chg,
                    propellant_density_g_per_cm3=b_rho, covolume_cm3_per_g=b_eta,
                    burn_rate_a=b_a, burn_rate_n=b_n, web_thickness_mm=b_web,
                    grain_geometry=b_geom, chamber_volume_cm3=b_vc,
                    bore_diameter_mm=b_bore, barrel_length_m=b_len,
                    projectile_mass_g=b_mp, start_pressure_mpa=b_ps,
                    sigma_p_per_k=sigma_p, service_limit_mpa=srv_lim, proof_limit_mpa=prf_lim,
                )
                st.session_state["mc_out"] = run_monte_carlo_ballistics(
                    force_constant_j_per_g=b_F, gamma=b_gam, charge_mass_g=b_chg,
                    propellant_density_g_per_cm3=b_rho, covolume_cm3_per_g=b_eta,
                    burn_rate_a=b_a, burn_rate_n=b_n, web_thickness_mm=b_web,
                    grain_geometry=b_geom, chamber_volume_cm3=b_vc,
                    bore_diameter_mm=b_bore, barrel_length_m=b_len,
                    projectile_mass_g=b_mp, start_pressure_mpa=b_ps,
                    n_trials=mc_trials, charge_mass_sigma_pct=tol_m,
                    web_sigma_pct=tol_w, burn_rate_sigma_pct=tol_a,
                    nc_nitration_sigma_pct_n=tol_n,
                    service_limit_mpa=srv_lim, proof_limit_mpa=prf_lim,
                )

        if "climatic_out" in st.session_state:
            clim = st.session_state["climatic_out"]
            if clim["passes_stanag_2895"]:
                st.success(f"✅ **STANAG 2895 Qualification PASS:** Extreme Hot Desert (+63 °C) P_max = **{clim['hot_pmax_mpa']:.1f} MPa** (Margin to Proof: **{clim['hot_safety_margin_to_proof_mpa']:+.1f} MPa**)")
            else:
                st.error(f"🚨 **STANAG 2895 Qualification FAIL:** Extreme Hot Desert (+63 °C) P_max = **{clim['hot_pmax_mpa']:.1f} MPa** exceeds Gun Proof Limit ({clim['proof_limit_mpa']:.0f} MPa)!")

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Arctic C2 (-40 °C) P_max", f"{clim['cold_pmax_mpa']:.1f} MPa")
            cm2.metric("Ambient (+21 °C) P_max", f"{clim['ambient_pmax_mpa']:.1f} MPa")
            cm3.metric("Hot Desert A1 (+63 °C) P_max", f"{clim['hot_pmax_mpa']:.1f} MPa")
            cm4.metric("Thermal Pressure Gradient", f"{clim['pressure_temp_gradient_mpa_per_k']:+.2f} MPa/K", delta=f"{clim['velocity_temp_gradient_m_s_per_k']:+.2f} m/s/K", delta_color="off")

            st.dataframe(clim["zones_df"].drop(columns=["Code"]), use_container_width=True, hide_index=True)

            fig_cl = go.Figure()
            zone_colors = {"C2": "#38bdf8", "C0": "#818cf8", "AMB": "#00ff88", "A2": "#ffaa00", "A1": "#ff5566"}
            for code, cv in clim["curves"].items():
                fig_cl.add_trace(go.Scatter(x=cv["time_ms"], y=cv["pressure_mpa"], name=cv["label"], line=dict(color=zone_colors.get(code, "#00f0ff"), width=2.5)))
            fig_cl.add_hline(y=clim["service_limit_mpa"], line_dash="dash", line_color="#ffaa00", annotation_text=f"Service MPOP ({clim['service_limit_mpa']:.0f} MPa)")
            fig_cl.add_hline(y=clim["proof_limit_mpa"], line_dash="dot", line_color="#ff4444", annotation_text=f"Proof Limit ({clim['proof_limit_mpa']:.0f} MPa)")
            fig_cl.update_layout(title="NATO STANAG 2895 Breech Pressure P(t) Across Climatic Conditioning Zones", xaxis_title="Time (ms)", yaxis_title="Breech Pressure (MPa)", height=390)
            st.plotly_chart(fig_cl, use_container_width=True)

        if "mc_out" in st.session_state:
            mc = st.session_state["mc_out"]
            st.divider()
            st.markdown(f"### 🎲 Monte Carlo Lot-to-Lot Ballistic Dispersion ({mc['n_trials']} Simulated Rounds)")
            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Mean P_max ± 1σ", f"{mc['mean_p_max_mpa']:.1f} ± {mc['std_p_max_mpa']:.1f} MPa", delta=f"P_max + 3σ = {mc['p_max_plus_3sigma_mpa']:.1f} MPa", delta_color="off")
            mm2.metric("Muzzle Velocity SD (σ_V₀)", f"{mc['std_v_muz_m_s']:.2f} m/s", delta=f"Extreme Spread = {mc['extreme_spread_v_m_s']:.1f} m/s", delta_color="off")
            mm3.metric("Proof Safety Capability Cpk", f"{mc['cpk_proof']:.2f}", delta="Safe ≥ 1.33" if mc["cpk_proof"] >= 1.33 else "Marginal < 1.33", delta_color="normal" if mc["cpk_proof"] >= 1.33 else "inverse")
            mm4.metric("Gaussian Proof Exceedance", f"{mc['gaussian_exceed_proof_pct']:.4f}%", delta=f"Service > MPOP: {mc['empirical_exceed_service_pct']:.1f}%", delta_color="off")

            mg1, mg2 = st.columns(2)
            tdf_mc = mc["trials_df"]
            with mg1:
                fig_hp = px.histogram(tdf_mc, x="p_max_mpa", nbins=22, title="Monte Carlo Peak Breech Pressure Distribution", labels={"p_max_mpa": "Peak Pressure P_max (MPa)"})
                fig_hp.add_vline(x=mc["p_max_plus_3sigma_mpa"], line_dash="dash", line_color="#00ff88", annotation_text="+3σ")
                fig_hp.add_vline(x=mc["service_limit_mpa"], line_dash="dash", line_color="#ffaa00", annotation_text="MPOP")
                fig_hp.add_vline(x=mc["proof_limit_mpa"], line_dash="dot", line_color="#ff4444", annotation_text="Proof")
                fig_hp.update_layout(height=370)
                st.plotly_chart(fig_hp, use_container_width=True)
            with mg2:
                sens_items = sorted(mc["sensitivity_correlations"].items(), key=lambda kv: abs(kv[1]))
                fig_tor = go.Figure(go.Bar(
                    x=[v for _, v in sens_items],
                    y=[k for k, _ in sens_items],
                    orientation="h",
                    marker_color=["#ff5566" if v < 0 else "#00f0ff" for _, v in sens_items],
                    text=[f"r = {v:+.2f}" for _, v in sens_items],
                    textposition="auto",
                ))
                fig_tor.update_layout(
                    title="Tornado Sensitivity Drivers for Peak Breech Pressure (Correlation r)",
                    xaxis=dict(title="Linear Correlation with P_max", range=[-1.05, 1.05]),
                    height=370,
                )
                st.plotly_chart(fig_tor, use_container_width=True)

    # --- Sub-tab 3: Closed-Vessel & Vivacity ---
    with sub_cv:
        render_closed_vessel_tab(formulation)

    # --- Sub-tab 4: Kamlet-Jacobs VoD Screening ---
    with sub_vod:
        st.subheader("💥 Kamlet-Jacobs Detonation Velocity (VoD) & CJ State Screening")
        vod_source = st.radio(
            "Input source",
            ["🧪 Active Formulation", "📋 Preset / Custom Explosive"],
            horizontal=True, key="vod_source", label_visibility="collapsed",
        )
        if vod_source.startswith("🧪"):
            if not formulation.components:
                st.warning("Add ingredients in the Formulation Editor first.")
            else:
                fkj, fkj_err = _formulation_kj_inputs(formulation)
                if fkj_err:
                    st.error(fkj_err)
                else:
                    ic1, ic2, ic3, ic4, ic5 = st.columns(5)
                    ic1.metric("C (mol/g)", f"{fkj['a']:.5f}")
                    ic2.metric("H (mol/g)", f"{fkj['b']:.5f}")
                    ic3.metric("N (mol/g)", f"{fkj['c']:.5f}")
                    ic4.metric("O (mol/g)", f"{fkj['d']:.5f}")
                    ic5.metric("Hf,mix (kJ/g)", f"{fkj['hf_kJ']:.3f}")

                    dc1, dc2, _ = st.columns([1, 1, 2])
                    with dc1:
                        fvod_rho = st.number_input("Loading Density ρ₀ (g/cm³)", 0.1, 2.5, 1.55, 0.05, format="%.3f", key="fvod_rho")
                    with dc2:
                        fvod_tmd = st.number_input("Plot up to TMD (g/cm³)", 0.5, 2.5, 1.85, 0.05, format="%.3f", key="fvod_tmd")
                    if st.button("▶ Estimate VoD from Active Formulation", type="primary", key="fvod_run"):
                        res_v = _kamlet_jacobs(fkj["a"], fkj["b"], fkj["c"], fkj["d"], fkj["mw"], fkj["hf_kJ"], fvod_rho)
                        st.session_state["vod_result"] = res_v
                        st.session_state["vod_inputs"] = dict(
                            formula=f"C{fkj['a']:.4f}H{fkj['b']:.4f}N{fkj['c']:.4f}O{fkj['d']:.4f}",
                            a=fkj["a"], b=fkj["b"], c=fkj["c"], d=fkj["d"], mw=1.0, hf_kJ=fkj["hf_kJ"], rho=fvod_rho, rho_tmd=fvod_tmd,
                        )
        else:
            preset_name = st.selectbox("Preset Explosive", list(_VOD_PRESETS.keys()), key="vod_preset")
            preset = _VOD_PRESETS[preset_name]
            vc1, vc2, vc3, vc4 = st.columns([1.2, 1, 1, 1])
            with vc1:
                vod_formula = st.text_input("Molecular Formula (CHNO)", value=preset["formula"] if preset else "C7H5N3O6", key="vod_formula")
            with vc2:
                vod_hf = st.number_input("Hf,298 (kJ/mol)", value=float(preset["hf_kJ"] if preset else -66.9), step=10.0, key="vod_hf")
            with vc3:
                vod_tmd = st.number_input("TMD (g/cm³)", 0.5, 2.5, float(preset["rho_tmd"] if preset else 1.654), 0.01, format="%.3f", key="vod_tmd")
            with vc4:
                vod_rho = st.number_input("Loading Density ρ₀ (g/cm³)", 0.1, float(vod_tmd), round(float(vod_tmd) * 0.95, 3), 0.01, format="%.3f", key="vod_rho")
            if st.button("▶ Estimate VoD", type="primary", key="vod_run"):
                a, b, c, d = _parse_chno(vod_formula)
                mw = a * 12.011 + b * 1.008 + c * 14.007 + d * 15.999
                if mw < 1.0:
                    st.error("Could not parse CHNO formula.")
                else:
                    st.session_state["vod_result"] = _kamlet_jacobs(a, b, c, d, mw, vod_hf, vod_rho)
                    st.session_state["vod_inputs"] = dict(formula=vod_formula, a=a, b=b, c=c, d=d, mw=mw, hf_kJ=vod_hf, rho=vod_rho, rho_tmd=vod_tmd)

        if "vod_result" in st.session_state and "vod_inputs" in st.session_state:
            rv = st.session_state["vod_result"]
            inp = st.session_state["vod_inputs"]
            if not rv["valid"]:
                st.error(f"Heat of detonation Q = {rv['Q_cal_per_g']:.1f} cal/g ≤ 0 — composition does not support KJ detonation.")
            else:
                km1, km2, km3, km4 = st.columns(4)
                km1.metric("Detonation Velocity VoD", f"{rv['D_km_s']:.3f} km/s")
                km2.metric("CJ Detonation Pressure P_CJ", f"{rv['P_GPa']:.2f} GPa")
                km3.metric("Heat of Detonation Q_det", f"{rv['Q_kJ_per_g']:.2f} kJ/g")
                km4.metric("Kamlet-Jacobs φ", f"{rv['phi']:.4f}")

                rho_arr = np.linspace(0.5, max(inp["rho_tmd"], inp["rho"] + 0.01), 80)
                d_arr = [_kamlet_jacobs(inp["a"], inp["b"], inp["c"], inp["d"], inp["mw"], inp["hf_kJ"], float(r))["D_km_s"] for r in rho_arr]
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=rho_arr, y=d_arr, mode="lines", name="D(ρ) — Kamlet-Jacobs", line=dict(color="#ff5566", width=2.5)))
                fig_dd.add_trace(go.Scatter(x=[inp["rho"]], y=[rv["D_km_s"]], mode="markers", marker=dict(size=12, color="#ffffff", line=dict(color="#ff5566", width=2.5)), name=f"ρ₀ = {inp['rho']:.3f} g/cm³"))
                fig_dd.update_layout(title="Kamlet-Jacobs Linear VoD vs Loading Density", xaxis_title="Loading Density ρ₀ (g/cm³)", yaxis_title="VoD (km/s)", height=350)
                st.plotly_chart(fig_dd, use_container_width=True)


# ---------------------------------------------------------------------------
# Workspace 3: Signature, Plume & Occupational Hygiene
# ---------------------------------------------------------------------------

def _render_workspace_signature_hygiene(formulation: Formulation) -> None:
    sub_flash, sub_tox = st.tabs([
        "✨ Muzzle Flash & Alkali Suppression",
        "🌫️ Plume Smoke (STANAG 4400) & Range Toxicity",
    ])
    conditions: CombustionConditions = st.session_state["active_conditions"]
    pah_config: PAHAllocationConfig = st.session_state["active_pah_config"]
    eq_temp: float = st.session_state["active_eq_temp"]

    with sub_flash:
        st.subheader("✨ Muzzle Flash, Mach Disk Recovery & Alkali Salt Suppression")
        if not formulation.components:
            st.info("Define a formulation first.")
        else:
            mf_c1, mf_c2, mf_c3 = st.columns(3)
            with mf_c1:
                mf_exp_ratio = st.slider("Barrel Volumetric Expansion Ratio (V_barrel / Vc)", 2.0, 35.0, 12.0, 0.5, key="mf_expr")
            with mf_c2:
                mf_mach = st.slider("Muzzle Exhaust Jet Mach Number", 1.2, 4.0, 2.4, 0.1, key="mf_mach")
            with mf_c3:
                mf_supp = st.slider("Alkali Flash Suppressant Salt (K₂SO₄ / KNO₃ wt%)", 0.0, 5.0, 0.5, 0.1, key="mf_supp")

            if st.button("▶ Run Muzzle Flash & Plume Afterburning Analysis", type="primary", key="btn_run_mf"):
                res_for_mf = st.session_state.results.get(formulation.name) or st.session_state.get("result")
                if res_for_mf is None:
                    res_for_mf = cached_solve_equilibrium_adiabatic(
                        formulation, conditions, pah_config, st.session_state.species_db, 2500.0
                    )
                    st.session_state["result"] = res_for_mf
                    st.session_state.results[formulation.name] = res_for_mf
                st.session_state["mf_out"] = analyze_muzzle_flash(
                    result=res_for_mf,
                    formulation=formulation,
                    expansion_ratio=mf_exp_ratio,
                    muzzle_mach=mf_mach,
                    suppressant_wt_pct=mf_supp,
                )

            if "mf_out" in st.session_state:
                mf_out = st.session_state["mf_out"]
                status_str = mf_out["flash_status"]
                if status_str.startswith("HIGH"):
                    st.error(f"**Flash Prediction:** {status_str}")
                elif status_str.startswith("MODERATE"):
                    st.warning(f"**Flash Prediction:** {status_str}")
                else:
                    st.success(f"**Flash Prediction:** {status_str}")

                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Muzzle Exit Temp", f"{mf_out['muzzle_exit_temp_k']:.0f} K")
                f2.metric("Mach Disk Recovery Temp", f"{mf_out['mach_disk_temp_k']:.0f} K")
                f3.metric("Radical Ignition Threshold", f"{mf_out['ignition_threshold_k']:.0f} K")
                f4.metric("Peak Plume Temp", f"{mf_out['peak_plume_temp_k']:.0f} K")

                f5, f6, f7 = st.columns(3)
                f5.metric("Combustible Exhaust Gas (CO+H₂+CH₄)", f"{mf_out['combustible_mole_pct']:.1f} mol%")
                f6.metric("Afterburn Heat Potential", f"{mf_out['afterburn_energy_j_per_g']:.0f} J/g")
                f7.metric("Soot Incandescence Index", f"{mf_out['incandescence_index']:.1f} / 100")

                plume_df = mf_out["plume_df"]
                fig_mf = go.Figure()
                fig_mf.add_trace(go.Scatter(x=plume_df["air_to_exhaust_ratio"], y=plume_df["potential_afterburn_temp_k"], name="Potential Fully-Ignited Plume Temp (K)", line=dict(color="#ffaa00", dash="dot", width=2)))
                fig_mf.add_trace(go.Scatter(x=plume_df["air_to_exhaust_ratio"], y=plume_df["actual_plume_temp_k"], name="Predicted Plume Temperature (K)", line=dict(color="#ff5566" if mf_out["ignited"] else "#00f0ff", width=3)))
                fig_mf.add_trace(go.Scatter(x=plume_df["air_to_exhaust_ratio"], y=plume_df["unreacted_mix_temp_k"], name="Unreacted Mixing Temp (K)", line=dict(color="#94a3b8", dash="dash", width=2)))
                fig_mf.add_hline(y=mf_out["ignition_threshold_k"], line_dash="dashdot", line_color="#00ff88", annotation_text=f"Ignition Threshold ({mf_out['ignition_threshold_k']:.0f} K)")
                fig_mf.update_layout(title="Muzzle Plume Temperature Envelope vs Entrained Air-to-Exhaust Ratio", xaxis_title="Entrained Air / Exhaust Mass Ratio α", yaxis_title="Temperature (K)", height=410)
                st.plotly_chart(fig_mf, use_container_width=True)

    with sub_tox:
        render_smoke_toxicity_tab(formulation, conditions, pah_config, eq_temp)


# ---------------------------------------------------------------------------
# Workspace 4: Design, Sensitivity & Surveillance
# ---------------------------------------------------------------------------

def _render_workspace_design_surveillance(formulation: Formulation) -> None:
    sub_opt, sub_sens, sub_stab = st.tabs([
        "🧬 SLSQP Optimizer & Ternary Maps",
        "📈 1D & 2D Sensitivity Sweeps",
        "⏳ STANAG 4582 Shelf-Life & Cook-Off",
    ])
    conditions: CombustionConditions = st.session_state["active_conditions"]
    pah_config: PAHAllocationConfig = st.session_state["active_pah_config"]
    eq_temp: float = st.session_state["active_eq_temp"]

    # --- Sub-tab 1: Optimizer & Ternary ---
    with sub_opt:
        st.subheader("🧬 Multi-Objective Constrained Formulation Optimizer (SLSQP Inverse Design)")
        if len(formulation.components) < 2:
            st.info("Add at least 2 components to the active formulation to run the optimizer.")
        else:
            opt_c1, opt_c2, opt_c3 = st.columns(3)
            with opt_c1:
                opt_goal_map = {
                    "force_constant": "Maximize Gun Impetus / Force F (J/g)",
                    "specific_impulse": "Maximize Specific Impulse Isp (s)",
                    "min_flame_temp": "Minimize Flame Temperature Tp (K)",
                    "min_soot": "Minimize Soot / PAH Carbon Yield",
                }
                opt_goal = st.selectbox("Optimization Objective", list(opt_goal_map.keys()), format_func=lambda k: opt_goal_map[k], key="opt_goal")
            with opt_c2:
                opt_max_t = st.slider("Max Allowed Flame Temp Tp (K)", 1800, 4200, 3000, 50, key="opt_max_t")
            with opt_c3:
                opt_max_pah_pct = st.slider("Max Allowed PAH/Soot Carbon (%)", 0.5, 25.0, 5.0, 0.5, key="opt_max_pah")

            if st.button("▶ Run Constrained Formulation Optimizer", type="primary", key="btn_run_opt"):
                with st.spinner("Optimizing ingredient mass fractions via SLSQP..."):
                    st.session_state["opt_res"] = optimize_formulation(
                        base_formulation=formulation,
                        conditions=conditions,
                        pah_config=pah_config,
                        species_db=st.session_state.species_db,
                        objective_metric=opt_goal,
                        max_temp_k=float(opt_max_t),
                        max_pah_frac=float(opt_max_pah_pct) / 100.0,
                        equilibrium_temp_k=eq_temp,
                    )

            if "opt_res" in st.session_state:
                opt_res = st.session_state["opt_res"]
                b_res, o_res = opt_res["baseline_result"], opt_res["optimized_result"]
                oc1, oc2, oc3, oc4 = st.columns(4)
                oc1.metric("Impetus F", f"{o_res.force_constant_j_per_g:.1f} J/g", delta=f"{o_res.force_constant_j_per_g - b_res.force_constant_j_per_g:+.1f} J/g")
                oc2.metric("Isobaric Flame Temp", f"{o_res.adiabatic_flame_temp_k:.0f} K", delta=f"{o_res.adiabatic_flame_temp_k - b_res.adiabatic_flame_temp_k:+.0f} K", delta_color="inverse")
                oc3.metric("Specific Impulse Isp", f"{o_res.specific_impulse_s:.1f} s", delta=f"{o_res.specific_impulse_s - b_res.specific_impulse_s:+.1f} s")
                b_pah = b_res.carbon_distribution.get("PAHs", 0.0) * 100.0
                o_pah = o_res.carbon_distribution.get("PAHs", 0.0) * 100.0
                oc4.metric("PAH / Soot Carbon", f"{o_pah:.2f}%", delta=f"{o_pah - b_pah:+.2f}%", delta_color="inverse")
                st.dataframe(opt_res["comparison_df"], use_container_width=True, hide_index=True)

                if st.button("✨ Apply Optimized Mass Fractions to Active Formulation", key="btn_apply_opt"):
                    st.session_state.formulations[st.session_state.selected_idx] = opt_res["optimized_formulation"]
                    st.session_state["result"] = o_res
                    st.session_state.results[opt_res["optimized_formulation"].name] = o_res
                    st.rerun()

        st.divider()
        st.subheader("🔺 3-Component Ternary Simplex Phase Diagram")
        if len(formulation.components) < 3:
            st.info("Add at least 3 components to the active formulation to generate a 3-component Ternary Diagram.")
        else:
            comp_names = [f"{i + 1}. {c.name} ({c.formula})" for i, c in enumerate(formulation.components)]
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                idx_a = st.selectbox("Apex A", range(len(comp_names)), index=0, format_func=lambda i: comp_names[i], key="tern_a")
            with tc2:
                idx_b = st.selectbox("Apex B", range(len(comp_names)), index=min(1, len(comp_names) - 1), format_func=lambda i: comp_names[i], key="tern_b")
            with tc3:
                idx_c = st.selectbox("Apex C", range(len(comp_names)), index=min(2, len(comp_names) - 1), format_func=lambda i: comp_names[i], key="tern_c")
            with tc4:
                tern_metric_map = {
                    "T_ad_K": "Isobaric Flame Temp (K)",
                    "force_J_per_g": "Impetus / Force F (J/g)",
                    "Isp_s": "Specific Impulse Isp (s)",
                    "PAH_carbon_frac": "PAH / Soot Carbon Fraction",
                    "OB_pct": "Oxygen Balance (%)",
                }
                tern_metric = st.selectbox("Ternary Color Metric", list(tern_metric_map.keys()), format_func=lambda k: tern_metric_map[k], key="tern_met")

            if st.button("▶ Compute Ternary Simplex Map", type="primary", key="btn_run_ternary"):
                if len({idx_a, idx_b, idx_c}) < 3:
                    st.error("Select 3 distinct components for Apex A, B, and C.")
                else:
                    with st.spinner("Solving equilibrium across 3-component simplex grid..."):
                        st.session_state["ternary_df"] = compute_ternary_grid(
                            formulation.components[idx_a], formulation.components[idx_b], formulation.components[idx_c],
                            conditions, pah_config, st.session_state.species_db, n_steps=8, equilibrium_temp_k=eq_temp,
                        )
                        st.session_state["ternary_labels"] = (
                            formulation.components[idx_a].name, formulation.components[idx_b].name, formulation.components[idx_c].name
                        )

            if "ternary_df" in st.session_state:
                t_df = st.session_state["ternary_df"]
                lbl_a, lbl_b, lbl_c = st.session_state.get("ternary_labels", ("A", "B", "C"))
                met_label = tern_metric_map[tern_metric]
                fig_tern = go.Figure(go.Scatterternary({
                    "mode": "markers",
                    "a": t_df["w_A"] * 100.0, "b": t_df["w_B"] * 100.0, "c": t_df["w_C"] * 100.0,
                    "marker": {
                        "symbol": "circle", "size": 16, "color": t_df[tern_metric],
                        "colorscale": "Turbo", "showscale": True, "colorbar": {"title": met_label},
                        "line": {"width": 1, "color": "#141724"},
                    },
                    "text": [
                        f"{lbl_a}: {r.w_A * 100:.0f}%<br>{lbl_b}: {r.w_B * 100:.0f}%<br>{lbl_c}: {r.w_C * 100:.0f}%<br>{met_label}: {getattr(r, tern_metric):.2f}"
                        for r in t_df.itertuples()
                    ],
                    "hoverinfo": "text",
                }))
                fig_tern.update_layout(
                    title=f"Ternary Composition Map — {met_label}",
                    ternary={"sum": 100, "aaxis": {"title": lbl_a, "min": 0}, "baxis": {"title": lbl_b, "min": 0}, "caxis": {"title": lbl_c, "min": 0}},
                    height=530,
                )
                st.plotly_chart(fig_tern, use_container_width=True)

    # --- Sub-tab 2: 1D & 2D Parametric Sensitivity ---
    with sub_sens:
        st.subheader("📈 1D & 2D Parametric Sensitivity Sweeps")
        if not formulation.components:
            st.info("Define a formulation first.")
        else:
            analysis_type = st.selectbox(
                "Sweep Type",
                ["Equivalence ratio sweep", "Temperature sweep", "PAH model comparison", "Multi-formulation comparison", "2D Sensitivity sweep"],
                key="sens_type_sel",
            )
            if analysis_type == "2D Sensitivity sweep":
                var_opts = {"Equivalence Ratio (φ)": "equivalence_ratio", "Temperature (K)": "temperature", "Pressure (atm)": "pressure"}
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    v1_label = st.selectbox("Variable 1 (X-Axis)", list(var_opts.keys()), index=0, key="sa2d_v1")
                    v1_name = var_opts[v1_label]
                    if v1_name == "equivalence_ratio":
                        v1_min, v1_max = st.slider("X Range (φ)", 0.5, 3.0, (0.5, 2.5), step=0.1, key="sa2d_v1_rng")
                    elif v1_name == "temperature":
                        v1_min, v1_max = st.slider("X Range (Temperature K)", 1000, 4500, (1500, 3500), step=100, key="sa2d_v1_rng")
                    else:
                        v1_min, v1_max = st.slider("X Range (Pressure atm)", 1, 500, (10, 150), step=10, key="sa2d_v1_rng")
                with col_v2:
                    rem_opts = {k: v for k, v in var_opts.items() if k != v1_label}
                    v2_label = st.selectbox("Variable 2 (Y-Axis)", list(rem_opts.keys()), index=0, key="sa2d_v2")
                    v2_name = rem_opts[v2_label]
                    if v2_name == "equivalence_ratio":
                        v2_min, v2_max = st.slider("Y Range (φ)", 0.5, 3.0, (0.8, 2.8), step=0.1, key="sa2d_v2_rng")
                    elif v2_name == "temperature":
                        v2_min, v2_max = st.slider("Y Range (Temperature K)", 1000, 4500, (2000, 4000), step=100, key="sa2d_v2_rng")
                    else:
                        v2_min, v2_max = st.slider("Y Range (Pressure atm)", 1, 500, (20, 200), step=10, key="sa2d_v2_rng")

                metric_opts = {
                    "Isobaric Flame Temperature (K)": "adiabatic_flame_temp_k",
                    "Gun Impetus / Force F (J/g)": "force_constant_j_per_g",
                    "Specific Impulse (s)": "specific_impulse_s",
                    "PAH Carbon Fraction": "PAH_carbon_frac",
                    "CO Mole Fraction": "CO_mole_frac",
                    "CO₂ Mole Fraction": "CO2_mole_frac",
                    "γ (Cp/Cv)": "gamma",
                }
                selected_metric_label = st.selectbox("Metric to Plot", list(metric_opts.keys()), index=0, key="sa2d_metric")
                selected_metric = metric_opts[selected_metric_label]

            if st.button("▶ Run Parametric Sweep", type="primary", key="btn_run_sens"):
                with st.spinner("Running parametric equilibrium sweep..."):
                    if analysis_type == "Equivalence ratio sweep":
                        st.session_state["sensitivity_df"] = sweep_equivalence_ratio(formulation, conditions, pah_config, st.session_state.species_db, eq_temp, np.linspace(0.5, 3.0, 25))
                        st.session_state["sensitivity_type"] = "phi"
                    elif analysis_type == "Temperature sweep":
                        st.session_state["sensitivity_df"] = sweep_temperature(formulation, conditions, pah_config, st.session_state.species_db, np.linspace(1000, 4500, 25))
                        st.session_state["sensitivity_type"] = "temp"
                    elif analysis_type == "PAH model comparison":
                        st.session_state["sensitivity_df"] = sweep_pah_models(formulation, conditions, st.session_state.species_db, eq_temp)
                        st.session_state["sensitivity_type"] = "pah"
                    elif analysis_type == "Multi-formulation comparison":
                        st.session_state["sensitivity_df"] = compare_formulations(st.session_state.formulations, conditions, pah_config, st.session_state.species_db, eq_temp)
                        st.session_state["sensitivity_type"] = "multi"
                    else:
                        st.session_state["sensitivity_df"] = sweep_2d(
                            formulation, conditions, pah_config, st.session_state.species_db,
                            v1_name, np.linspace(v1_min, v1_max, 10), v2_name, np.linspace(v2_min, v2_max, 10),
                        )
                        st.session_state["sensitivity_type"] = "sweep2d"
                        st.session_state["sa2d_v1_name"] = v1_name
                        st.session_state["sa2d_v2_name"] = v2_name
                        st.session_state["sa2d_metric"] = selected_metric
                        st.session_state["sa2d_metric_label"] = selected_metric_label

            if "sensitivity_df" in st.session_state:
                df = st.session_state["sensitivity_df"]
                stype = st.session_state["sensitivity_type"]
                if stype == "phi":
                    g1, g2 = st.columns(2)
                    with g1:
                        fig = go.Figure()
                        for col_s, lbl_s in [("CO2_mole_frac", "CO₂"), ("CO_mole_frac", "CO"), ("H2O_mole_frac", "H₂O"), ("H2_mole_frac", "H₂")]:
                            fig.add_trace(go.Scatter(x=df["phi_target"], y=df[col_s], name=lbl_s, mode="lines+markers"))
                        fig.update_layout(title="Major Products vs Equivalence Ratio φ", xaxis_title="φ", yaxis_title="Mole Fraction", height=380)
                        st.plotly_chart(fig, use_container_width=True)
                    with g2:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=df["phi_target"], y=df["PAH_carbon_frac"], name="PAH C fraction", fill="tozeroy"))
                        fig2.add_trace(go.Scatter(x=df["phi_target"], y=df["CO_carbon_frac"], name="CO C fraction"))
                        fig2.add_trace(go.Scatter(x=df["phi_target"], y=df["CO2_carbon_frac"], name="CO₂ C fraction"))
                        fig2.update_layout(title="Carbon Partitioning vs φ", xaxis_title="φ", yaxis_title="Carbon Share", height=380)
                        st.plotly_chart(fig2, use_container_width=True)
                elif stype == "temp":
                    g1, g2 = st.columns(2)
                    with g1:
                        fig = go.Figure()
                        for col_s, lbl_s in [("CO2_mole_frac", "CO₂"), ("CO_mole_frac", "CO"), ("H2O_mole_frac", "H₂O"), ("H2_mole_frac", "H₂")]:
                            fig.add_trace(go.Scatter(x=df["sweep_temperature_k"], y=df[col_s], name=lbl_s, mode="lines+markers"))
                        fig.update_layout(title="Equilibrium Products vs Temperature", xaxis_title="T (K)", yaxis_title="Mole Fraction", height=380)
                        st.plotly_chart(fig, use_container_width=True)
                    with g2:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=df["sweep_temperature_k"], y=df["PAH_carbon_frac"], name="PAH C fraction", fill="tozeroy"))
                        fig2.update_layout(title="PAH Carbon Share vs Temperature", xaxis_title="T (K)", yaxis_title="PAH C Share", height=380)
                        st.plotly_chart(fig2, use_container_width=True)
                elif stype == "pah":
                    fig = px.bar(df, x="pah_model", y=["CO2_carbon_frac", "CO_carbon_frac", "PAH_carbon_frac"], barmode="group", title="Carbon Distribution by PAH Model")
                    st.plotly_chart(fig, use_container_width=True)
                elif stype == "multi":
                    fig = px.bar(df, x="formulation", y=["CO2_mole_frac", "CO_mole_frac", "H2O_mole_frac", "H2_mole_frac", "N2_mole_frac"], barmode="group", title="Multi-Formulation Exhaust Comparison")
                    fig.update_layout(xaxis_tickangle=-35, height=430)
                    st.plotly_chart(fig, use_container_width=True)
                elif stype == "sweep2d":
                    v1_n = st.session_state.get("sa2d_v1_name", "param1")
                    v2_n = st.session_state.get("sa2d_v2_name", "param2")
                    met = st.session_state.get("sa2d_metric", "adiabatic_flame_temp_k")
                    met_lbl = st.session_state.get("sa2d_metric_label", "Value")
                    x_vals = sorted(df[f"param_{v1_n}"].unique())
                    y_vals = sorted(df[f"param_{v2_n}"].unique())
                    z_grid = np.zeros((len(y_vals), len(x_vals)))
                    for iy, yv in enumerate(y_vals):
                        for ix, xv in enumerate(x_vals):
                            match = df[(df[f"param_{v1_n}"] == xv) & (df[f"param_{v2_n}"] == yv)]
                            if not match.empty:
                                z_grid[iy, ix] = match.iloc[0][met]
                    g1, g2 = st.columns(2)
                    with g1:
                        fig = go.Figure(data=[go.Surface(z=z_grid, x=x_vals, y=y_vals, colorscale="Viridis")])
                        fig.update_layout(title=f"3D Surface: {met_lbl}", scene=dict(xaxis_title=v1_n, yaxis_title=v2_n, zaxis_title=met_lbl), height=460)
                        st.plotly_chart(fig, use_container_width=True)
                    with g2:
                        fig2 = go.Figure(data=go.Contour(z=z_grid, x=x_vals, y=y_vals, colorscale="Viridis", contours=dict(showlabels=True)))
                        fig2.update_layout(title=f"2D Contour Map: {met_lbl}", xaxis_title=v1_n, yaxis_title=v2_n, height=460)
                        st.plotly_chart(fig2, use_container_width=True)

                with st.expander("Raw Sweep Dataset & CSV Export"):
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button("Download Sweep CSV", df.to_csv(index=False), file_name="combusto_sensitivity.csv", mime="text/csv")

    # --- Sub-tab 3: STANAG 4582 Shelf-Life & Cook-Off ---
    with sub_stab:
        render_stability_tab(formulation)


# ---------------------------------------------------------------------------
# Main Workbench Entry Point
# ---------------------------------------------------------------------------

WORKSPACES = [
    "🪪 Passport & Thermochemistry",
    "🎯 Ballistics & Chamber Tolerance",
    "🔥 Signature, Plume & Hygiene",
    "🛡️ Design, Sensitivity & Surveillance",
    "📚 Benchmarks & Provenance",
]


def render_main_workbench(
    extra_editor_hook: Optional[Callable[[Formulation], None]] = None,
) -> None:
    """Render the shared Combusto 5-Workspace engineering workbench."""
    _ensure_session_state()
    formulation = _render_formulation_manager(extra_editor_hook=extra_editor_hook)

    st.markdown("#### Engineering Workspaces")
    selected_ws = st.radio(
        "Engineering Workspace",
        WORKSPACES,
        horizontal=True,
        label_visibility="collapsed",
        key="active_workspace_nav",
    )

    if selected_ws == WORKSPACES[0]:
        _render_workspace_passport_thermo(formulation)
    elif selected_ws == WORKSPACES[1]:
        _render_workspace_ballistics_tolerance(formulation)
    elif selected_ws == WORKSPACES[2]:
        _render_workspace_signature_hygiene(formulation)
    elif selected_ws == WORKSPACES[3]:
        _render_workspace_design_surveillance(formulation)
    else:
        render_benchmarks_and_grounding_tab()
