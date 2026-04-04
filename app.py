"""Combusto â€” Gun Propellant Combustion Product Predictor.

Run with: streamlit run app.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from combusto.models import (
    CombustionConditions,
    Formulation,
    PAHAllocationConfig,
    PropellantComponent,
)
from combusto.chemistry import equivalence_ratio, oxygen_balance
from combusto.equilibrium import solve_equilibrium, solve_equilibrium_adiabatic
from combusto.export import (
    result_to_dataframe,
    pah_details_to_dataframe,
    carbon_distribution_to_dataframe,
    element_balance_to_dataframe,
    result_to_csv,
    result_to_excel_bytes,
)
from combusto.pah_allocation import PRESETS
from combusto.presets import (
    load_presets,
    save_formulations,
    load_formulations,
    formulation_to_dict,
    formulation_from_dict,
)
from combusto.sensitivity import (
    sweep_equivalence_ratio,
    sweep_temperature,
    sweep_pah_models,
    compare_formulations,
)
from combusto.species_db import load_species_db

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Combusto",
    page_icon=":fire:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, compact, work-friendly layout
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* —— Global typography & base —— */
html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Background gradient to ensure a deep space-lab theme */
.stApp {
    background: radial-gradient(circle at top left, #141724 0%, #0a0b10 100%) !important;
}

/* —— Main content area —— */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* —— App title —— */
h1 {
    font-size: 2.5rem !important;
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
    font-size: 1.1rem !important;
    letter-spacing: 0.05em;
    margin-bottom: 1rem !important;
}

/* —— Glassmorphism Metric cards —— */
div[data-testid="stMetric"] {
    background: rgba(20, 24, 39, 0.6);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 240, 255, 0.1);
    border-radius: 12px;
    padding: 1rem !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    border: 1px solid rgba(0, 240, 255, 0.4);
    box-shadow: 0 10px 20px rgba(0, 240, 255, 0.15);
}

div[data-testid="stMetric"] label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6ee6f5 !important;
    opacity: 0.9;
}

div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    text-shadow: 0 0 10px rgba(255,255,255,0.1);
}

/* —— Primary button (Run) with Neon Gradient —— */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #00f0ff 0%, #0077ff 100%) !important;
    border: none !important;
    color: #000 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.6rem 2rem !important;
    border-radius: 8px !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.4) !important;
    margin: 1rem auto 0 auto !important;
    max-width: 500px !important;
    display: block;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #33f3ff 0%, #3399ff 100%) !important;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.8) !important;
    transform: scale(1.02);
}

/* —— Regular buttons —— */
div[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
    transition: background 0.2s;
}

div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}

/* —— Expanders —— */
div[data-testid="stExpander"] {
    background: rgba(10, 12, 20, 0.4);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
}

div[data-testid="stExpander"] summary {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* —— Input fields —— */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
    color: #fff !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border: 1px solid #00f0ff !important;
    box-shadow: 0 0 8px rgba(0, 240, 255, 0.3) !important;
}

/* Hide Streamlit Footer & Hamburger */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Tab folder-style via component iframe JS ────────────────────────────
# JS Injection for Tab styles (Streamlit component isolation workaround)
import streamlit.components.v1 as components

components.html("""
<script>
(function() {
  const TAB_CSS = `
    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        background: rgba(0,0,0,0.2) !important;
        border-radius: 8px !important;
        padding: 6px !important;
        border-bottom: none !important;
    }
    button[data-baseweb="tab"] {
        padding: 0.6rem 1.5rem !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: #8a94a6 !important;
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        transition: all .2s !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255,255,255,0.05) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(0, 240, 255, 0.9) !important;
        color: #000000 !important;
        border-bottom: none !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4) !important;
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


st.title("Combusto")
st.caption("Gun Propellant Combustion Product Predictor with PAH/Soot Modeling")

# ---------------------------------------------------------------------------
# Password Protection
# ---------------------------------------------------------------------------
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "combusto2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Please enter the password to access this app", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Please enter the password to access this app", type="password", on_change=password_entered, key="password")
        st.error("ðŸ˜• Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "formulations" not in st.session_state:
    from combusto.models import Formulation
    st.session_state.formulations = [Formulation(name="Custom Formulation")]
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "compare_idx" not in st.session_state:
    st.session_state.compare_idx = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "species_db" not in st.session_state:
    st.session_state.species_db = load_species_db()

# ---------------------------------------------------------------------------
# Sidebar â€” formulation management
# ---------------------------------------------------------------------------
import re as _re

_VOD_PRESETS = {
    "Custom": None,
    "TNT  (C\u2087H\u2085N\u2083O\u2086)":          dict(formula="C7H5N3O6",   hf_kJ=-66.9,   rho_tmd=1.654),
    "RDX  (C\u2083H\u2086N\u2086O\u2086)":          dict(formula="C3H6N6O6",   hf_kJ=92.6,    rho_tmd=1.806),
    "HMX  (C\u2084H\u2088N\u2088O\u2088)":          dict(formula="C4H8N8O8",   hf_kJ=74.4,    rho_tmd=1.906),
    "PETN (C\u2085H\u2088N\u2084O\u2081\u2082)":     dict(formula="C5H8N4O12",  hf_kJ=-538.5,  rho_tmd=1.778),
    "TATB (C\u2086H\u2086N\u2086O\u2086)":          dict(formula="C6H6N6O6",   hf_kJ=-154.0,  rho_tmd=1.937),
    "Tetryl (C\u2087H\u2085N\u2085O\u2088)":        dict(formula="C7H5N5O8",   hf_kJ=-38.9,   rho_tmd=1.730),
    "NQ / Nitroguanidine (CH\u2084N\u2084O\u2082)": dict(formula="CH4N4O2",    hf_kJ=-90.4,   rho_tmd=1.715),
    "FOX-7 (C\u2082H\u2084N\u2084O\u2084)":         dict(formula="C2H4N4O4",   hf_kJ=-133.4,  rho_tmd=1.885),
    "DINA  (C\u2084H\u2088N\u2082O\u2086)":          dict(formula="C4H8N2O6",   hf_kJ=-392.0,  rho_tmd=1.490),
}


def _parse_chno(formula: str):
    """Parse a CHNO formula string e.g. 'C7H5N3O6' -> (a, b, c, d) counts."""
    counts = {"C": 0.0, "H": 0.0, "N": 0.0, "O": 0.0}
    for m in _re.finditer(r"([CHNO])(\d+\.?\d*)?", formula.upper()):
        el = m.group(1)
        counts[el] += float(m.group(2)) if m.group(2) else 1.0
    return counts["C"], counts["H"], counts["N"], counts["O"]


# Common atomic weights for MW calculation of propellant components
_AW = {
    "C": 12.011, "H": 1.008,  "N": 14.007, "O": 15.999,
    "S": 32.06,  "K": 39.098, "NA": 22.990, "CU": 63.546,
    "FE": 55.845,"AL": 26.982,"PB": 207.2,  "CR": 51.996,
    "MN": 54.938,"BA": 137.33,"SR": 87.62,  "CA": 40.078,
    "CL": 35.45, "F": 18.998, "B": 10.81,   "SI": 28.085,
    "P": 30.974, "ZN": 65.38, "NI": 58.693, "MG": 24.305,
}


def _formula_mw_general(formula: str) -> float:
    """Molecular weight of any formula string using common atomic weights."""
    mw = 0.0
    for m in _re.finditer(r"([A-Z][a-z]?)(\d+\.?\d*)?", formula):
        el = m.group(1).upper()
        cnt = float(m.group(2)) if m.group(2) else 1.0
        mw += _AW.get(el, 0.0) * cnt
    return mw


def _formulation_kj_inputs(form):
    """Extract (a, b, c, d, mw=1, hf_kJ_per_g) from a Formulation object.

    All values are normalised to 1 gram of propellant mixture, so mw=1.0
    can be passed directly to _kamlet_jacobs().
    """
    try:
        elem = form.elemental_composition()   # mol / g  {element: value}
    except Exception:
        return None, "Could not compute elemental composition."

    a = elem.get("C", 0.0)
    b = elem.get("H", 0.0)
    c = elem.get("N", 0.0)
    d = elem.get("O", 0.0)

    if a + b + c + d < 1e-9:
        return None, "Elemental composition is zero â€” check formulation components."

    # Effective Hf per gram of mixture  (kJ/g)
    hf_per_g = 0.0
    for comp in form.components:
        mw_comp = _formula_mw_general(comp.formula)
        if mw_comp > 0:
            hf_per_g += comp.mass_fraction * comp.hf_298 / mw_comp
        # else: skip components whose MW can't be parsed

    return dict(a=a, b=b, c=c, d=d, mw=1.0, hf_kJ=hf_per_g), None



def _kamlet_jacobs(a, b, c, d, mw, hf_kJ_per_mol, rho):
    """Kamlet-Jacobs VoD/P_CJ estimator for CHNO explosives.

    Parameters
    ----------
    a,b,c,d      : C, H, N, O atomic counts per formula unit
    mw           : molecular weight (g/mol)
    hf_kJ_per_mol: heat of formation (kJ/mol), positive = endothermic
    rho          : loading density (g/cm3)

    Returns dict with D (km/s), P (GPa) and all intermediates.
    Reference: Kamlet & Jacobs, J. Chem. Phys. 48, 23 (1968).
    """
    HF_H2O = -57_800.0   # cal/mol (gas)
    HF_CO2 = -94_050.0   # cal/mol
    HF_CO  = -26_420.0   # cal/mol
    hf_cal  = hf_kJ_per_mol * 1000.0 / 4.184

    # H2O-CO2 arbitrary product hierarchy
    n_N2  = c / 2.0
    n_H2O = b / 2.0
    O_rem = d - n_H2O          # O remaining after forming H2O

    if O_rem >= 2 * a:          # excess O2
        n_CO2, n_CO, n_Cs, n_O2 = a, 0.0, 0.0, (O_rem - 2*a) / 2.0
    elif O_rem >= a:            # mix CO2 + CO
        n_CO2, n_CO, n_Cs, n_O2 = O_rem - a, 2*a - O_rem, 0.0, 0.0
    elif O_rem >= 0:            # CO + C(s)
        n_CO2, n_CO, n_Cs, n_O2 = 0.0, O_rem, a - O_rem, 0.0
    else:                       # severely O-deficient
        n_H2O = max(0.0, d)
        n_CO2, n_CO, n_Cs, n_O2 = 0.0, 0.0, a, 0.0

    gas_mol = n_N2 + n_H2O + n_CO2 + n_CO + n_O2
    N = gas_mol / mw             # mol gas / g explosive

    if gas_mol > 0:
        M_avg = (28*n_N2 + 18*n_H2O + 44*n_CO2 + 28*n_CO + 32*n_O2) / gas_mol
    else:
        M_avg = 28.0

    # Heat of detonation Q (cal/g) â€” positive = exothermic
    Q = (hf_cal - (n_CO2*HF_CO2 + n_H2O*HF_H2O + n_CO*HF_CO)) / mw

    base_info = dict(
        N=N, M_avg=M_avg, Q_cal_per_g=Q,
        n_N2=n_N2, n_H2O=n_H2O, n_CO2=n_CO2,
        n_CO=n_CO, n_Cs=n_Cs, n_O2=n_O2,
        OB_pct=(d - 2*a - b/2) / mw * 1600,
    )

    if Q <= 0:
        return dict(valid=False, **base_info)

    phi = N * M_avg**0.5 * Q**0.5        # Kamlet-Jacobs phi
    D   = 1.01 * phi**0.5 * (1 + 1.30 * rho)   # km/s
    P   = 15.58 * rho**2 * phi                  # GPa

    # CJ derived quantities (polytropic approx, gamma_CJ ~ 3)
    gamma   = 3.0
    rho_CJ  = rho * (gamma + 1) / gamma         # g/cm3
    u_CJ    = D / (gamma + 1)                   # km/s particle velocity
    c_CJ    = D * gamma / (gamma + 1)           # km/s sound speed at CJ
    T_CJ    = (Q * 4.184) / (N * 30.0)          # K  (Cv_avg ~ 30 J/mol/K)
    dD_drho = 1.01 * 1.30 * phi**0.5            # km/s per g/cm3

    return dict(
        valid=True,
        D_km_s=D, P_GPa=P, phi=phi,
        N_mol_per_g=N, M_avg_g_per_mol=M_avg,
        Q_cal_per_g=Q, Q_kJ_per_g=Q * 4.184 / 1000,
        rho_CJ=rho_CJ, u_CJ=u_CJ, c_CJ=c_CJ, T_CJ_K=T_CJ,
        dD_drho=dD_drho,
        N=N, M_avg=M_avg,
        n_N2=n_N2, n_H2O=n_H2O, n_CO2=n_CO2,
        n_CO=n_CO, n_Cs=n_Cs, n_O2=n_O2,
        OB_pct=(d - 2*a - b/2) / mw * 1600,
    )



# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

# --- Formulation Control Bar ---
_top_col1, _top_col2 = st.columns(2)

formulation_names = [f.name for f in st.session_state.formulations]

with _top_col1:
    st.markdown("**Select Active Formulation:**")
    selected_name = st.selectbox(
        "Select Active Formulation",
        formulation_names,
        index=min(st.session_state.selected_idx, len(formulation_names) - 1),
        key="formulation_select",
        label_visibility="collapsed"
    )
    st.session_state.selected_idx = formulation_names.index(selected_name)

with _top_col2:
    st.markdown("**Compare against:**")
    compare_options = ["None"] + [n for i, n in enumerate(formulation_names) if i != st.session_state.selected_idx]
    compare_choice = st.selectbox("Compare with", compare_options, label_visibility="collapsed")
    if compare_choice != "None":
        st.session_state.compare_idx = formulation_names.index(compare_choice)
    else:
        st.session_state.compare_idx = None

st.markdown("**Library Import:**")
_lib_col1, _lib_col2 = st.columns([3, 1])
with _lib_col1:
    import sys, importlib
    import combusto.presets
    if "combusto.presets" in sys.modules:
        importlib.reload(sys.modules["combusto.presets"])
    from combusto.presets import get_all_library_templates
    lib_templates = get_all_library_templates()
    lib_names = [e.name for e in lib_templates]
    selected_lib_idx = st.selectbox("Standard Explosive Library", range(len(lib_names)), format_func=lambda i: f"{lib_templates[i].name} — {lib_templates[i].description}", label_visibility="collapsed")
with _lib_col2:
    if st.button("➕ Inject into Session", use_container_width=True):
        from combusto.models import Formulation, PropellantComponent
        src = lib_templates[selected_lib_idx]
        new_f = Formulation(
            name=src.name,
            components=[PropellantComponent(c.name, c.formula, c.mass_fraction, c.hf_298) for c in src.components],
            description=src.description,
        )
        st.session_state.formulations.append(new_f)
        st.session_state.selected_idx = len(st.session_state.formulations) - 1
        st.rerun()

st.markdown("**Formulation Controls:**")
_b1, _b2, _b3 = st.columns(3)
with _b1:
    if st.button("Copy Formulation", use_container_width=True):
        src = st.session_state.formulations[st.session_state.selected_idx]
        from combusto.models import Formulation, PropellantComponent
        new_f = Formulation(
            name=f"{src.name} (copy)",
            components=[PropellantComponent(c.name, c.formula, c.mass_fraction, c.hf_298) for c in src.components],
            description=src.description,
        )
        st.session_state.formulations.append(new_f)
        st.session_state.selected_idx = len(st.session_state.formulations) - 1
        st.rerun()
with _b2:
    if st.button("New Formulation", use_container_width=True):
        from combusto.models import Formulation
        st.session_state.formulations.append(Formulation(name=f"Custom {len(st.session_state.formulations) + 1}"))
        st.session_state.selected_idx = len(st.session_state.formulations) - 1
        st.rerun()
with _b3:
    if len(st.session_state.formulations) > 1:
        if st.button("Delete Formulation", use_container_width=True):
            st.session_state.formulations.pop(st.session_state.selected_idx)
            st.session_state.selected_idx = max(0, st.session_state.selected_idx - 1)
            st.rerun()

with st.expander("💾 Save & Load JSON Formulations"):
    _c_save, _c_load = st.columns(2)
    with _c_save:
        st.markdown("**Export to internal storage:**")
        if st.button("Save Current Session", use_container_width=True):
            from pathlib import Path
            save_path = Path("data/custom_formulations.json")
            from combusto.presets import save_formulations
            save_formulations(st.session_state.formulations, save_path)
            st.success(f"Saved formulations")
    with _c_load:
        st.markdown("**Import raw JSON:**")
        import json
        uploaded = st.file_uploader("Load JSON", type="json", key="load_file", label_visibility="collapsed")
        if uploaded:
            try:
                from combusto.presets import formulation_from_dict
                data = json.load(uploaded)
                loaded = [formulation_from_dict(e) for e in data["formulations"]]
                st.session_state.formulations.extend(loaded)
                st.success(f"Loaded {len(loaded)} formulations")
                st.rerun()
            except Exception as e:
                st.error("Invalid file")

with st.expander("📊 View All Loaded Formulations"):
    import pandas as pd
    from combusto.chemistry import equivalence_ratio, oxygen_balance
    
    summary_data = []
    for idx, f in enumerate(st.session_state.formulations):
        try:
            phi_val = equivalence_ratio(f)
            ob_val = oxygen_balance(f)
        except Exception:
            phi_val = 0.0
            ob_val = 0.0
            
        components_str = ", ".join([f"{c.name} ({c.mass_fraction*100:.1f}%)" for c in f.components])
        
        summary_data.append({
            "idx": idx,
            "Name": f.name,
            "Equivalence Ratio (φ)": round(phi_val, 3),
            "Oxygen Balance (%)": round(ob_val, 1),
            "Components": components_str
        })
        
    st.dataframe(pd.DataFrame(summary_data).set_index("idx"), use_container_width=True)

formulation = st.session_state.formulations[st.session_state.selected_idx]

# ===== Formulation Editor (Top Level) =====
with st.expander(f"⚙️ Formulation Editor ({formulation.name})", expanded=True):
    with st.expander("ℹ️  How to use — Formulation Editor", expanded=False):
        st.markdown("""
        **Define your gun propellant ingredients here.**
        - Add one row per component (Name, molecular formula, mass fraction, Hfâ‚‚â‚‰â‚ˆ).
        - Mass fractions must **sum to 1.0** \u2014 the metric strip below confirms this.
        - **Formula syntax**: standard notation e.g. `C6H7N3O8` (no spaces/brackets).
        - **Hfâ‚‚â‚‰â‚ˆ** is the standard heat of formation in kJ/mol (negative = exothermic formation).
        - Equivalence ratio \u03c6 and oxygen balance are computed automatically.
        - Use the **sidebar** to create, duplicate, or delete named formulations.
        """)
    col_name, col_desc = st.columns([1, 2])
    with col_name:
        new_name = st.text_input(
            "Formulation name", 
            value=formulation.name, 
            key=f"form_name_{st.session_state.selected_idx}"
        )
        formulation.name = new_name
    with col_desc:
        new_desc = st.text_input(
            "Description", 
            value=formulation.description, 
            key=f"form_desc_{st.session_state.selected_idx}"
        )
        formulation.description = new_desc
    st.session_state.formulations[st.session_state.selected_idx] = formulation

    st.markdown("**Components** â€” mass fractions should sum to 1.0")

    comp_data = []
    for c in formulation.components:
        comp_data.append({
            "Name": c.name,
            "Formula": c.formula,
            "Mass Fraction": c.mass_fraction,
            "Hf298 (kJ/mol)": c.hf_298,
        })
    if not comp_data:
        comp_data = [{"Name": "", "Formula": "", "Mass Fraction": 0.0, "Hf298 (kJ/mol)": 0.0}]

    edited_df = st.data_editor(
        pd.DataFrame(comp_data),
        num_rows="dynamic",
        use_container_width=True,
        key=f"comp_editor_{st.session_state.selected_idx}",
        column_config={
            "Mass Fraction": st.column_config.NumberColumn(
                min_value=0.0, max_value=1.0, step=0.01, format="%.4f",
            ),
            "Hf298 (kJ/mol)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    new_components = []
    for _, row in edited_df.iterrows():
        if row["Name"] and row["Formula"]:
            try:
                new_components.append(
                    PropellantComponent(
                        name=str(row["Name"]),
                        formula=str(row["Formula"]),
                        mass_fraction=float(row["Mass Fraction"]),
                        hf_298=float(row["Hf298 (kJ/mol)"]),
                    )
                )
            except Exception:
                pass
    formulation.components = new_components
    st.session_state.formulations[st.session_state.selected_idx] = formulation

    if formulation.components:
        total_frac = sum(c.mass_fraction for c in formulation.components)
        frac_delta = total_frac - 1.0

        m_frac, m_phi, m_ob = st.columns(3)
        m_frac.metric(
            "Total mass fraction",
            f"{total_frac:.4f}",
            delta=f"{frac_delta:+.4f}" if abs(frac_delta) > 0.001 else "OK",
            delta_color="inverse" if abs(frac_delta) > 0.001 else "off",
        )

        try:
            phi = equivalence_ratio(formulation)
            ob = oxygen_balance(formulation)
            m_phi.metric(
                "Equivalence ratio (phi)",
                f"{phi:.3f}",
                delta="Fuel-rich" if phi > 1 else "O2-rich",
            )
            m_ob.metric("Oxygen balance", f"{ob:.2f}%")
        except Exception:
            m_phi.metric("Equivalence ratio", "Error")
            m_ob.metric("Oxygen balance", "Error")

        col_elem, col_mass = st.columns(2)
        with col_elem:
            st.markdown("**Elemental composition (mol/g)**")
            try:
                elem = formulation.elemental_composition()
                elem_df = pd.DataFrame(
                    [{"Element": k, "mol/g": f"{v:.6f}"} for k, v in sorted(elem.items())]
                )
                st.dataframe(elem_df, use_container_width=True, hide_index=True, height=200)
            except Exception as e:
                st.error(str(e))
        with col_mass:
            st.markdown("**Elemental mass fractions**")
            try:
                emf = formulation.elemental_mass_fractions()
                emf_df = pd.DataFrame(
                    [{"Element": k, "Mass %": f"{v * 100:.2f}"} for k, v in sorted(emf.items())]
                )
                st.dataframe(emf_df, use_container_width=True, hide_index=True, height=200)
            except Exception as e:
                st.error(str(e))

st.markdown("<br>", unsafe_allow_html=True)
tab2, tab3, tab_cal, tab_perf, tab4, tab5 = st.tabs([
    "Conditions & Run", "Results", "Isothermal Calorimetry (Qv)", "Isentropic Expansion (Isp)", "Sensitivity", "VoD Estimator"
])

# ===== Tab 2: Conditions & Run =====
with tab2:
    with st.expander("ℹ️  How to use — Conditions & Run", expanded=False):
        st.markdown("""
        **Configure combustion conditions and launch the equilibrium solver.**
        - **Mode**: *isobaric* (constant P), *isochoric* (constant V, closed chamber), or *adiabatic* (T solved iteratively).
        - **Pressure**: chamber pressure in atm \u2014 typical gun values are 100\u20134 000 atm.
        - **Equilibrium temperature**: fixed T for non-adiabatic modes (disabled in adiabatic mode).
        - **Loading density** *(isochoric only)*: propellant mass per unit chamber volume (g/cm\u00b3).
        - **PAH Allocation Model**: controls soot/PAH carbon apportionment. *Moderate* is a good default.
        - Click **\u25b6 Run Combustion Calculation** \u2014 results appear in the **Results** tab.
        """)
    st.subheader("Combustion Conditions")

    col1, col2, col3, _spacer = st.columns([1, 1, 1, 1])
    with col1:
        mode = st.selectbox("Mode", ["isobaric", "isochoric", "adiabatic (self-consistent)"])
        pressure = st.number_input("Pressure (atm)", 0.1, 10000.0, 100.0, 10.0)
    with col2:
        eq_temp = st.number_input(
            "Equilibrium temperature (K)",
            500.0, 6000.0, 2500.0, 100.0,
            disabled=(mode == "adiabatic (self-consistent)"),
            help="Disabled in adiabatic mode â€” T is solved iteratively",
        )
        init_temp = st.number_input("Initial temperature (K)", 200.0, 500.0, 298.15, 1.0)
    with col3:
        loading_density = None
        covolume = 1.0e-3
        if mode == "isochoric":
            loading_density = st.number_input(
                "Loading density (g/cmÂ³)", 0.01, 2.0, 0.2, 0.01,
                help="Propellant mass per unit chamber volume",
            )
            covolume = st.number_input(
                "Covolume (mÂ³/mol)", 1e-5, 1e-2, 1e-3, 1e-4, format="%.4f",
                help="Noble-Abel covolume parameter",
            )

    st.divider()
    st.subheader("PAH Allocation Model")

    col_pah1, col_pah2 = st.columns([1, 2])
    with col_pah1:
        pah_preset = st.selectbox(
            "Preset", list(PRESETS.keys()),
            index=list(PRESETS.keys()).index("moderate"),
        )
        pah_config = PRESETS[pah_preset]

    with col_pah2:
        with st.expander("Customize PAH parameters", expanded=False):
            c1, c2, c3 = st.columns(3)
            custom_base = c1.slider("Base C fraction", 0.0, 0.3, pah_config.base_fraction, 0.01)
            custom_exp = c2.slider("phi exponent", 0.5, 5.0, pah_config.phi_exponent, 0.1)
            custom_max = c3.slider("Max C fraction", 0.0, 0.8, pah_config.max_fraction, 0.05)
            if (custom_base != pah_config.base_fraction or
                    custom_exp != pah_config.phi_exponent or
                    custom_max != pah_config.max_fraction):
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
    )

    st.divider()
    run_btn = st.button("â–¶  Run Combustion Calculation", type="primary")

    if run_btn:
        if not formulation.components:
            st.error("Add at least one component to the formulation.")
        else:
            with st.spinner("Solving equilibrium..."):
                try:
                    if mode == "adiabatic (self-consistent)":
                        result = solve_equilibrium_adiabatic(
                            formulation, conditions, pah_config,
                            species_db=st.session_state.species_db,
                            t_guess=2500.0,
                        )
                        if result.converged:
                            st.success(f"Converged in {result.iterations} iterations (T = {result.adiabatic_flame_temp_k:.0f} K)")
                        else:
                            st.warning(f"Did not converge after {result.iterations} iterations (T ~ {result.temperature_k:.0f} K)")
                    else:
                        result = solve_equilibrium(
                            formulation, conditions, pah_config,
                            species_db=st.session_state.species_db,
                            temperature_k=eq_temp,
                        )
                        st.success("Calculation complete!")

                    st.session_state.results[formulation.name] = result

                    # Run comparison if selected
                    if st.session_state.compare_idx is not None:
                        comp_form = st.session_state.formulations[st.session_state.compare_idx]
                        if mode == "adiabatic (self-consistent)":
                            result_b = solve_equilibrium_adiabatic(
                                comp_form, conditions, pah_config,
                                species_db=st.session_state.species_db,
                            )
                        else:
                            result_b = solve_equilibrium(
                                comp_form, conditions, pah_config,
                                species_db=st.session_state.species_db,
                                temperature_k=eq_temp,
                            )
                        st.session_state.results[comp_form.name] = result_b
                except Exception as e:
                    st.error(f"Calculation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ===== Tab 3: Results =====
with tab3:
    with st.expander("\u2139\ufe0f  How to use \u2014 Results", expanded=False):
        st.markdown("""
        **Displays equilibrium combustion output for the active formulation.**
        - Run a calculation from the **Conditions & Run** tab first.
        - **Metric cards**: key scalars \u2014 \u03c6, OB%, T_ad, Force constant, \u03b3, mean MW, â€¦
        - **Product Distribution**: full species table, sortable by mole/mass fraction.
        - **Mole Fraction chart** (top-10 species) and **Carbon Distribution** pie chart.
        - **PAH/Soot Species** section appears when soot products are non-negligible.
        - **Element Balance Check** (expander) verifies mass conservation \u2014 errors < 1 % are normal.
        - Use **Compare with** in the sidebar to overlay a second formulation side-by-side.
        - **Export** results as CSV, Excel, or JSON at the bottom of the tab.
        """)
    active_name = formulation.name
    result = st.session_state.results.get(active_name)

    if result is None:
        st.info("Run a calculation first (Conditions & Run tab).")
    else:
        st.subheader(f"Results: {active_name}")

        # Key metrics row 1
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("phi", f"{result.equivalence_ratio:.3f}")
        m2.metric("OB%", f"{result.oxygen_balance_pct:.2f}")
        m3.metric("T_ad (K)", f"{result.adiabatic_flame_temp_k:.0f}")
        m4.metric("Impetus (J/g)", f"{result.force_constant_j_per_g:.0f}")
        m5.metric("gamma", f"{result.gamma:.3f}")

        # Key metrics row 2
        m6, m7, m8, m9, m10 = st.columns(5)
        m6.metric("Cp_mix (J/mol K)", f"{result.mixture_cp:.2f}")
        m7.metric("Mean MW", f"{result.mean_molecular_weight:.2f}")
        m8.metric("Gas moles/g", f"{result.gas_moles_total:.5f}")
        if result.computed_pressure_atm > 0 and result.computed_pressure_atm != float("inf"):
            m9.metric("P_chamber (atm)", f"{result.computed_pressure_atm:.0f}")
        else:
            m9.metric("P_chamber (atm)", "N/A" if result.computed_pressure_atm == 0 else "Overflow")
        m10.metric("T_equil (K)", f"{result.temperature_k:.0f}")

        # Product distribution
        st.markdown("### Product Distribution")
        prod_df = result_to_dataframe(result)
        st.dataframe(
            prod_df.style.format({
                "Moles": "{:.6e}",
                "Mole Fraction": "{:.6f}",
                "Mass Fraction": "{:.6f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### Mole Fractions (Top 10)")
            top10 = prod_df.nlargest(10, "Mole Fraction")
            fig = px.bar(
                top10, x="Species", y="Mole Fraction",
                color="Mole Fraction",
                color_continuous_scale="Viridis",
                text_auto=".4f",
            )
            fig.update_layout(showlegend=False, height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            st.markdown("#### Carbon Distribution")
            if result.carbon_distribution:
                c_df = carbon_distribution_to_dataframe(result)
                c_df = c_df[c_df["Fraction"] > 0.001]
                fig2 = px.pie(
                    c_df, names="Category", values="Fraction", hole=0.35,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

        # PAH details
        if result.pah_details:
            st.markdown("### PAH / Soot Species")
            pah_df = pah_details_to_dataframe(result)
            col_pah_tbl, col_pah_chart = st.columns([1, 1])
            with col_pah_tbl:
                st.dataframe(
                    pah_df.style.format({
                        "Mole Fraction": "{:.6e}",
                        "Mass Fraction": "{:.6e}",
                        "Carbon Share": "{:.4f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_pah_chart:
                if len(pah_df) > 0:
                    fig_pah = px.bar(
                        pah_df, x="Species", y="Carbon Share",
                        color="Species", text_auto=".4f",
                    )
                    fig_pah.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_pah, use_container_width=True)

        # Element balance
        with st.expander("Element Balance Check"):
            eb_df = element_balance_to_dataframe(result)
            max_err = eb_df["Error %"].max() if len(eb_df) > 0 else 0
            if max_err < 1.0:
                st.success(f"Max element balance error: {max_err:.4f}%")
            elif max_err < 5.0:
                st.warning(f"Max element balance error: {max_err:.2f}%")
            else:
                st.error(f"Max element balance error: {max_err:.1f}% â€” results may be unreliable")
            st.dataframe(
                eb_df.style.format({"Error %": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        # Comparison
        if st.session_state.compare_idx is not None:
            comp_name = st.session_state.formulations[st.session_state.compare_idx].name
            result_b = st.session_state.results.get(comp_name)
            if result_b:
                st.divider()
                st.subheader(f"Comparison: {active_name} vs {comp_name}")

                metrics_data = {
                    "Metric": [
                        "Equivalence Ratio", "OB %", "T_ad (K)", "Force (J/g)",
                        "gamma", "Cp_mix (J/mol K)", "Mean MW", "Gas moles/g",
                    ],
                    active_name: [
                        f"{result.equivalence_ratio:.3f}", f"{result.oxygen_balance_pct:.2f}",
                        f"{result.adiabatic_flame_temp_k:.0f}", f"{result.force_constant_j_per_g:.0f}",
                        f"{result.gamma:.3f}", f"{result.mixture_cp:.2f}",
                        f"{result.mean_molecular_weight:.2f}", f"{result.gas_moles_total:.5f}",
                    ],
                    comp_name: [
                        f"{result_b.equivalence_ratio:.3f}", f"{result_b.oxygen_balance_pct:.2f}",
                        f"{result_b.adiabatic_flame_temp_k:.0f}", f"{result_b.force_constant_j_per_g:.0f}",
                        f"{result_b.gamma:.3f}", f"{result_b.mixture_cp:.2f}",
                        f"{result_b.mean_molecular_weight:.2f}", f"{result_b.gas_moles_total:.5f}",
                    ],
                }
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

                # Side-by-side carbon distribution
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{active_name}**")
                    if result.carbon_distribution:
                        cd1 = carbon_distribution_to_dataframe(result)
                        cd1 = cd1[cd1["Fraction"] > 0.001]
                        fig_c1 = px.pie(cd1, names="Category", values="Fraction", hole=0.3)
                        st.plotly_chart(fig_c1, use_container_width=True)
                with c2:
                    st.markdown(f"**{comp_name}**")
                    if result_b.carbon_distribution:
                        cd2 = carbon_distribution_to_dataframe(result_b)
                        cd2 = cd2[cd2["Fraction"] > 0.001]
                        fig_c2 = px.pie(cd2, names="Category", values="Fraction", hole=0.3)
                        st.plotly_chart(fig_c2, use_container_width=True)

        # Export
        st.divider()
        st.markdown("### Export")
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            csv_data = result_to_csv(result)
            st.download_button(
                "Download CSV", csv_data,
                file_name=f"combusto_{active_name.replace(' ', '_')}.csv",
                mime="text/csv", use_container_width=True,
            )
        with col_exp2:
            xlsx_data = result_to_excel_bytes(result)
            st.download_button(
                "Download Excel", xlsx_data,
                file_name=f"combusto_{active_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with col_exp3:
            json_str = json.dumps({
                "formulation": formulation_to_dict(formulation),
                "conditions": {
                    "temperature_k": conditions.temperature_k,
                    "pressure_atm": conditions.pressure_atm,
                    "mode": conditions.mode,
                },
                "results_summary": {
                    "phi": result.equivalence_ratio,
                    "OB_pct": result.oxygen_balance_pct,
                    "T_ad_K": result.adiabatic_flame_temp_k,
                    "force_J_per_g": result.force_constant_j_per_g,
                    "gamma": result.gamma,
                    "mean_MW": result.mean_molecular_weight,
                },
            }, indent=2)
            st.download_button(
                "Download JSON", json_str,
                file_name=f"combusto_{active_name.replace(' ', '_')}.json",
                mime="application/json", use_container_width=True,
            )

# ===== Tab_Cal: Isothermal Calorimetry =====
with tab_cal:
    with st.expander("ℹ️  How to use — Isothermal Calorimetry", expanded=False):
        st.markdown("""
        **Simulate a constant volume bomb calorimeter test.**
        - Calculates the theoretical Heat of Explosion ($Q_v$) for the active formulation.
        - Automatically forces the system to an isothermal state at **298.15 K**, ensuring that water and other condensables are correctly handled.
        - Click **▶ Run Isothermal Calorimetry Simulation** to compute energy density.
        """)
    st.subheader("Isothermal Calorimetry Test")
    st.markdown("Simulate a constant volume calorimetry test (isothermal at 298.15 K).")
    run_cal = st.button("▶ Run Isothermal Calorimetry Simulation", type="primary")
    if run_cal:
        if not formulation.components:
            st.error("Add at least one component to the formulation.")
        else:
            with st.spinner("Computing theoretical heat of explosion (Qv)..."):
                cal_conditions = CombustionConditions(
                    temperature_k=298.15,
                    pressure_atm=1.0,
                    mode="bomb_calorimeter",
                    loading_density=0.1,
                    covolume_m3_per_mol=1e-3,
                )
                res_cal = solve_equilibrium(
                    formulation, cal_conditions, PRESETS["moderate"],
                    species_db=st.session_state.species_db,
                    temperature_k=298.15,
                )
                st.success("Simulation complete.")
                m1, m2 = st.columns(2)
                m1.metric("Calorimetric Heat of Explosion (Q_v)", f"{res_cal.heat_of_explosion_j_per_g:.0f} J/g")
                m2.metric("Gas Moles / g (at 298K)", f"{res_cal.gas_moles_total:.5f}")
                
                st.markdown("#### Product States at 298.15 K")
                cal_df = result_to_dataframe(res_cal)
                st.dataframe(cal_df, use_container_width=True, hide_index=True)

# ===== Tab_Perf: Isentropic Expansion =====
with tab_perf:
    with st.expander("ℹ️  How to use — Isentropic Expansion", expanded=False):
        st.markdown("""
        **Calculate ideal rocket performance parameters.**
        - Uses the 1D equilibrium model to simulate adiabatic combustion at the given Chamber Pressure ($P_c$).
        - Expands the gases isentropically to the Exit Pressure ($P_e$).
        - Computes the theoretical Vacuum Specific Impulse ($I_{sp}$) and Gas generation Volume ($V_0$).
        """)
    st.subheader("Isentropic Expansion Performance")
    c1, c2 = st.columns(2)
    with c1:
        pc_atm = st.number_input("Chamber Pressure Pc (atm)", 10.0, 500.0, 68.0, 1.0)
    with c2:
        pe_atm = st.number_input("Exit Pressure Pe (atm)", 0.001, 10.0, 1.0, 0.1)

    run_perf = st.button("▶ Calculate Performance", type="primary", key="btn_perf")
    if run_perf:
        if not formulation.components:
            st.error("Add at least one component to the formulation.")
        else:
            with st.spinner("Solving adiabatic combustion..."):
                perf_cond = CombustionConditions(temperature_k=298.15, pressure_atm=pc_atm, mode="isobaric")
                res_perf = solve_equilibrium_adiabatic(
                    formulation, perf_cond, PRESETS["moderate"],
                    species_db=st.session_state.species_db,
                    t_guess=2500.0,
                )
                if res_perf.converged:
                    st.success("Calculation complete.")
                    
                    gamma = res_perf.gamma
                    mean_mw_kg = res_perf.mean_molecular_weight / 1000.0
                    t_ad = res_perf.adiabatic_flame_temp_k
                    isp_s = 0.0
                    if gamma > 1.0 and mean_mw_kg > 0:
                        r_spec = 8.314462 / mean_mw_kg
                        term1 = (2 * gamma / (gamma - 1)) * r_spec * t_ad
                        term2 = 1.0 - (pe_atm / pc_atm)**((gamma - 1) / gamma)
                        if term1 * term2 > 0:
                            v_e = math.sqrt(term1 * term2)
                            isp_s = v_e / 9.80665
                            
                    st.metric("Vacuum Specific Impulse (I_sp)", f"{isp_s:.1f} s")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Tc (K)", f"{t_ad:.0f}")
                    m2.metric("gamma", f"{gamma:.3f}")
                    m3.metric("Mw (g/mol)", f"{res_perf.mean_molecular_weight:.2f}")
                    m4.metric("Gas Volume V_0 (L/kg)", f"{res_perf.gas_volume_l_per_kg:.1f}")
                else:
                    st.error("Adiabatic flame temperature failed to converge.")

# ===== Tab 4: Sensitivity Analysis =====
with tab4:
    with st.expander("\u2139\ufe0f  How to use \u2014 Sensitivity Analysis", expanded=False):
        st.markdown("""
        **Sweep one variable while holding others fixed to explore parametric trends.**
        - Requires a loaded formulation and conditions set in the **Conditions & Run** tab.
        - **Equivalence ratio sweep**: varies \u03c6 from 0.5 \u2192 3.0 \u2014 shows CO/CO\u2082 crossover.
        - **Temperature sweep**: varies T from 1\u202f000 \u2192 4\u202f500 K \u2014 assesses kinetic freeze points.
        - **PAH model comparison**: runs all built-in soot models and compares carbon partitioning.
        - **Multi-formulation comparison**: evaluates all loaded formulations under identical conditions.
        - Results are plotted automatically; download raw data from the *Raw sweep data* expander.
        """)
    if not formulation.components:
        st.info("Define a formulation first.")
    else:
        _sa_col, _sa_spacer = st.columns([1, 1])
        with _sa_col:
            analysis_type = st.selectbox(
                "Analysis type",
                ["Equivalence ratio sweep", "Temperature sweep", "PAH model comparison", "Multi-formulation comparison"],
            )

        run_sensitivity = st.button("\u25b6  Run Sensitivity Analysis", type="primary")

        if run_sensitivity:
            with st.spinner("Running sweep..."):
                try:
                    if analysis_type == "Equivalence ratio sweep":
                        phi_range = np.linspace(0.5, 3.0, 25)
                        df = sweep_equivalence_ratio(
                            formulation, conditions, pah_config,
                            st.session_state.species_db, eq_temp, phi_range,
                        )
                        st.session_state["sensitivity_df"] = df
                        st.session_state["sensitivity_type"] = "phi"

                    elif analysis_type == "Temperature sweep":
                        t_range = np.linspace(1000, 4500, 25)
                        df = sweep_temperature(
                            formulation, conditions, pah_config,
                            st.session_state.species_db, t_range,
                        )
                        st.session_state["sensitivity_df"] = df
                        st.session_state["sensitivity_type"] = "temp"

                    elif analysis_type == "PAH model comparison":
                        df = sweep_pah_models(
                            formulation, conditions,
                            st.session_state.species_db, eq_temp,
                        )
                        st.session_state["sensitivity_df"] = df
                        st.session_state["sensitivity_type"] = "pah"

                    elif analysis_type == "Multi-formulation comparison":
                        df = compare_formulations(
                            st.session_state.formulations, conditions, pah_config,
                            st.session_state.species_db, eq_temp,
                        )
                        st.session_state["sensitivity_df"] = df
                        st.session_state["sensitivity_type"] = "multi"

                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display results
        if "sensitivity_df" in st.session_state:
            df = st.session_state["sensitivity_df"]
            stype = st.session_state["sensitivity_type"]

            if stype == "phi":
                st.markdown("### Equivalence Ratio Sweep")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["phi_target"], y=df["CO2_mole_frac"], name="CO2", mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=df["phi_target"], y=df["CO_mole_frac"], name="CO", mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=df["phi_target"], y=df["H2O_mole_frac"], name="H2O", mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=df["phi_target"], y=df["H2_mole_frac"], name="H2", mode="lines+markers"))
                    fig.update_layout(title="Major Products vs phi", xaxis_title="phi", yaxis_title="Mole Fraction", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col_g2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df["phi_target"], y=df["PAH_carbon_frac"], name="PAH C fraction", fill="tozeroy"))
                    fig2.add_trace(go.Scatter(x=df["phi_target"], y=df["CO_carbon_frac"], name="CO C fraction"))
                    fig2.add_trace(go.Scatter(x=df["phi_target"], y=df["CO2_carbon_frac"], name="CO2 C fraction"))
                    fig2.update_layout(title="Carbon Distribution vs phi", xaxis_title="phi", yaxis_title="Fraction of C", height=400)
                    st.plotly_chart(fig2, use_container_width=True)

            elif stype == "temp":
                st.markdown("### Temperature Sweep")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["sweep_temperature_k"], y=df["CO2_mole_frac"], name="CO2", mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=df["sweep_temperature_k"], y=df["CO_mole_frac"], name="CO", mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=df["sweep_temperature_k"], y=df["H2O_mole_frac"], name="H2O", mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=df["sweep_temperature_k"], y=df["H2_mole_frac"], name="H2", mode="lines+markers"))
                    fig.update_layout(title="Products vs Temperature", xaxis_title="T (K)", yaxis_title="Mole Fraction", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col_g2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df["sweep_temperature_k"], y=df["PAH_carbon_frac"], name="PAH C fraction", fill="tozeroy"))
                    fig2.update_layout(title="PAH Carbon vs Temperature", xaxis_title="T (K)", yaxis_title="PAH C fraction", height=400)
                    st.plotly_chart(fig2, use_container_width=True)

            elif stype == "pah":
                st.markdown("### PAH Model Comparison")
                fig = px.bar(
                    df, x="pah_model",
                    y=["CO2_carbon_frac", "CO_carbon_frac", "PAH_carbon_frac"],
                    barmode="group",
                    labels={"value": "Fraction of C", "variable": "Category"},
                    title="Carbon Distribution by PAH Model",
                )
                st.plotly_chart(fig, use_container_width=True)

            elif stype == "multi":
                st.markdown("### Multi-Formulation Comparison")
                fig = px.bar(
                    df, x="formulation",
                    y=["CO2_mole_frac", "CO_mole_frac", "H2O_mole_frac", "H2_mole_frac", "N2_mole_frac"],
                    barmode="group",
                    title="Major Products Comparison",
                )
                fig.update_layout(xaxis_tickangle=-45, height=450)
                st.plotly_chart(fig, use_container_width=True)

            # Show raw data
            with st.expander("Raw sweep data"):
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download sweep CSV", csv,
                    file_name="combusto_sensitivity.csv",
                    mime="text/csv",
                )

# ===== Tab 5: VoD Estimator =====
with tab5:
    with st.expander("\u2139\ufe0f  How to use \u2014 VoD Estimator", expanded=False):
        st.markdown("""
        **Estimates detonation velocity and CJ state using the Kamlet-Jacobs empirical equations.**

        Two source modes are available â€” choose via the toggle below:

        **Active Formulation**: Derives CHNO composition and Hf directly from the propellant
        defined in the Formulation tab. Only a loading density is needed. Ideal for assessing
        whether a gun propellant has detonation sensitivity.

        **Preset / Custom**: Enter any CHNO explosive formula manually or pick from the
        preset library (TNT, RDX, HMX, PETN, TATB, â€¦).

        *KJ accuracy*: typically \u00b12\u20134 % of measured VoD for pure CHNO explosives.
        The density dependence chart shows the linear D vs \u03c1 relationship from the KJ model.
        """)

    # â”€â”€ Source selector â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vod_source = st.radio(
        "Input source",
        ["\U0001f9ea  Active Formulation", "\U0001f4cb  Preset / Custom"],
        horizontal=True,
        key="vod_source",
        label_visibility="collapsed",
    )
    use_formulation = vod_source.startswith("\U0001f9ea")

    st.divider()

    # â”€â”€ Branch A: from the active formulation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if use_formulation:
        st.subheader(f"Source: \u201c{formulation.name}\u201d")

        if not formulation.components:
            st.warning("No components defined in the active formulation. Go to the **Formulation** tab and add ingredients first.")
        else:
            _fkj, _fkj_err = _formulation_kj_inputs(formulation)
            if _fkj_err:
                st.error(_fkj_err)
            else:
                # Show derived composition as read-only info cards
                ic1, ic2, ic3, ic4, ic5 = st.columns(5)
                ic1.metric("C (mol/g)",  f"{_fkj['a']:.5f}")
                ic2.metric("H (mol/g)",  f"{_fkj['b']:.5f}")
                ic3.metric("N (mol/g)",  f"{_fkj['c']:.5f}")
                ic4.metric("O (mol/g)",  f"{_fkj['d']:.5f}")
                ic5.metric("Hf\u1d39\u02e3 (kJ/g)", f"{_fkj['hf_kJ']:.3f}",
                           help="Effective heat of formation per gram = \u03a3(w\u1d62 \u00b7 Hf\u1d62 / MW\u1d62)")

                st.caption(
                    "CHNO values are the elemental composition (mol/g) from the formulation. "
                    "Hf\u1d39\u02e3 is the mass-weighted sum of component heats of formation. "
                    "Non-CHNO elements are excluded from the detonation model."
                )

                dc1, dc2, _ = st.columns([1, 1, 2])
                with dc1:
                    fvod_rho = st.number_input(
                        "Loading density \u03c1\u2080 (g/cm\u00b3)",
                        min_value=0.1, max_value=2.5,
                        value=0.2, step=0.01, format="%.3f",
                        key="fvod_rho",
                        help="As-loaded propellant density (g/cm\u00b3).",
                    )
                with dc2:
                    fvod_tmd = st.number_input(
                        "Plot up to TMD (g/cm\u00b3)",
                        min_value=0.5, max_value=2.5,
                        value=1.80, step=0.01, format="%.3f",
                        key="fvod_tmd",
                        help="Upper bound for the density dependence chart.",
                    )

                fvod_run = st.button("\u25b6  Estimate VoD from Formulation", type="primary", key="fvod_run")

                if fvod_run:
                    try:
                        _res = _kamlet_jacobs(
                            _fkj["a"], _fkj["b"], _fkj["c"], _fkj["d"],
                            _fkj["mw"], _fkj["hf_kJ"], fvod_rho,
                        )
                        _label = (
                            f"C{_fkj['a']:.4f} H{_fkj['b']:.4f} "
                            f"N{_fkj['c']:.4f} O{_fkj['d']:.4f}"
                        )
                        st.session_state["vod_result"] = _res
                        st.session_state["vod_inputs"] = dict(
                            formula=_label,
                            a=_fkj["a"], b=_fkj["b"], c=_fkj["c"], d=_fkj["d"],
                            mw=1.0, hf_kJ=_fkj["hf_kJ"],
                            rho=fvod_rho, rho_tmd=fvod_tmd,
                        )
                    except Exception as _e:
                        st.error(f"Calculation failed: {_e}")

    # â”€â”€ Branch B: preset / custom manual entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    else:
        pre_col, _ = st.columns([1, 1])
        with pre_col:
            preset_name = st.selectbox("Preset explosive", list(_VOD_PRESETS.keys()), key="vod_preset")

        preset = _VOD_PRESETS[preset_name]
        default_formula = preset["formula"]  if preset else "C7H5N3O6"
        default_hf      = preset["hf_kJ"]   if preset else -66.9
        default_tmd     = preset["rho_tmd"] if preset else 1.654

        st.subheader("Explosive Parameters")
        vc1, vc2, vc3, vc4 = st.columns([1.2, 1, 1, 1])
        with vc1:
            vod_formula = st.text_input(
                "Molecular formula (CHNO)", value=default_formula, key="vod_formula",
                help="e.g. C3H6N6O6 for RDX. Non-CHNO atoms are ignored.",
            )
        with vc2:
            vod_hf = st.number_input(
                "Hf (kJ/mol)", value=float(default_hf), step=10.0, format="%.1f", key="vod_hf",
                help="Standard heat of formation at 298 K. Negative = exothermic.",
            )
        with vc3:
            vod_tmd = st.number_input(
                "TMD (g/cm\u00b3)", min_value=0.5, max_value=2.5,
                value=float(default_tmd), step=0.01, format="%.3f", key="vod_tmd",
                help="Theoretical maximum density of the pure explosive.",
            )
        with vc4:
            vod_rho = st.number_input(
                "Loading density \u03c1\u2080 (g/cm\u00b3)",
                min_value=0.1, max_value=float(vod_tmd),
                value=round(min(float(default_tmd) * 0.95, float(vod_tmd)), 3),
                step=0.01, format="%.3f", key="vod_rho",
                help="As-loaded density \u2014 must be \u2264 TMD.",
            )

        vod_run = st.button("\u25b6  Estimate VoD", type="primary", key="vod_run")

        if vod_run:
            try:
                _a, _b, _c, _d = _parse_chno(vod_formula)
                _mw = _a*12.011 + _b*1.008 + _c*14.007 + _d*15.999
                if _mw < 1.0:
                    st.error("Could not parse formula \u2014 check CHNO notation (e.g. C3H6N6O6).")
                else:
                    _res = _kamlet_jacobs(_a, _b, _c, _d, _mw, vod_hf, vod_rho)
                    st.session_state["vod_result"] = _res
                    st.session_state["vod_inputs"] = dict(
                        formula=vod_formula, a=_a, b=_b, c=_c, d=_d,
                        mw=_mw, hf_kJ=vod_hf, rho=vod_rho, rho_tmd=vod_tmd,
                    )
            except Exception as _e:
                st.error(f"Calculation failed: {_e}")

    # â”€â”€ Shared results section â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "vod_result" in st.session_state and "vod_inputs" in st.session_state:
        _res = st.session_state["vod_result"]
        _inp = st.session_state["vod_inputs"]

        st.divider()

        if not _res["valid"]:
            st.error(
                f"\u26a0\ufe0f  Heat of detonation Q = {_res['Q_cal_per_g']:.1f} cal/g \u2264 0. "
                "This composition is not detonable under the KJ model \u2014 "
                "check formula / Hf sign, or note that very fuel-lean propellants may not support detonation."
            )
        else:
            # â”€â”€ KJ primary results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.subheader("Kamlet-Jacobs Results")
            km1, km2, km3, km4 = st.columns(4)
            km1.metric("VoD",              f"{_res['D_km_s']:.3f} km/s",    help="Detonation velocity (Kamlet-Jacobs)")
            km2.metric("P_CJ",             f"{_res['P_GPa']:.2f} GPa",      help="Chapman-Jouguet detonation pressure")
            km3.metric("Q_det",            f"{_res['Q_kJ_per_g']:.2f} kJ/g",help="Specific heat of detonation")
            km4.metric("\u03c6  (KJ param)", f"{_res['phi']:.4f}",           help="\u03c6 = N\u00b7\u221aM\u00b7\u221aQ")

            ki1, ki2, ki3, ki4 = st.columns(4)
            ki1.metric("N (mol gas/g)",    f"{_res['N_mol_per_g']:.5f}")
            ki2.metric("M\u0305 (g/mol)",  f"{_res['M_avg_g_per_mol']:.2f}")
            ki3.metric("Q (cal/g)",        f"{_res['Q_cal_per_g']:.1f}")
            ki4.metric("O.B. (%)",         f"{_res['OB_pct']:.1f}")

            # â”€â”€ CJ state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.divider()
            st.subheader("Chapman-Jouguet State  (\u03b3_CJ \u2248 3)")
            cj1, cj2, cj3, cj4 = st.columns(4)
            cj1.metric("\u03c1_CJ (g/cm\u00b3)", f"{_res['rho_CJ']:.3f}",  help="Product density at CJ point")
            cj2.metric("u_CJ (km/s)",            f"{_res['u_CJ']:.3f}",    help="Particle velocity at CJ = D/(\u03b3+1)")
            cj3.metric("c_CJ (km/s)",            f"{_res['c_CJ']:.3f}",    help="Sound speed at CJ point")
            cj4.metric("T_CJ (K)",               f"{_res['T_CJ_K']:.0f}",  help="Approx. CJ temperature (rough estimate)")

            # â”€â”€ Detonation products â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.divider()
            st.subheader("Detonation Products  (H\u2082O\u2013CO\u2082 arbitrary)")
            prod_col, pie_col = st.columns([1, 1])

            with prod_col:
                _gas = _res["n_N2"] + _res["n_H2O"] + _res["n_CO2"] + _res["n_CO"] + _res["n_O2"]
                def _pct(x):
                    return f"{100*x/_gas:.1f}" if _gas > 0 else "\u2014"
                prod_rows = [
                    {"Species": "N\u2082",  "Mol / g propellant": f"{_res['n_N2']:.5f}",  "Mol %": _pct(_res['n_N2'])},
                    {"Species": "H\u2082O", "Mol / g propellant": f"{_res['n_H2O']:.5f}", "Mol %": _pct(_res['n_H2O'])},
                    {"Species": "CO\u2082", "Mol / g propellant": f"{_res['n_CO2']:.5f}", "Mol %": _pct(_res['n_CO2'])},
                    {"Species": "CO",       "Mol / g propellant": f"{_res['n_CO']:.5f}",  "Mol %": _pct(_res['n_CO'])},
                    {"Species": "O\u2082",  "Mol / g propellant": f"{_res['n_O2']:.5f}",  "Mol %": _pct(_res['n_O2'])},
                    {"Species": "C(s)",     "Mol / g propellant": f"{_res['n_Cs']:.5f}",  "Mol %": "solid"},
                ]
                st.dataframe(pd.DataFrame(prod_rows), use_container_width=True, hide_index=True)
                st.caption(
                    f"Composition: {_inp['formula']}  |  "
                    f"Hf = {_inp['hf_kJ']:.3f} kJ/g  |  "
                    f"\u03c1\u2080 = {_inp['rho']:.3f} g/cm\u00b3"
                )

            with pie_col:
                _gas_vals = [max(_res["n_N2"], 0), max(_res["n_H2O"], 0),
                             max(_res["n_CO2"], 0), max(_res["n_CO"], 0), max(_res["n_O2"], 0)]
                if sum(_gas_vals) > 0:
                    fig_pie = go.Figure(go.Pie(
                        labels=["N\u2082", "H\u2082O", "CO\u2082", "CO", "O\u2082"],
                        values=_gas_vals,
                        hole=0.38, textinfo="label+percent",
                        marker_colors=["#6ea8d8","#5bc0a0","#e07b54","#e0c454","#a095d4"],
                    ))
                    fig_pie.update_layout(showlegend=False, height=260,
                                          margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_pie, use_container_width=True)

            # â”€â”€ Density dependence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            st.divider()
            st.subheader("Density Dependence  (KJ linear law:  D = 1.01\u00b7\u221a\u03c6\u00b7(1 + 1.30\u03c1))")
            _rho_arr = np.linspace(0.5, max(_inp["rho_tmd"], _inp["rho"] + 0.01), 100)
            _D_arr = np.array([
                _kamlet_jacobs(
                    _inp["a"], _inp["b"], _inp["c"], _inp["d"],
                    _inp["mw"], _inp["hf_kJ"], r,
                )["D_km_s"] for r in _rho_arr
            ])

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=_rho_arr, y=_D_arr, mode="lines",
                line=dict(color="#e05252", width=2.5), name="D(\u03c1) \u2014 KJ",
            ))
            fig_dd.add_trace(go.Scatter(
                x=[_inp["rho"]], y=[_res["D_km_s"]],
                mode="markers",
                marker=dict(size=12, color="#ffffff", line=dict(color="#e05252", width=2.5)),
                name=f"\u03c1\u2080 = {_inp['rho']:.3f} \u2192 {_res['D_km_s']:.3f} km/s",
            ))
            fig_dd.update_layout(
                xaxis_title="Loading density \u03c1\u2080 (g/cm\u00b3)",
                yaxis_title="Detonation velocity D (km/s)",
                height=360,
                margin=dict(t=20, b=40, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_dd, use_container_width=True)
            st.caption(
                f"KJ density slope  dD/d\u03c1 = **{_res['dD_drho']:.3f} km\u00b7s\u207b\u00b9\u00b7(g\u00b7cm\u207b\u00b3)\u207b\u00b9**.  "
                "Typical CHNO range: 2.8 \u2013 3.5 km\u00b7s\u207b\u00b9/(g\u00b7cm\u207b\u00b3)."
            )

