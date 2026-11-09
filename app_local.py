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
    sweep_2d,
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


st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Combusto Local Admin")
admin_action = st.sidebar.radio("Admin Mode", ["Main Workbench", "Database Editor", "API Key Manager"])

if admin_action == "Database Editor":
    st.title("Combusto Admin / 🗄️ Master Database Editor")
    st.markdown("Directly edit backend empirical dictionaries. Saves securely to disk.")
    
    db_choice = st.selectbox("Select Database File", ["data/ingredients.json", "data/propellants.json", "data/species.json"])
    
    import json
    from pathlib import Path
    import pandas as pd
    
    db_path = Path(db_choice)
    if db_path.exists():
        with open(db_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # Determine the root key
        root_key = list(raw_data.keys())[-1]
        df = pd.DataFrame(raw_data[root_key])
        
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, height=600)
        
        if st.button(f"💾 Save {db_choice} to Disk", type="primary"):
            raw_data[root_key] = edited_df.to_dict(orient="records")
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2)
            st.success(f"Successfully overwrote {db_choice}")
    else:
        st.error("Database file missing!")
        
    st.stop() # Stops rendering the main app

elif admin_action == "API Key Manager":
    st.title("Combusto Admin / 🔑 Local API Key Manager")
    st.markdown("Inject secrets into local `.streamlit/secrets.toml` for environmental access.")
    
    import os
    import toml
    from pathlib import Path
    
    secrets_dir = Path(".streamlit")
    secrets_file = secrets_dir / "secrets.toml"
    
    if not secrets_dir.exists():
        secrets_dir.mkdir(parents=True, exist_ok=True)
    
    current_secrets = {}
    if secrets_file.exists():
        try:
            current_secrets = toml.load(secrets_file)
        except Exception:
            pass
            
    nist_key = st.text_input("NIST WebBook API Key", value=current_secrets.get("NIST_API_KEY", ""), type="password")
    mat_key = st.text_input("Materials Project API Key", value=current_secrets.get("MATERIALSPROJECT_API_KEY", ""), type="password")
    pub_key = st.text_input("PubChem PUG REST Auth Token", value=current_secrets.get("PUBCHEM_API_KEY", ""), type="password")
    
    if st.button("🔒 Save Environment Secrets", type="primary"):
        new_secrets = {
            "NIST_API_KEY": nist_key,
            "MATERIALSPROJECT_API_KEY": mat_key,
            "PUBCHEM_API_KEY": pub_key
        }
        with open(secrets_file, "w") as f:
            toml.dump(new_secrets, f)
        st.success("Secrets successfully committed to local storage!")
        
    st.stop()


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
    from combusto.presets import load_presets
    from combusto.models import Formulation
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
formulation_names = [f.name for f in st.session_state.formulations]

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

st.markdown("---")
_top_col1, _top_col2 = st.columns(2)

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

    st.markdown("**Empirical Ingredient Database Search**")
    _ing_col1, _ing_col2 = st.columns([3, 1])
    with _ing_col1:
        from combusto.presets import load_ingredients
        raw_ingredients = load_ingredients()
        ing_opts = [
            f"{ing['name']}  ({ing['formula']} | {ing['hf_298']} kJ/mol | CAS {ing.get('cas', 'N/A')})"
            for ing in raw_ingredients
        ]
        selected_ing_idx = st.selectbox(
            "Search and inject standardized ingredients",
            range(len(ing_opts)),
            format_func=lambda i: ing_opts[i],
            label_visibility="collapsed",
            key=f"ing_search_{st.session_state.selected_idx}"
        )
    with _ing_col2:
        if st.button("➕ Add to Formulation", use_container_width=True, key=f"add_ing_{st.session_state.selected_idx}"):
            if raw_ingredients:
                ing = raw_ingredients[selected_ing_idx]
                from combusto.models import PropellantComponent
                formulation.components.append(PropellantComponent(
                    name=ing["name"],
                    formula=ing["formula"],
                    mass_fraction=0.0,
                    hf_298=ing["hf_298"]
                ))
                st.session_state.formulations[st.session_state.selected_idx] = formulation
                st.rerun()

    from combusto.ui_extensions import render_nc_nitration_calculator
    render_nc_nitration_calculator(formulation)

    with st.expander("🔍 External API Chemical Lookup (PubChem & NIST)", expanded=False):
        api_col1, api_col2 = st.columns([3, 1])
        with api_col1:
            search_query = st.text_input(
                "Search compound by name or SMILES (e.g. 'RDX' or 'CC1=C...')",
                key=f"api_search_query_{st.session_state.selected_idx}"
            )
        with api_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_search = st.button("Search API", key=f"run_api_search_{st.session_state.selected_idx}", use_container_width=True)
            
        if search_query and run_search:
            from combusto.lookup import query_pubchem, query_nist_enthalpy
            
            # Determine if it's SMILES
            is_smiles = any(c in search_query for c in ['=', '#', '@', '\\', '/', '[', ']']) or (len(search_query) > 12 and any(c.islower() for c in search_query))
            search_type = "smiles" if is_smiles else "name"
            
            with st.spinner("Querying PubChem..."):
                pubchem_res = query_pubchem(search_query, search_type)
                
            if pubchem_res:
                st.success(f"Found in PubChem: **{pubchem_res['iupac_name'] or pubchem_res['name']}**")
                st.write(f"- Formula: `{pubchem_res['formula']}`")
                st.write(f"- MW: `{pubchem_res['mw']:.2f} g/mol`")
                
                # Query NIST WebBook
                with st.spinner("Scraping NIST Chemistry WebBook..."):
                    nist_hf = query_nist_enthalpy(pubchem_res['formula'])
                    if nist_hf is None:
                        nist_hf = query_nist_enthalpy(pubchem_res['name'])
                        
                if nist_hf is not None:
                    st.write(f"- NIST Hf_298: `{nist_hf:.1f} kJ/mol`")
                    hf_val = nist_hf
                else:
                    st.warning("Could not retrieve Hf_298 from NIST WebBook. You can enter it manually below.")
                    hf_val = 0.0
                    
                st.session_state[f"api_lookup_res_{st.session_state.selected_idx}"] = {
                    "name": pubchem_res['name'],
                    "formula": pubchem_res['formula'],
                    "hf_298": hf_val
                }
            else:
                st.error("Compound not found on PubChem. Please check spelling or SMILES format.")
                if f"api_lookup_res_{st.session_state.selected_idx}" in st.session_state:
                    del st.session_state[f"api_lookup_res_{st.session_state.selected_idx}"]

        lookup_key = f"api_lookup_res_{st.session_state.selected_idx}"
        if lookup_key in st.session_state:
            cached_res = st.session_state[lookup_key]
            st.info(f"Buffered Compound: **{cached_res['name']}** ({cached_res['formula']} | {cached_res['hf_298']:.1f} kJ/mol)")
            if st.button("➕ Inject into Component Table", key=f"inject_api_res_{st.session_state.selected_idx}", use_container_width=True):
                from combusto.models import PropellantComponent
                formulation.components.append(PropellantComponent(
                    name=cached_res['name'],
                    formula=cached_res['formula'],
                    mass_fraction=0.0,
                    hf_298=cached_res['hf_298']
                ))
                st.session_state.formulations[st.session_state.selected_idx] = formulation
                del st.session_state[lookup_key]
                st.success(f"Injected {cached_res['name']}!")
                st.rerun()

    st.markdown("**Components** — mass fractions should sum to 1.0")

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
tab2, tab3, tab_cal, tab_perf, tab4, tab_ballistics, tab_cv, tab_flash, tab_tox, tab_stab, tab_opt, tab5, tab_grounding = st.tabs([
    "Conditions & Run", "Results", "Isothermal Calorimetry (Qv)", "Isentropic Expansion (Isp)", "Sensitivity", "Internal Ballistics & Erosion", "Closed-Vessel & Vivacity", "Muzzle Flash & Plume", "Smoke & Range Toxicity", "Shelf-Life & Thermal Safety", "Optimizer & Ternary", "VoD Estimator", "📚 Benchmarks & Grounding"
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
        from combusto.grounding import EPA_AP42_CALIBRATED_PAH
        _ui_pah_presets = dict(PRESETS)
        _ui_pah_presets["epa_ap42_calibrated (Trace VOC/PAH)"] = EPA_AP42_CALIBRATED_PAH
        pah_preset = st.selectbox(
            "Preset", list(_ui_pah_presets.keys()),
            index=list(_ui_pah_presets.keys()).index("epa_ap42_calibrated (Trace VOC/PAH)"),
        )
        pah_config = _ui_pah_presets[pah_preset]

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
                    st.session_state["result"] = result

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
        m3.metric("T_p Isobaric (K)", f"{result.adiabatic_flame_temp_k:.0f}", help="Constant-pressure flame temperature (ΔH = 0, NASA CEA)")
        m4.metric("T_v Closed-Vessel (K)", f"{result.isochoric_flame_temp_k:.0f}", help="Constant-volume gun chamber flame temperature (ΔU = 0, STANAG 4115)")
        m5.metric("Impetus F (J/g)", f"{result.force_constant_j_per_g:.0f}", help="n_gas · R · T_v (US Army BRL / STANAG 4115)")

        # Key metrics row 2
        m6, m7, m8, m9, m10 = st.columns(5)
        m6.metric("Covolume η (cm³/g)", f"{result.covolume_cm3_per_g:.3f}", help="BLAKE / Hirschfelder composition-derived Noble-Abel covolume Σ nᵢbᵢ")
        m7.metric("gamma (Cp/Cv)", f"{result.gamma:.3f}")
        m8.metric("Mean Gas MW", f"{result.mean_molecular_weight:.2f}")
        if result.computed_pressure_atm > 0 and result.computed_pressure_atm != float("inf"):
            m9.metric("P_chamber (atm)", f"{result.computed_pressure_atm:.0f}")
        else:
            m9.metric("Q_v Calorimetric (J/g)", f"{result.heat_of_explosion_j_per_g:.0f}")
        m10.metric("Gas moles/g", f"{result.gas_moles_total:.5f}")

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
        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
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
        with col_exp4:
            from combusto.export_pdf import export_result_pdf
            pdf_bytes = export_result_pdf(result, formulation)
            st.download_button(
                "📄 Download PDF Report", pdf_bytes,
                file_name=f"combusto_{active_name.replace(' ', '_')}_report.pdf",
                mime="application/pdf", use_container_width=True,
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
                    st.session_state["res_perf_cached"] = res_perf
                    
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

    from combusto.ui_extensions import render_noble_abel_isentrope_section
    render_noble_abel_isentrope_section(formulation, conditions, pah_config, eq_temp)

# ===== Tab 4: Sensitivity Analysis =====
with tab4:
    with st.expander("ℹ️  How to use — Sensitivity Analysis", expanded=False):
        st.markdown("""
        **Sweep one or two variables while holding others fixed to explore parametric trends.**
        - Requires a loaded formulation and conditions set in the **Conditions & Run** tab.
        - **Equivalence ratio sweep**: varies φ from 0.5 → 3.0 — shows CO/CO₂ crossover.
        - **Temperature sweep**: varies T from 1 000 → 4 500 K — assesses kinetic freeze points.
        - **PAH model comparison**: runs all built-in soot models and compares carbon partitioning.
        - **Multi-formulation comparison**: evaluates all loaded formulations under identical conditions.
        - **2D Sensitivity sweep**: sweeps two variables simultaneously and renders interactive 3D surface and 2D contour maps.
        """)
    if not formulation.components:
        st.info("Define a formulation first.")
    else:
        _sa_col, _sa_spacer = st.columns([1, 1])
        with _sa_col:
            analysis_type = st.selectbox(
                "Analysis type",
                ["Equivalence ratio sweep", "Temperature sweep", "PAH model comparison", "Multi-formulation comparison", "2D Sensitivity sweep"],
            )

        # 2D Sweep parameter selection
        if analysis_type == "2D Sensitivity sweep":
            var_opts = {
                "Equivalence ratio (phi)": "equivalence_ratio",
                "Temperature (K)": "temperature",
                "Pressure (atm)": "pressure"
            }
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                v1_label = st.selectbox("Variable 1 (X-Axis)", list(var_opts.keys()), index=0, key="sa2d_v1")
                v1_name = var_opts[v1_label]
                if v1_name == "equivalence_ratio":
                    v1_min, v1_max = st.slider("X Range (phi)", 0.5, 3.0, (0.5, 2.5), step=0.1, key="sa2d_v1_rng")
                elif v1_name == "temperature":
                    v1_min, v1_max = st.slider("X Range (Temperature K)", 1000, 4500, (1500, 3500), step=100, key="sa2d_v1_rng")
                else:
                    v1_min, v1_max = st.slider("X Range (Pressure atm)", 1, 500, (10, 150), step=10, key="sa2d_v1_rng")
            with col_v2:
                remaining_opts = {k: v for k, v in var_opts.items() if k != v1_label}
                v2_label = st.selectbox("Variable 2 (Y-Axis)", list(remaining_opts.keys()), index=0, key="sa2d_v2")
                v2_name = remaining_opts[v2_label]
                if v2_name == "equivalence_ratio":
                    v2_min, v2_max = st.slider("Y Range (phi)", 0.5, 3.0, (0.8, 2.8), step=0.1, key="sa2d_v2_rng")
                elif v2_name == "temperature":
                    v2_min, v2_max = st.slider("Y Range (Temperature K)", 1000, 4500, (2000, 4000), step=100, key="sa2d_v2_rng")
                else:
                    v2_min, v2_max = st.slider("Y Range (Pressure atm)", 1, 500, (20, 200), step=10, key="sa2d_v2_rng")
            
            metric_opts = {
                "Adiabatic Flame Temperature (K)": "adiabatic_flame_temp_k",
                "PAH Carbon Fraction": "PAH_carbon_frac",
                "CO Carbon Fraction": "CO_carbon_frac",
                "CO2 Carbon Fraction": "CO2_carbon_frac",
                "CO Mole Fraction": "CO_mole_frac",
                "CO2 Mole Fraction": "CO2_mole_frac",
                "H2 Mole Fraction": "H2_mole_frac",
                "H2O Mole Fraction": "H2O_mole_frac",
                "Force Constant (J/g)": "force_constant_j_per_g",
                "Gamma (Cp/Cv)": "gamma",
                "Specific Impulse (s)": "specific_impulse_s",
            }
            selected_metric_label = st.selectbox("Select Metric to Plot", list(metric_opts.keys()), index=0, key="sa2d_metric")
            selected_metric = metric_opts[selected_metric_label]

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

                    elif analysis_type == "2D Sensitivity sweep":
                        r1 = np.linspace(v1_min, v1_max, 10)
                        r2 = np.linspace(v2_min, v2_max, 10)
                        df = sweep_2d(
                            formulation, conditions, pah_config,
                            st.session_state.species_db, v1_name, r1, v2_name, r2
                        )
                        st.session_state["sensitivity_df"] = df
                        st.session_state["sensitivity_type"] = "sweep2d"
                        st.session_state["sa2d_v1_name"] = v1_name
                        st.session_state["sa2d_v2_name"] = v2_name
                        st.session_state["sa2d_metric"] = selected_metric
                        st.session_state["sa2d_metric_label"] = selected_metric_label
 
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

            elif stype == "sweep2d":
                v1_n = st.session_state.get("sa2d_v1_name", "param1")
                v2_n = st.session_state.get("sa2d_v2_name", "param2")
                met = st.session_state.get("sa2d_metric", "adiabatic_flame_temp_k")
                met_lbl = st.session_state.get("sa2d_metric_label", "Value")
                
                st.markdown(f"### 2D Sensitivity Sweep: {met_lbl}")
                
                x_vals = sorted(df[f"param_{v1_n}"].unique())
                y_vals = sorted(df[f"param_{v2_n}"].unique())
                
                z_grid = np.zeros((len(y_vals), len(x_vals)))
                for idx_y, y_val in enumerate(y_vals):
                    for idx_x, x_val in enumerate(x_vals):
                        match = df[(df[f"param_{v1_n}"] == x_val) & (df[f"param_{v2_n}"] == y_val)]
                        if not match.empty:
                            z_grid[idx_y, idx_x] = match.iloc[0][met]
                            
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig = go.Figure(data=[go.Surface(z=z_grid, x=x_vals, y=y_vals, colorscale="Viridis")])
                    fig.update_layout(
                        title=f"3D Surface: {met_lbl}",
                        scene=dict(
                            xaxis_title=v1_n,
                            yaxis_title=v2_n,
                            zaxis_title=met_lbl
                        ),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col_g2:
                    fig2 = go.Figure(data=go.Contour(
                        z=z_grid,
                        x=x_vals,
                        y=y_vals,
                        colorscale="Viridis",
                        line_width=0.5,
                        contours=dict(showlabels=True)
                    ))
                    fig2.update_layout(
                        title=f"2D Contour: {met_lbl}",
                        xaxis_title=v1_n,
                        yaxis_title=v2_n,
                        height=500
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # Show raw data
            with st.expander("Raw sweep data"):
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download sweep CSV", csv,
                    file_name="combusto_sensitivity.csv",
                    mime="text/csv",
                )

# ===== Tab_Ballistics: Internal Ballistics, Multi-Perf Grain Visualizer & Barrel Erosion =====
with tab_ballistics:
    with st.expander("ℹ️  How to use — Internal Ballistics, Grain Burnback & Barrel Erosion", expanded=False):
        st.markdown(r"""
        **Simulate transient interior ballistics, progressive multi-perforated grain burnback, and bore wall thermal erosion.**
        - Uses **Noble-Abel Equation of State**, **Saint-Robert's burn rate law** ($r = a P^n$), and **grain geometry form functions** including progressive **7-perforated** and **19-perforated** rosette grains with post-web-breakup sliver phase.
        - **2D Grain Cross-Section Visualizer**: Inspect concentric burning surfaces and form function $z(u)$ across normalized burn depth $u = 2 x_b / D_0$.
        - **Barrel Thermal Erosion & Wear Estimator**: Computes transient bore wall surface temperature $T_{wall}(t)$, convective heat flux, chemical gas wash ($\text{CO}/\text{CO}_2$ and $\text{H}_2/\text{H}_2\text{O}$), and estimated barrel wear life in Equivalent Full Charges (EFC).
        """)
    st.subheader("Transient Gun Interior Ballistics & Barrel Erosion Simulator")

    if not formulation.components:
        st.info("Define a formulation first.")
    else:
        from combusto.ballistics import (
            simulate_ballistics,
            generate_grain_cross_section,
            grain_form_function,
            estimate_barrel_erosion,
        )

        default_F = 1050.0
        default_gamma = 1.24
        if "result" in st.session_state and st.session_state["result"] is not None:
            res_cur = st.session_state["result"]
            if res_cur.force_constant_j_per_g > 0:
                default_F = float(res_cur.force_constant_j_per_g)
            if res_cur.gamma > 1.0:
                default_gamma = float(res_cur.gamma)

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.markdown("**Propellant & Grain Parameters**")
            ib_F = st.number_input("Force Constant F (J/g)", 100.0, 2500.0, default_F, 10.0, key="ib_F")
            ib_gamma = st.number_input("Ratio of Specific Heats γ", 1.05, 1.67, default_gamma, 0.01, key="ib_gamma")
            ib_charge_g = st.number_input("Charge Mass (g)", 0.5, 50000.0, 50.0, 1.0, key="ib_charge")
            ib_rho_p = st.number_input("Propellant Solid Density (g/cm³)", 0.8, 2.2, 1.60, 0.05, key="ib_rhop")
            ib_covol = st.number_input("Gas Covolume η (cm³/g)", 0.5, 2.0, 1.0, 0.05, key="ib_covol")

        with col_b2:
            st.markdown("**Burn Rate & Geometry**")
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
            st.markdown("**Chamber & Barrel Hardware**")
            ib_vc_cm3 = st.number_input("Chamber Volume Vc (cm³)", 1.0, 100000.0, 65.0, 5.0, key="ib_vc")
            ib_bore_mm = st.number_input("Bore Diameter (mm)", 4.0, 200.0, 30.0, 1.0, key="ib_bore")
            ib_len_m = st.number_input("Barrel Travel Length (m)", 0.1, 10.0, 1.5, 0.1, key="ib_len")
            ib_mproj_g = st.number_input("Projectile Mass (g)", 1.0, 100000.0, 300.0, 10.0, key="ib_mproj")
            ib_pstart_mpa = st.number_input("Shot-Start Pressure (MPa)", 1.0, 150.0, 30.0, 5.0, key="ib_pstart")

        # 2D Grain Cross-Section & Form Function Visualizer
        with st.expander("🔬 2D Grain Cross-Section Burnback & Form Function Visualizer", expanded=True):
            u_max_vis = 1.18 if ib_geom == "7_perf" else (1.15 if ib_geom == "19_perf" else 1.0)
            u_vis = st.slider("Inspect Normalized Web Burn Depth u = 2·x_b / D₀", 0.0, float(u_max_vis), 0.35, 0.02, key="ib_u_vis")
            cs = generate_grain_cross_section(ib_geom, ib_web_mm, u_vis)
            z_cur, dz_du_cur = grain_form_function(u_vis, ib_geom)

            vis_c1, vis_c2 = st.columns(2)
            with vis_c1:
                theta = np.linspace(0, 2 * np.pi, 120)
                fig_cs = go.Figure()
                r_oi = cs["outer_r_init"]
                fig_cs.add_trace(go.Scatter(
                    x=r_oi * np.cos(theta), y=r_oi * np.sin(theta),
                    mode="lines", name="Initial Outer Boundary",
                    line=dict(color="#64748b", dash="dot", width=1.5),
                ))
                r_oc = cs["outer_r_cur"]
                fig_cs.add_trace(go.Scatter(
                    x=r_oc * np.cos(theta), y=r_oc * np.sin(theta),
                    mode="lines", name="Active Outer Burn Front",
                    line=dict(color="#00f0ff", width=2.5),
                    fill="toself", fillcolor="rgba(0, 240, 255, 0.10)",
                ))
                for idx_p, (cx, cy) in enumerate(cs["perf_centers"]):
                    r_pi = cs["perf_r_init"]
                    r_pc = cs["perf_r_cur"]
                    fig_cs.add_trace(go.Scatter(
                        x=cx + r_pi * np.cos(theta), y=cy + r_pi * np.sin(theta),
                        mode="lines", name="Initial Perf" if idx_p == 0 else None,
                        showlegend=(idx_p == 0),
                        line=dict(color="#64748b", dash="dot", width=1.0),
                    ))
                    fig_cs.add_trace(go.Scatter(
                        x=cx + r_pc * np.cos(theta), y=cy + r_pc * np.sin(theta),
                        mode="lines", name="Active Perf Burn Front" if idx_p == 0 else None,
                        showlegend=(idx_p == 0),
                        line=dict(color="#ffaa00", width=2.0),
                        fill="toself", fillcolor="rgba(20, 23, 36, 0.85)",
                    ))
                lim = r_oi * 1.15
                fig_cs.update_layout(
                    title=f"Cross-Section at u={u_vis:.2f} (Mass Burned z={z_cur*100:.1f}%)",
                    xaxis=dict(title="mm", range=[-lim, lim], scaleanchor="y", scaleratio=1),
                    yaxis=dict(title="mm", range=[-lim, lim]),
                    height=380,
                )
                st.plotly_chart(fig_cs, use_container_width=True)

            with vis_c2:
                u_arr = np.linspace(0.0, u_max_vis, 100)
                z_vals = [grain_form_function(float(uu), ib_geom)[0] * 100.0 for uu in u_arr]
                dz_vals = [grain_form_function(float(uu), ib_geom)[1] for uu in u_arr]
                fig_ff = go.Figure()
                fig_ff.add_trace(go.Scatter(x=u_arr, y=z_vals, name="Mass Burned z (%)", line=dict(color="#00ff88", width=2.5)))
                fig_ff.add_trace(go.Scatter(x=u_arr, y=dz_vals, name="Relative Surface Area dz/du", yaxis="y2", line=dict(color="#ffaa00", dash="dash", width=2.0)))
                fig_ff.add_vline(x=u_vis, line_dash="dot", line_color="#00f0ff")
                fig_ff.update_layout(
                    title="Grain Form Function z(u) & Relative Burning Area dz/du",
                    xaxis_title="Normalized Web Burn Depth u = 2·x_b / D₀",
                    yaxis=dict(title="Mass Burned z (%)", range=[0, 105]),
                    yaxis2=dict(title="Relative Burn Surface Area dz/du", overlaying="y", side="right"),
                    height=380,
                )
                st.plotly_chart(fig_ff, use_container_width=True)

        if st.button("▶ Run Interior Ballistics & Erosion Simulation", type="primary", key="btn_run_ib"):
            with st.spinner("Integrating interior ballistics ODEs and bore wall heat transfer..."):
                ib_out = simulate_ballistics(
                    force_constant_j_per_g=ib_F,
                    gamma=ib_gamma,
                    charge_mass_g=ib_charge_g,
                    propellant_density_g_per_cm3=ib_rho_p,
                    covolume_cm3_per_g=ib_covol,
                    burn_rate_a=ib_a,
                    burn_rate_n=ib_n,
                    web_thickness_mm=ib_web_mm,
                    grain_geometry=ib_geom,
                    chamber_volume_cm3=ib_vc_cm3,
                    bore_diameter_mm=ib_bore_mm,
                    barrel_length_m=ib_len_m,
                    projectile_mass_g=ib_mproj_g,
                    start_pressure_mpa=ib_pstart_mpa,
                )
                ero_out = estimate_barrel_erosion(
                    ib_out=ib_out,
                    result=st.session_state.get("result"),
                    bore_diameter_mm=ib_bore_mm,
                    charge_mass_g=ib_charge_g,
                )
                st.session_state["ib_out"] = ib_out
                st.session_state["ero_out"] = ero_out

        if "ib_out" in st.session_state:
            ib_out = st.session_state["ib_out"]
            m_ib1, m_ib2, m_ib3, m_ib4 = st.columns(4)
            m_ib1.metric("Peak Chamber Pressure", f"{ib_out['max_pressure_mpa']:.1f} MPa")
            m_ib2.metric("Muzzle Velocity", f"{ib_out['muzzle_velocity_m_s']:.1f} m/s")
            ke_kj = 0.5 * (ib_mproj_g / 1000.0) * (ib_out['muzzle_velocity_m_s'] ** 2) / 1000.0
            m_ib3.metric("Muzzle Kinetic Energy", f"{ke_kj:.1f} kJ")
            m_ib4.metric("Action Time", f"{ib_out['burn_time_ms']:.2f} ms")

            ib_g1, ib_g2 = st.columns(2)
            with ib_g1:
                fig_ib1 = go.Figure()
                fig_ib1.add_trace(go.Scatter(x=ib_out["time"] * 1000.0, y=ib_out["pressure_mpa"], name="Pressure (MPa)", line=dict(color="#00f0ff", width=2.5)))
                fig_ib1.add_trace(go.Scatter(x=ib_out["time"] * 1000.0, y=ib_out["fraction_burned"] * 100.0, name="Burned %", yaxis="y2", line=dict(color="#ffaa00", dash="dot", width=2)))
                fig_ib1.update_layout(
                    title="Chamber Pressure & Burn Fraction vs Time",
                    xaxis_title="Time (ms)",
                    yaxis=dict(title="Pressure (MPa)"),
                    yaxis2=dict(title="Propellant Burned (%)", overlaying="y", side="right", range=[0, 105]),
                    height=400,
                )
                st.plotly_chart(fig_ib1, use_container_width=True)
            with ib_g2:
                fig_ib2 = go.Figure()
                fig_ib2.add_trace(go.Scatter(x=ib_out["travel_m"], y=ib_out["velocity_m_s"], name="Velocity (m/s)", line=dict(color="#00ff88", width=2.5)))
                fig_ib2.add_trace(go.Scatter(x=ib_out["travel_m"], y=ib_out["pressure_mpa"], name="Pressure (MPa)", yaxis="y2", line=dict(color="#00f0ff", dash="dash", width=2)))
                fig_ib2.update_layout(
                    title="Projectile Velocity & Pressure vs Barrel Travel",
                    xaxis_title="Projectile Travel (m)",
                    yaxis=dict(title="Velocity (m/s)"),
                    yaxis2=dict(title="Pressure (MPa)", overlaying="y", side="right"),
                    height=400,
                )
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
                fig_ero.add_trace(go.Scatter(
                    x=ib_out["time"] * 1000.0, y=ero_out["wall_temp_k"],
                    name="Bore Wall Surface Temp (K)", line=dict(color="#ff5555", width=2.5),
                ))
                fig_ero.add_hline(y=1000.0, line_dash="dash", line_color="#ffaa00", annotation_text="Steel Austenitization Onset (~1000 K)")
                fig_ero.add_hline(y=1720.0, line_dash="dot", line_color="#ff2222", annotation_text="Gun Steel Melt Threshold (~1720 K)")
                fig_ero.update_layout(
                    title="Transient Bore Wall Surface Temperature at Origin of Rifling",
                    xaxis_title="Time (ms)",
                    yaxis_title="Wall Temperature (K)",
                    height=380,
                )
                st.plotly_chart(fig_ero, use_container_width=True)

# ===== Tab_Flash: Muzzle Flash & Plume Afterburning Predictor =====
with tab_flash:
    with st.expander("ℹ️  How to use — Muzzle Flash & Plume Predictor", expanded=False):
        st.markdown("""
        **Predict primary, intermediate (Mach disk), and secondary muzzle flash.**
        - Evaluates adiabatic gas expansion at the muzzle, normal-shock temperature recovery behind the Mach disk, and turbulent mixing of unburned $\\text{CO}, \\text{H}_2, \\text{CH}_4$ with ambient $\\text{O}_2$.
        - Models chemical radical scavenging from alkali flash-suppressant salts ($\\text{K}_2\\text{SO}_4, \\text{KNO}_3$), which elevate the $\\text{H}_2/\\text{CO}$ auto-ignition threshold in the plume shear layer.
        """)
    st.subheader("Muzzle Flash & Afterburning Plume Predictor")

    if not formulation.components:
        st.info("Define a formulation first.")
    else:
        from combusto.muzzle_flash import analyze_muzzle_flash

        mf_c1, mf_c2, mf_c3 = st.columns(3)
        with mf_c1:
            mf_exp_ratio = st.slider("Barrel Volumetric Expansion Ratio (V_barrel / V_c)", 2.0, 35.0, 12.0, 0.5, key="mf_expr")
        with mf_c2:
            mf_mach = st.slider("Muzzle Exhaust Jet Mach Number", 1.2, 4.0, 2.4, 0.1, key="mf_mach")
        with mf_c3:
            mf_supp = st.slider("Alkali Flash Suppressant Salt (K₂SO₄ / KNO₃ wt%)", 0.0, 5.0, 0.5, 0.1, key="mf_supp")

        if st.button("▶ Run Muzzle Flash & Plume Analysis", type="primary", key="btn_run_mf"):
            with st.spinner("Evaluating muzzle exhaust expansion and plume afterburning..."):
                res_for_mf = st.session_state.get("result")
                if res_for_mf is None:
                    res_for_mf = solve_equilibrium(
                        formulation, conditions, pah_config, st.session_state.species_db, eq_temp
                    )
                    st.session_state["result"] = res_for_mf
                mf_out = analyze_muzzle_flash(
                    result=res_for_mf,
                    formulation=formulation,
                    expansion_ratio=mf_exp_ratio,
                    muzzle_mach=mf_mach,
                    suppressant_wt_pct=mf_supp,
                )
                st.session_state["mf_out"] = mf_out

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
            fig_mf.add_trace(go.Scatter(
                x=plume_df["air_to_exhaust_ratio"], y=plume_df["potential_afterburn_temp_k"],
                name="Potential Fully-Ignited Plume Temp (K)",
                line=dict(color="#ffaa00", dash="dot", width=2),
            ))
            fig_mf.add_trace(go.Scatter(
                x=plume_df["air_to_exhaust_ratio"], y=plume_df["actual_plume_temp_k"],
                name="Predicted Plume Temperature (K)",
                line=dict(color="#ff4444" if mf_out["ignited"] else "#00f0ff", width=3),
            ))
            fig_mf.add_trace(go.Scatter(
                x=plume_df["air_to_exhaust_ratio"], y=plume_df["unreacted_mix_temp_k"],
                name="Unreacted Air-Exhaust Mixing Temp (K)",
                line=dict(color="#8899aa", dash="dash", width=2),
            ))
            fig_mf.add_hline(
                y=mf_out["ignition_threshold_k"],
                line_dash="dashdot", line_color="#00ff88",
                annotation_text=f"Auto-Ignition Threshold ({mf_out['ignition_threshold_k']:.0f} K)",
            )
            fig_mf.update_layout(
                title="Muzzle Plume Temperature Envelope vs Entrained Air-to-Exhaust Mass Ratio",
                xaxis_title="Entrained Air / Exhaust Mass Ratio (α = m_air / m_exhaust)",
                yaxis_title="Temperature (K)",
                height=430,
            )
            st.plotly_chart(fig_mf, use_container_width=True)

# ===== Tab_Opt: Multi-Objective Optimizer & Ternary Composition Diagrams =====
with tab_opt:
    with st.expander("ℹ️  How to use — Formulation Optimizer & Ternary Maps", expanded=False):
        st.markdown("""
        **Inverse-design propellant formulations and explore 3-component composition spaces.**
        - **Multi-Objective Optimizer**: Uses constrained SLSQP optimization to adjust ingredient mass fractions to maximize Force Constant ($F$) or Specific Impulse ($I_{sp}$) while obeying upper bounds on Adiabatic Flame Temperature ($T_{ad}$) and Soot/PAH yield.
        - **Ternary Composition Diagram**: Evaluates the complete 3-component simplex ($w_A + w_B + w_C = 1$) and plots interactive ternary contour/scatter maps.
        """)
    st.subheader("Multi-Objective Formulation Optimizer (Inverse Design)")

    if len(formulation.components) < 2:
        st.info("Add at least 2 components to the active formulation to run the optimizer.")
    else:
        from combusto.optimizer import optimize_formulation, compute_ternary_grid

        opt_c1, opt_c2, opt_c3 = st.columns(3)
        with opt_c1:
            opt_goal_map = {
                "force_constant": "Maximize Force Constant F (J/g)",
                "specific_impulse": "Maximize Specific Impulse Isp (s)",
                "min_flame_temp": "Minimize Flame Temperature T_ad (K)",
                "min_soot": "Minimize Soot / PAH Carbon Yield",
            }
            opt_goal = st.selectbox("Optimization Objective", list(opt_goal_map.keys()), format_func=lambda k: opt_goal_map[k], key="opt_goal")
        with opt_c2:
            opt_max_t = st.slider("Max Allowed Flame Temp T_ad (K)", 1800, 4200, 3000, 50, key="opt_max_t")
        with opt_c3:
            opt_max_pah_pct = st.slider("Max Allowed PAH/Soot Carbon (%)", 0.5, 25.0, 5.0, 0.5, key="opt_max_pah")

        if st.button("▶ Run Constrained Formulation Optimizer", type="primary", key="btn_run_opt"):
            with st.spinner("Optimizing ingredient mass fractions via SLSQP..."):
                opt_res = optimize_formulation(
                    base_formulation=formulation,
                    conditions=conditions,
                    pah_config=pah_config,
                    species_db=st.session_state.species_db,
                    objective_metric=opt_goal,
                    max_temp_k=float(opt_max_t),
                    max_pah_frac=float(opt_max_pah_pct) / 100.0,
                    equilibrium_temp_k=eq_temp,
                )
                st.session_state["opt_res"] = opt_res

        if "opt_res" in st.session_state:
            opt_res = st.session_state["opt_res"]
            b_res = opt_res["baseline_result"]
            o_res = opt_res["optimized_result"]

            st.success("Optimization complete!")
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Force Constant F", f"{o_res.force_constant_j_per_g:.1f} J/g", delta=f"{o_res.force_constant_j_per_g - b_res.force_constant_j_per_g:+.1f} J/g")
            oc2.metric("Adiabatic Flame Temp", f"{o_res.adiabatic_flame_temp_k:.0f} K", delta=f"{o_res.adiabatic_flame_temp_k - b_res.adiabatic_flame_temp_k:+.0f} K", delta_color="inverse")
            oc3.metric("Specific Impulse Isp", f"{o_res.specific_impulse_s:.1f} s", delta=f"{o_res.specific_impulse_s - b_res.specific_impulse_s:+.1f} s")
            b_pah = b_res.carbon_distribution.get("PAHs", 0.0) * 100.0
            o_pah = o_res.carbon_distribution.get("PAHs", 0.0) * 100.0
            oc4.metric("PAH / Soot Carbon", f"{o_pah:.2f}%", delta=f"{o_pah - b_pah:+.2f}%", delta_color="inverse")

            st.dataframe(opt_res["comparison_df"], use_container_width=True, hide_index=True)

            if st.button("✨ Apply Optimized Mass Fractions to Active Formulation", key="btn_apply_opt"):
                st.session_state.formulations[st.session_state.selected_idx] = opt_res["optimized_formulation"]
                st.session_state["result"] = o_res
                st.rerun()

    st.divider()
    st.subheader("🔺 3-Component Ternary Phase Diagram")
    if len(formulation.components) < 3:
        st.info("Add at least 3 components to the active formulation to generate a 3-component Ternary Phase Diagram.")
    else:
        from combusto.optimizer import compute_ternary_grid

        comp_names = [f"{i+1}. {c.name} ({c.formula})" for i, c in enumerate(formulation.components)]
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            idx_a = st.selectbox("Apex A", range(len(comp_names)), index=0, format_func=lambda i: comp_names[i], key="tern_a")
        with tc2:
            idx_b = st.selectbox("Apex B", range(len(comp_names)), index=min(1, len(comp_names)-1), format_func=lambda i: comp_names[i], key="tern_b")
        with tc3:
            idx_c = st.selectbox("Apex C", range(len(comp_names)), index=min(2, len(comp_names)-1), format_func=lambda i: comp_names[i], key="tern_c")
        with tc4:
            tern_metric_map = {
                "T_ad_K": "Adiabatic Flame Temp (K)",
                "force_J_per_g": "Force Constant F (J/g)",
                "Isp_s": "Specific Impulse Isp (s)",
                "PAH_carbon_frac": "PAH / Soot Carbon Fraction",
                "OB_pct": "Oxygen Balance (%)",
            }
            tern_metric = st.selectbox("Ternary Color Metric", list(tern_metric_map.keys()), format_func=lambda k: tern_metric_map[k], key="tern_met")

        if st.button("▶ Compute Ternary Simplex Map", type="primary", key="btn_run_ternary"):
            if len({idx_a, idx_b, idx_c}) < 3:
                st.error("Please select 3 distinct components for Apex A, B, and C.")
            else:
                with st.spinner("Solving equilibrium across 3-component simplex grid..."):
                    t_df = compute_ternary_grid(
                        formulation.components[idx_a],
                        formulation.components[idx_b],
                        formulation.components[idx_c],
                        conditions,
                        pah_config,
                        st.session_state.species_db,
                        n_steps=8,
                        equilibrium_temp_k=eq_temp,
                    )
                    st.session_state["ternary_df"] = t_df
                    st.session_state["ternary_labels"] = (
                        formulation.components[idx_a].name,
                        formulation.components[idx_b].name,
                        formulation.components[idx_c].name,
                    )

        if "ternary_df" in st.session_state:
            t_df = st.session_state["ternary_df"]
            lbl_a, lbl_b, lbl_c = st.session_state.get("ternary_labels", ("A", "B", "C"))
            met_label = tern_metric_map[tern_metric]

            fig_tern = go.Figure(go.Scatterternary({
                "mode": "markers",
                "a": t_df["w_A"] * 100.0,
                "b": t_df["w_B"] * 100.0,
                "c": t_df["w_C"] * 100.0,
                "marker": {
                    "symbol": "circle",
                    "size": 16,
                    "color": t_df[tern_metric],
                    "colorscale": "Turbo",
                    "showscale": True,
                    "colorbar": {"title": met_label},
                    "line": {"width": 1, "color": "#141724"},
                },
                "text": [
                    f"{lbl_a}: {r.w_A*100:.0f}%<br>{lbl_b}: {r.w_B*100:.0f}%<br>{lbl_c}: {r.w_C*100:.0f}%<br>{met_label}: {getattr(r, tern_metric):.2f}"
                    for r in t_df.itertuples()
                ],
                "hoverinfo": "text",
            }))
            fig_tern.update_layout(
                title=f"Ternary Composition Map — {met_label}",
                ternary={
                    "sum": 100,
                    "aaxis": {"title": lbl_a, "min": 0},
                    "baxis": {"title": lbl_b, "min": 0},
                    "caxis": {"title": lbl_c, "min": 0},
                },
                height=560,
            )
            st.plotly_chart(fig_tern, use_container_width=True)

# ===== Tab_CV: Closed-Vessel & Vivacity =====
with tab_cv:
    from combusto.ui_extensions import render_closed_vessel_tab
    render_closed_vessel_tab(formulation)

# ===== Tab_Tox: Plume Smoke & Range Toxicity =====
with tab_tox:
    from combusto.ui_extensions import render_smoke_toxicity_tab
    render_smoke_toxicity_tab(formulation, conditions, pah_config, eq_temp)

# ===== Tab_Stab: Propellant Shelf-Life & Thermal Safety =====
with tab_stab:
    from combusto.ui_extensions import render_stability_tab
    render_stability_tab(formulation)

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


# ===== Tab_Grounding: Experimental Benchmarks & Literature Provenance =====
with tab_grounding:
    from combusto.ui_extensions import render_benchmarks_and_grounding_tab
    render_benchmarks_and_grounding_tab()
