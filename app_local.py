"""Combusto Desktop Pro — Local Propellant Engineering Workbench & Database Admin.

Local desktop entry point with Master Database Editor, API Key Manager, and
live PubChem / NIST Chemistry WebBook compound lookup.
Run with: streamlit run app_local.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import toml

from combusto.lookup import query_nist_enthalpy, query_pubchem
from combusto.models import Formulation, PropellantComponent
from combusto.species_db import load_species_db
from combusto.workspace_ui import (
    check_web_password,
    inject_cyber_styles,
    render_main_workbench,
)

st.set_page_config(
    page_title="Combusto Desktop Pro",
    page_icon=":fire:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_cyber_styles()

st.sidebar.markdown("## ⚙️ Combusto Local Admin")
admin_action = st.sidebar.radio(
    "Admin Mode",
    ["Main Workbench", "Database Editor", "API Key Manager"],
)

if admin_action == "Database Editor":
    st.title("Combusto Admin — 🗄️ Master Database Editor")
    st.markdown("Directly inspect and edit empirical thermochemical JSON libraries on local disk.")

    db_choice = st.selectbox(
        "Select Database File",
        ["data/ingredients.json", "data/propellants.json", "data/species.json"],
    )
    db_path = Path(__file__).parent / db_choice
    if db_path.exists():
        with open(db_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        root_key = list(raw_data.keys())[-1]
        df = pd.DataFrame(raw_data[root_key])
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, height=600)

        if st.button(f"💾 Save {db_choice} to Disk", type="primary"):
            raw_data[root_key] = edited_df.to_dict(orient="records")
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2)
            if "species_db" in st.session_state:
                st.session_state.species_db = load_species_db()
            st.success(f"Successfully saved {db_choice}")
    else:
        st.error(f"Database file not found: {db_path}")
    st.stop()

elif admin_action == "API Key Manager":
    st.title("Combusto Admin — 🔑 Local API Key Manager")
    st.markdown("Configure API keys stored locally in `.streamlit/secrets.toml` for external chemical services.")

    secrets_dir = Path(__file__).parent / ".streamlit"
    secrets_file = secrets_dir / "secrets.toml"
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
            "PUBCHEM_API_KEY": pub_key,
        }
        with open(secrets_file, "w", encoding="utf-8") as f:
            toml.dump(new_secrets, f)
        st.success("Secrets committed to local `.streamlit/secrets.toml`!")
    st.stop()


def _render_local_pubchem_nist_lookup(formulation: Formulation) -> None:
    """Local Desktop Pro hook for live PubChem PUG REST & NIST WebBook lookup."""
    idx = st.session_state.selected_idx
    with st.expander("🔍 External API Chemical Lookup (PubChem & NIST WebBook — Local Pro)", expanded=False):
        api_col1, api_col2 = st.columns([3, 1])
        with api_col1:
            search_query = st.text_input(
                "Search compound by name or SMILES (e.g. 'RDX' or 'O=[N+]([O-])N1CN(...)')",
                key=f"api_search_query_{idx}",
            )
        with api_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_search = st.button("Search API", key=f"run_api_search_{idx}", use_container_width=True)

        if search_query and run_search:
            is_smiles = (
                any(c in search_query for c in ["=", "#", "@", "\\", "/", "[", "]"])
                or (len(search_query) > 12 and any(c.islower() for c in search_query))
            )
            search_type = "smiles" if is_smiles else "name"
            with st.spinner("Querying PubChem PUG REST..."):
                pubchem_res = query_pubchem(search_query, search_type)

            if pubchem_res:
                st.success(f"Found in PubChem: **{pubchem_res['iupac_name'] or pubchem_res['name']}**")
                st.write(f"- Formula: `{pubchem_res['formula']}`  |  MW: `{pubchem_res['mw']:.2f} g/mol`")
                with st.spinner("Querying NIST Chemistry WebBook..."):
                    nist_hf = query_nist_enthalpy(pubchem_res["formula"])
                    if nist_hf is None:
                        nist_hf = query_nist_enthalpy(pubchem_res["name"])
                if nist_hf is not None:
                    st.write(f"- NIST ΔHf,298: `{nist_hf:.1f} kJ/mol`")
                    hf_val = nist_hf
                else:
                    st.warning("Could not retrieve ΔHf,298 from NIST WebBook — enter manually after injection.")
                    hf_val = 0.0
                st.session_state[f"api_lookup_res_{idx}"] = {
                    "name": pubchem_res["name"],
                    "formula": pubchem_res["formula"],
                    "hf_298": hf_val,
                }
            else:
                st.error("Compound not found on PubChem. Check spelling or SMILES syntax.")
                st.session_state.pop(f"api_lookup_res_{idx}", None)

        lookup_key = f"api_lookup_res_{idx}"
        if lookup_key in st.session_state:
            cached_res = st.session_state[lookup_key]
            st.info(f"Buffered Compound: **{cached_res['name']}** ({cached_res['formula']} | {cached_res['hf_298']:.1f} kJ/mol)")
            if st.button("➕ Inject into Component Table", key=f"inject_api_res_{idx}", use_container_width=True):
                formulation.components.append(PropellantComponent(
                    name=cached_res["name"],
                    formula=cached_res["formula"],
                    mass_fraction=0.0,
                    hf_298=cached_res["hf_298"],
                ))
                st.session_state.formulations[idx] = formulation
                del st.session_state[lookup_key]
                st.rerun()


st.title("Combusto")
st.caption("Gun & Rocket Propellant Thermochemical, Ballistic & Environmental Engineering Suite — Desktop Pro")

if not check_web_password():
    st.stop()

render_main_workbench(extra_editor_hook=_render_local_pubchem_nist_lookup)
