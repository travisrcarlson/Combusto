"""Combusto Desktop Pro — Local Propellant Engineering Workbench & Database Admin.

Local desktop entry point with Master Database Editor, API Key Manager, and
live PubChem / NIST Chemistry WebBook compound lookup.
Run with: streamlit run app_local.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import streamlit as st

try:
    import tomllib  # Python 3.11+ stdlib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

from combusto.lookup import query_nist_enthalpy, query_pubchem
from combusto.models import Formulation, PropellantComponent
from combusto.species_db import load_species_db
from combusto.workspace_ui import (
    check_web_password,
    inject_cyber_styles,
    render_main_workbench,
)


def load_local_secrets_toml(secrets_file: Path) -> Dict[str, str]:
    """Read key-value string pairs from `.streamlit/secrets.toml` using stdlib only."""
    if not secrets_file.is_file():
        return {}
    raw_text = secrets_file.read_text(encoding="utf-8")
    if tomllib is not None:
        try:
            parsed = tomllib.loads(raw_text)
            return {str(k): str(v) for k, v in parsed.items() if isinstance(v, (str, int, float, bool))}
        except Exception:
            pass

    fallback: Dict[str, str] = {}
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            try:
                v = json.loads(v) if v[0] == '"' else v[1:-1]
            except Exception:
                v = v[1:-1]
        fallback[k] = v
    return fallback


def save_local_secrets_toml(secrets_file: Path, secrets: Dict[str, str]) -> None:
    """Write key-value string pairs to `.streamlit/secrets.toml` without external dependencies."""
    secrets_file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Combusto Desktop Pro Local API Secrets (strictly offline)\n"]
    for key, val in secrets.items():
        lines.append(f"{key} = {json.dumps(str(val or ''))}\n")
    secrets_file.write_text("".join(lines), encoding="utf-8")


def prepare_records_for_editor(records: List[Dict[str, Any]]) -> tuple[pd.DataFrame, Set[str], Dict[str, Any]]:
    """Serialize nested list/dict columns to JSON strings so `st.data_editor` preserves them cleanly."""
    nested_cols: Set[str] = set()
    col_defaults: Dict[str, Any] = {}
    for rec in records:
        for col, val in rec.items():
            if isinstance(val, (list, dict)):
                nested_cols.add(col)
                col_defaults.setdefault(col, [] if isinstance(val, list) else {})
            elif val is not None and col not in col_defaults:
                if isinstance(val, bool):
                    col_defaults[col] = False
                elif isinstance(val, int):
                    col_defaults[col] = 0
                elif isinstance(val, float):
                    col_defaults[col] = 0.0
                else:
                    col_defaults[col] = ""

    serialized_rows: List[Dict[str, Any]] = []
    for rec in records:
        row_copy = dict(rec)
        for col in nested_cols:
            if col in row_copy and row_copy[col] is not None:
                row_copy[col] = json.dumps(row_copy[col], ensure_ascii=False)
            else:
                row_copy[col] = json.dumps(col_defaults.get(col, []))
        serialized_rows.append(row_copy)

    return pd.DataFrame(serialized_rows), nested_cols, col_defaults


def restore_records_from_editor(
    edited_df: pd.DataFrame,
    nested_cols: Set[str],
    col_defaults: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deserialize JSON columns and sanitize NaN values from `st.data_editor` back into strict JSON records."""
    clean_records: List[Dict[str, Any]] = []
    for row_idx, raw_row in enumerate(edited_df.to_dict(orient="records"), start=1):
        name_val = raw_row.get("name")
        if name_val is None or (isinstance(name_val, float) and math.isnan(name_val)) or not str(name_val).strip():
            # Skip empty trailing rows added in dynamic editor
            continue

        restored: Dict[str, Any] = {}
        for col, val in raw_row.items():
            is_missing = val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val)
            if col in nested_cols:
                if is_missing or not str(val).strip():
                    restored[col] = col_defaults.get(col, [])
                elif isinstance(val, (list, dict)):
                    restored[col] = val
                else:
                    try:
                        parsed_val = json.loads(str(val))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Row {row_idx} ('{name_val}'), column '{col}' contains invalid JSON: {exc}") from exc
                    if not isinstance(parsed_val, (list, dict)):
                        raise ValueError(f"Row {row_idx} ('{name_val}'), column '{col}' must decode to a JSON array or object.")
                    restored[col] = parsed_val
            else:
                if is_missing:
                    default_v = col_defaults.get(col, "")
                    restored[col] = default_v
                elif isinstance(val, (int, float, bool, str)):
                    restored[col] = val
                else:
                    restored[col] = val.item() if hasattr(val, "item") else str(val)
        clean_records.append(restored)

    return clean_records


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


def _main() -> None:
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
        db_path = Path(__file__).resolve().parent / db_choice
        if db_path.exists():
            with open(db_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            root_key = list(raw_data.keys())[-1]
            df_display, nested_cols, col_defaults = prepare_records_for_editor(raw_data[root_key])
            if nested_cols:
                st.caption(
                    f"Structured array/object columns ({', '.join(sorted(nested_cols))}) are displayed as editable JSON strings and validated prior to saving."
                )
            edited_df = st.data_editor(
                df_display,
                num_rows="dynamic",
                use_container_width=True,
                height=600,
                key=f"db_editor_{db_choice}",
            )

            if st.button(f"💾 Save {db_choice} to Disk", type="primary"):
                try:
                    clean_records = restore_records_from_editor(edited_df, nested_cols, col_defaults)
                except ValueError as exc:
                    st.error(f"Validation error — file NOT modified: {exc}")
                else:
                    raw_data[root_key] = clean_records
                    with open(db_path, "w", encoding="utf-8") as f:
                        json.dump(raw_data, f, indent=2, ensure_ascii=False, allow_nan=False)
                    st.cache_data.clear()
                    if "species_db" in st.session_state:
                        st.session_state.species_db = load_species_db()
                    st.success(f"Successfully validated and saved {len(clean_records)} records to {db_choice}")
        else:
            st.error(f"Database file not found: {db_path}")
        st.stop()

    elif admin_action == "API Key Manager":
        st.title("Combusto Admin — 🔑 Local API Key Manager")
        st.markdown("Configure API keys stored locally in `.streamlit/secrets.toml` for external chemical services.")

        secrets_file = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
        current_secrets = load_local_secrets_toml(secrets_file)

        nist_key = st.text_input("NIST WebBook API Key", value=current_secrets.get("NIST_API_KEY", ""), type="password")
        mat_key = st.text_input("Materials Project API Key", value=current_secrets.get("MATERIALSPROJECT_API_KEY", ""), type="password")
        pub_key = st.text_input("PubChem PUG REST Auth Token", value=current_secrets.get("PUBCHEM_API_KEY", ""), type="password")

        if st.button("🔒 Save Environment Secrets", type="primary"):
            new_secrets = {
                "NIST_API_KEY": nist_key,
                "MATERIALSPROJECT_API_KEY": mat_key,
                "PUBCHEM_API_KEY": pub_key,
            }
            save_local_secrets_toml(secrets_file, new_secrets)
            st.success("Secrets committed to local `.streamlit/secrets.toml`!")
        st.stop()

    st.title("Combusto")
    st.caption("Gun & Rocket Propellant Thermochemical, Ballistic & Environmental Engineering Suite — Desktop Pro")

    if not check_web_password():
        st.stop()

    render_main_workbench(extra_editor_hook=_render_local_pubchem_nist_lookup)


if __name__ == "__main__" or st.runtime.exists():
    _main()
