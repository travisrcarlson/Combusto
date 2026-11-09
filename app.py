"""Combusto — Gun & Rocket Propellant Thermochemical & Ballistic Engineering Suite.

Hosted web application entry point.
Run with: streamlit run app.py
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import streamlit as st

from combusto.workspace_ui import (
    check_web_password,
    inject_cyber_styles,
    render_main_workbench,
)

st.set_page_config(
    page_title="Combusto",
    page_icon=":fire:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_cyber_styles()

st.title("Combusto")
st.caption("Gun & Rocket Propellant Thermochemical, Ballistic & Environmental Engineering Suite")

with st.sidebar.expander("🛠️ Developer Pro Portal", expanded=False):
    dev_pass = st.text_input("Enter Developer Password:", type="password", key="dev_pass")
    if dev_pass == "Freya":
        st.success("Authorized. Local Desktop Pro Build Unlocked.")

        def generate_zip() -> io.BytesIO:
            memory_file = io.BytesIO()
            base_dir = Path(__file__).parent
            with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _dirs, files in os.walk(base_dir):
                    if any(skip in root for skip in (".git", "__pycache__", ".pytest_cache")):
                        continue
                    for file in files:
                        if (
                            file.endswith(".pyc")
                            or file in (".env", "secrets.toml")
                            or ".sqlite" in file
                        ):
                            continue
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(base_dir)
                        zf.write(file_path, arcname)
            memory_file.seek(0)
            return memory_file

        st.download_button(
            label="📥 Download Combusto Desktop (.zip)",
            data=generate_zip(),
            file_name="Combusto_Pro_Desktop.zip",
            mime="application/zip",
            type="primary",
        )
    elif dev_pass:
        st.error("Incorrect password.")

if not check_web_password():
    st.stop()

render_main_workbench(extra_editor_hook=None)
