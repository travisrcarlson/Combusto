"""Combusto — Gun & Rocket Propellant Thermochemical & Ballistic Engineering Suite.

Hosted web application entry point.
Run with: streamlit run app.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import streamlit as st

from combusto.workspace_ui import (
    check_web_password,
    inject_cyber_styles,
    render_main_workbench,
)

_ALLOWED_ROOT_FILES = (
    "app.py",
    "app_local.py",
    "requirements.txt",
    "README.md",
    "INSTRUCTIONS.md",
)
_ALLOWED_DATA_FILES = (
    "ingredients.json",
    "propellants.json",
    "species.json",
)


@st.cache_data(show_spinner=False)
def build_desktop_zip_bytes() -> bytes:
    """Build a clean, allowlisted Combusto Desktop Pro archive in memory.

    Strictly packages source modules, grounded JSON databases, unit tests, and
    documentation while excluding local tunnel binaries, logs, caches, and secrets.
    """
    base_dir = Path(__file__).resolve().parent
    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in _ALLOWED_ROOT_FILES:
            fpath = base_dir / fname
            if fpath.is_file():
                zf.write(fpath, arcname=fname)

        combusto_dir = base_dir / "combusto"
        if combusto_dir.is_dir():
            for py_file in sorted(combusto_dir.glob("*.py")):
                zf.write(py_file, arcname=f"combusto/{py_file.name}")

        data_dir = base_dir / "data"
        if data_dir.is_dir():
            for dname in _ALLOWED_DATA_FILES:
                dpath = data_dir / dname
                if dpath.is_file():
                    zf.write(dpath, arcname=f"data/{dname}")

        tests_dir = base_dir / "tests"
        if tests_dir.is_dir():
            for test_file in sorted(tests_dir.glob("*.py")):
                zf.write(test_file, arcname=f"tests/{test_file.name}")

    return memory_file.getvalue()


def _main() -> None:
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
            st.download_button(
                label="📥 Download Combusto Desktop (.zip)",
                data=build_desktop_zip_bytes(),
                file_name="Combusto_Pro_Desktop.zip",
                mime="application/zip",
                type="primary",
            )
        elif dev_pass:
            st.error("Incorrect password.")

    if not check_web_password():
        st.stop()

    render_main_workbench(extra_editor_hook=None)


if __name__ == "__main__" or st.runtime.exists():
    _main()
