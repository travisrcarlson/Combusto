"""Streamlit AppTest end-to-end UI smoke tests for web (`app.py`) and Desktop Pro (`app_local.py`)."""

from __future__ import annotations

import logging
from pathlib import Path
import warnings

from streamlit.testing.v1 import AppTest

from combusto.workspace_ui import WORKSPACES


REPO_ROOT = Path(__file__).resolve().parent.parent


def _collect_deprecation_warnings(rec_warnings, caplog_records) -> list[str]:
    bad: list[str] = []
    for w in rec_warnings:
        msg = str(w.message)
        if "use_container_width" in msg:
            bad.append(msg)
    for rec in caplog_records:
        msg = rec.getMessage()
        if "use_container_width" in msg:
            bad.append(msg)
    return bad


def test_web_app_navigation_sweeps_and_dev_portal(monkeypatch, caplog):
    """Verify web app login, all 5 workspaces, 2D sweep axis switch, Dev Portal, and zero deprecations."""
    monkeypatch.chdir(REPO_ROOT)
    caplog.set_level(logging.WARNING)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=45)
        at.run()
        assert not at.exception
        # Must stop on password gate before rendering workbench radio
        assert len(at.radio) == 0

        # Unlock with web password
        at.text_input(key="password").input("combusto2026").run()
        assert not at.exception
        assert len(at.radio) >= 1

        # Cycle across all 5 engineering workspaces
        for ws in WORKSPACES:
            at.radio(key="active_workspace_nav").set_value(ws).run()
            assert not at.exception, f"UI exception in workspace {ws}: {at.exception}"

        # In Workspace 4 (Design, Sensitivity & Surveillance), exercise 2D sweep axis switcher
        at.radio(key="active_workspace_nav").set_value(WORKSPACES[3]).run()
        assert not at.exception
        at.selectbox(key="sens_type_sel").set_value("2D Sensitivity sweep").run()
        assert not at.exception

        for axis_label in ("Temperature (K)", "Pressure (atm)", "Equivalence Ratio (φ)"):
            at.selectbox(key="sa2d_v1").set_value(axis_label).run()
            assert not at.exception, f"Duplicate widget key failure switching X axis to {axis_label}"

        # Developer Pro Portal must unlock Desktop .zip download button
        at.sidebar.text_input(key="dev_pass").input("Freya").run()
        assert not at.exception
        assert len(at.sidebar.success) >= 1

    bad = _collect_deprecation_warnings(rec, caplog.records)
    assert not bad, f"Deprecated use_container_width warnings detected: {bad}"


def test_local_app_admin_auth_and_views(monkeypatch, caplog):
    """Verify Desktop Pro admin sidebar is hidden until auth and Database/API Key views load cleanly."""
    monkeypatch.chdir(REPO_ROOT)
    caplog.set_level(logging.WARNING)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        at = AppTest.from_file(str(REPO_ROOT / "app_local.py"), default_timeout=45)
        at.run()
        assert not at.exception
        # Before login, Admin Mode sidebar radio MUST NOT be exposed
        assert len(at.sidebar.radio) == 0

        # Unlock
        at.text_input(key="password").input("combusto2026").run()
        assert not at.exception
        assert len(at.sidebar.radio) == 1

        for view_name in ("Database Editor", "API Key Manager", "Main Workbench"):
            at.sidebar.radio[0].set_value(view_name).run()
            assert not at.exception, f"Desktop Pro view {view_name} raised exception: {at.exception}"

    bad = _collect_deprecation_warnings(rec, caplog.records)
    assert not bad, f"Deprecated use_container_width warnings detected in app_local.py: {bad}"
