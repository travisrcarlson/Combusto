"""Streamlit UI extension tabs for Combusto.

Provides modular rendering functions shared by both `app.py` and `app_local.py` for:
1. Noble-Abel Real-Gas vs. Ideal Polytropic Expansion Isentrope
2. Closed-Vessel (Manometric Bomb) Vivacity & Burn-Rate Calibration Suite
3. Tactical Plume Smoke (AGARD / STANAG) & Range Toxicity (OSHA/NIOSH) Analyzer
4. Propellant Thermal Stability, STANAG 4582 Shelf-Life & Cook-Off Safety Analyzer
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from .cache_helpers import cached_solve_equilibrium, cached_solve_equilibrium_adiabatic
from .models import CombustionConditions, Formulation, PAHAllocationConfig
from .thermo_cycles import compute_expansion_isentrope
from .closed_vessel import simulate_closed_vessel, calibrate_abel_eos_and_burn_rate
from .toxicity_smoke import analyze_toxicity_and_smoke
from .stability import analyze_thermal_stability, detect_stabilizer_wt_pct


def get_formulation_suite(form_name: str) -> Dict[str, Any]:
    """Return the isolated per-formulation suite cache dictionary in `st.session_state`."""
    suite_cache: Dict[str, Dict[str, Any]] = st.session_state.setdefault("suite_cache", {})
    if form_name not in suite_cache:
        suite_cache[form_name] = {}
    return suite_cache[form_name]


def rename_formulation_state(old_name: str, new_name: str) -> None:
    """Preserve per-formulation equilibrium results and suite outputs across a name edit."""
    if not old_name or old_name == new_name:
        return
    results = st.session_state.setdefault("results", {})
    if old_name in results and new_name not in results:
        results[new_name] = results.pop(old_name)
    suite_cache = st.session_state.setdefault("suite_cache", {})
    if old_name in suite_cache and new_name not in suite_cache:
        suite_cache[new_name] = suite_cache.pop(old_name)


def render_noble_abel_isentrope_section(
    formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    eq_temp: float,
) -> None:
    """Render the Real-Gas Noble-Abel vs. Ideal Expansion Isentrope section."""
    st.divider()
    st.subheader("📈 Noble-Abel Real-Gas vs. Ideal Polytropic Expansion Isentrope")
    if not formulation.components:
        st.info("Define a formulation first.")
        return

    suite = get_formulation_suite(formulation.name)
    res_iso_src = suite.get("res_perf_cached") or st.session_state.get("results", {}).get(formulation.name)
    default_iso_eta = 1.00
    if res_iso_src is not None and getattr(res_iso_src, "covolume_cm3_per_g", 0.0) > 0.3:
        default_iso_eta = float(min(1.60, max(0.40, round(res_iso_src.covolume_cm3_per_g, 2))))

    iso_c1, iso_c2, iso_c3 = st.columns(3)
    with iso_c1:
        iso_delta = st.slider(
            "Initial Loading Density Δ (g/cm³)", 0.05, 0.55, 0.22, 0.01, key="iso_delta"
        )
    with iso_c2:
        iso_eta = st.slider(
            "Gas Covolume η (cm³/g, BLAKE Σnᵢbᵢ)",
            0.40,
            1.60,
            default_iso_eta,
            0.02,
            key=f"iso_eta_{formulation.name}",
        )
    with iso_c3:
        iso_vmax = st.slider(
            "Max Expansion Ratio (V / V₀)", 4.0, 30.0, 15.0, 1.0, key="iso_vmax"
        )
    if res_iso_src is None and st.button("▶ Compute Expansion Isentrope", type="primary", key="btn_iso_calc"):
        if st.session_state.get("active_mode_label") == "adiabatic (self-consistent)":
            res_iso_src = cached_solve_equilibrium_adiabatic(
                formulation, conditions, pah_config, st.session_state.species_db, 2500.0
            )
        else:
            res_iso_src = cached_solve_equilibrium(
                formulation, conditions, pah_config, st.session_state.species_db, eq_temp
            )
        st.session_state.setdefault("results", {})[formulation.name] = res_iso_src

    if res_iso_src is not None:
        iso_out = compute_expansion_isentrope(
            result=res_iso_src,
            initial_loading_density_g_cm3=iso_delta,
            covolume_cm3_per_g=iso_eta,
            max_expansion_ratio=iso_vmax,
        )
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Initial Noble-Abel Pressure", f"{iso_out['p0_noble_abel_mpa']:.1f} MPa")
        ic2.metric("Real-Gas Compressibility Z₀", f"{iso_out['z0_compressibility']:.3f}")
        ic3.metric("Reversible Expansion Work", f"{iso_out['max_work_noble_j_g']:.1f} J/g")
        ic4.metric("Initial Sound Speed a₀", f"{iso_out['initial_sound_speed_m_s']:.0f} m/s")

        iso_df = iso_out["isentrope_df"]
        ig1, ig2 = st.columns(2)
        with ig1:
            fig_pv = go.Figure()
            fig_pv.add_trace(go.Scatter(
                x=iso_df["volume_ratio_v_v0"], y=iso_df["p_noble_abel_mpa"],
                name="Noble-Abel EOS P(V)", line=dict(color="#00f0ff", width=2.5),
            ))
            fig_pv.add_trace(go.Scatter(
                x=iso_df["volume_ratio_v_v0"], y=iso_df["p_ideal_gas_mpa"],
                name="Ideal Gas EOS P(V)", line=dict(color="#8899aa", dash="dash", width=2),
            ))
            fig_pv.update_layout(
                title="Adiabatic Expansion Isentrope (Log P vs V/V₀)",
                xaxis_title="Expansion Ratio V / V₀",
                yaxis=dict(title="Pressure (MPa)", type="log"),
                height=380,
            )
            st.plotly_chart(fig_pv, use_container_width=True)
        with ig2:
            fig_tw = go.Figure()
            fig_tw.add_trace(go.Scatter(
                x=iso_df["volume_ratio_v_v0"], y=iso_df["cumulative_work_noble_j_g"],
                name="Cumulative Work ∫P dV (J/g)", line=dict(color="#00ff88", width=2.5),
            ))
            fig_tw.add_trace(go.Scatter(
                x=iso_df["volume_ratio_v_v0"], y=iso_df["t_noble_abel_k"],
                name="Gas Temperature T (K)", yaxis="y2", line=dict(color="#ffaa00", dash="dot", width=2),
            ))
            fig_tw.update_layout(
                title="Cumulative Expansion Work & Temperature Drop vs V/V₀",
                xaxis_title="Expansion Ratio V / V₀",
                yaxis=dict(title="Cumulative Expansion Work (J/g)"),
                yaxis2=dict(title="Gas Temperature (K)", overlaying="y", side="right"),
                height=380,
            )
            st.plotly_chart(fig_tw, use_container_width=True)


def render_closed_vessel_tab(formulation: Formulation) -> None:
    """Render the Closed-Vessel (Manometric Bomb) Vivacity & Calibration Suite."""
    with st.expander("ℹ️  How to use — Closed-Vessel (Manometric Bomb) & Vivacity Suite", expanded=False):
        st.markdown(r"""
        **Simulate NATO STANAG 4115 closed-vessel tests and calibrate Noble-Abel & Saint-Robert parameters.**
        - **Closed-Bomb Simulator**: Predicts constant-volume pressure $P(t)$, pressurization rate $dP/dt$, and **Dynamic Vivacity** $L(P/P_{\max}) = \frac{1}{P \cdot P_{\max}}\frac{dP}{dt}$.
        - **Abel EOS & Burn Rate Calibrator**: Fits Force Constant $F$ and Covolume $\eta$ from multi-density peak pressures ($\frac{P_{\max}}{\Delta} = F + \eta P_{\max}$) and fits $r = a P^n$ via log-log regression.
        """)
    st.subheader("Closed-Vessel (Manometric Bomb) Dynamic Vivacity Simulator")

    suite = get_formulation_suite(formulation.name)
    res_cur = st.session_state.get("results", {}).get(formulation.name)
    default_F = 1050.0
    default_eta = 1.00
    if res_cur is not None:
        if res_cur.force_constant_j_per_g > 0:
            default_F = float(res_cur.force_constant_j_per_g)
        if getattr(res_cur, "covolume_cm3_per_g", 0.0) > 0.3:
            default_eta = float(min(1.80, max(0.40, round(res_cur.covolume_cm3_per_g, 3))))

    cv1, cv2, cv3 = st.columns(3)
    with cv1:
        cv_F = st.number_input("Force Constant F (J/g)", 200.0, 2500.0, default_F, 10.0, key=f"cv_F_{formulation.name}")
        cv_eta = st.number_input("Gas Covolume η (cm³/g, BLAKE Σnᵢbᵢ)", 0.40, 1.80, default_eta, 0.02, key=f"cv_eta_{formulation.name}")
        cv_delta = st.number_input("Loading Density Δ (g/cm³)", 0.05, 0.45, 0.20, 0.02, key="cv_delta")
    with cv2:
        geom_labels = {
            "7_perf": "7-Perforated Cylinder (Progressive)",
            "19_perf": "19-Perforated Rosette (Highly Progressive)",
            "single_perf": "Single-Perforated Tube (Neutral)",
            "cord": "Cylindrical Cord (Regressive)",
            "spherical": "Spherical Ball (Regressive)",
        }
        cv_geom = st.selectbox("Grain Geometry", list(geom_labels.keys()), format_func=lambda g: geom_labels[g], key="cv_geom")
        cv_web = st.number_input("Web Thickness D₀ (mm)", 0.1, 15.0, 1.20, 0.1, key="cv_web")
        cv_loss = st.slider("Wall Heat Loss Fraction", 0.0, 0.20, 0.03, 0.01, key="cv_loss")
    with cv3:
        cv_a = st.number_input("Burn Rate Coeff a (mm/s/MPaⁿ)", 0.05, 20.0, 1.25, 0.05, key="cv_a")
        cv_n = st.number_input("Pressure Exponent n", 0.30, 1.20, 0.85, 0.02, key="cv_n")
        cv_rhop = st.number_input("Propellant Solid Density (g/cm³)", 0.90, 2.20, 1.60, 0.05, key="cv_rhop")

    if st.button("▶ Run Closed-Vessel Bomb Simulation", type="primary", key="btn_run_cv"):
        cv_out = simulate_closed_vessel(
            force_constant_j_per_g=cv_F,
            covolume_cm3_per_g=cv_eta,
            propellant_density_g_per_cm3=cv_rhop,
            loading_density_g_per_cm3=cv_delta,
            burn_rate_a=cv_a,
            burn_rate_n=cv_n,
            web_thickness_mm=cv_web,
            grain_geometry=cv_geom,
            heat_loss_fraction=cv_loss,
        )
        suite["cv_out"] = cv_out

    if "cv_out" in suite:
        cv_out = suite["cv_out"]
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Peak Bomb Pressure P_max", f"{cv_out['p_max_mpa']:.1f} MPa")
        cm2.metric("Peak dP/dt", f"{cv_out['peak_dp_dt_gpa_s']:.2f} GPa/s")
        cm3.metric("Mean Vivacity (30–70% P_max)", f"{cv_out['mean_vivacity_30_70']:.3f} MPa⁻¹·s⁻¹")
        cm4.metric("Burnout Time", f"{cv_out['burnout_time_ms']:.2f} ms")

        tdf = cv_out["trace_df"]
        cg1, cg2 = st.columns(2)
        with cg1:
            fig_cv1 = go.Figure()
            fig_cv1.add_trace(go.Scatter(
                x=tdf["time_ms"], y=tdf["pressure_mpa"],
                name="Bomb Pressure P(t) (MPa)", line=dict(color="#00f0ff", width=2.5),
            ))
            fig_cv1.add_trace(go.Scatter(
                x=tdf["time_ms"], y=tdf["dp_dt_gpa_s"],
                name="dP/dt (GPa/s)", yaxis="y2", line=dict(color="#ffaa00", dash="dot", width=2),
            ))
            fig_cv1.update_layout(
                title="Closed-Vessel Pressure P(t) & dP/dt vs Time",
                xaxis_title="Time (ms)",
                yaxis=dict(title="Pressure (MPa)"),
                yaxis2=dict(title="dP/dt (GPa/s)", overlaying="y", side="right"),
                height=390,
            )
            st.plotly_chart(fig_cv1, use_container_width=True)
        with cg2:
            fig_cv2 = go.Figure()
            fig_cv2.add_trace(go.Scatter(
                x=tdf["p_over_pmax"], y=tdf["vivacity_per_mpa_s"],
                name="Dynamic Vivacity L", line=dict(color="#00ff88", width=2.5),
            ))
            fig_cv2.add_vrect(
                x0=0.30, x1=0.70, fillcolor="rgba(0, 240, 255, 0.08)",
                line_width=0, annotation_text="STANAG 30–70% Window",
            )
            fig_cv2.update_layout(
                title="Dynamic Vivacity L = (1 / (P·P_max))·(dP/dt) vs P / P_max",
                xaxis_title="Normalized Pressure P / P_max",
                yaxis_title="Dynamic Vivacity L (MPa⁻¹·s⁻¹)",
                height=390,
            )
            st.plotly_chart(fig_cv2, use_container_width=True)

    st.divider()
    st.subheader("📐 Experimental Abel EOS (F, η) & Burn Rate (a, n) Linear Calibrator")
    st.caption("Edit the multi-density closed-vessel test table below to fit Force Constant F, Covolume η, and Saint-Robert's law r = a·Pⁿ.")
    default_cal_df = pd.DataFrame([
        {"Loading Density Δ (g/cm³)": 0.10, "Peak Pressure P_max (MPa)": 118.5, "Burn Pressure P (MPa)": 50.0, "Measured Burn Rate r (mm/s)": 34.5},
        {"Loading Density Δ (g/cm³)": 0.15, "Peak Pressure P_max (MPa)": 192.0, "Burn Pressure P (MPa)": 100.0, "Measured Burn Rate r (mm/s)": 62.2},
        {"Loading Density Δ (g/cm³)": 0.20, "Peak Pressure P_max (MPa)": 278.5, "Burn Pressure P (MPa)": 175.0, "Measured Burn Rate r (mm/s)": 99.8},
        {"Loading Density Δ (g/cm³)": 0.25, "Peak Pressure P_max (MPa)": 382.0, "Burn Pressure P (MPa)": 260.0, "Measured Burn Rate r (mm/s)": 140.1},
    ])
    edited_cal_df = st.data_editor(default_cal_df, num_rows="dynamic", use_container_width=True, key="cv_cal_editor")

    if st.button("▶ Fit Noble-Abel (F, η) & Saint-Robert (a, n)", type="primary", key="btn_fit_cv"):
        if len(edited_cal_df) < 2:
            st.error("At least 2 rows are required for linear regression.")
        else:
            fit_out = calibrate_abel_eos_and_burn_rate(
                loading_densities_g_cm3=edited_cal_df["Loading Density Δ (g/cm³)"].tolist(),
                peak_pressures_mpa=edited_cal_df["Peak Pressure P_max (MPa)"].tolist(),
                burn_pressures_mpa=edited_cal_df["Burn Pressure P (MPa)"].tolist(),
                burn_rates_mm_s=edited_cal_df["Measured Burn Rate r (mm/s)"].tolist(),
            )
            st.session_state["cv_fit_out"] = fit_out

    if "cv_fit_out" in st.session_state:
        fit_out = st.session_state["cv_fit_out"]
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Fitted Force Constant F", f"{fit_out['fitted_F_j_per_g']:.1f} J/g", delta=f"R² = {fit_out['r2_abel']:.4f}")
        fc2.metric("Fitted Covolume η", f"{fit_out['fitted_covolume_cm3_per_g']:.3f} cm³/g")
        fc3.metric("Fitted Burn Rate Coeff a", f"{fit_out['fitted_a_mm_s_mpa_n']:.3f} mm/s/MPaⁿ", delta=f"R² = {fit_out['r2_burn']:.4f}")
        fc4.metric("Fitted Pressure Exponent n", f"{fit_out['fitted_n']:.3f}")

        fg1, fg2 = st.columns(2)
        with fg1:
            adf = fit_out["abel_df"]
            fig_ab = go.Figure()
            fig_ab.add_trace(go.Scatter(
                x=adf["p_max_mpa"], y=adf["p_over_delta_j_g"],
                mode="markers", name="Measured P_max / Δ",
                marker=dict(size=10, color="#00f0ff"),
            ))
            fig_ab.add_trace(go.Scatter(
                x=adf["p_max_mpa"], y=adf["fitted_p_over_delta_j_g"],
                mode="lines", name="Abel Fit: F + η·P_max",
                line=dict(color="#ffaa00", width=2),
            ))
            fig_ab.update_layout(
                title="Abel Equation of State Plot (P_max / Δ vs P_max)",
                xaxis_title="Peak Pressure P_max (MPa)",
                yaxis_title="P_max / Δ (J/g)",
                height=360,
            )
            st.plotly_chart(fig_ab, use_container_width=True)
        with fg2:
            bdf = fit_out["burn_df"]
            fig_br = go.Figure()
            fig_br.add_trace(go.Scatter(
                x=bdf["pressure_mpa"], y=bdf["measured_burn_rate_mm_s"],
                mode="markers", name="Measured r(P)",
                marker=dict(size=10, color="#00ff88"),
            ))
            fig_br.add_trace(go.Scatter(
                x=bdf["pressure_mpa"], y=bdf["fitted_burn_rate_mm_s"],
                mode="lines", name=f"Saint-Robert Fit: r = {fit_out['fitted_a_mm_s_mpa_n']:.2f}·P^{fit_out['fitted_n']:.2f}",
                line=dict(color="#00f0ff", width=2),
            ))
            fig_br.update_layout(
                title="Log-Log Strand Burn Rate Law Fit",
                xaxis=dict(title="Pressure P (MPa)", type="log"),
                yaxis=dict(title="Burn Rate r (mm/s)", type="log"),
                height=360,
            )
            st.plotly_chart(fig_br, use_container_width=True)


def render_smoke_toxicity_tab(
    formulation: Formulation,
    conditions: CombustionConditions,
    pah_config: PAHAllocationConfig,
    eq_temp: float,
) -> None:
    """Render the Tactical Smoke Classification & Range Toxicity (OSHA/NIOSH) Analyzer."""
    with st.expander("ℹ️  How to use — Plume Smoke & Occupational Range Toxicity Analyzer", expanded=False):
        st.markdown("""
        **Evaluate NATO AGARD / STANAG 4400 smoke classification and indoor firing range occupational health.**
        - **Primary & Secondary Smoke**: Accounts for solid particulates (soot, PAHs, metal oxides) and water/acid vapor condensation across ambient temperature and relative humidity.
        - **OSHA PEL / NIOSH REL / IDLH Exposure**: Computes enclosed firing point concentrations of $\\text{CO}, \\text{NO}, \\text{NO}_2, \\text{HCN}, \\text{NH}_3, \\text{HCl}, \\text{HF}$ and EPA **Benzo[a]pyrene Toxic Equivalence (BaP-TEQ)** across all 12 PAH species, plus required fresh-air ventilation rates.
        """)
    st.subheader("Tactical Plume Smoke & Occupational Range Toxicity (OSHA / NIOSH)")

    if not formulation.components:
        st.info("Define a formulation first.")
        return

    tx1, tx2, tx3 = st.columns(3)
    with tx1:
        tx_charge_g = st.number_input("Charge Mass per Round (g)", 0.5, 25000.0, 45.0, 5.0, key="tx_charge")
        tx_rounds = st.number_input("Number of Rounds Fired in Burst", 1, 500, 5, 1, key="tx_rounds")
    with tx2:
        tx_vol_m3 = st.number_input("Firing Enclosure / Bay Volume (m³)", 10.0, 10000.0, 120.0, 10.0, key="tx_vol")
        tx_ach = st.slider("Ventilation Rate (Air Changes / Hour)", 0.0, 30.0, 6.0, 0.5, key="tx_ach")
    with tx3:
        tx_temp_c = st.slider("Ambient Temperature (°C)", -20.0, 45.0, 15.0, 1.0, key="tx_temp")
        tx_rh_pct = st.slider("Ambient Relative Humidity (%)", 5.0, 100.0, 65.0, 5.0, key="tx_rh")

    suite = get_formulation_suite(formulation.name)
    if st.button("▶ Run Smoke & Range Toxicity Assessment", type="primary", key="btn_run_tox"):
        res_src = st.session_state.get("results", {}).get(formulation.name)
        if res_src is None:
            if st.session_state.get("active_mode_label") == "adiabatic (self-consistent)":
                res_src = cached_solve_equilibrium_adiabatic(
                    formulation, conditions, pah_config, st.session_state.species_db, 2500.0
                )
            else:
                res_src = cached_solve_equilibrium(
                    formulation, conditions, pah_config, st.session_state.species_db, eq_temp
                )
            st.session_state.setdefault("results", {})[formulation.name] = res_src
        tox_out = analyze_toxicity_and_smoke(
            result=res_src,
            charge_mass_g=tx_charge_g,
            rounds_fired=int(tx_rounds),
            enclosure_volume_m3=tx_vol_m3,
            air_changes_per_hour=tx_ach,
            ambient_temp_c=tx_temp_c,
            relative_humidity_pct=tx_rh_pct,
        )
        suite["tox_out"] = tox_out

    if "tox_out" in suite:
        tox_out = suite["tox_out"]
        hq = tox_out["max_hazard_quotient"]
        if hq >= 1.0:
            st.error(f"**Occupational Exposure Alert:** Peak Hazard Quotient = **{hq:.2f}× OSHA PEL** (Limiting Gas: **{tox_out['limiting_pollutant']}**)")
        elif hq >= 0.5:
            st.warning(f"**Occupational Exposure Caution:** Peak Hazard Quotient = **{hq:.2f}× OSHA PEL** (Limiting Gas: **{tox_out['limiting_pollutant']}**)")
        else:
            st.success(f"**Occupational Exposure Within Safe Limits:** Peak Hazard Quotient = **{hq:.2f}× OSHA PEL**")

        st.info(f"**NATO AGARD / STANAG 4400 Plume Classification:** {tox_out['agard_smoke_class']}")

        tm1, tm2, tm3, tm4 = st.columns(4)
        tm1.metric("Primary Particulate Smoke", f"{tox_out['primary_smoke_g_per_kg']:.1f} g/kg")
        tm2.metric("Secondary Condensation Index", f"{tox_out['secondary_smoke_index']:.1f} / 100")
        tm3.metric("Carcinogenic PAH BaP-TEQ", f"{tox_out['bap_teq_ug_per_g']:.2f} µg/g")
        tm4.metric("Min Recommended Ventilation", f"{tox_out['recommended_ventilation_m3_min']:.1f} m³/min")

        st.markdown("#### Workplace Gaseous Pollutant Inventory vs. NIOSH REL / OSHA PEL / IDLH")
        st.dataframe(tox_out["tox_df"], use_container_width=True, hide_index=True)

        tg1, tg2 = st.columns(2)
        with tg1:
            tdf = tox_out["tox_df"]
            fig_tx = go.Figure()
            fig_tx.add_trace(go.Bar(
                x=tdf["Species"], y=tdf["Range Conc (mg/m³)"],
                name="Predicted Bay Conc (mg/m³)", marker_color="#00f0ff",
            ))
            fig_tx.add_trace(go.Bar(
                x=tdf["Species"], y=tdf["OSHA PEL (mg/m³)"],
                name="OSHA PEL Limit (mg/m³)", marker_color="#ffaa00",
            ))
            fig_tx.update_layout(
                title="Enclosed Firing Point Concentration vs. OSHA PEL",
                xaxis_title="Toxic Gas Species",
                yaxis=dict(title="Concentration (mg/m³)", type="log"),
                barmode="group",
                height=390,
            )
            st.plotly_chart(fig_tx, use_container_width=True)
        with tg2:
            fig_smk = go.Figure(data=go.Contour(
                z=tox_out["smoke_matrix"],
                x=tox_out["t_grid_c"],
                y=tox_out["rh_grid_pct"],
                colorscale="Blues",
                contours=dict(showlabels=True),
                colorbar=dict(title="Fog Index"),
            ))
            fig_smk.add_trace(go.Scatter(
                x=[tx_temp_c], y=[tx_rh_pct],
                mode="markers", name="Current Ambient State",
                marker=dict(size=12, color="#ff4444", symbol="x"),
            ))
            fig_smk.update_layout(
                title="Secondary Plume Condensation Fog Envelope (Temp vs Humidity)",
                xaxis_title="Ambient Temperature (°C)",
                yaxis_title="Relative Humidity (%)",
                height=390,
            )
            st.plotly_chart(fig_smk, use_container_width=True)


def render_stability_tab(formulation: Formulation) -> None:
    """Render the Propellant Thermal Stability, STANAG 4582 Shelf-Life & Cook-Off Safety tab."""
    with st.expander("ℹ️  How to use — Thermal Stability, STANAG 4582 Shelf-Life & Cook-Off Safety", expanded=False):
        st.markdown("""
        **Predict stabilizer depletion shelf-life and prevent self-heating thermal runaway during storage.**
        - **NATO STANAG 4582 Stabilizer Depletion**: Models Arrhenius $\\text{NO}_x$ cleavage and stabilizer consumption (DPA, Centralite, Akardite) to predict safe storage life across magazine temperatures.
        - **Frank-Kamenetskii Critical Temperature ($T_{\\text{crit}}$)**: Calculates the critical ambient self-heating threshold as a function of storage container diameter.
        """)
    st.subheader("Propellant Thermal Stability, STANAG 4582 Shelf-Life & Cook-Off Safety")

    if not formulation.components:
        st.info("Define a formulation first.")
        return

    suite = get_formulation_suite(formulation.name)
    detected_stab = detect_stabilizer_wt_pct(formulation)
    default_stab = detected_stab if detected_stab > 0.05 else 1.50

    st1, st2, st3 = st.columns(3)
    with st1:
        sb_temp_c = st.slider("Magazine Storage Temperature (°C)", 10.0, 65.0, 25.0, 1.0, key="sb_temp")
        sb_stab_pct = st.number_input(
            "Initial Stabilizer Content (wt%)",
            0.10,
            6.00,
            float(default_stab),
            0.10,
            key=f"sb_stab_{formulation.name}",
        )
    with st2:
        sb_ea = st.number_input("Arrhenius Activation Energy E_a (kJ/mol)", 110.0, 165.0, 135.0, 1.0, key="sb_ea")
        sb_diam_mm = st.number_input("Storage Container / Cartridge Diameter (mm)", 10.0, 400.0, 105.0, 5.0, key="sb_diam")
    with st3:
        sb_cond = st.number_input("Propellant Bed Thermal Conductivity λ (W/m·K)", 0.08, 0.80, 0.24, 0.02, key="sb_cond")
        sb_rho = st.number_input("Bulk Storage Density (g/cm³)", 0.60, 1.90, 1.55, 0.05, key="sb_rho")

    if st.button("▶ Run Thermal Stability & Shelf-Life Analysis", type="primary", key="btn_run_stab"):
        stab_out = analyze_thermal_stability(
            formulation=formulation,
            storage_temp_c=sb_temp_c,
            stabilizer_wt_pct=sb_stab_pct,
            activation_energy_kj_mol=sb_ea,
            charge_diameter_mm=sb_diam_mm,
            thermal_conductivity_w_m_k=sb_cond,
            propellant_density_g_cm3=sb_rho,
        )
        suite["stab_out"] = stab_out

    if "stab_out" in suite:
        stab_out = suite["stab_out"]
        if stab_out["stanag_4582_compliant"]:
            st.success("✅ **NATO STANAG 4582 Compliant:** Exceeds 10-year equivalent thermal stability benchmark at 25 °C with safe cook-off margin.")
        else:
            st.warning("⚠️ **Surveillance Alert:** Stabilizer content or thermal margin falls below NATO STANAG 4582 long-term storage criteria.")

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric(f"Safe Shelf-Life at {sb_temp_c:.0f} °C", f"{stab_out['safe_shelf_life_years']:.1f} years")
        sm2.metric("25 °C Benchmark Life", f"{stab_out['shelf_life_at_25c_years']:.1f} years")
        sm3.metric("Critical Cook-Off Temp T_crit", f"{stab_out['critical_cookoff_temp_c']:.1f} °C ({stab_out['critical_cookoff_temp_k']:.0f} K)")
        sm4.metric("Safety Margin (T_crit − T_store)", f"{stab_out['thermal_margin_c']:+.1f} °C")

        sg1, sg2 = st.columns(2)
        with sg1:
            ddf = stab_out["depletion_df"]
            fig_dep = go.Figure()
            fig_dep.add_trace(go.Scatter(
                x=ddf["time_years"], y=ddf["remaining_stabilizer_pct"],
                name="Remaining Stabilizer (wt%)", line=dict(color="#00f0ff", width=2.5),
            ))
            fig_dep.add_trace(go.Scatter(
                x=ddf["time_years"], y=ddf["critical_limit_pct"],
                name="STANAG Surveillance Condemnation Limit", line=dict(color="#ff4444", dash="dash", width=2),
            ))
            fig_dep.update_layout(
                title=f"Stabilizer Depletion Trajectory at {sb_temp_c:.0f} °C",
                xaxis_title="Storage Time (Years)",
                yaxis_title="Stabilizer Mass Fraction (wt%)",
                height=380,
            )
            st.plotly_chart(fig_dep, use_container_width=True)
        with sg2:
            cdf = stab_out["cookoff_df"]
            fig_ck = go.Figure()
            fig_ck.add_trace(go.Scatter(
                x=cdf["charge_diameter_mm"], y=cdf["critical_cookoff_temp_c"],
                name="Frank-Kamenetskii T_crit (°C)", line=dict(color="#ffaa00", width=2.5),
            ))
            fig_ck.add_trace(go.Scatter(
                x=[sb_diam_mm], y=[stab_out["critical_cookoff_temp_c"]],
                mode="markers", name=f"Current Diameter ({sb_diam_mm:.0f} mm)",
                marker=dict(size=12, color="#00ff88"),
            ))
            fig_ck.add_hline(
                y=sb_temp_c, line_dash="dot", line_color="#00f0ff",
                annotation_text=f"Storage Temp ({sb_temp_c:.0f} °C)",
            )
            fig_ck.update_layout(
                title="Critical Self-Heating Temperature T_crit vs Container Diameter",
                xaxis_title="Charge / Container Diameter (mm)",
                yaxis_title="Critical Temperature T_crit (°C)",
                height=380,
            )
            st.plotly_chart(fig_ck, use_container_width=True)


def render_nc_nitration_calculator(formulation: Formulation) -> None:
    """Render the US Army BRL / Kubota Nitrocellulose Nitration (%N) Stoichiometry Synthesizer."""
    from .grounding import synthesize_nitrocellulose
    from .models import PropellantComponent

    with st.expander("🧪 Nitrocellulose Nitration (%N) Continuum Synthesizer (US Army BRL BLAKE / MIL-DTL-244C)", expanded=False):
        st.caption(
            "Nitrocellulose is a non-stoichiometric nitrate ester polymer whose formula C₆H₁₀₋ₓNₓO₅₊₂ₓ and "
            "ΔHf,298 = (−4179.0 + 141.25·[%N]) J/g depend continuously on Nitrogen content (%N) per Freedman (ARBRL-TR-02411)."
        )
        nc1, nc2, nc3 = st.columns([2, 1, 1])
        with nc1:
            nc_pct = st.slider(
                "Target Nitrogen Content (%N by mass)",
                min_value=10.50,
                max_value=14.10,
                value=13.15,
                step=0.05,
                key="nc_synth_pct",
            )
        with nc2:
            nc_mass_frac = st.number_input(
                "Mass Fraction to Inject",
                min_value=0.01,
                max_value=1.00,
                value=0.50,
                step=0.05,
                key="nc_synth_mf",
            )
        nc_info = synthesize_nitrocellulose(nc_pct)
        with nc3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Inject NC Grade", use_container_width=True, key="btn_inject_nc"):
                formulation.components.append(
                    PropellantComponent(
                        name=f"Nitrocellulose ({nc_pct:.2f}% N)",
                        formula=nc_info["formula"],
                        mass_fraction=float(nc_mass_frac),
                        hf_298=float(nc_info["hf_kj_per_mol"]),
                    )
                )
                st.session_state.formulations[st.session_state.selected_idx] = formulation
                st.rerun()

        nm1, nm2, nm3, nm4 = st.columns(4)
        nm1.metric("Anhydroglucose Formula", nc_info["formula"])
        nm2.metric("ΔHf,298 (Molar / Specific)", f"{nc_info['hf_kj_per_mol']:.1f} kJ/mol", delta=f"{nc_info['hf_j_per_g']:.0f} J/g", delta_color="off")
        nm3.metric("Substitution Degree x", f"{nc_info['degree_of_substitution_x']:.3f} / 3.0")
        nm4.metric("Oxygen Balance", f"{nc_info['oxygen_balance_pct']:.1f}%")
        st.caption(f"**Military Classification:** {nc_info['military_grade']}  |  **Solid Density:** {nc_info['density_g_cm3']:.3f} g/cm³")


def render_benchmarks_and_grounding_tab() -> None:
    """Render the Experimental Benchmarks, NASA CEA / BRL Validation & Provenance Explorer tab."""
    from .grounding import (
        METHODOLOGY_CITATIONS,
        get_ingredient_provenance_df,
        run_benchmark_validation_suite,
        validate_epa_ap42_emissions,
    )

    with st.expander("ℹ️  How to use — Experimental Benchmarks, Code Parity & Data Provenance", expanded=False):
        st.markdown("""
        **Verify Combusto's first-principles thermochemical and emission models against peer-reviewed experimental and reference code benchmarks.**
        - **US Army BRL (BLAKE / IBHVG2), Fraunhofer ICT & NASA CEA Parity**: Executes Combusto live against 8 standard military cannon, LOVA, small-arms, and space-booster propellants (`M1`, `M6`, `M9`, `JA2`, `M30`, `WC 844`, `M43`, `TP-H1148`) and measures parity across Flame Temperature ($T_v / T_p$), Impetus ($F$), Covolume ($\\eta$), $\\gamma$, and Calorimetric $Q_v$.
        - **US EPA AP-42 Chapter 15 Enclosed-Chamber Emission Factors**: Compares muzzle-freeze gaseous and trace-PAH emission factors ($\\text{g/kg}$) for `WC 844 (5.56×45mm NATO)` against measured US Army AEC / EPA AP-42 firing-point datasets.
        - **Traceable Provenance & Literature Browser**: Inspect CAS numbers, crystal densities, NASA Glenn / NIST-JANAF polynomials, Hirschfelder/BLAKE molar covolumes $b_i$, and full primary citations.
        """)

    st.subheader("📚 US Army BRL (BLAKE / IBHVG2), ICT & NASA CEA Benchmark Validation")
    st.caption("Click below to run Combusto's Gibbs minimization & Noble-Abel solver live across all 8 canonical reference propellants.")

    if st.button("▶ Run Live 8-Propellant Reference Benchmark Suite", type="primary", key="btn_run_benchmarks"):
        from .cache_helpers import cached_benchmark_validation_suite
        with st.spinner("Solving self-consistent Gibbs equilibrium across M1, M6, M9, JA2, M30, WC844, M43, and TP-H1148..."):
            st.session_state["bm_suite_out"] = cached_benchmark_validation_suite(st.session_state.species_db)

    if "bm_suite_out" in st.session_state:
        bm_out = st.session_state["bm_suite_out"]
        bdf = bm_out["benchmark_df"]

        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("Flame Temp MAPE (Tv / Tp)", f"{bm_out['mape_temp_pct']:.2f}%", delta="vs BRL / NASA CEA", delta_color="off")
        bm2.metric("Impetus / Force F MAPE", f"{bm_out['mape_force_pct']:.2f}%", delta="vs STANAG 4115 / IBHVG2", delta_color="off")
        bm3.metric("BLAKE Covolume η MAPE", f"{bm_out['mape_covolume_pct']:.2f}%", delta="Σ nᵢbᵢ vs Measured", delta_color="off")
        bm4.metric("Calorimetric Qv MAPE", f"{bm_out['mape_qv_pct']:.2f}%", delta="vs Bomb Calorimetry", delta_color="off")

        st.dataframe(bdf, use_container_width=True, hide_index=True)

        pg1, pg2 = st.columns(2)
        with pg1:
            t_min, t_max = 2200.0, 4100.0
            fig_pt = go.Figure()
            fig_pt.add_trace(go.Scatter(
                x=[t_min, t_max], y=[t_min, t_max],
                mode="lines", name="1:1 Exact Parity",
                line=dict(color="#8899aa", dash="dash", width=2),
            ))
            fig_pt.add_trace(go.Scatter(
                x=bdf["Ref T (K)"], y=bdf["Combusto T (K)"],
                mode="markers+text",
                text=[p.split(" ")[0] for p in bdf["Propellant"]],
                textposition="top center",
                name="Benchmark Propellants",
                marker=dict(size=11, color="#00f0ff", line=dict(width=1, color="#ffffff")),
            ))
            fig_pt.update_layout(
                title="Flame Temperature Parity: Combusto vs. BRL / ICT / NASA CEA",
                xaxis_title="Published Reference T (K)",
                yaxis_title="Combusto Predicted T (K)",
                height=390,
            )
            st.plotly_chart(fig_pt, use_container_width=True)

        with pg2:
            f_min, f_max = 850.0, 1280.0
            fig_pf = go.Figure()
            fig_pf.add_trace(go.Scatter(
                x=[f_min, f_max], y=[f_min, f_max],
                mode="lines", name="1:1 Exact Parity",
                line=dict(color="#8899aa", dash="dash", width=2),
            ))
            fig_pf.add_trace(go.Scatter(
                x=bdf["Ref F (J/g)"], y=bdf["Combusto F (J/g)"],
                mode="markers+text",
                text=[p.split(" ")[0] for p in bdf["Propellant"]],
                textposition="top center",
                name="Propellant Impetus F",
                marker=dict(size=11, color="#00ff88", line=dict(width=1, color="#ffffff")),
            ))
            fig_pf.update_layout(
                title="Propellant Impetus (Force Constant F) Parity: Combusto vs. BRL",
                xaxis_title="Published Reference F (J/g)",
                yaxis_title="Combusto Predicted F (J/g)",
                height=390,
            )
            st.plotly_chart(fig_pf, use_container_width=True)

    st.divider()
    st.subheader("🌫️ US EPA AP-42 Chapter 15 & US Army AEC Emission Factor Grounding")
    st.caption(
        "Muzzle gases freeze kinetically during rapid barrel expansion (~1500–1800 K). Compare Combusto's "
        "muzzle-freeze equilibrium + EPA-calibrated trace PAH model against enclosed-chamber firing measurements on WC 844 (5.56×45mm NATO)."
    )
    ep1, ep2, ep3 = st.columns([1, 1, 1])
    with ep1:
        ep_t_freeze = st.slider("Kinetic Freeze-Out Temperature (K)", 1350.0, 2100.0, 1650.0, 25.0, key="ep_tfreeze")
    with ep2:
        ep_p_freeze = st.slider("Muzzle Exit Freeze Pressure (atm)", 5.0, 120.0, 35.0, 5.0, key="ep_pfreeze")
    with ep3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_epa = st.button("▶ Compare vs. EPA AP-42 Table 15.1", use_container_width=True, key="btn_run_epa")

    if run_epa or "epa_val_out" in st.session_state:
        if run_epa:
            st.session_state["epa_val_out"] = validate_epa_ap42_emissions(
                freeze_temp_k=ep_t_freeze,
                pressure_atm=ep_p_freeze,
                species_db=st.session_state.species_db,
            )
        epa_df = st.session_state["epa_val_out"]["emissions_df"]
        st.dataframe(epa_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔎 Traceable Thermochemical Provenance & Primary Bibliography")
    prov_tab1, prov_tab2, prov_tab3 = st.tabs([
        "🧪 Grounded Ingredients (CAS, ρ₀, ΔHf)",
        "⚛️ Product Species & BLAKE Covolumes (59 Species)",
        "📖 Peer-Reviewed Methodology Citations",
    ])
    with prov_tab1:
        ing_df = get_ingredient_provenance_df()
        st.dataframe(ing_df, use_container_width=True, hide_index=True)
    with prov_tab2:
        sp_rows = []
        for sp in st.session_state.species_db.values():
            sp_rows.append({
                "Species": sp.name,
                "Formula": sp.formula,
                "Phase": sp.phase,
                "Category": sp.category,
                "CAS RN": sp.cas or "—",
                "ΔHf,298 (kJ/mol)": sp.hf_298,
                "Valid T_max (K)": sp.t_high,
                "Molar Covolume bᵢ (cm³/mol)": sp.molar_covolume_cm3_mol,
                "Thermochemical Source": sp.source,
            })
        st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)
    with prov_tab3:
        st.dataframe(pd.DataFrame(METHODOLOGY_CITATIONS), use_container_width=True, hide_index=True)
