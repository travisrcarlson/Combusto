# Combusto

**Gun & Rocket Propellant Thermochemical, Ballistic, Signature & Surveillance Engineering Suite**

Combusto is a multi-physics computational workbench for energetic materials, gun interior ballistics, rocket nozzle expansion, plume signature/hygiene, and shelf-life surveillance. Unlike equilibrium-only codes that assume complete relaxation to ideal small gases, Combusto couples **vectorized Gibbs free-energy minimization with analytical Gibbs-Duhem Jacobians**, **US Army BRL BLAKE real-gas virial fugacity**, and **calibrated polycyclic aromatic hydrocarbon (PAH) / soot precursor allocation** across five integrated engineering workspaces.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the hosted web workbench (default session password: combusto2026)
streamlit run app.py

# Launch Desktop Pro (local Database Editor, API Key Manager & PubChem/NIST lookup)
streamlit run app_local.py

# Run the full automated verification suite
python -m pytest tests/ -v
```

---

## Architecture

```text
Combusto/
├── app.py                      # Hosted Streamlit web UI + password-gated Desktop Pro ZIP builder
├── app_local.py                # Local Desktop Pro UI (Master JSON Database Editor, API Keys, PubChem/NIST)
├── INSTRUCTIONS.md             # Desktop Pro setup, database administration & workspace guide
├── requirements.txt            # Runtime & test dependencies
├── combusto/
│   ├── models.py               # Core dataclasses, IUPAC atomic masses & parenthetical formula parser
│   ├── chemistry.py            # Multi-element CHNO/Cl/F/metal oxygen balance, φ & element conservation
│   ├── thermodynamics.py       # NASA 7-coefficient polynomials with C¹ constant-Cp boundary clamping
│   ├── equilibrium.py          # Vectorized SLSQP Gibbs solver (analytical Gibbs-Duhem Jacobian) + BLAKE EOS
│   ├── pah_allocation.py       # Incomplete-combustion PAH & soot carbon partitioning presets
│   ├── nozzle.py               # 1D NASA CEA-style Shifting Equilibrium vs. Frozen rocket nozzle solver
│   ├── thermo_cycles.py        # Noble-Abel real-gas vs. ideal polytropic expansion isentropes
│   ├── ballistics.py           # Lagrange gun ODE solver, 7/19-perf rosette burnback & Lawton bore erosion
│   ├── closed_vessel.py        # NATO STANAG 4115 manometric bomb vivacity & Abel/Saint-Robert calibrator
│   ├── uncertainty.py          # NATO STANAG 2895 climatic zones (-40 to +63 °C) & Monte Carlo dispersion
│   ├── detonation.py           # Kamlet-Jacobs (1968) CJ detonation velocity (VoD) & pressure screening
│   ├── muzzle_flash.py         # Mach-disk shock recovery & K₂SO₄/KNO₃ radical-scavenging flash model
│   ├── toxicity_smoke.py       # NATO STANAG 4400 primary/secondary smoke, OSHA PEL/IDLH & EPA BaP-TEQ
│   ├── stability.py            # NATO STANAG 4582 Arrhenius stabilizer depletion & Frank-Kamenetskii T_crit
│   ├── optimizer.py            # Constrained SLSQP inverse formulation optimizer & 3-apex ternary maps
│   ├── sensitivity.py          # 1D & 2D parametric sweeps (φ, T, P, PAH model, multi-formulation)
│   ├── grounding.py            # NC %N continuum synthesizer + BRL/ICT/NASA CEA & EPA AP-42 validation
│   ├── export.py               # Structured CSV, multi-sheet Excel & JSON exports
│   ├── export_pdf.py           # Lab Summary PDF & Multi-Suite Engineering Dossier PDF generator
│   ├── species_db.py           # 59-species NASA Glenn / NIST-JANAF loader with BLAKE molar covolumes
│   ├── presets.py              # Standard military/aerospace propellant & 48-ingredient library loader
│   ├── cache_helpers.py        # Deterministic SHA-256 Streamlit memoization & dark cyber Plotly theme
│   ├── workspace_ui.py         # Shared 5-Workspace UI with per-formulation state isolation
│   └── ui_extensions.py        # Modular tabs (Isentrope, Closed-Vessel, Smoke/Toxicity, Stability, Grounding)
├── data/
│   ├── ingredients.json        # 48 grounded energetics, binders, plasticizers & stabilizers with CAS & ρ₀
│   ├── propellants.json        # Canonical NATO/US gun, LOVA, small-arms & rocket booster formulations
│   └── species.json            # 59 gas/condensed species (small molecules, 12 PAHs, alkali/metal salts)
└── tests/
    ├── test_chemistry.py       # Formula parsing, OB%, φ & element conservation
    ├── test_equilibrium.py     # Gibbs equilibrium solver & element conservation checks
    ├── test_pah.py             # PAH/soot carbon allocation rules
    ├── test_advanced.py        # Self-consistent adiabatic, Noble-Abel isochoric & sweeps
    ├── test_enhancements.py    # Ballistics, closed-vessel, muzzle flash, smoke/toxicity, stability & PDF
    ├── test_grounding.py       # NC %N continuum, BRL/CEA 8-propellant parity & EPA AP-42 validation
    ├── test_refinements.py     # BLAKE fugacity, shifting vs. frozen nozzle, 19-perf, STANAG 2895 & MC
    └── test_audit_wave5.py     # Kamlet-Jacobs VoD, halogen OB, analytical Jacobian speed, ZIP & isolation
```

---

## Core Engineering Capabilities

### 1. Passport & Thermochemistry
- **Vectorized Gibbs Minimization**: Exact analytical Gibbs-Duhem chemical potential gradient ($\nabla_n G / RT = \boldsymbol{\mu} / RT$) with high-pressure **BLAKE / truncated Virial real-gas fugacity** ($\ln \phi_i = b_i P / RT$), initial grain sensible enthalpy conditioning, and optional inert chamber purge gases (`N2`, `Ar`, `He`, `CO2`).
- **1D Rocket Nozzle Expansion**: Solves **Shifting Chemical Equilibrium** ($S(T,P) = S_c$) alongside **Frozen Composition** through the sonic throat to exit pressure $P_e$ for $C^*$, $C_F$, $\epsilon = A_e/A_t$, and Ambient/Vacuum $I_{sp}$.
- **Executive Propellant Passport**: Single-click coupled execution across all physics engines with a 6-axis survivability radar chart and downloadable **Multi-Suite Engineering Dossier (PDF)**.

### 2. Ballistics & Chamber Tolerance
- **Lagrange Interior Ballistics & Bore Erosion**: Adaptive `RK45` dense-output integration with exact geometric **7-perf cylinder** and **19-perf rosette** progressive burnback + slivering, paired with **Lawton/Krause** bore wall heat-transfer and EFC barrel life estimation.
- **STANAG 2895 Climatic & Monte Carlo Dispersion**: Evaluates `-40 °C` (Arctic C2) through `+63 °C` (Hot Desert A1) conditioning against Service MPOP and Gun Proof limits, plus stochastic lot-tolerance dispersion ($P_{\max} + 3\sigma$, $C_{pk}$, $\sigma_{V_0}$).
- **Closed-Vessel & Kamlet-Jacobs VoD**: NATO STANAG 4115 dynamic vivacity $L(P/P_{\max})$, linear Abel EOS ($F, \eta$) / Saint-Robert ($a, n$) regression, and **Kamlet & Jacobs (1968)** detonation velocity $D$ and Chapman-Jouguet pressure $P_{CJ}$ screening.

### 3. Signature, Plume & Occupational Hygiene
- **Muzzle Flash Suppression**: Supersonic expansion, Mach-disk normal-shock temperature recovery, and $\text{K}_2\text{SO}_4 / \text{KNO}_3$ alkali free-radical scavenging ignition thresholds.
- **STANAG 4400 Smoke & Range Toxicity**: Primary particulate smoke (soot, PAHs, condensed metal oxides, and plume-condensible $\text{KOH}/\text{KCl}$ alkali salts), secondary humidity fog contours, enclosed firing-bay **OSHA PEL / NIOSH REL / IDLH** hazard quotients, and **EPA BaP-TEQ** carcinogenic PAH inventory.

### 4. Design, Sensitivity & Surveillance
- **Inverse Optimization & Ternary Simplex Maps**: Constrained SLSQP ingredient optimization (maximizing $F$ or $I_{sp}$, or minimizing $T_p$ / soot subject to thermal and emission ceilings) and 3-component ternary phase diagrams.
- **STANAG 4582 Thermal Stability & Cook-Off**: Arrhenius stabilizer depletion shelf-life (50% depletion / `0.20 wt%` condemnation rule) and **Frank-Kamenetskii** critical self-heating temperature $T_{\text{crit}}$ vs. cartridge diameter.

### 5. Experimental Benchmarks & Provenance
- **Live 8-Propellant Code Parity**: Validates Combusto directly against published **US Army BRL (`BLAKE` / `IBHVG2`)**, **Fraunhofer ICT**, and **NASA CEA** benchmarks across `M1`, `M6`, `M9`, `JA2`, `M30`, `WC 844`, `M43`, and `TP-H1148` (`~2.3%` flame temperature MAPE, `~2.6%` impetus MAPE in `< 3 s` total runtime).
- **US EPA AP-42 Ch. 15 Grounding**: Validates kinetic muzzle-freeze CO, $\text{CO}_2$, $\text{CH}_4$, $\text{HCN}$, $\text{NH}_3$, benzene, and trace PAH emission factors against enclosed-chamber `WC 844` firing data.

---

## Web vs. Desktop Pro Editions

| Capability | Web Workbench (`app.py`) | Desktop Pro (`app_local.py`) |
| :--- | :---: | :---: |
| Full 5-Workspace Physics, Ballistics & PDF Dossiers | Yes | Yes |
| Developer Portal `.zip` Export (`Freya`) | Yes | — |
| Master JSON Database Editor (`ingredients`, `propellants`, `species`) | Isolated (Excluded) | Yes |
| Local `.streamlit/secrets.toml` API Key Manager | Isolated (Excluded) | Yes |
| Live PubChem PUG REST & NIST WebBook Scraper | Isolated (Excluded) | Yes |

---

## Dependencies

- Python 3.11+
- `streamlit`, `numpy`, `scipy`, `pandas`, `openpyxl`, `plotly`, `reportlab`, `pytest`
