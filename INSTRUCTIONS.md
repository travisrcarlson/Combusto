# Combusto: Desktop Pro Edition
**Developer, Ballistician & Administrative Setup Guide**

Welcome to the standalone Desktop Pro edition of **Combusto**. This localized build (`app_local.py`) unlocks administrative editing of the core empirical thermochemical databases, live PubChem/NIST chemical scraping, and local API key management, while providing the full 5-Workspace multi-physics thermochemical, rocket nozzle, interior ballistic, climatic/Monte-Carlo tolerance, signature/hygiene, and surveillance engineering suite.

---

## 1. Installation & Boot Protocol

Ensure you have **Python 3.9+** installed and access to your system terminal (PowerShell, Command Prompt, or macOS/Linux Terminal).

**Step 1:** Open your terminal and navigate to this extracted folder:
```bash
cd path/to/Combusto_Pro_Desktop
```

**Step 2:** Install all scientific, optimization, and PDF reporting dependencies:
```bash
pip install -r requirements.txt
```
*(Or manually: `pip install streamlit numpy scipy pandas openpyxl plotly reportlab toml requests`)*

**Step 3:** Launch the Desktop Pro application using Streamlit:
```bash
streamlit run app_local.py
```
*(Note: Running `app.py` launches the public web version without the Database Editor, API Key Manager, or live PubChem/NIST scraper. Always run `app_local.py` for full local administrative capabilities. Default session access password: `combusto2026`.)*

---

## 2. Master Database Editor & Live PubChem/NIST Lookup (`app_local.py` Exclusive)

In the left-hand sidebar under **⚙️ Combusto Local Admin**:
- **Database Editor**: Inspect, edit, add, or delete entries directly in:
  1. `data/ingredients.json` — 48 grounded energetic oxidizers, nitrate esters (MIL-DTL-244C NC grades), binders, plasticizers, and stabilizers with CAS Registry Numbers, crystal densities $\rho_0$, and condensed-phase $\Delta H^\circ_{f,298}$.
  2. `data/propellants.json` — Standardized NATO/US military cannon, mortar, small-arms, LOVA, and space-booster formulations (`M1`, `M6`, `M9`, `JA2`, `M30`, `WC 844`, `M43`, `TP-H1148`).
  3. `data/species.json` — 59-species NASA Glenn / NIST-JANAF 7-coefficient polynomial dictionary with Hirschfelder/BLAKE molar covolumes $b_i$ and 12 priority PAH species.
- **External API Chemical Lookup (PubChem & NIST WebBook)**: Inside the **Formulation Editor** on `Main Workbench`, expand **🔍 External API Chemical Lookup** to query PubChem PUG REST by compound name or SMILES and retrieve $\Delta H^\circ_{f,298}$ from NIST Chemistry WebBook, then inject directly into your active formulation.

---

## 3. API Key Manager & Secrets Storage (`app_local.py` Exclusive)

In the left-hand sidebar under **⚙️ Combusto Local Admin**, select **"API Key Manager"**:
1. Enter your optional API tokens for NIST WebBook, Materials Project (`mp-api`), or PubChem.
2. Click **"🔒 Save Environment Secrets"**.
3. Keys are written locally to `.streamlit/secrets.toml` (excluded from Git and excluded from ZIP exports), keeping your credentials strictly offline on your workstation.

---

## 4. Guide to the 5 Engineering Workspaces

1. **🪪 Passport & Thermochemistry**
   - **Executive Propellant Passport**: One-click full-suite characterization across all physics engines, rendering a 6-axis survivability/performance radar footprint, qualification summary table, and **📘 Full Multi-Suite Engineering Dossier (PDF)** download.
   - **Conditions & Equilibrium Solver**: Isobaric, closed-vessel isochoric, and self-consistent adiabatic Gibbs minimization with optional **BLAKE / Virial Real-Gas Fugacity** ($\ln \phi_i = b_i P / RT$) and PAH/soot partitioning.
   - **Equilibrium Results & Exports**: Isobaric $T_p$, closed-vessel $T_v$, gun Impetus $F = n_{\text{gas}} R T_v$, composition-derived BLAKE covolume $\eta = \sum n_i b_i$, real-gas compressibility $Z$, species mole/mass tables, and CSV/Excel/JSON/PDF exports.
   - **Nozzle Isp (Shifting vs. Frozen) & Calorimetry**: NASA CEA-style 1D rocket nozzle expansion solving **both Shifting Equilibrium** ($S(n_{\text{eq}}(T,P), T, P) = S_c$) and **Frozen Composition** for $C^*$, $C_F$, $\epsilon = A_e/A_t$, $T_e$, Ambient & Vacuum $I_{sp}$, Noble-Abel real-gas expansion isentropes, and $298.15\text{ K}$ liquid-$\text{H}_2\text{O}$ bomb calorimetry ($Q_v$).
2. **🎯 Ballistics & Chamber Tolerance**
   - **Interior Ballistics & Barrel Erosion**: Lagrange-gradient gun ODE solver with progressive **7-perf & 19-perf rosette** burnback visualizers, sliver burnout, Lawton/Krause transient bore wall heating, and EFC barrel wear life.
   - **STANAG 2895 Climatic & Monte Carlo Dispersion**: Evaluates burn-rate temperature sensitivity $a(T_0) = a_{\text{ref}} \exp[\sigma_p(T_0 - T_{\text{ref}})]$ across **NATO STANAG 2895 climatic zones** (`-40 °C` Arctic C2 to `+63 °C` Hot Desert A1) against Service MPOP & Proof limits, plus **Monte Carlo lot-tolerance ballistic dispersion** ($P_{\max} + 3\sigma$, $C_{pk}$, $\sigma_{V_0}$, Extreme Spread, and Tornado correlation drivers).
   - **Closed-Vessel & Vivacity**: NATO STANAG 4115 manometric bomb simulator and linear Abel EOS ($F, \eta$) + Saint-Robert ($a, n$) calibrator.
   - **VoD Detonation Screening**: Kamlet-Jacobs detonation velocity and Chapman-Jouguet ($P_{CJ}$) screening.
3. **🔥 Signature, Plume & Hygiene**
   - **Muzzle Flash & Alkali Suppression**: Mach-disk normal-shock temperature recovery and $\text{K}_2\text{SO}_4 / \text{KNO}_3$ free-radical scavenging plume auto-ignition predictor.
   - **Plume Smoke (STANAG 4400) & Range Toxicity**: AGARD / STANAG 4400 primary/secondary smoke transmittance, indoor firing-bay **OSHA PEL / NIOSH REL / IDLH** gaseous hazard quotients, and **BaP-TEQ** carcinogenic PAH inventory.
4. **🛡️ Design, Sensitivity & Surveillance**
   - **SLSQP Optimizer & Ternary Maps**: Constrained inverse formulation design and 3-component simplex ternary phase diagrams.
   - **1D & 2D Sensitivity Sweeps**: 1D sweeps and 2-variable 3D surface / 2D contour maps across $\phi$, $T$, and $P$.
   - **STANAG 4582 Shelf-Life & Cook-Off**: Arrhenius stabilizer depletion shelf-life and Frank-Kamenetskii critical cook-off temperature $T_{\text{crit}}$.
5. **📚 Benchmarks & Provenance**
   - Live 8-propellant parity suite against **US Army BRL (BLAKE / IBHVG2)**, **Fraunhofer ICT**, and **NASA CEA**, **US EPA AP-42 Ch. 15** firing emission validation, and searchable CAS / NIST-JANAF / NASA Glenn provenance tables.
