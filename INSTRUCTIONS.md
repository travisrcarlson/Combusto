# Combusto: Desktop Pro Edition
**Developer, Ballistician & Administrative Setup Guide**

Welcome to the standalone Desktop Pro edition of **Combusto**. This localized build (`app_local.py`) unlocks administrative editing of the core empirical thermochemical databases, live PubChem/NIST chemical scraping, and local API key management, while providing the full 12-module thermochemical, ballistic, plume, stability, and range-hygiene suite.

---

## 1. Installation & Boot Protocol

Before launching Combusto, ensure you have **Python 3.9+** installed and access to your system terminal (PowerShell, Command Prompt, or macOS/Linux Terminal).

**Step 1:** Open your terminal and navigate to this extracted folder:
```bash
cd path/to/Combusto_Pro_Desktop
```

**Step 2:** Install all mathematical, optimization, and PDF reporting dependencies:
```bash
pip install -r requirements.txt
```
*(Or manually: `pip install streamlit numpy scipy pandas openpyxl plotly reportlab`)*

**Step 3:** Launch the Desktop Pro application using Streamlit:
```bash
streamlit run app_local.py
```
*(Note: Running `app.py` launches the public web version without the Database Editor or API Key Manager. Always run `app_local.py` for full administrative capabilities.)*

---

## 2. Master Database Editor & Live PubChem/NIST Lookup (`app_local.py` Exclusive)

In the left-hand sidebar under **⚙️ Combusto Local Admin**, select **"Database Editor"**.

Combusto is grounded in three core JSON databases:
1. `data/ingredients.json` — Empirical formulas, molecular weights, densities, and standard heats of formation ($\Delta H^\circ_{f,298}$) for energetic oxidizers, binders, plasticizers, and stabilizers.
2. `data/propellants.json` — Standardized single-base, double-base, triple-base, and composite propellant formulations.
3. `data/species.json` — NASA 7-coefficient polynomial dictionary for gaseous, condensed, and 16-EPA-priority PAH product species.

**Live Chemical Thermochemistry Scraper:**
- When editing `ingredients.json`, expand **🌐 Fetch Ingredient from PubChem & NIST WebBook**.
- Enter any chemical name or CAS number (e.g., `RDX`, `Diphenylamine`, `Triacetin`) to query PubChem PUG-REST for molecular formula/MW and scrape NIST Chemistry WebBook for condensed/gas-phase $\Delta H^\circ_{f,298}$, then append directly to `ingredients.json`.

**Manual Editing:**
- Double-click any cell in the interactive table to modify values, add rows, or delete rows, then click **"💾 Save to Disk"** to persist changes permanently.

---

## 3. API Key Manager & Secrets Storage (`app_local.py` Exclusive)

In the left-hand sidebar under **⚙️ Combusto Local Admin**, select **"API Key Manager"**.

1. Obtain your API tokens from Materials Project (`mp-api`), NIST, PubChem, or ChemSpider.
2. Paste them into the masked input fields and click **"🔒 Save Environment Secrets"**.
3. Combusto writes these keys locally to `.streamlit/secrets.toml` (excluded from Git via `.gitignore`), keeping your credentials strictly offline on your machine.

---

## 4. Overview of Analysis Suites

1. **Formulation Builder**: Build multi-ingredient propellants with live Oxygen Balance ($\text{OB}\%$), elemental mass fractions, and density checks.
2. **Conditions & Run / Results**: Gibbs free-energy minimization solver (isobaric, isochoric Noble-Abel, or self-consistent adiabatic) with PAH/soot carbon partitioning and formatted **PDF Lab Report** export.
3. **Isothermal Calorimetry ($Q_v$)**: Bomb-calorimeter heat of explosion and product water phase state at $298.15\text{ K}$.
4. **Isentropic Expansion ($I_{sp}$) & Real-Gas Cycles**: Rocket nozzle $I_{sp}$ plus **Noble-Abel Real-Gas vs. Ideal Gas** $P\text{–}V$ expansion isentropes, compressibility factor $Z$, sound speed, and reversible $\int P\,dV$ work.
5. **Sensitivity Analysis (1D & 2D)**: Parametric 1D sweeps and simultaneous 2-variable 3D surface / 2D contour maps across $\phi$, $T$, and $P$.
6. **Internal Ballistics & Barrel Erosion**: Coupled Lagrange-gradient interior ballistics ODE solver with **7-perf & 19-perf rosette** burnback cross-section visualizers, sliver phase modeling, transient bore wall temperature, and EFC barrel wear life estimation.
7. **Closed-Vessel & Vivacity**: NATO STANAG 4115 manometric bomb simulator ($P(t)$, $dP/dt$, dynamic vivacity $L$) plus linear **Abel EOS ($F, \eta$)** and **Saint-Robert ($a, n$)** multi-density regression calibrator.
8. **Muzzle Flash & Plume**: Mach-disk normal-shock recovery temperature and $\text{K}_2\text{SO}_4 / \text{KNO}_3$ alkali radical-scavenging afterburn suppression predictor.
9. **Smoke & Range Toxicity**: **NATO AGARD / STANAG 4400** primary/secondary plume condensation classifier plus indoor firing-bay **OSHA PEL / NIOSH REL / IDLH** gaseous hazard quotients and **EPA BaP-TEQ** carcinogenic PAH inventory.
10. **Shelf-Life & Thermal Safety**: **NATO STANAG 4582** Arrhenius stabilizer depletion shelf-life model and **Frank-Kamenetskii** critical cook-off temperature ($T_{\text{crit}}$) vs. container diameter.
11. **Optimizer & Ternary**: Constrained SLSQP formulation optimizer and interactive 3-component simplex ternary phase diagrams.
12. **VoD Estimator**: Kamlet-Jacobs detonation velocity and Chapman-Jouguet ($P_{CJ}$) screening calculator.
