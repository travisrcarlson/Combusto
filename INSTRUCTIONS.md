# Combusto: Desktop Pro Edition
**Developer & Administrative Setup Guide**

Welcome to the standalone internal developer version of Combusto. This localized fork (`app_local.py`) bypasses the main web platform restrictions, allowing you root-level control over the physical empirical databases and system API protocols.

---

## 1. Installation & Boot Protocol

Before executing Combusto, ensure you have Python 3.9+ installed and access to your system terminal (or Anaconda Prompt).

**Step 1:** Open your terminal and navigate to this exact extracted folder.
**Step 2:** Install the rigid mathematical dependencies required for the thermochemical engine:
```bash
pip install streamlit pandas numpy plotly setuptools
```
**Step 3:** Ignite the Local Pro version explicitly using Streamlit:
```bash
streamlit run app_local.py
```
*(Note: Attempting to run `app.py` will invoke the restricted public web version. Always use `app_local.py` for administrative access).*

---

## 2. Master Database Editor Mode

In the left-hand sidebar under **⚙️ Combusto Local Admin**, select **"Database Editor"**.

Combusto is fundamentally grounded in three core JSON databases:
1. `data/ingredients.json` — Empirical thermodynamic identifiers ($H_{f,298}$) for foundational precursors (AP, HMX, HTPB, AL).
2. `data/propellants.json` — Major standardized gun and rocket propellant composite presets.
3. `data/species.json` — Exhaustive dictionary of NASA 7-coefficient product decomposition species parameters.

**How to Edit:**
1. Upon selecting a database file from the dropdown, an interactive editable matrix will render.
2. You may double click any cell to manually overwrite names, mass fractions, or Heat of Formation variables.
3. You may add or delete entire rows to inject novel chemistry into the app.
4. Click **"💾 Save to Disk"**. This physically commands the application to overwrite the backend `.json` file on your hard drive, locking in your changes permanently for all future simulation runs.

---

## 3. API Key Manager & Secrets Implementation

In the left-hand sidebar under **⚙️ Combusto Local Admin**, select **"API Key Manager"**.

Combusto's future interoperability requires secure connections to major institutions like NIST, PubChem, and the Materials Project. However, hardcoding authorization keys into raw python scripts is an extreme security risk. 

**Secure Local Injection:**
1. Generate your private developer tokens from the respective institutions.
2. Paste them into the designated masked password fields in the Local API Key Manager tab.
3. Click **"🔒 Save Environment Secrets"**. 

**What happens next?**
The application will automatically write these exact keys into a hidden systems folder at `.streamlit/secrets.toml`. This is a strict local environment variable configuration file. When Combusto boots up, it natively parses this file, ingesting your API keys deeply into the script memory without ever exposing them to your source code or potential GitHub uploads.
