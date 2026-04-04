# Combusto

Gun propellant combustion product predictor with PAH/soot modeling.

Unlike classical thermochemical codes (ICT, NASA CEA, EXPLO5) that assume equilibrium to simple small molecules, Combusto explicitly models polycyclic aromatic hydrocarbons (PAHs) and soot precursors as incomplete-combustion products.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web UI
streamlit run app.py

# Run tests
python -m pytest tests/ -v
```

## Architecture

```
combusto/
├── app.py                  # Streamlit web UI (4 tabs)
├── combusto/
│   ├── models.py           # Data models (Species, Formulation, CombustionResult, etc.)
│   ├── chemistry.py        # Element balance, oxygen balance, equivalence ratio
│   ├── thermodynamics.py   # NASA 7-coefficient polynomial evaluation (Cp, H, S, G)
│   ├── equilibrium.py      # Gibbs minimization + adiabatic T solver + isochoric mode
│   ├── pah_allocation.py   # PAH/soot carbon allocation rules (4 presets)
│   ├── sensitivity.py      # Parametric sweeps (phi, T, PAH model, multi-formulation)
│   ├── species_db.py       # Species database loader
│   ├── presets.py          # Preset formulations + save/load utilities
│   └── export.py           # CSV/Excel/JSON export
├── data/
│   ├── species.json        # 30+ species: small molecules, PAHs, metal oxides, soot
│   └── propellants.json    # 5 preset propellant formulations
└── tests/
    ├── test_chemistry.py   # Formula parsing, OB%, phi, element balance (9 tests)
    ├── test_equilibrium.py # Equilibrium solver sanity checks (3 tests)
    ├── test_pah.py         # PAH allocation rules (5 tests)
    └── test_advanced.py    # Adiabatic solver, isochoric, sensitivity, serialization (7 tests)
```

## Features

### Combustion Solver
- **Gibbs free energy minimization** via scipy SLSQP with element-conservation constraints
- **Self-consistent adiabatic flame temperature**: iterates T-composition until converged (damped relaxation)
- **Isobaric and isochoric modes**: isochoric uses Noble-Abel equation of state for chamber pressure
- **Ballistic outputs**: propellant force constant (J/g), gamma (Cp/Cv), mean MW, gas moles

### PAH/Soot Modeling
- Four allocation presets (none/conservative/moderate/aggressive)
- Configurable base fraction, phi exponent, max fraction
- Distribution among 6+ PAH species (benzene through coronene) + soot precursor
- Carbon accounting: CO2 vs CO vs CH4 vs PAHs vs soot

### Species Database (data/species.json)
- 15 small molecules: CO2, CO, H2O, H2, N2, NO, NO2, O2, CH4, OH, H, O, NH3, HCN, C2H2, C2H4, HCl, HF
- 10 PAH species: benzene, toluene, naphthalene, acenaphthylene, fluorene, phenanthrene, anthracene, fluoranthene, pyrene, chrysene, benzo[a]pyrene, coronene
- Lumped soot species (C20H10, solid C)
- Metal oxides: Al2O3, PbO, CuO, Fe2O3
- Extensible via JSON config

### Sensitivity Analysis
- **Equivalence ratio sweep**: vary oxygen content across phi range
- **Temperature sweep**: product distribution vs equilibrium temperature
- **PAH model comparison**: side-by-side of all 4 presets
- **Multi-formulation comparison**: run all formulations under same conditions

### UI
- **Formulation editor**: add/remove/edit components with live validation
- **Conditions panel**: isobaric, isochoric, or self-consistent adiabatic mode
- **Results panel**: metrics, product tables, mole fraction bar charts, carbon distribution pie charts, PAH details, element balance diagnostics
- **Sensitivity panel**: interactive sweep plots with Plotly
- **Comparison**: side-by-side metrics and charts for two formulations
- **Export**: CSV, Excel (multi-sheet), JSON
- **Save/Load**: persist custom formulations to JSON

## Preset Propellants

| Name | Description |
|------|-------------|
| Single-base (NC) | Nitrocellulose with DPA stabilizer and DBP plasticizer |
| Double-base (NC/NG) | NC + nitroglycerin with centralite stabilizer |
| Triple-base (NC/NG/NQ) | NC + NG + nitroguanidine |
| Composite (NC/NG/Al) | Aluminized double-base with AP |
| Ball powder (WC 844) | Spherical rifle powder with flash suppressant |

## Extending the Species Database

Edit `data/species.json` to add new species:

```json
{
  "name": "species_name",
  "formula": "C10H8",
  "phase": "gas",
  "category": "pah",
  "hf_298": 150.96,
  "t_mid": 1000.0,
  "thermo_low": [a1, a2, a3, a4, a5, a6, a7],
  "thermo_high": [a1, a2, a3, a4, a5, a6, a7]
}
```

Categories: `small`, `pah`, `lump`, `metal_oxide`.
NASA polynomial coefficients can be sourced from Burcat/Ruscic thermodynamic tables or GRI-Mech.

## Dependencies

- Python 3.10+
- streamlit, numpy, scipy, pandas, openpyxl, plotly, pytest
