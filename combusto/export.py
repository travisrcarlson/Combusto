"""Export utilities for combustion results."""

from __future__ import annotations

import io
from typing import Dict

import pandas as pd

from .models import CombustionResult


def result_to_dataframe(result: CombustionResult) -> pd.DataFrame:
    """Convert species results to a pandas DataFrame."""
    rows = []
    for name in sorted(result.species_moles.keys(), key=lambda n: -result.species_moles[n]):
        rows.append({
            "Species": name,
            "Moles": result.species_moles[name],
            "Mole Fraction": result.species_mole_fractions.get(name, 0.0),
            "Mass Fraction": result.species_mass_fractions.get(name, 0.0),
        })
    return pd.DataFrame(rows)


def pah_details_to_dataframe(result: CombustionResult) -> pd.DataFrame:
    """Convert PAH details to a DataFrame."""
    if not result.pah_details:
        return pd.DataFrame(columns=["Species", "Formula", "Mole Fraction", "Mass Fraction", "Carbon Share"])
    rows = []
    for detail in result.pah_details:
        rows.append({
            "Species": detail["name"],
            "Formula": detail["formula"],
            "Mole Fraction": detail["mole_fraction"],
            "Mass Fraction": detail["mass_fraction"],
            "Carbon Share": detail["carbon_share"],
        })
    return pd.DataFrame(rows)


def carbon_distribution_to_dataframe(result: CombustionResult) -> pd.DataFrame:
    """Convert carbon distribution to a DataFrame."""
    rows = [{"Category": k, "Fraction": v} for k, v in result.carbon_distribution.items()]
    return pd.DataFrame(rows)


def element_balance_to_dataframe(result: CombustionResult) -> pd.DataFrame:
    """Convert element balance errors to a DataFrame."""
    rows = [{"Element": k, "Error %": v} for k, v in sorted(result.element_balance.items())]
    return pd.DataFrame(rows)


def result_to_csv(result: CombustionResult) -> str:
    """Export all results to CSV string."""
    df = result_to_dataframe(result)
    return df.to_csv(index=False)


def result_to_excel_bytes(result: CombustionResult) -> bytes:
    """Export results to Excel file bytes (multiple sheets)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        result_to_dataframe(result).to_excel(writer, sheet_name="Products", index=False)
        pah_details_to_dataframe(result).to_excel(writer, sheet_name="PAH Details", index=False)
        carbon_distribution_to_dataframe(result).to_excel(writer, sheet_name="Carbon Distribution", index=False)
        element_balance_to_dataframe(result).to_excel(writer, sheet_name="Element Balance", index=False)

        # Summary sheet
        summary = pd.DataFrame([{
            "Equivalence Ratio": result.equivalence_ratio,
            "Oxygen Balance %": result.oxygen_balance_pct,
            "Adiabatic Flame Temp (K)": result.adiabatic_flame_temp_k,
            "Mixture Cp (J/mol·K)": result.mixture_cp,
            "Pressure (atm)": result.pressure_atm,
        }])
        summary.to_excel(writer, sheet_name="Summary", index=False)
    return buf.getvalue()
