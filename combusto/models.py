"""Data models for Combusto."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Element helpers
# ---------------------------------------------------------------------------

# Standard atomic masses (g/mol) — IUPAC 2021
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Ti": 47.867, "Fe": 55.845, "Cu": 63.546, "Zn": 65.38, "Pb": 207.2,
    "Ba": 137.327, "Bi": 208.980, "Sn": 118.710, "W": 183.84,
}


def parse_formula(formula: str) -> Dict[str, float]:
    """Parse a chemical formula string into {element: count}.

    Supports parenthetical groups and fractional subscripts.
    Examples: "C6H7N3O9", "Al2O3", "Ca(NO3)2", "C6.1H7.3N2.8O9.5"
    """
    return _parse_group(formula, 1.0)


def _parse_group(s: str, multiplier: float) -> Dict[str, float]:
    result: Dict[str, float] = {}
    i = 0
    while i < len(s):
        if s[i] == "(":
            # Find matching closing paren
            depth = 1
            j = i + 1
            while j < len(s) and depth > 0:
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                j += 1
            inner = s[i + 1 : j - 1]
            # Read subscript after ")"
            num_str = ""
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                num_str += s[j]
                j += 1
            sub = float(num_str) if num_str else 1.0
            for elem, cnt in _parse_group(inner, 1.0).items():
                result[elem] = result.get(elem, 0.0) + cnt * sub * multiplier
            i = j
        elif s[i].isupper():
            elem = s[i]
            i += 1
            while i < len(s) and s[i].islower():
                elem += s[i]
                i += 1
            num_str = ""
            while i < len(s) and (s[i].isdigit() or s[i] == "."):
                num_str += s[i]
                i += 1
            count = float(num_str) if num_str else 1.0
            result[elem] = result.get(elem, 0.0) + count * multiplier
        else:
            i += 1
    return result


def molecular_weight(formula: str) -> float:
    """Compute molecular weight from a chemical formula string (g/mol)."""
    composition = parse_formula(formula)
    mw = 0.0
    for elem, count in composition.items():
        if elem not in ATOMIC_MASSES:
            raise ValueError(f"Unknown element: {elem}")
        mw += ATOMIC_MASSES[elem] * count
    return mw


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

@dataclass
class Species:
    """A chemical species in the product database."""
    name: str
    formula: str
    phase: str = "gas"  # "gas" or "condensed"
    category: str = "small"  # "small", "pah", "lump", "metal_oxide"
    # NASA 7-coefficient polynomial: [a1..a7] for two T ranges
    thermo_low: List[float] = field(default_factory=lambda: [0.0] * 7)
    thermo_high: List[float] = field(default_factory=lambda: [0.0] * 7)
    t_mid: float = 1000.0  # Transition temperature (K)
    t_low: float = 200.0
    t_high: float = 6000.0
    hf_298: float = 0.0  # Standard enthalpy of formation kJ/mol (backup)
    cas: str = ""
    source: str = "NASA Glenn TP-2002-211556 / Burcat"
    molar_covolume_cm3_mol: float = 0.0

    @property
    def composition(self) -> Dict[str, float]:
        return parse_formula(self.formula)

    @property
    def mw(self) -> float:
        return molecular_weight(self.formula)


@dataclass
class PropellantComponent:
    """A single ingredient in a propellant formulation."""
    name: str
    formula: str
    mass_fraction: float  # 0..1
    hf_298: float = 0.0  # Enthalpy of formation kJ/mol

    @property
    def composition(self) -> Dict[str, float]:
        return parse_formula(self.formula)

    @property
    def mw(self) -> float:
        return molecular_weight(self.formula)


@dataclass
class Formulation:
    """A propellant formulation (collection of components)."""
    name: str
    components: List[PropellantComponent] = field(default_factory=list)
    description: str = ""

    def elemental_composition(self) -> Dict[str, float]:
        """Compute overall moles of each element per gram of propellant."""
        totals: Dict[str, float] = {}
        for comp in self.components:
            mw = comp.mw
            moles_per_gram = comp.mass_fraction / mw  # mol component per g propellant
            for elem, count in comp.composition.items():
                totals[elem] = totals.get(elem, 0.0) + moles_per_gram * count
        return totals

    def elemental_mass_fractions(self) -> Dict[str, float]:
        """Mass fraction of each element in the propellant."""
        elem_moles = self.elemental_composition()
        total_mass = sum(
            moles * ATOMIC_MASSES.get(e, 0.0) for e, moles in elem_moles.items()
        )
        if total_mass == 0:
            return {}
        return {
            e: moles * ATOMIC_MASSES.get(e, 0.0) / total_mass
            for e, moles in elem_moles.items()
        }


@dataclass
class CombustionConditions:
    """Operating conditions for a combustion scenario."""
    temperature_k: float = 298.15  # Initial temperature (K)
    pressure_atm: float = 1.0  # Pressure (atm)
    mode: str = "isobaric"  # "isobaric", "isochoric", or "bomb_calorimeter"
    loading_density: Optional[float] = None  # g/cm³ (for isochoric / bomb_calorimeter)
    chamber_volume_cm3: Optional[float] = None
    propellant_mass_g: Optional[float] = None
    inert_gas: Optional[str] = None  # e.g. "N2", "Ar"
    inert_moles: float = 0.0
    covolume_m3_per_mol: float = 1.0e-3  # Noble-Abel covolume (m³/mol)
    real_gas_fugacity: bool = False  # Virial / BLAKE covolume fugacity ln(phi_i) = b_i * P / (R * T)


@dataclass
class PAHAllocationConfig:
    """Configuration for PAH/soot carbon allocation."""
    model_name: str = "moderate"
    # Fraction of available C allocated to PAHs when equivalence ratio >= threshold
    base_fraction: float = 0.05  # 5% of carbon to PAHs at phi=1
    phi_exponent: float = 2.0  # scaling with (phi - 1)
    max_fraction: float = 0.40  # cap at 40%
    # Distribution among PAH species (weights, normalized internally)
    pah_distribution: Dict[str, float] = field(default_factory=lambda: {
        "C6H6": 0.30,    # benzene
        "C7H8": 0.15,    # toluene
        "C10H8": 0.25,   # naphthalene
        "C14H10": 0.15,  # phenanthrene
        "C16H10": 0.10,  # pyrene
        "C24H12": 0.05,  # coronene
    })


@dataclass
class CombustionResult:
    """Full result of a combustion calculation."""
    species_moles: Dict[str, float] = field(default_factory=dict)
    species_mole_fractions: Dict[str, float] = field(default_factory=dict)
    species_mass_fractions: Dict[str, float] = field(default_factory=dict)
    temperature_k: float = 0.0
    pressure_atm: float = 0.0
    equivalence_ratio: float = 0.0
    oxygen_balance_pct: float = 0.0
    element_balance: Dict[str, float] = field(default_factory=dict)
    adiabatic_flame_temp_k: float = 0.0
    mixture_cp: float = 0.0  # J/(mol·K)
    enthalpy_change_kj: float = 0.0
    carbon_distribution: Dict[str, float] = field(default_factory=dict)
    pah_details: List[Dict] = field(default_factory=list)
    # Extended fields for ballistic / isochoric calculations
    computed_pressure_atm: float = 0.0
    gas_moles_total: float = 0.0
    force_constant_j_per_g: float = 0.0  # propellant impetus F = n_gas * R * T_v per gram
    isobaric_force_j_per_g: float = 0.0  # isobaric RT product = n_gas * R * T_p per gram
    isochoric_flame_temp_k: float = 0.0  # closed-vessel constant-volume flame temp T_v (K)
    covolume_cm3_per_g: float = 0.0      # composition-derived Noble-Abel covolume eta (cm3/g)
    mean_molecular_weight: float = 0.0
    gamma: float = 0.0  # ratio of specific heats Cp/Cv for gas mixture
    heat_of_explosion_j_per_g: float = 0.0
    specific_impulse_s: float = 0.0
    gas_volume_l_per_kg: float = 0.0
    compressibility_z: float = 1.0
    converged: bool = False
    iterations: int = 0
