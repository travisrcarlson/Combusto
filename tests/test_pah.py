"""Tests for PAH allocation module."""

import pytest
from combusto.pah_allocation import (
    PRESETS,
    allocate_pahs,
    carbon_fraction_to_pahs,
)
from combusto.species_db import load_species_db


def test_no_pahs_at_stoichiometric():
    """At phi <= 1, no carbon should go to PAHs."""
    config = PRESETS["moderate"]
    f = carbon_fraction_to_pahs(1.0, config)
    assert f == 0.0

    f = carbon_fraction_to_pahs(0.8, config)
    assert f == 0.0


def test_pahs_increase_with_phi():
    """PAH fraction should increase with equivalence ratio."""
    config = PRESETS["moderate"]
    f1 = carbon_fraction_to_pahs(1.5, config)
    f2 = carbon_fraction_to_pahs(2.0, config)
    f3 = carbon_fraction_to_pahs(3.0, config)
    assert f1 < f2 < f3


def test_pahs_capped_at_max():
    """PAH fraction should not exceed max_fraction."""
    config = PRESETS["moderate"]
    f = carbon_fraction_to_pahs(100.0, config)
    assert f <= config.max_fraction


def test_allocate_pahs_conserves_carbon():
    """Carbon allocated + remaining should equal total."""
    db = load_species_db()
    pah_species = {n: s for n, s in db.items() if s.category in ("pah", "lump")}
    config = PRESETS["moderate"]

    total_c = 0.05  # mol
    total_h = 0.10  # mol
    phi = 2.0

    pah_moles, rem_c, rem_h = allocate_pahs(total_c, total_h, phi, config, pah_species)

    from combusto.models import parse_formula
    c_in_pahs = sum(
        moles * parse_formula(pah_species[n].formula).get("C", 0.0)
        for n, moles in pah_moles.items()
        if n in pah_species
    )

    assert abs(c_in_pahs + rem_c - total_c) < 1e-10, "Carbon not conserved"


def test_none_preset_produces_no_pahs():
    """The 'none' preset should allocate zero PAHs."""
    config = PRESETS["none"]
    f = carbon_fraction_to_pahs(5.0, config)
    assert f == 0.0

    db = load_species_db()
    pah_species = {n: s for n, s in db.items() if s.category in ("pah", "lump")}
    pah_moles, rem_c, _ = allocate_pahs(0.1, 0.2, 5.0, config, pah_species)
    assert len(pah_moles) == 0
    assert abs(rem_c - 0.1) < 1e-10
