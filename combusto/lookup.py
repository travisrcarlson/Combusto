"""Chemical lookup module for Combusto.

Queries PubChem and NIST WebBook to retrieve chemical formulas, molecular weights,
SMILES, and standard enthalpies of formation.
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional

def query_pubchem(identifier: str, search_type: str = "name") -> Optional[Dict[str, Any]]:
    """Query PubChem for formula, weight, SMILES, and name.
    
    *identifier* can be a name or SMILES string.
    *search_type* can be 'name' or 'smiles'.
    """
    safe_id = urllib.parse.quote(identifier)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{search_type}/{safe_id}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            properties = data.get("PropertyTable", {}).get("Properties", [])
            if properties:
                prop = properties[0]
                return {
                    "formula": prop.get("MolecularFormula"),
                    "mw": float(prop.get("MolecularWeight", 0.0)),
                    "iupac_name": prop.get("IUPACName"),
                    "smiles": prop.get("CanonicalSMILES"),
                    "name": identifier,
                }
    except Exception:
        pass
    return None

def query_nist_enthalpy(name_or_formula: str) -> Optional[float]:
    """Scrape the NIST Chemistry WebBook for the standard enthalpy of formation.
    
    Returns the value in kJ/mol if found (preferring solid/liquid/gas in that order).
    """
    safe_name = urllib.parse.quote(name_or_formula)
    # Search by Name
    url = f"https://webbook.nist.gov/cgi/cbook.cgi?Name={safe_name}&Units=SI"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode("utf-8")
            
            # If it's a redirect or a compound page, parse it directly
            hf = _parse_nist_html(html)
            if hf is not None:
                return hf
            
            # Check if it was a search results list page. 
            # Look for links like /cgi/cbook.cgi?ID=C...
            matches = re.findall(r'/cgi/cbook\.cgi\?ID=(C\d+)', html)
            if matches:
                # Try the first match
                first_id = matches[0]
                detail_url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID={first_id}&Units=SI"
                req_detail = urllib.request.Request(detail_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req_detail, timeout=5) as detail_response:
                    detail_html = detail_response.read().decode("utf-8")
                    return _parse_nist_html(detail_html)
                    
    except Exception:
        pass
    
    # Try searching by Formula as fallback
    url_form = f"https://webbook.nist.gov/cgi/cbook.cgi?Formula={safe_name}&Units=SI"
    try:
        req = urllib.request.Request(url_form, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode("utf-8")
            hf = _parse_nist_html(html)
            if hf is not None:
                return hf
            matches = re.findall(r'/cgi/cbook\.cgi\?ID=(C\d+)', html)
            if matches:
                first_id = matches[0]
                detail_url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID={first_id}&Units=SI"
                req_detail = urllib.request.Request(detail_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req_detail, timeout=5) as detail_response:
                    detail_html = detail_response.read().decode("utf-8")
                    return _parse_nist_html(detail_html)
    except Exception:
        pass
        
    return None

def _parse_nist_html(html: str) -> Optional[float]:
    """Parse NIST HTML to find standard enthalpy of formation (Hf_298)."""
    # Look for table rows or lists with delta_f H_solid, delta_f H_liquid, or delta_f H_gas
    # e.g., Gas phase: Hf_gas or delta_f H°_gas
    # We will search for standard enthalpy patterns.
    # Typical: <li><strong>Δ<sub>f</sub>H°<sub>gas</sub></strong>: -74.6 kJ/mol</li>
    # Or in tables: <td>Standard enthalpy of formation (solid)</td><td>-74.6</td>
    
    # Regex 1: List items with Δ_f H°
    # We match: Δ<sub>f</sub>H°<sub>(solid|liquid|gas|condensed)</sub> ... value ... kJ/mol
    pattern_list = r'Δ<sub>f</sub>H°<sub>(solid|liquid|gas|condensed)</sub>.*?(-?\d+(?:\.\d+)?)\s*kJ/mol'
    matches = re.findall(pattern_list, html, re.IGNORECASE)
    
    # Sort phases to prefer solid/condensed first, then liquid, then gas
    phase_preference = {"solid": 0, "condensed": 1, "liquid": 2, "gas": 3}
    if matches:
        sorted_matches = sorted(matches, key=lambda x: phase_preference.get(x[0].lower(), 4))
        return float(sorted_matches[0][1])
        
    # Regex 2: Table data cells
    # We match td: Standard enthalpy of formation (solid/liquid/gas) followed by another td with the value
    pattern_table = r'Standard enthalpy of formation\s*\((solid|liquid|gas|condensed)\).*?class="num".*?(-?\d+(?:\.\d+)?)'
    table_matches = re.findall(pattern_table, html, re.DOTALL | re.IGNORECASE)
    if table_matches:
        sorted_matches = sorted(table_matches, key=lambda x: phase_preference.get(x[0].lower(), 4))
        return float(sorted_matches[0][1])

    # Regex 3: Simple check for "Standard enthalpy of formation" in text
    # e.g. "Standard enthalpy of formation ... -234.3 kJ/mol"
    pattern_simple = r'Standard enthalpy of formation.*?(-?\d+(?:\.\d+)?)\s*kJ/mol'
    simple_match = re.search(pattern_simple, html, re.DOTALL | re.IGNORECASE)
    if simple_match:
        return float(simple_match.group(1))
        
    return None
