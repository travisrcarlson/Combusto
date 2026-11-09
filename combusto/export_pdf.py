"""PDF Export module for Combusto.

Generates a professional PDF report containing the formulation details,
oxygen balance, calculated metrics, product composition, and carbon distribution.
"""

from __future__ import annotations

import io
from typing import Dict, Any, List, Optional
from xml.sax.saxutils import escape as _xml_escape
from combusto.models import CombustionResult, Formulation

# Safely import ReportLab modules
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.graphics.shapes import Drawing, String, Rect
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


_PDF_TRANSLIT = str.maketrans({
    "φ": "phi", "η": "eta", "γ": "gamma", "Δ": "Delta", "±": "+/-",
    "—": "-", "–": "-", "•": "*", "µ": "u", "°": " deg", "³": "3",
    "²": "2", "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₈": "8", "₉": "9", "σ": "sigma", "ε": "eps", "→": "->",
    "≥": ">=", "≤": "<=", "×": "x", "≈": "~",
})


def _pdf_safe_str(s: str) -> str:
    txt = str(s).translate(_PDF_TRANSLIT)
    txt = txt.encode("latin-1", errors="replace").decode("latin-1")
    return txt.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pure_python_pdf(lines: List[tuple]) -> bytes:
    """Zero-dependency PDF 1.4 generator fallback using built-in Type1 Helvetica."""
    lines_per_page = 46
    pages = [lines[i : i + lines_per_page] for i in range(0, max(1, len(lines)), lines_per_page)]

    objects: List[bytes] = []
    # Obj 1: Catalog, Obj 2: Pages, Obj 3: Helvetica, Obj 4: Helvetica-Bold
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")  # placeholder for /Pages
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_obj_ids: List[int] = []
    for pg_lines in pages:
        stream_cmds = [b"BT"]
        y = 748
        for idx_l, (raw_txt, is_bold, font_sz) in enumerate(pg_lines):
            f_tag = b"/F2" if is_bold else b"/F1"
            safe_b = _pdf_safe_str(raw_txt).encode("latin-1")
            stream_cmds.append(
                f"1 0 0 1 42 {y} Tm {f_tag} {font_sz} Tf ({safe_b.decode('latin-1')}) Tj".encode("latin-1")
            )
            y -= (font_sz + 5)
        stream_cmds.append(b"ET")
        stream_bytes = b"\n".join(stream_cmds)

        content_id = len(objects) + 1
        objects.append(f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1") + stream_bytes + b"\nendstream")

        page_id = len(objects) + 1
        page_obj_ids.append(page_id)
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")
        objects.append(page_dict)

    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_obj_ids)} >>".encode("latin-1")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj_body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{idx} 0 obj\n".encode("latin-1"))
        out.write(obj_body)
        out.write(b"\nendobj\n")

    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    return out.getvalue()

def export_result_pdf(result: CombustionResult, formulation: Formulation) -> bytes:
    """Compile combustion results and formulation details into a PDF document.
    
    Returns raw PDF bytes.
    """
    if not REPORTLAB_AVAILABLE:
        fb_lines = [
            ("Combusto Lab Report", True, 16),
            (f"Formulation: {formulation.name}", True, 11),
            (f"Description: {formulation.description or 'Standard energetic formulation'}", False, 10),
            ("", False, 8),
            ("1. Formulation Ingredients", True, 12),
        ]
        for c in formulation.components:
            fb_lines.append((f"  - {c.name} ({c.formula}): {c.mass_fraction*100:.2f}% | Hf298 = {c.hf_298:.1f} kJ/mol", False, 9))
        fb_lines.extend([
            ("", False, 8),
            ("2. Thermochemical & Performance Indicators", True, 12),
            (f"  Equivalence Ratio (phi): {result.equivalence_ratio:.3f}  |  Oxygen Balance: {result.oxygen_balance_pct:+.2f}%", False, 9),
            (f"  Isobaric Tp: {result.adiabatic_flame_temp_k:.1f} K  |  Closed-Vessel Tv: {result.isochoric_flame_temp_k:.1f} K", False, 9),
            (f"  Gun Impetus F: {result.force_constant_j_per_g:.1f} J/g  |  BLAKE Covolume eta: {result.covolume_cm3_per_g:.3f} cm3/g", False, 9),
            (f"  Calorimetric Qv: {result.heat_of_explosion_j_per_g:.0f} J/g  |  Gamma: {result.gamma:.3f}  |  MW: {result.mean_molecular_weight:.2f} g/mol", False, 9),
        ])
        return _build_pure_python_pdf(fb_lines)
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#00bbff'),
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'DocSection',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#141724'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#333333')
    )
    
    body_bold_style = ParagraphStyle(
        'DocBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=body_style,
        fontSize=9,
        leading=11
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=body_style,
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )
    
    story = []
    
    # Title Block
    story.append(Paragraph("Combusto Lab Report", title_style))
    story.append(Paragraph("Gun Propellant Combustion Product & Performance Analytics", body_bold_style))
    story.append(Spacer(1, 15))
    
    # Formulation Information
    story.append(Paragraph("1. Propellant Formulation Details", section_style))
    story.append(Paragraph(f"<b>Name:</b> {formulation.name}", body_style))
    if formulation.description:
        story.append(Paragraph(f"<b>Description:</b> {formulation.description}", body_style))
        
    story.append(Spacer(1, 8))
    
    # Formulation Components Table
    form_data = [[Paragraph("Component Name", table_header_style), 
                  Paragraph("Stoichiometric Formula", table_header_style), 
                  Paragraph("Mass Fraction (%)", table_header_style), 
                  Paragraph("Hf_298 (kJ/mol)", table_header_style)]]
                  
    for c in formulation.components:
        form_data.append([
            Paragraph(c.name, table_text_style),
            Paragraph(c.formula, table_text_style),
            Paragraph(f"{c.mass_fraction*100.0:.2f}%", table_text_style),
            Paragraph(f"{c.hf_298:.1f}", table_text_style)
        ])
        
    form_table = Table(form_data, colWidths=[200, 130, 100, 100])
    form_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#141724')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    story.append(form_table)
    story.append(Spacer(1, 15))
    
    # Performance & Diagnostics Block
    story.append(Paragraph("2. Thermochemical & Performance Indicators", section_style))
    
    metric_data = [
        [Paragraph("Parameter", table_header_style), Paragraph("Calculated Value", table_header_style), Paragraph("Meaning / Reference Standard", table_header_style)],
        [Paragraph("Equivalence Ratio (&phi;)", table_text_style), Paragraph(f"{result.equivalence_ratio:.3f}", table_text_style), Paragraph("&phi; > 1 indicates fuel-rich stoichiometry", table_text_style)],
        [Paragraph("Oxygen Balance (OB%)", table_text_style), Paragraph(f"{result.oxygen_balance_pct:.2f}%", table_text_style), Paragraph("Stoichiometric oxygen balance to CO2/H2O", table_text_style)],
        [Paragraph("Isobaric Flame Temp (T_p)", table_text_style), Paragraph(f"{result.adiabatic_flame_temp_k:.1f} K", table_text_style), Paragraph("Constant-pressure flame temp (&Delta;H = 0, NASA CEA)", table_text_style)],
        [Paragraph("Isochoric Flame Temp (T_v)", table_text_style), Paragraph(f"{result.isochoric_flame_temp_k:.1f} K", table_text_style), Paragraph("Closed-vessel gun chamber temp (&Delta;U = 0, STANAG 4115)", table_text_style)],
        [Paragraph("Propellant Force / Impetus (F)", table_text_style), Paragraph(f"{result.force_constant_j_per_g:.1f} J/g", table_text_style), Paragraph("n_gas * R * T_v (US Army BRL / STANAG 4115)", table_text_style)],
        [Paragraph("Noble-Abel Covolume (&eta;)", table_text_style), Paragraph(f"{result.covolume_cm3_per_g:.3f} cm3/g", table_text_style), Paragraph("Composition-derived &Sigma; n_i b_i (BLAKE ARBRL-TR-02411)", table_text_style)],
        [Paragraph("Heat of Explosion (Q_v)", table_text_style), Paragraph(f"{result.heat_of_explosion_j_per_g:.0f} J/g", table_text_style), Paragraph("Isothermal 298.15 K bomb calorimetry (H2O liquid)", table_text_style)],
        [Paragraph("Specific Impulse (Isp)", table_text_style), Paragraph(f"{result.specific_impulse_s:.1f} s", table_text_style), Paragraph("Ideal nozzle expansion thrust per unit weight flow", table_text_style)],
        [Paragraph("Ratio of Specific Heats (&gamma;)", table_text_style), Paragraph(f"{result.gamma:.3f}", table_text_style), Paragraph("Gas-phase Cp/Cv at chamber temperature", table_text_style)],
        [Paragraph("Mean Molecular Weight", table_text_style), Paragraph(f"{result.mean_molecular_weight:.2f} g/mol", table_text_style), Paragraph("Molar-weighted average gas product weight", table_text_style)]
    ]
    
    metric_table = Table(metric_data, colWidths=[180, 130, 220])
    metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#141724')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 15))
    
    # Product Species Distribution Block
    story.append(Paragraph("3. Major Product Species Mole Fractions", section_style))
    
    # Gather major gas species mole fractions
    sorted_species = sorted(result.species_mole_fractions.items(), key=lambda x: x[1], reverse=True)
    
    spec_data = [[Paragraph("Species Formula", table_header_style), 
                  Paragraph("Phase", table_header_style), 
                  Paragraph("Mole Fraction", table_header_style), 
                  Paragraph("Mass Fraction", table_header_style)]]
                  
    for sp_name, mole_frac in sorted_species[:10]: # Top 10 species
        mass_frac = result.species_mass_fractions.get(sp_name, 0.0)
        # Check phase
        phase_str = "Gas"
        if sp_name in ("C(s)", "Al2O3(s)", "CuO(s)", "PbO(s)", "Fe2O3(s)"):
            phase_str = "Solid"
            
        spec_data.append([
            Paragraph(sp_name, table_text_style),
            Paragraph(phase_str, table_text_style),
            Paragraph(f"{mole_frac:.5f}", table_text_style),
            Paragraph(f"{mass_frac:.5f}", table_text_style)
        ])
        
    spec_table = Table(spec_data, colWidths=[130, 100, 150, 150])
    spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#141724')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    story.append(spec_table)
    story.append(Spacer(1, 15))
    
    # Carbon Distribution Native Chart (Only if we have carbon)
    if result.carbon_distribution:
        story.append(Paragraph("4. Carbon Element Budget Allocation", section_style))
        story.append(Paragraph("Distribution of carbon atoms across final exhaust species:", body_style))
        story.append(Spacer(1, 5))
        
        # Table showing carbon allocation
        c_data = [[Paragraph("Exhaust Family", table_header_style), Paragraph("Carbon Atom Fraction (%)", table_header_style)]]
        for name, frac in result.carbon_distribution.items():
            c_data.append([
                Paragraph(name, table_text_style),
                Paragraph(f"{frac*100.0:.1f}%", table_text_style)
            ])
            
        c_table = Table(c_data, colWidths=[200, 150])
        c_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#141724')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')])
        ]))
        
        # Add visual layout (Pie chart side-by-side with table)
        d = Drawing(180, 100)
        pc = Pie()
        pc.x = 20
        pc.y = 10
        pc.width = 80
        pc.height = 80
        
        # Load data to pie chart
        pie_labels = []
        pie_data = []
        for name, frac in result.carbon_distribution.items():
            if frac > 0.01:
                pie_labels.append(name)
                pie_data.append(frac)
                
        pc.data = pie_data
        pc.labels = pie_labels
        pc.sideLabels = False
        d.add(pc)
        
        # Wrap side-by-side in double columns table
        double_cols = [[c_table, d]]
        cols_table = Table(double_cols, colWidths=[350, 180])
        cols_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0)
        ]))
        story.append(cols_table)

    # Section 5: Data Provenance & Primary Literature Citations
    story.append(Spacer(1, 14))
    story.append(Paragraph("5. Thermochemical Grounding & Primary References", section_style))
    refs = [
        "1. McBride, B. J., Zehe, M. J., & Gordon, S. (2002). NASA Glenn Coefficients for Calculating Thermodynamic Properties of Individual Species, NASA/TP-2002-211556.",
        "2. Chase, M. W. (1998). NIST-JANAF Thermochemical Tables (4th Ed.), J. Phys. Chem. Ref. Data, Monograph 9.",
        "3. Freedman, E. (1982). BLAKE: A Thermodynamics Code Based on TIGER, US Army BRL Report ARBRL-TR-02411; Anderson & Fickie (1987), IBHVG2, BRL-TR-2829.",
        "4. Fraunhofer ICT Database of Thermochemical Values (7th Ed.) & NATO STANAG 4115 / 4400 / 4582.",
        "5. US EPA AP-42 Chapter 15 (Ordnance Detonation & Firing Emission Factors) & NIOSH / OSHA 29 CFR 1910.1000."
    ]
    for r_txt in refs:
        story.append(Paragraph(r_txt, table_text_style))
        story.append(Spacer(1, 2))
        
    # Build Document
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def _styled_table(rows: List[List[Any]], col_widths: List[float]) -> "Table":
    """Helper to build a consistently styled ReportLab Table."""
    tbl = Table(rows, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#141724')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d8dee9')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fb')]),
    ]))
    return tbl


def export_full_dossier_pdf(
    result: CombustionResult,
    formulation: Formulation,
    suite_data: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Generate a comprehensive Multi-Suite Engineering Dossier PDF.

    Combines Thermochemistry, Shifting/Frozen Nozzle Performance, Interior Ballistics
    & Barrel Erosion, NATO STANAG 2895 Climatic & Monte Carlo Dispersion, Muzzle Flash,
    STANAG 4400 Plume Smoke / OSHA Range Toxicity, and STANAG 4582 Shelf-Life / Cook-Off.
    """
    if not REPORTLAB_AVAILABLE:
        suite = suite_data or {}
        fb_lines = [
            ("COMBUSTO - Multi-Suite Propellant Engineering Dossier", True, 15),
            (f"Propellant Passport: {formulation.name}", True, 11),
            (f"Specification: {formulation.description or 'Multi-physics engineering characterization'}", False, 9),
            ("", False, 6),
            ("1. Formulation Composition", True, 11),
        ]
        for c in formulation.components:
            fb_lines.append((f"  * {c.name} ({c.formula}) — {c.mass_fraction*100:.2f}% | Hf298 = {c.hf_298:.1f} kJ/mol", False, 9))
        fb_lines.extend([
            ("", False, 6),
            ("2. Gibbs Equilibrium Thermochemistry & Real-Gas State", True, 11),
            (f"  phi = {result.equivalence_ratio:.3f} | OB = {result.oxygen_balance_pct:+.2f}% | Z = {getattr(result, 'compressibility_z', 1.0):.3f}", False, 9),
            (f"  Tp = {result.adiabatic_flame_temp_k:.1f} K | Tv = {result.isochoric_flame_temp_k:.1f} K | Impetus F = {result.force_constant_j_per_g:.1f} J/g", False, 9),
            (f"  Covolume eta = {result.covolume_cm3_per_g:.3f} cm3/g | Qv = {result.heat_of_explosion_j_per_g:.0f} J/g | gamma = {result.gamma:.3f}", False, 9),
        ])
        for sp_n, x_i in sorted(result.species_mole_fractions.items(), key=lambda kv: kv[1], reverse=True)[:8]:
            fb_lines.append((f"    Species {sp_n}: X = {x_i:.5f}, Y = {result.species_mass_fractions.get(sp_n, 0.0):.5f}", False, 8.5))
        noz = suite.get("nozzle_out")
        if noz and "frozen" in noz and "shifting" in noz:
            fz, sh = noz["frozen"], noz["shifting"]
            fb_lines.extend([
                ("", False, 6),
                ("3. Rocket Nozzle Expansion (Frozen vs Shifting)", True, 11),
                (f"  C*: Frozen {fz['c_star_m_s']:.1f} m/s vs Shifting {sh['c_star_m_s']:.1f} m/s", False, 9),
                (f"  Isp,vac: Frozen {fz['isp_vacuum_s']:.2f} s vs Shifting {sh['isp_vacuum_s']:.2f} s (Gain {sh['isp_vacuum_s']-fz['isp_vacuum_s']:+.2f} s)", False, 9),
                (f"  Exit Temp Te: Frozen {fz['exit_temp_k']:.0f} K vs Shifting {sh['exit_temp_k']:.0f} K | Ae/At = {sh['area_ratio_ae_at']:.2f}", False, 9),
            ])
        ib, ero = suite.get("ib_out"), suite.get("ero_out")
        clim, mc = suite.get("climatic_out"), suite.get("mc_out")
        if any(x is not None for x in (ib, ero, clim, mc)):
            fb_lines.extend([("", False, 6), ("4. Interior Ballistics, Barrel Erosion & Tolerance Dispersion", True, 11)])
            if ib:
                fb_lines.append((f"  P_max = {ib['max_pressure_mpa']:.1f} MPa | Muzzle Velocity V0 = {ib['muzzle_velocity_m_s']:.1f} m/s", False, 9))
            if ero:
                fb_lines.append((f"  Bore Wall Peak = {ero['peak_wall_temp_k']:.0f} K | Wear = {ero['wear_um_per_shot']:.2f} um/shot | Life = {ero['estimated_barrel_life_rounds']} EFC", False, 9))
            if clim:
                fb_lines.append((f"  STANAG 2895 (-40C -> +63C): {clim['cold_pmax_mpa']:.1f} -> {clim['hot_pmax_mpa']:.1f} MPa", False, 9))
            if mc:
                fb_lines.append((f"  Monte Carlo: P_max+3sigma = {mc['p_max_plus_3sigma_mpa']:.1f} MPa | sigma(V0) = {mc['std_v_muz_m_s']:.2f} m/s | Cpk = {mc['cpk_proof']:.2f}", False, 9))
        mf, tox, stab = suite.get("mf_out"), suite.get("tox_out"), suite.get("stab_out")
        if mf or tox or stab:
            fb_lines.extend([("", False, 6), ("5. Signature, Range Hygiene & STANAG 4582 Surveillance", True, 11)])
            if mf:
                fb_lines.append((f"  Muzzle Flash: {mf.get('flash_status', 'N/A')} | Peak Plume = {mf.get('peak_plume_temp_k', 0):.0f} K", False, 9))
            if tox:
                fb_lines.append((f"  Smoke: {tox.get('agard_smoke_class', 'N/A')} | OSHA HQ = {tox.get('max_hazard_quotient', 0):.2f}x PEL", False, 9))
            if stab:
                fb_lines.append((f"  STANAG 4582 Life @ 25C = {stab.get('shelf_life_at_25c_years', 0):.1f} yr | Cook-Off T_crit = {stab.get('critical_cookoff_temp_c', 0):.1f} C", False, 9))
        fb_lines.extend([
            ("", False, 6),
            ("6. Primary References", True, 11),
            ("  NASA/TP-2002-211556; NIST-JANAF 4th Ed.; US Army BRL BLAKE (ARBRL-TR-02411); STANAG 4115/2895/4400/4582.", False, 8.5),
        ])
        return _build_pure_python_pdf(fb_lines)

    suite = suite_data or {}
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DossierTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=25,
        textColor=colors.HexColor('#0077aa'),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'DossierSub',
        parent=styles['BodyText'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        'DossierSec',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=11,
        spaceAfter=5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        'DossierBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor('#1e293b'),
    )
    th_style = ParagraphStyle(
        'DossierTH',
        parent=body_style,
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=10.5,
        textColor=colors.white,
    )
    td_style = ParagraphStyle(
        'DossierTD',
        parent=body_style,
        fontSize=8.5,
        leading=10.5,
    )

    def P_th(txt: str) -> Paragraph:
        return Paragraph(txt, th_style)

    def P_td(txt: str) -> Paragraph:
        return Paragraph(_xml_escape(str(txt)), td_style)

    story: List[Any] = []
    story.append(Paragraph("COMBUSTO — Multi-Suite Propellant Engineering Dossier", title_style))
    story.append(Paragraph(
        f"Propellant Passport: {_xml_escape(formulation.name)}  |  "
        "Thermochemistry • Ballistics • Nozzle • Signature • Hygiene • Surveillance",
        subtitle_style,
    ))

    # --- 1. Formulation & Stoichiometry ---
    story.append(Paragraph("1. Formulation Composition & Energetic Stoichiometry", section_style))
    if formulation.description:
        story.append(Paragraph(f"<b>Specification / Notes:</b> {_xml_escape(formulation.description)}", body_style))
        story.append(Spacer(1, 4))

    f_rows = [[P_th("Ingredient"), P_th("Formula"), P_th("Mass %"), P_th("ΔHf,298 (kJ/mol)")]]
    for c in formulation.components:
        f_rows.append([
            P_td(c.name),
            P_td(c.formula),
            P_td(f"{c.mass_fraction * 100.0:.2f}%"),
            P_td(f"{c.hf_298:.1f}"),
        ])
    story.append(_styled_table(f_rows, [220, 120, 95, 105]))
    story.append(Spacer(1, 8))

    # --- 2. Thermochemistry & Real-Gas State ---
    story.append(Paragraph("2. Gibbs Equilibrium Thermochemistry & Real-Gas Parameters", section_style))
    t_rows = [
        [P_th("Metric"), P_th("Value"), P_th("Metric"), P_th("Value")],
        [P_td("Equivalence Ratio (φ)"), P_td(f"{result.equivalence_ratio:.3f}"),
         P_td("Oxygen Balance (OB%)"), P_td(f"{result.oxygen_balance_pct:+.2f}%")],
        [P_td("Isobaric Flame Temp Tp"), P_td(f"{result.adiabatic_flame_temp_k:.1f} K"),
         P_td("Closed-Vessel Temp Tv"), P_td(f"{result.isochoric_flame_temp_k:.1f} K")],
        [P_td("Gun Impetus / Force F"), P_td(f"{result.force_constant_j_per_g:.1f} J/g"),
         P_td("BLAKE Covolume η"), P_td(f"{result.covolume_cm3_per_g:.3f} cm³/g")],
        [P_td("Calorimetric Heat Qv"), P_td(f"{result.heat_of_explosion_j_per_g:.0f} J/g"),
         P_td("Specific Heat Ratio γ"), P_td(f"{result.gamma:.3f}")],
        [P_td("Mean Gas MW"), P_td(f"{result.mean_molecular_weight:.2f} g/mol"),
         P_td("Real-Gas Compressibility Z"), P_td(f"{getattr(result, 'compressibility_z', 1.0):.3f}")],
    ]
    story.append(_styled_table(t_rows, [145, 125, 145, 125]))
    story.append(Spacer(1, 6))

    sorted_sp = sorted(result.species_mole_fractions.items(), key=lambda kv: kv[1], reverse=True)[:8]
    sp_rows = [[P_th("Product Species"), P_th("Mole Fraction"), P_th("Mass Fraction")]]
    for sp_n, x_i in sorted_sp:
        sp_rows.append([
            P_td(sp_n),
            P_td(f"{x_i:.5f}"),
            P_td(f"{result.species_mass_fractions.get(sp_n, 0.0):.5f}"),
        ])
    story.append(_styled_table(sp_rows, [200, 170, 170]))

    # --- 3. Shifting vs. Frozen Rocket Nozzle Expansion ---
    noz = suite.get("nozzle_out")
    if noz and "frozen" in noz and "shifting" in noz:
        story.append(Paragraph("3. Rocket Nozzle Expansion (Frozen vs. Shifting Equilibrium)", section_style))
        fz, sh = noz["frozen"], noz["shifting"]
        n_rows = [
            [P_th("Nozzle Metric"), P_th("Frozen Composition"), P_th("Shifting Equilibrium"), P_th("Recombination Shift")],
            [P_td("Characteristic Velocity C*"), P_td(f"{fz['c_star_m_s']:.1f} m/s"), P_td(f"{sh['c_star_m_s']:.1f} m/s"), P_td(f"{sh['c_star_m_s'] - fz['c_star_m_s']:+.1f} m/s")],
            [P_td("Exit Temperature Te"), P_td(f"{fz['exit_temp_k']:.0f} K"), P_td(f"{sh['exit_temp_k']:.0f} K"), P_td(f"{sh['exit_temp_k'] - fz['exit_temp_k']:+.0f} K")],
            [P_td("Expansion Area Ratio Ae/At"), P_td(f"{fz['area_ratio_ae_at']:.2f}"), P_td(f"{sh['area_ratio_ae_at']:.2f}"), P_td(f"{sh['area_ratio_ae_at'] - fz['area_ratio_ae_at']:+.2f}")],
            [P_td("Ambient Specific Impulse Isp"), P_td(f"{fz['isp_ambient_s']:.1f} s"), P_td(f"{sh['isp_ambient_s']:.1f} s"), P_td(f"{sh['isp_ambient_s'] - fz['isp_ambient_s']:+.2f} s")],
            [P_td("Vacuum Specific Impulse Isp,vac"), P_td(f"{fz['isp_vacuum_s']:.1f} s"), P_td(f"{sh['isp_vacuum_s']:.1f} s"), P_td(f"{sh['isp_vacuum_s'] - fz['isp_vacuum_s']:+.2f} s")],
        ]
        story.append(_styled_table(n_rows, [175, 125, 125, 115]))

    # --- 4. Interior Ballistics, Erosion & STANAG 2895 / Monte Carlo ---
    ib = suite.get("ib_out")
    ero = suite.get("ero_out")
    clim = suite.get("climatic_out")
    mc = suite.get("mc_out")
    if any(x is not None for x in (ib, ero, clim, mc)):
        story.append(Paragraph("4. Interior Ballistics, Barrel Erosion & Climatic/Lot Dispersion", section_style))
        b_rows = [[P_th("Subsystem Parameter"), P_th("Predicted Value"), P_th("Engineering Assessment")]]
        if ib:
            b_rows.append([P_td("Peak Breech Pressure P_max"), P_td(f"{ib['max_pressure_mpa']:.1f} MPa"), P_td(f"Action Time: {ib['burn_time_ms']:.2f} ms")])
            b_rows.append([P_td("Muzzle Velocity V_0"), P_td(f"{ib['muzzle_velocity_m_s']:.1f} m/s"), P_td("Lagrange / Resal Coupled 1D Ballistics")])
        if ero:
            b_rows.append([P_td("Peak Bore Wall Temp"), P_td(f"{ero['peak_wall_temp_k']:.0f} K ({ero['peak_wall_temp_c']:.0f} °C)"), P_td(ero['erosion_regime'])])
            b_rows.append([P_td("Throat Wear / Barrel Life"), P_td(f"{ero['wear_um_per_shot']:.2f} µm/shot"), P_td(f"Est. {ero['estimated_barrel_life_rounds']:,} EFC rounds")])
        if clim:
            b_rows.append([
                P_td("STANAG 2895 Cold (-40°C) -> Hot (+63°C)"),
                P_td(f"{clim['cold_pmax_mpa']:.1f} -> {clim['hot_pmax_mpa']:.1f} MPa"),
                P_td(f"dP/dT = {clim['pressure_temp_gradient_mpa_per_k']:+.2f} MPa/K | Margin to Proof: {clim['hot_safety_margin_to_proof_mpa']:+.1f} MPa"),
            ])
        if mc:
            b_rows.append([
                P_td(f"Monte Carlo Dispersion (N={mc['n_trials']})"),
                P_td(f"P_max+3σ = {mc['p_max_plus_3sigma_mpa']:.1f} MPa"),
                P_td(f"σ(V0) = {mc['std_v_muz_m_s']:.2f} m/s | ES = {mc['extreme_spread_v_m_s']:.1f} m/s | Cpk = {mc['cpk_proof']:.2f}"),
            ])
        story.append(_styled_table(b_rows, [185, 135, 220]))

    # --- 5. Muzzle Flash, STANAG 4400 Smoke & OSHA/NIOSH Toxicity ---
    mf = suite.get("mf_out")
    tox = suite.get("tox_out")
    if mf or tox:
        story.append(Paragraph("5. Signature (Muzzle Flash & Smoke) & Firing-Bay Occupational Hygiene", section_style))
        s_rows = [[P_th("Signature / Hygiene Metric"), P_th("Predicted Value"), P_th("Standard / Classification")]]
        if mf:
            s_rows.append([P_td("Muzzle Flash Prediction"), P_td(f"Peak Plume {mf.get('peak_plume_temp_k', 0.0):.0f} K"), P_td(mf.get('flash_status', 'N/A'))])
            s_rows.append([P_td("Mach Disk vs Ignition Threshold"), P_td(f"{mf.get('mach_disk_temp_k', 0.0):.0f} K vs {mf.get('ignition_threshold_k', 0.0):.0f} K"), P_td(f"Combustible Gas: {mf.get('combustible_mole_pct', 0.0):.1f} mol%")])
        if tox:
            smk_cls = tox.get("agard_smoke_class", tox.get("smoke_class", "Evaluated"))
            prim_smk = tox.get("primary_smoke_g_per_kg", 0.0)
            sec_smk = tox.get("secondary_smoke_index", 0.0)
            hq = tox.get("max_hazard_quotient", tox.get("hazard_index_osha", 0.0))
            lim_gas = tox.get("limiting_pollutant", tox.get("limiting_gas", "CO"))
            bap = tox.get("bap_teq_ug_per_g", tox.get("bap_teq_mg_per_kg", 0.0))
            vent_req = tox.get("recommended_ventilation_m3_min", 0.0)
            s_rows.append([P_td("NATO AGARD / STANAG 4400 Smoke"), P_td(f"Primary {prim_smk:.1f} g/kg | Fog Idx {sec_smk:.1f}"), P_td(smk_cls)])
            s_rows.append([P_td("OSHA / NIOSH Peak Hazard Quotient"), P_td(f"{hq:.2f}x PEL (Limit 1.0)"), P_td(f"Driver: {lim_gas} | Min Vent: {vent_req:.1f} m³/min")])
            s_rows.append([P_td("Carcinogenic PAH Yield (BaP-TEQ)"), P_td(f"{bap:.2f} µg/g"), P_td("Nisbet & LaGoy Toxicity Equivalency")])
        story.append(_styled_table(s_rows, [185, 145, 210]))

    # --- 6. STANAG 4582 Surveillance & Cook-Off ---
    stab = suite.get("stab_out")
    if stab:
        story.append(Paragraph("6. STANAG 4582 Chemical Shelf-Life & Frank-Kamenetskii Cook-Off", section_style))
        life_25 = stab.get("shelf_life_at_25c_years", stab.get("safe_shelf_life_years", stab.get("safe_life_years", 0.0)))
        t_crit_c = stab.get("critical_cookoff_temp_c", stab.get("t_crit_cookoff_c", 0.0))
        t_crit_k = stab.get("critical_cookoff_temp_k", stab.get("t_crit_cookoff_k", t_crit_c + 273.15))
        margin_c = stab.get("thermal_margin_c", 0.0)
        comp_flag = stab.get("stanag_4582_compliant", True)
        status_4582 = "STANAG 4582 Compliant (>=10 yr @ 25 °C)" if comp_flag else "Surveillance Alert (<10 yr Benchmark)"
        st_rows = [
            [P_th("Thermal Surveillance Metric"), P_th("Value"), P_th("Assessment")],
            [P_td("25 °C Benchmark Shelf-Life"), P_td(f"{life_25:.1f} years"), P_td(status_4582)],
            [P_td("Critical Cook-Off Temp T_crit"), P_td(f"{t_crit_c:.1f} °C ({t_crit_k:.0f} K)"), P_td(f"Cook-Off Margin: {margin_c:+.1f} °C")],
        ]
        story.append(_styled_table(st_rows, [185, 145, 210]))

    # --- 7. Primary Bibliography ---
    story.append(Paragraph("7. Primary Literature & Standards Provenance", section_style))
    refs = [
        "1. McBride, Zehe & Gordon (2002), NASA Glenn Thermodynamic Coefficients, NASA/TP-2002-211556; Gordon & McBride (1994), NASA RP-1311.",
        "2. Freedman (1982), BLAKE Real-Gas Code, US Army BRL ARBRL-TR-02411; Anderson & Fickie (1987), IBHVG2, BRL-TR-2829.",
        "3. NATO STANAG 4115 (Closed-Vessel), STANAG 2895 (Climatic Zones), STANAG 4400 (Plume Smoke), STANAG 4582 (Microcalorimetry).",
        "4. US EPA AP-42 Ch. 15 (Ordnance Firing Emissions) & OSHA 29 CFR 1910.1000 / NIOSH Pocket Guide.",
    ]
    for r_txt in refs:
        story.append(Paragraph(_xml_escape(r_txt), td_style))
        story.append(Spacer(1, 2))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
