"""PDF Export module for Combusto.

Generates a professional PDF report containing the formulation details,
oxygen balance, calculated metrics, product composition, and carbon distribution.
"""

from __future__ import annotations

import io
from typing import Dict, Any, List
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

def export_result_pdf(result: CombustionResult, formulation: Formulation) -> bytes:
    """Compile combustion results and formulation details into a PDF document.
    
    Returns raw PDF bytes.
    """
    if not REPORTLAB_AVAILABLE:
        # Fallback if reportlab is not installed
        buffer = io.BytesIO()
        buffer.write(b"Error: ReportLab library is not installed or available on this system.")
        return buffer.getvalue()
        
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
        [Paragraph("Parameter", table_header_style), Paragraph("Calculated Value", table_header_style), Paragraph("Meaning / Units", table_header_style)],
        [Paragraph("Equivalence Ratio (&phi;)", table_text_style), Paragraph(f"{result.equivalence_ratio:.3f}", table_text_style), Paragraph("&phi; > 1 indicates fuel-rich mixture", table_text_style)],
        [Paragraph("Oxygen Balance (OB%)", table_text_style), Paragraph(f"{result.oxygen_balance_pct:.2f}%", table_text_style), Paragraph("Oxygen availability relative to total combustion", table_text_style)],
        [Paragraph("Adiabatic Flame Temp", table_text_style), Paragraph(f"{result.adiabatic_flame_temp_k:.1f} K", table_text_style), Paragraph("Max combustion temp under no heat loss", table_text_style)],
        [Paragraph("Specific Impulse (Isp)", table_text_style), Paragraph(f"{result.specific_impulse_s:.1f} s", table_text_style), Paragraph("Total vacuum thrust force over mass flow rate", table_text_style)],
        [Paragraph("Force Constant (F)", table_text_style), Paragraph(f"{result.force_constant_j_per_g:.1f} J/g", table_text_style), Paragraph("Propellant energy constant (noble force)", table_text_style)],
        [Paragraph("Ratio of Specific Heats (&gamma;)", table_text_style), Paragraph(f"{result.gamma:.3f}", table_text_style), Paragraph("Cp/Cv of exhaust gas mixture", table_text_style)],
        [Paragraph("Mean Molecular Weight", table_text_style), Paragraph(f"{result.mean_molecular_weight:.2f} g/mol", table_text_style), Paragraph("Average weight of gas phase products", table_text_style)]
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
        
    # Build Document
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
