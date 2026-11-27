"""Build the four deliverable documents (zh/en x docx) for the stage report.

Numbers are the measured ones from results/RESULTS_*.md; nothing here is
recomputed, so the paper cannot drift from the archived experiments.
"""
from __future__ import annotations

import pathlib
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "paper"
OUT.mkdir(exist_ok=True)

FIG_SAMPLES = ROOT / "results" / "compare" / "samples_compare.png"
FIG_OBS = ROOT / "results" / "compare" / "observables_compare.png"

REFS_EN = [
    "Y. Zhu, W. Guo, J. Choi, G.-H. Liu, Y. Chen, and M. Tao, \u201cMDNS: Masked Diffusion "
    "Neural Sampler via Stochastic Optimal Control,\u201d in Advances in Neural Information "
    "Processing Systems (NeurIPS), 2025.",
    "R. H. Swendsen and J.-S. Wang, \u201cNonuniversal critical dynamics in Monte Carlo "
    "simulations,\u201d Physical Review Letters, vol. 58, no. 2, pp. 86\u201388, 1987.",
    "C. M. Fortuin and P. W. Kasteleyn, \u201cOn the random-cluster model: I. Introduction and "
    "relation to the Ising and Potts models,\u201d Physica, vol. 57, no. 4, pp. 536\u2013564, 1972.",
    "F. Y. Wu, \u201cThe Potts model,\u201d Reviews of Modern Physics, vol. 54, no. 1, "
    "pp. 235\u2013268, 1982.",
    "E. Hoogeboom, A. A. Gritsenko, J. Bastings, B. Poole, R. van den Berg, and T. Salimans, "
    "\u201cAutoregressive Diffusion Models,\u201d in International Conference on Learning "
    "Representations (ICLR), 2022.",
    "J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. van den Berg, \u201cStructured Denoising "
    "Diffusion Models in Discrete State-Spaces,\u201d in NeurIPS, 2021.",
    "A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, "
    "M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, \u201cAn Image "
    "is Worth 16x16 Words: Transformers for Image Recognition at Scale,\u201d in ICLR, 2021.",
    "J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, \u201cRoFormer: Enhanced Transformer "
    "with Rotary Position Embedding,\u201d arXiv:2104.09864, 2021; Neurocomputing, 2024.",
    "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, \u201cMobileNetV2: Inverted "
    "Residuals and Linear Bottlenecks,\u201d in CVPR, 2018.",
    "I. Loshchilov and F. Hutter, \u201cDecoupled Weight Decay Regularization,\u201d in ICLR, 2019.",
    "K. Jordan, \u201cMuon: An optimizer for hidden layers in neural networks,\u201d technical "
    "report, 2024.",
    "A. B. Owen, Monte Carlo Theory, Methods and Examples. Stanford, 2013.",
    "R. G. Edwards and A. D. Sokal, \u201cGeneralization of the Fortuin-Kasteleyn-Swendsen-Wang "
    "representation and Monte Carlo algorithm,\u201d Physical Review D, vol. 38, no. 6, "
    "pp. 2009\u20132012, 1988.",
    "B. T. Polyak and A. B. Juditsky, \u201cAcceleration of stochastic approximation by "
    "averaging,\u201d SIAM Journal on Control and Optimization, vol. 30, no. 4, pp. 838\u2013855, 1992.",
    "A. Tarvainen and H. Valpola, \u201cMean teachers are better role models: Weight-averaged "
    "consistency targets improve semi-supervised deep learning results,\u201d in NeurIPS, 2017.",
    "N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller, "
    "\u201cEquation of state calculations by fast computing machines,\u201d Journal of Chemical "
    "Physics, vol. 21, no. 6, pp. 1087\u20131092, 1953.",
]

REFS_ZH = [
    "Y. Zhu, W. Guo, J. Choi, G.-H. Liu, Y. Chen, M. Tao, \u201cMDNS: Masked Diffusion Neural "
    "Sampler via Stochastic Optimal Control\u201d, NeurIPS, 2025.",
    "R. H. Swendsen, J.-S. Wang, \u201cNonuniversal critical dynamics in Monte Carlo "
    "simulations\u201d, Physical Review Letters, 58(2): 86\u201388, 1987.",
    "C. M. Fortuin, P. W. Kasteleyn, \u201cOn the random-cluster model: I. Introduction and "
    "relation to the Ising and Potts models\u201d, Physica, 57(4): 536\u2013564, 1972.",
    "F. Y. Wu, \u201cThe Potts model\u201d, Reviews of Modern Physics, 54(1): 235\u2013268, 1982.",
    "E. Hoogeboom, A. A. Gritsenko, J. Bastings, B. Poole, R. van den Berg, T. Salimans, "
    "\u201cAutoregressive Diffusion Models\u201d, ICLR, 2022.",
    "J. Austin, D. D. Johnson, J. Ho, D. Tarlow, R. van den Berg, \u201cStructured Denoising "
    "Diffusion Models in Discrete State-Spaces\u201d, NeurIPS, 2021.",
    "A. Dosovitskiy, L. Beyer, A. Kolesnikov, et al., \u201cAn Image is Worth 16x16 Words: "
    "Transformers for Image Recognition at Scale\u201d, ICLR, 2021.",
    "J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, Y. Liu, \u201cRoFormer: Enhanced Transformer with "
    "Rotary Position Embedding\u201d, arXiv:2104.09864, 2021; Neurocomputing, 2024.",
    "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, \u201cMobileNetV2: Inverted "
    "Residuals and Linear Bottlenecks\u201d, CVPR, 2018.",
    "I. Loshchilov, F. Hutter, \u201cDecoupled Weight Decay Regularization\u201d, ICLR, 2019.",
    "K. Jordan, \u201cMuon: An optimizer for hidden layers in neural networks\u201d, 技术报告, 2024.",
    "A. B. Owen, Monte Carlo Theory, Methods and Examples, Stanford, 2013.",
    "R. G. Edwards, A. D. Sokal, \u201cGeneralization of the Fortuin-Kasteleyn-Swendsen-Wang "
    "representation and Monte Carlo algorithm\u201d, Physical Review D, 38(6): 2009\u20132012, 1988.",
    "B. T. Polyak, A. B. Juditsky, \u201cAcceleration of stochastic approximation by "
    "averaging\u201d, SIAM Journal on Control and Optimization, 30(4): 838\u2013855, 1992.",
    "A. Tarvainen, H. Valpola, \u201cMean teachers are better role models\u201d, NeurIPS, 2017.",
    "N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, E. Teller, \u201cEquation of "
    "state calculations by fast computing machines\u201d, Journal of Chemical Physics, "
    "21(6): 1087\u20131092, 1953.",
]


def setup(doc: Document, lang: str) -> None:
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    for attr in ("top_margin", "bottom_margin"):
        setattr(sec, attr, Cm(2.4))
    sec.left_margin = sec.right_margin = Cm(2.5)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.space_after = Pt(5)
    pf.space_before = Pt(0)

    for name, size, east in (("Heading 1", 13, "SimHei"), ("Heading 2", 11.5, "SimHei"),
                             ("Heading 3", 10.5, "SimHei")):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), east)
        st.paragraph_format.space_before = Pt(11)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.keep_with_next = True


def para(doc, text, *, size=None, italic=False, bold=False, align=None,
         space_after=None, indent_head=None, style=None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    if size:
        run.font.size = Pt(size)
    run.italic = italic
    run.bold = bold
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if indent_head is not None:
        p.paragraph_format.first_line_indent = Cm(indent_head)
    return p


def body(doc, text, lang):
    return para(doc, text, align="justify", indent_head=0.0 if lang == "en" else 0.74)


def caption(doc, text):
    return para(doc, text, size=9, italic=True, align="center", space_after=9)


def table(doc, header, rows, widths=None, font=8.8, align_first_left=True):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(font)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(v))
            r.font.size = Pt(font)
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if (i == 0 and align_first_left)
                           else WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.space_after = Pt(1)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return t


def figure(doc, path, width_cm=15.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
