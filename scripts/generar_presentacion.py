"""Genera la presentación de sustentación (.pptx) del proyecto para el concurso
Datos al Ecosistema 2026 — IA para Colombia (nivel intermedio).

Sigue la estructura ejecutiva exigida por los lineamientos oficiales
(data/Lineamientos_Sustentacion_Concurso_Datos_Ecosistema_2026.pptx): un pitch
de 6-9 diapositivas -> portada, problema, datos abiertos, solución e IA,
resultados (módulos A y B), impacto, repositorio y cierre.

Las cifras se leen en vivo de outputs/ (métricas, ranking, SHAP) para no
hardcodearlas; las figuras PNG de outputs/ se incrustan tal cual.

Uso:
    .venv/Scripts/python scripts/generar_presentacion.py
Salida:
    outputs/Sustentacion_Datos_Ecosistema_2026.pptx
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PPTX_PATH = OUT / "Sustentacion_Datos_Ecosistema_2026.pptx"

# --------------------------------------------------------------------------- #
# Paleta institucional y tipografía
# --------------------------------------------------------------------------- #
NAVY = RGBColor(0x0B, 0x3C, 0x5D)      # azul profundo (primario)
TEAL = RGBColor(0x14, 0x7D, 0x88)      # verde azulado (secundario)
GOLD = RGBColor(0xE0, 0x9F, 0x2C)      # acento cálido (destacados)
BG_LIGHT = RGBColor(0xF5, 0xF7, 0xFA)  # fondo claro de paneles
INK = RGBColor(0x22, 0x2B, 0x33)       # texto principal
MUTE = RGBColor(0x5C, 0x6B, 0x7A)      # texto secundario
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NAVY_SOFT = RGBColor(0x16, 0x50, 0x76)

FONT = "Calibri"

# Lienzo 16:9
SW = Inches(13.333)
SH = Inches(7.5)


# --------------------------------------------------------------------------- #
# Helpers de dibujo
# --------------------------------------------------------------------------- #
def _no_line(shape):
    shape.line.fill.background()


def rect(slide, left, top, width, height, color, shape=MSO_SHAPE.RECTANGLE):
    sp = slide.shapes.add_shape(shape, left, top, width, height)
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    _no_line(sp)
    sp.shadow.inherit = False
    return sp


def _set_run(run, text, size, color, bold=False, italic=False, font=FONT):
    run.text = text
    f = run.font
    f.size = Pt(size)
    f.color.rgb = color
    f.bold = bold
    f.italic = italic
    f.name = font


def textbox(slide, left, top, width, height, lines, *, anchor=MSO_ANCHOR.TOP,
            align=PP_ALIGN.LEFT, wrap=True):
    """`lines` es una lista de párrafos; cada párrafo es una lista de runs
    (dict con text/size/color/bold/italic) o un dict simple.
    space_before/space_after/line/level opcionales por párrafo vía tupla
    (runs, opts)."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    for i, para in enumerate(lines):
        opts = {}
        if isinstance(para, tuple):
            para, opts = para
        if isinstance(para, dict):
            para = [para]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opts.get("align", align)
        if "space_after" in opts:
            p.space_after = Pt(opts["space_after"])
        if "space_before" in opts:
            p.space_before = Pt(opts["space_before"])
        if "line" in opts:
            p.line_spacing = opts["line"]
        p.level = opts.get("level", 0)
        for run_spec in para:
            r = p.add_run()
            _set_run(r, run_spec.get("text", ""), run_spec.get("size", 16),
                     run_spec.get("color", INK), run_spec.get("bold", False),
                     run_spec.get("italic", False), run_spec.get("font", FONT))
    return tb


def bullets(slide, left, top, width, height, items, *, size=15,
            color=INK, bullet_color=TEAL, gap=6, line=1.05, anchor=MSO_ANCHOR.TOP):
    """items: lista de str o (str, level). Viñeta coloreada + texto."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    for i, item in enumerate(items):
        level = 0
        if isinstance(item, tuple):
            item, level = item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = line
        glyph = "•" if level == 0 else "–"
        rb = p.add_run()
        _set_run(rb, f"{'   ' * level}{glyph}  ",
                 size, GOLD if level == 0 else MUTE, bold=True)
        rt = p.add_run()
        _set_run(rt, item, size, color)
    return tb


def title_bar(slide, title, page_no, eyebrow=None):
    """Banda superior de navegación con título y numeración."""
    rect(slide, 0, 0, SW, Inches(1.12), NAVY)
    rect(slide, 0, Inches(1.12), SW, Inches(0.06), GOLD)
    if eyebrow:
        textbox(slide, Inches(0.55), Inches(0.16), Inches(11), Inches(0.3),
                [[{"text": eyebrow.upper(), "size": 11, "color": GOLD, "bold": True}]])
        ttop = Inches(0.42)
    else:
        ttop = Inches(0.30)
    textbox(slide, Inches(0.55), ttop, Inches(11.2), Inches(0.7),
            [[{"text": title, "size": 27, "color": WHITE, "bold": True}]],
            anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, Inches(12.3), Inches(0.30), Inches(0.7), Inches(0.55),
            [[{"text": f"{page_no:02d}", "size": 20, "color": GOLD, "bold": True}]],
            anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.RIGHT)


def footer(slide):
    rect(slide, 0, Inches(7.16), SW, Inches(0.34), BG_LIGHT)
    textbox(slide, Inches(0.55), Inches(7.16), Inches(9), Inches(0.34),
            [[{"text": "Concurso Datos al Ecosistema 2026 — IA para Colombia · Nivel Intermedio",
               "size": 9.5, "color": MUTE}]], anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, Inches(9.6), Inches(7.16), Inches(3.18), Inches(0.34),
            [[{"text": "Fuente: datos.gov.co", "size": 9.5, "color": MUTE}]],
            anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.RIGHT)


def panel(slide, left, top, width, height, fill=BG_LIGHT, bar=None):
    """Tarjeta redondeada opcionalmente con barra de color a la izquierda."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = fill
    _no_line(card)
    card.shadow.inherit = False
    try:
        card.adjustments[0] = 0.045
    except Exception:
        pass
    if bar is not None:
        rect(slide, left, top + Emu(int(height * 0.12)), Inches(0.09),
             Emu(int(height * 0.76)), bar, MSO_SHAPE.ROUNDED_RECTANGLE)
    return card


def kpi(slide, left, top, width, height, number, label, *,
        fill=NAVY, num_color=GOLD, label_color=WHITE):
    panel(slide, left, top, width, height, fill=fill)
    textbox(slide, left, top + Inches(0.14), width, Inches(0.62),
            [[{"text": number, "size": 30, "color": num_color, "bold": True}]],
            anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    textbox(slide, left + Inches(0.12), top + Inches(0.74), width - Inches(0.24),
            height - Inches(0.82),
            [[{"text": label, "size": 11, "color": label_color}]],
            anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.CENTER)


def image_fit(slide, path, left, top, box_w, box_h, *, frame=True):
    """Incrusta la imagen escalada al recuadro conservando proporción, centrada."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(box_w / iw, box_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    lft = int(left + (box_w - w) / 2)
    tp = int(top + (box_h - h) / 2)
    if frame:
        pad = Inches(0.08)
        rect(slide, lft - pad, tp - pad, w + 2 * pad, h + 2 * pad, WHITE,
             MSO_SHAPE.ROUNDED_RECTANGLE)
    slide.shapes.add_picture(str(path), lft, tp, width=w, height=h)


def field_pill(slide, left, top, width, label, *, dropdown=False,
               height=Inches(0.32), size=9.5):
    """Campo de formulario simulado (mockup): caja blanca con borde suave,
    etiqueta a la izquierda y, opcionalmente, un cursor de desplegable."""
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    sp.fill.solid()
    sp.fill.fore_color.rgb = WHITE
    sp.line.color.rgb = RGBColor(0xC9, 0xD2, 0xDB)
    sp.line.width = Pt(0.75)
    sp.shadow.inherit = False
    try:
        sp.adjustments[0] = 0.2
    except Exception:
        pass
    textbox(slide, left + Inches(0.12), top, width - Inches(0.34), height,
            [[{"text": label, "size": size, "color": INK}]], anchor=MSO_ANCHOR.MIDDLE)
    if dropdown:
        textbox(slide, left + width - Inches(0.26), top, Inches(0.2), height,
                [[{"text": "▾", "size": size, "color": MUTE}]],
                anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    return sp


def semaforo(slide, left, top, width, height, pos):
    """Barra semáforo (verde→ámbar→rojo, de menor a mayor riesgo) con un
    marcador blanco en la posición fraccional `pos` (0..1)."""
    seg = int(width / 3)
    colores = [RGBColor(0x2E, 0x8B, 0x57), GOLD, RGBColor(0xC0, 0x39, 0x2B)]
    for i, col in enumerate(colores):
        rect(slide, left + i * seg, top, seg, height, col)
    mx = int(left + pos * width)
    rect(slide, mx - Inches(0.017), top - Inches(0.05),
         Inches(0.034), height + Inches(0.1), WHITE)


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def solid_bg(slide, color):
    rect(slide, 0, 0, SW, SH, color)


# --------------------------------------------------------------------------- #
# Formato numérico (español: coma decimal, punto de miles)
# --------------------------------------------------------------------------- #
def miles(n):
    return f"{int(round(n)):,}".replace(",", ".")


def coma(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


# --------------------------------------------------------------------------- #
# Cifras en vivo desde outputs/
# --------------------------------------------------------------------------- #
def cargar_cifras():
    d = {}
    met = pd.read_csv(OUT / "tabla_metricas.csv")
    log = met[met["modelo"].str.contains("logística", case=False)].iloc[0]
    d["roc_heldout"] = coma(log["heldout_roc_auc"], 2)
    d["roc_cv"] = coma(log["cv_roc_auc"], 2)

    rank = pd.read_csv(OUT / "ranking_brechas.csv", dtype={"COD_DANE": str})
    top = rank.sort_values("INDICE_BRECHA", ascending=False).head(5)
    d["top_brecha"] = [(str(r.NOMBRE_MUNI).title(), coma(r.INDICE_BRECHA, 1))
                       for r in top.itertuples()]
    # resumen por tipología (conteo + brecha media)
    g = (rank.groupby("TIPOLOGIA")
         .agg(n=("INDICE_BRECHA", "size"), brecha=("INDICE_BRECHA", "mean"))
         .sort_values("brecha", ascending=False))
    d["tipologias"] = [(t.split("·")[-1].strip(), int(row.n), coma(row.brecha, 1))
                       for t, row in g.iterrows()]
    d["n_muni"] = int(rank.shape[0])

    shap = pd.read_csv(OUT / "importancia_shap.csv")
    d["shap_top"] = list(shap.sort_values("importancia_media_abs_shap",
                                          ascending=False)["variable"].head(6))
    return d


# --------------------------------------------------------------------------- #
# Diapositivas
# --------------------------------------------------------------------------- #
def s1_portada(prs):
    s = blank(prs)
    solid_bg(s, NAVY)
    rect(s, 0, 0, Inches(0.28), SH, GOLD)
    rect(s, Inches(0.28), 0, Inches(0.10), SH, TEAL)
    textbox(s, Inches(0.95), Inches(0.85), Inches(11.5), Inches(0.4),
            [[{"text": "CONCURSO DATOS AL ECOSISTEMA 2026 · IA PARA COLOMBIA",
               "size": 14, "color": GOLD, "bold": True}]])
    textbox(s, Inches(0.9), Inches(1.5), Inches(11.6), Inches(1.9),
            [[{"text": "Del colegio a la universidad", "size": 46, "color": WHITE,
               "bold": True}],
             ({"text": "Inteligencia territorial para cerrar brechas de acceso y "
                       "permanencia en la educación superior de Antioquia",
               "size": 20, "color": RGBColor(0xCF, 0xDD, 0xE8)},
              {"space_before": 10, "line": 1.1})])
    # panel de datos del equipo (placeholders editables)
    panel(s, Inches(0.9), Inches(4.35), Inches(11.5), Inches(1.9),
          fill=NAVY_SOFT, bar=GOLD)
    filas = [
        ("Nivel", "Intermedio"),
        ("ID del equipo", "[completar]"),
        ("Integrantes", "[Nombre 1] · [Nombre 2] · [Nombre 3]"),
        ("Institución", "[Universidad / entidad]"),
        ("Fecha de sustentación", "[dd/mm/2026]"),
    ]
    lines = []
    for k, v in filas:
        lines.append(([{"text": f"{k}:  ", "size": 14, "color": GOLD, "bold": True},
                       {"text": v, "size": 14, "color": WHITE}],
                      {"space_after": 5}))
    textbox(s, Inches(1.25), Inches(4.55), Inches(10.9), Inches(1.6), lines)
    textbox(s, Inches(0.95), Inches(6.7), Inches(11.5), Inches(0.4),
            [[{"text": "Datos abiertos + IA + impacto público  ·  Repositorio y "
                       "app desplegable incluidos", "size": 12,
               "color": RGBColor(0x9F, 0xB4, 0xC4), "italic": True}]])


def s2_problema(prs):
    s = blank(prs)
    solid_bg(s, WHITE)
    title_bar(s, "Problema y objetivo", 2, eyebrow="Necesidad pública y territorial")
    footer(s)
    # Problema
    panel(s, Inches(0.55), Inches(1.5), Inches(6.05), Inches(3.9), bar=TEAL)
    textbox(s, Inches(0.9), Inches(1.7), Inches(5.6), Inches(0.4),
            [[{"text": "El problema", "size": 18, "color": NAVY, "bold": True}]])
    bullets(s, Inches(0.9), Inches(2.25), Inches(5.5), Inches(3.0), [
        "En Antioquia el acceso y la permanencia en educación superior son "
        "muy desiguales entre municipios urbanos y rurales.",
        "No hay una focalización basada en datos que conecte el territorio "
        "(de dónde viene el estudiante) con el aula (cómo le va).",
        "El acompañamiento institucional (Semestre Cero, Soñares, PIES) tiene "
        "capacidad limitada y necesita priorizar a quién atiende.",
    ], size=14.5)
    # Objetivo
    panel(s, Inches(6.75), Inches(1.5), Inches(6.03), Inches(3.9), bar=GOLD)
    textbox(s, Inches(7.1), Inches(1.7), Inches(5.6), Inches(0.4),
            [[{"text": "El objetivo", "size": 18, "color": NAVY, "bold": True}]])
    bullets(s, Inches(7.1), Inches(2.25), Inches(5.5), Inches(3.0), [
        "Identificar los municipios con mayor brecha de acceso (módulo "
        "territorial no supervisado).",
        "Detectar tempranamente a los estudiantes con mayor riesgo académico "
        "(módulo supervisado).",
        "Entregar insumos accionables para priorizar el acompañamiento, con "
        "datos abiertos + IA de forma reproducible.",
    ], size=14.5, bullet_color=GOLD)
    # frase-síntesis
    panel(s, Inches(0.55), Inches(5.65), Inches(12.23), Inches(1.25), fill=NAVY)
    textbox(s, Inches(0.9), Inches(5.65), Inches(11.5), Inches(1.25),
            [[{"text": "Problema real  +  datos abiertos  +  IA  +  impacto público",
               "size": 21, "color": WHITE, "bold": True}],
             ({"text": "una sola herramienta reproducible que va del territorio al aula",
               "size": 13, "color": RGBColor(0xCF, 0xDD, 0xE8), "italic": True},
              {"space_before": 4})],
            anchor=MSO_ANCHOR.MIDDLE)


def s3_datos(prs, c):
    s = blank(prs)
    solid_bg(s, WHITE)
    title_bar(s, "Datos abiertos", 3, eyebrow="Fuentes, variables y tratamiento")
    footer(s)
    # tabla de conjuntos
    filas = [
        ("Conjunto", "Fuente", "Registros"),
        ("Matrícula UdeA — sedes regionales 2026-1", "Universidad de Antioquia", "8.157"),
        ("Beneficiarios de acompañamiento", "datos.gov.co · Gob. Antioquia", "13.778"),
        ("Población censada 2018", "datos.gov.co · DANE", "12.825"),
        ("Saber 11 — ICFES (2018+)", "datos.gov.co · ICFES (API)", "297.962"),
        ("DIVIPOLA — municipios", "datos.gov.co · DANE (API)", "1.122"),
    ]
    rows, cols = len(filas), 3
    gt = s.shapes.add_table(rows, cols, Inches(0.55), Inches(1.5),
                            Inches(8.1), Inches(3.7)).table
    gt.columns[0].width = Inches(4.35)
    gt.columns[1].width = Inches(2.75)
    gt.columns[2].width = Inches(1.0)
    gt.first_row = False
    gt.horz_banding = False
    for r in range(rows):
        for cc in range(cols):
            cell = gt.cell(r, cc)
            cell.margin_left = Inches(0.1)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = NAVY
            else:
                cell.fill.fore_color.rgb = BG_LIGHT if r % 2 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if cc == 2 else PP_ALIGN.LEFT
            run = p.add_run()
            _set_run(run, filas[r][cc], 11.5 if r else 12,
                     WHITE if r == 0 else INK, bold=(r == 0))
    # panel lateral con tratamiento / llave
    panel(s, Inches(8.85), Inches(1.5), Inches(3.93), Inches(3.7), bar=TEAL)
    textbox(s, Inches(9.15), Inches(1.68), Inches(3.4), Inches(0.4),
            [[{"text": "Integración y calidad", "size": 15, "color": NAVY, "bold": True}]])
    bullets(s, Inches(9.15), Inches(2.2), Inches(3.35), Inches(2.9), [
        f"Llave común: código DANE de municipio (5 dígitos) — {c['n_muni']} "
        "municipios integrados.",
        "Cruce del 100 % (auditoría difusa de nombres con rapidfuzz).",
        "Limpieza: mojibake (ftfy), fechas mixtas y API Socrata paginada.",
        "Matrices analíticas: 125×26 (municipal) y 8.157×23 (estudiantes).",
    ], size=12)
    # KPIs inferiores
    kpi(s, Inches(0.55), Inches(5.5), Inches(3.9), Inches(1.35), "4 / 5",
        "conjuntos provienen de datos.gov.co", fill=NAVY)
    kpi(s, Inches(4.72), Inches(5.5), Inches(3.9), Inches(1.35), "~334 mil",
        "registros integrados de 5 fuentes", fill=TEAL, num_color=WHITE)
    kpi(s, Inches(8.9), Inches(5.5), Inches(3.88), Inches(1.35), "100 %",
        "de cruce por código DANE de municipio", fill=NAVY)


def s4_solucion(prs):
    s = blank(prs)
    solid_bg(s, WHITE)
    title_bar(s, "Solución e IA", 4, eyebrow="Funcionamiento y componente técnico")
    footer(s)
    # pipeline
    etapas = ["acquire", "clean", "integrate", "module A / B"]
    x = Inches(0.55)
    cw = Inches(2.7)
    for i, e in enumerate(etapas):
        col = NAVY if i < 3 else TEAL
        panel(s, x, Inches(1.5), cw, Inches(0.62), fill=col)
        textbox(s, x, Inches(1.5), cw, Inches(0.62),
                [[{"text": e, "size": 14, "color": WHITE, "bold": True}]],
                anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
        if i < 3:
            textbox(s, x + cw, Inches(1.5), Inches(0.42), Inches(0.62),
                    [[{"text": "▶", "size": 14, "color": GOLD, "bold": True}]],
                    anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
        x = x + cw + Inches(0.42)
    textbox(s, Inches(10.6), Inches(1.5), Inches(2.3), Inches(0.62),
            [[{"text": "Pipeline reproducible", "size": 11, "color": MUTE,
               "italic": True}]], anchor=MSO_ANCHOR.MIDDLE)
    # dos módulos
    panel(s, Inches(0.55), Inches(2.45), Inches(6.05), Inches(3.55), bar=TEAL)
    textbox(s, Inches(0.9), Inches(2.62), Inches(5.5), Inches(0.5),
            [[{"text": "Módulo A · Tipologías territoriales", "size": 16,
               "color": NAVY, "bold": True}],
             [{"text": "Aprendizaje NO supervisado", "size": 11, "color": TEAL,
               "bold": True}]])
    bullets(s, Inches(0.9), Inches(3.5), Inches(5.5), Inches(2.4), [
        "KMeans sobre 7 variables municipales (matrícula, Saber 11, ruralidad, "
        "cobertura, brechas de género).",
        "Índice de brecha compuesto 0–100 y mapa interactivo.",
        "Validación: silhouette (k≥3) y estabilidad ARI multi-semilla.",
    ], size=13)
    panel(s, Inches(6.75), Inches(2.45), Inches(6.03), Inches(3.55), bar=GOLD)
    textbox(s, Inches(7.1), Inches(2.62), Inches(5.5), Inches(0.5),
            [[{"text": "Módulo B · Riesgo académico", "size": 16, "color": NAVY,
               "bold": True}],
             [{"text": "Aprendizaje SUPERVISADO", "size": 11, "color": GOLD,
               "bold": True}]])
    bullets(s, Inches(7.1), Inches(3.5), Inches(5.5), Inches(2.4), [
        "15 variables sin fuga de información; 3 modelos comparados "
        "(Reg. logística, Random Forest, LightGBM).",
        "Validación cruzada estratificada (PR-AUC por desbalance).",
        "Interpretabilidad con SHAP y scoring por deciles de riesgo.",
    ], size=13, bullet_color=GOLD)
    panel(s, Inches(0.55), Inches(6.2), Inches(12.23), Inches(0.72), fill=NAVY)
    textbox(s, Inches(0.9), Inches(6.2), Inches(11.5), Inches(0.72),
            [[{"text": "Prototipo desplegable: app Streamlit dockerizada "
                       "(puerto 8501) — pipeline 100 % reproducible.",
               "size": 13.5, "color": WHITE, "bold": True}]],
            anchor=MSO_ANCHOR.MIDDLE)


def s5_resultados_a(prs, c):
    s = blank(prs)
    solid_bg(s, WHITE)
    title_bar(s, "Resultados · Módulo A (territorial)", 5,
              eyebrow="Dónde está la mayor brecha de acceso")
    footer(s)
    # columna izquierda: KPIs + ranking
    kpi(s, Inches(0.55), Inches(1.5), Inches(2.9), Inches(1.3), "3",
        "tipologías de municipios", fill=NAVY)
    kpi(s, Inches(3.6), Inches(1.5), Inches(2.9), Inches(1.3), "0,94",
        "estabilidad ARI (silhouette 0,32)", fill=TEAL, num_color=WHITE)
    panel(s, Inches(0.55), Inches(3.0), Inches(5.95), Inches(3.35), bar=GOLD)
    textbox(s, Inches(0.9), Inches(3.18), Inches(5.4), Inches(0.4),
            [[{"text": "Ranking de brecha (índice 0–100)", "size": 15,
               "color": NAVY, "bold": True}]])
    rank_lines = []
    for i, (nom, val) in enumerate(c["top_brecha"], 1):
        rank_lines.append(([
            {"text": f"{i}.  ", "size": 14, "color": GOLD, "bold": True},
            {"text": f"{nom}", "size": 14, "color": INK, "bold": True},
            {"text": f"   {val}", "size": 14, "color": TEAL, "bold": True},
        ], {"space_after": 5}))
    textbox(s, Inches(0.95), Inches(3.7), Inches(3.4), Inches(2.5), rank_lines)
    bullets(s, Inches(4.35), Inches(3.72), Inches(2.05), Inches(2.5), [
        "Alta ruralidad + bajo Saber 11 + baja matrícula + poca cobertura.",
        "La sede regional UdeA reduce la brecha.",
    ], size=11.5)
    # columna derecha: figura de perfiles de clúster
    fig = OUT / "fig_pca_clusters.png"
    if fig.exists():
        image_fit(s, fig, Inches(6.75), Inches(1.5), Inches(6.03), Inches(4.85))
    textbox(s, Inches(6.75), Inches(6.42), Inches(6.03), Inches(0.5),
            [[{"text": "Municipios proyectados (PCA) y coloreados por tipología. "
                       "Demo: mapa interactivo mapa_brechas.html.",
               "size": 10.5, "color": MUTE, "italic": True}]],
            align=PP_ALIGN.CENTER)


def s6_resultados_b(prs, c):
    s = blank(prs)
    solid_bg(s, WHITE)
    title_bar(s, "Resultados · Módulo B (riesgo académico)", 6,
              eyebrow="A quién priorizar para el acompañamiento")
    footer(s)
    kpi(s, Inches(0.55), Inches(1.5), Inches(2.9), Inches(1.45),
        c["roc_heldout"], "ROC-AUC en datos no vistos (reg. logística)", fill=NAVY)
    kpi(s, Inches(3.6), Inches(1.5), Inches(2.9), Inches(1.45), "6,2×",
        "lift: el decil superior captura 63 % de los casos", fill=GOLD,
        num_color=NAVY, label_color=NAVY)
    panel(s, Inches(0.55), Inches(3.15), Inches(5.95), Inches(3.25), bar=TEAL)
    textbox(s, Inches(0.9), Inches(3.32), Inches(5.4), Inches(0.4),
            [[{"text": "Factores de riesgo (SHAP)", "size": 15, "color": NAVY,
               "bold": True}]])
    bullets(s, Inches(0.9), Inches(3.85), Inches(5.45), Inches(2.4), [
        "Antigüedad sin avance de nivel — el predictor más fuerte.",
        "Facultad (Ingeniería y Educación elevan el riesgo).",
        "Ser mujer aparece como factor protector.",
        "El contexto municipal (ruralidad, Saber 11 local) aporta señal: "
        "la brecha territorial también se expresa dentro del aula.",
    ], size=12.5)
    # figura SHAP a la derecha
    fig = OUT / "fig_shap_summary.png"
    if fig.exists():
        image_fit(s, fig, Inches(6.75), Inches(1.5), Inches(6.03), Inches(3.35))
    # comparación de modelos como banda inferior
    figc = OUT / "fig_comparacion_modelos.png"
    if figc.exists():
        image_fit(s, figc, Inches(6.75), Inches(4.95), Inches(6.03), Inches(1.5))
    textbox(s, Inches(6.75), Inches(6.45), Inches(6.03), Inches(0.4),
            [[{"text": "Arriba: importancia SHAP.  Abajo: comparación de 3 modelos "
                       "(CV estratificada).", "size": 10, "color": MUTE,
               "italic": True}]], align=PP_ALIGN.CENTER)


def s7_demo(prs):
    s = blank(prs)
    solid_bg(s, WHITE)
    title_bar(s, "Demo · Calculadora de riesgo académico", 7,
              eyebrow="Prototipo interactivo — del dato al acompañamiento")
    footer(s)

    # -------- Panel izquierdo: mockup del formulario --------
    panel(s, Inches(0.55), Inches(1.42), Inches(5.55), Inches(4.9), bar=TEAL)
    textbox(s, Inches(0.85), Inches(1.55), Inches(5.0), Inches(0.4),
            [[{"text": "Formulario · Predicción individual", "size": 15,
               "color": NAVY, "bold": True}]])
    textbox(s, Inches(0.85), Inches(2.02), Inches(5.0), Inches(0.3),
            [[{"text": "PERFIL ACADÉMICO", "size": 10, "color": TEAL, "bold": True}]])
    col_a = ["Sexo", "Sede", "Facultad", "Tipo de aceptación",
             "Naturaleza del colegio", "Nivel de pregrado"]
    col_b = [("Edad", False), ("Estrato", True), ("Antigüedad (sem.)", False),
             ("Créditos últ. sem.", False), ("¿Vive fuera de Antioquia?", False)]
    y0, pitch = Inches(2.34), Inches(0.38)
    for i, lab in enumerate(col_a):
        field_pill(s, Inches(0.85), y0 + i * pitch, Inches(2.35), lab, dropdown=True)
    for i, (lab, dd) in enumerate(col_b):
        field_pill(s, Inches(3.35), y0 + i * pitch, Inches(2.5), lab, dropdown=dd)
    # contexto municipal (autocompletado)
    textbox(s, Inches(0.85), Inches(4.64), Inches(5.0), Inches(0.3),
            [[{"text": "CONTEXTO DEL MUNICIPIO (AUTOCOMPLETADO)", "size": 10,
               "color": TEAL, "bold": True}]])
    field_pill(s, Inches(0.85), Inches(4.96), Inches(2.35), "Municipio", dropdown=True)
    rect(s, Inches(3.35), Inches(4.96), Inches(1.05), Inches(0.32), GOLD,
         MSO_SHAPE.ROUNDED_RECTANGLE)
    textbox(s, Inches(3.35), Inches(4.96), Inches(1.05), Inches(0.32),
            [[{"text": "auto", "size": 10, "color": NAVY, "bold": True}]],
            anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    textbox(s, Inches(0.85), Inches(5.34), Inches(5.1), Inches(0.35),
            [[{"text": "% rural · Saber 11 municipal · tasa de acompañamiento se "
                       "completan solos", "size": 9, "color": MUTE, "italic": True}]])
    rect(s, Inches(0.85), Inches(5.74), Inches(2.7), Inches(0.44), NAVY,
         MSO_SHAPE.ROUNDED_RECTANGLE)
    textbox(s, Inches(0.85), Inches(5.74), Inches(2.7), Inches(0.44),
            [[{"text": "▶  Calcular riesgo", "size": 13, "color": WHITE, "bold": True}]],
            anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)

    # -------- Flecha de flujo --------
    textbox(s, Inches(6.02), Inches(3.0), Inches(0.75), Inches(0.9),
            [[{"text": "▶", "size": 32, "color": GOLD, "bold": True}]],
            anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)

    # -------- Columna derecha: salida del modelo --------
    textbox(s, Inches(6.75), Inches(1.45), Inches(6.03), Inches(0.35),
            [[{"text": "SALIDA DEL MODELO", "size": 10, "color": MUTE, "bold": True},
              {"text": "   ·   modelo_riesgo.pkl → predict_proba", "size": 10,
               "color": MUTE}]])
    panel(s, Inches(6.75), Inches(1.88), Inches(6.03), Inches(2.3), fill=NAVY)
    textbox(s, Inches(7.1), Inches(2.05), Inches(5.4), Inches(0.35),
            [[{"text": "Probabilidad de riesgo académico", "size": 13, "color": WHITE}]])
    textbox(s, Inches(7.1), Inches(2.4), Inches(5.4), Inches(0.8),
            [[{"text": "72,3 %", "size": 44, "color": GOLD, "bold": True},
              {"text": "   (ejemplo)", "size": 13,
               "color": RGBColor(0x9F, 0xB4, 0xC4), "italic": True}]],
            anchor=MSO_ANCHOR.MIDDLE)
    semaforo(s, Inches(7.1), Inches(3.55), Inches(5.35), Inches(0.26), 0.723)
    textbox(s, Inches(7.1), Inches(3.86), Inches(2.6), Inches(0.24),
            [[{"text": "menor riesgo", "size": 8.5,
               "color": RGBColor(0x9F, 0xB4, 0xC4)}]])
    textbox(s, Inches(9.85), Inches(3.86), Inches(2.6), Inches(0.24),
            [[{"text": "mayor riesgo", "size": 8.5,
               "color": RGBColor(0x9F, 0xB4, 0xC4)}]], align=PP_ALIGN.RIGHT)
    panel(s, Inches(6.75), Inches(4.4), Inches(6.03), Inches(1.3), fill=BG_LIGHT,
          bar=GOLD)
    textbox(s, Inches(7.1), Inches(4.55), Inches(5.5), Inches(1.05),
            [[{"text": "Alerta estadística, no diagnóstico individual.",
               "size": 12.5, "color": NAVY, "bold": True}],
             ({"text": "Cuando la probabilidad es ≥ 50 %, se lee junto con el "
                       "acompañamiento humano de Bienestar.", "size": 11,
               "color": INK}, {"space_before": 3, "line": 1.05})])
    textbox(s, Inches(6.75), Inches(5.82), Inches(6.03), Inches(0.4),
            [[{"text": "Demo en vivo durante la sustentación (app Streamlit).",
               "size": 10, "color": MUTE, "italic": True}]])

    # -------- Banda inferior: salvaguardas --------
    panel(s, Inches(0.55), Inches(6.5), Inches(12.23), Inches(0.55), fill=NAVY)
    textbox(s, Inches(0.9), Inches(6.5), Inches(11.6), Inches(0.55),
            [[{"text": "Salvaguardas:  las opciones categóricas provienen del modelo "
                       "entrenado (nunca ofrece valores no vistos)   ·   el contexto "
                       "municipal se autocompleta desde el Módulo A.",
               "size": 11.5, "color": WHITE}]], anchor=MSO_ANCHOR.MIDDLE)


def s8_impacto(prs):
    s = blank(prs)
    solid_bg(s, WHITE)
    title_bar(s, "Impacto esperado", 8, eyebrow="Valor público y sostenibilidad")
    footer(s)
    tarjetas = [
        ("Beneficiarios", TEAL,
         ["Estudiantes de sedes regionales y municipios rurales con mayor brecha.",
          "Equipos de Bienestar que focalizan el acompañamiento."]),
        ("Uso institucional", NAVY,
         ["Prioriza Semestre Cero / Soñares / PIES (UdeA) y política territorial "
          "(Gobernación).",
          "Con capacidad para el 10 % de la matrícula se cubre 2/3 del riesgo."]),
        ("Escalabilidad", GOLD,
         ["Pipeline reproducible y app desplegable en contenedor.",
          "Replicable a otros departamentos con las mismas fuentes abiertas."]),
        ("Ética", NAVY_SOFT,
         ["El scoring prioriza acompañamiento, nunca restringe el acceso.",
          "Datos anonimizados; hace explícitas las desigualdades históricas."]),
    ]
    xs = [Inches(0.55), Inches(6.75)]
    ys = [Inches(1.5), Inches(3.9)]
    for i, (tit, col, items) in enumerate(tarjetas):
        x = xs[i % 2]
        y = ys[i // 2]
        panel(s, x, y, Inches(6.03), Inches(2.2), bar=col)
        textbox(s, x + Inches(0.35), y + Inches(0.15), Inches(5.4), Inches(0.4),
                [[{"text": tit, "size": 16, "color": NAVY, "bold": True}]])
        bullets(s, x + Inches(0.35), y + Inches(0.68), Inches(5.45), Inches(1.4),
                items, size=12, bullet_color=col)
    panel(s, Inches(0.55), Inches(6.35), Inches(12.23), Inches(0.6), fill=GOLD)
    textbox(s, Inches(0.9), Inches(6.35), Inches(11.5), Inches(0.6),
            [[{"text": "Focalización accionable: menos recursos, mejor dirigidos, "
                       "con evidencia de datos abiertos.", "size": 14,
               "color": NAVY, "bold": True}]], anchor=MSO_ANCHOR.MIDDLE)


def s9_repositorio(prs):
    s = blank(prs)
    solid_bg(s, WHITE)
    title_bar(s, "Repositorio y validación técnica", 9,
              eyebrow="Recursos para revisión del jurado")
    footer(s)
    panel(s, Inches(0.55), Inches(1.5), Inches(7.6), Inches(4.9), bar=TEAL)
    textbox(s, Inches(0.9), Inches(1.7), Inches(7), Inches(0.4),
            [[{"text": "Checklist de entregables", "size": 17, "color": NAVY,
               "bold": True}]])
    items = [
        ("Repositorio", "[URL GitHub / GitLab del equipo]"),
        ("Código fuente organizado", "src/ — acquire · clean · integrate · module_a · module_b"),
        ("Datos y enlaces oficiales", "datos.gov.co — Saber 11, Censo, Beneficiarios, DIVIPOLA"),
        ("Documentación técnica", "README.md con reproducción paso a paso"),
        ("Notebooks narrativos", "01 limpieza · 02 módulo A · 03 módulo B"),
        ("Demo / prototipo", "app Streamlit dockerizada (:8501) — calculadora de riesgo individual"),
        ("Recursos de resultados", "outputs/ — modelo, métricas, ranking, mapa, figuras"),
    ]
    lines = []
    for k, v in items:
        lines.append(([
            {"text": "✔  ", "size": 14, "color": TEAL, "bold": True},
            {"text": f"{k}:  ", "size": 13.5, "color": INK, "bold": True},
            {"text": v, "size": 13.5, "color": MUTE},
        ], {"space_after": 9}))
    textbox(s, Inches(0.95), Inches(2.3), Inches(7.05), Inches(4.0), lines)
    # panel lateral
    panel(s, Inches(8.35), Inches(1.5), Inches(4.43), Inches(4.9), fill=NAVY)
    textbox(s, Inches(8.7), Inches(1.85), Inches(3.8), Inches(0.5),
            [[{"text": "Reproducible de punta a punta", "size": 15, "color": GOLD,
               "bold": True}]])
    bullets(s, Inches(8.7), Inches(2.55), Inches(3.75), Inches(2.6), [
        "Python 3.12 · scikit-learn · LightGBM · SHAP · Streamlit.",
        "Un comando por etapa regenera datos, modelos y figuras.",
        "Imagen Docker/Podman lista para desplegar la demo.",
    ], size=12.5, color=WHITE, bullet_color=GOLD)
    textbox(s, Inches(8.7), Inches(5.35), Inches(3.8), Inches(0.9),
            [[{"text": "Enlaces completos = condiciones para validar recursos, "
                       "bases, código y resultados.", "size": 11.5,
               "color": RGBColor(0xCF, 0xDD, 0xE8), "italic": True}]])


def s10_cierre(prs):
    s = blank(prs)
    solid_bg(s, NAVY)
    rect(s, 0, 0, SW, Inches(0.16), GOLD)
    rect(s, 0, Inches(7.34), SW, Inches(0.16), TEAL)
    textbox(s, Inches(0.9), Inches(1.15), Inches(11.5), Inches(0.4),
            [[{"text": "CIERRE · VALOR DIFERENCIAL", "size": 14, "color": GOLD,
               "bold": True}]])
    textbox(s, Inches(0.9), Inches(1.75), Inches(11.6), Inches(1.8),
            [[{"text": "Una sola inteligencia sobre datos abiertos:",
               "size": 34, "color": WHITE, "bold": True}],
             ({"text": "del territorio al aula.", "size": 34, "color": GOLD,
               "bold": True}, {"space_before": 2})])
    bullets(s, Inches(0.95), Inches(3.75), Inches(11.4), Inches(2.0), [
        "Integra el módulo territorial (dónde) y el de riesgo (a quién) en una "
        "sola herramienta accionable.",
        "5 conjuntos integrados con cruce del 100 % y validación estadística "
        "en ambos módulos.",
        "Reproducible y desplegable: del notebook a la app en contenedor.",
    ], size=15, color=WHITE, bullet_color=GOLD, gap=8)
    panel(s, Inches(0.9), Inches(6.05), Inches(11.53), Inches(0.95), fill=NAVY_SOFT,
          bar=GOLD)
    textbox(s, Inches(1.25), Inches(6.05), Inches(11), Inches(0.95),
            [[{"text": "Problema real  +  datos abiertos  +  IA  +  impacto público",
               "size": 19, "color": WHITE, "bold": True}]],
            anchor=MSO_ANCHOR.MIDDLE)


# --------------------------------------------------------------------------- #
def main():
    cifras = cargar_cifras()
    prs = Presentation()
    prs.slide_width = SW
    prs.slide_height = SH

    s1_portada(prs)
    s2_problema(prs)
    s3_datos(prs, cifras)
    s4_solucion(prs)
    s5_resultados_a(prs, cifras)
    s6_resultados_b(prs, cifras)
    s7_demo(prs)
    s8_impacto(prs)
    s9_repositorio(prs)
    s10_cierre(prs)

    PPTX_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(PPTX_PATH)
    print(f"OK -> {PPTX_PATH}  ({len(prs.slides._sldIdLst)} diapositivas)")


if __name__ == "__main__":
    main()
