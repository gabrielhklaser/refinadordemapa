"""
Gerador de mapa paleoclimático vetorial de exemplo (demonstração).

Cria um PDF 100% vetorizado: América do Sul estilizada no Último Máximo
Glacial (~21 ka AP), com zonas climáticas, calota de gelo andina, linha de
costa, graticule, pontos de amostragem, legenda e blocos de texto.

O mapa é gerado COM descontinuidades visuais intencionais (bordas
serrilhadas em degraus e specks de ruído próximos às fronteiras das zonas),
exatamente o tipo de artefato que o refinamento deve corrigir sem alterar
as geometrias geográficas.
"""

from __future__ import annotations

import math
import random

import pymupdf as fitz
import numpy as np
from shapely.geometry import Polygon

PAGE_W, PAGE_H = 1200.0, 850.0

# Área útil do mapa (margens)
MAP_L, MAP_T, MAP_R, MAP_B = 70.0, 128.0, 1130.0, 780.0

# Paleta das zonas climáticas (LGM, ilustrativa)
ZONE_COLORS = {
    "oceano": (0.78, 0.88, 0.95),
    "tropical_umido": (0.13, 0.42, 0.24),
    "tropical": (0.42, 0.66, 0.39),
    "semiarido": (0.91, 0.72, 0.29),
    "arido": (0.85, 0.53, 0.20),
    "temperado": (0.58, 0.66, 0.47),
    "glacial": (0.78, 0.83, 0.88),
    "calota_gelo": (0.94, 0.96, 0.98),
}

ZONE_LABELS = {
    "oceano": "Oceano",
    "tropical_umido": "Tropical Úmido (floresta refúgio)",
    "tropical": "Tropical sazonal",
    "semiarido": "Semiárido",
    "arido": "Árido / desértico",
    "temperado": "Temperado oceânico",
    "glacial": "Zona glacial periglacial",
    "calota_gelo": "Calota de gelo andina",
}

# Contorno estilizado da América do Sul (normalizado 0-1; y cresce para baixo)
SA_OUTLINE = [
    (0.42, 0.015), (0.47, 0.03), (0.53, 0.045), (0.585, 0.075),
    (0.62, 0.13), (0.635, 0.20), (0.625, 0.28), (0.595, 0.35),
    (0.55, 0.42), (0.50, 0.48), (0.455, 0.55), (0.425, 0.62),
    (0.40, 0.69), (0.375, 0.76), (0.35, 0.83), (0.325, 0.90),
    (0.305, 0.955), (0.285, 0.985), (0.265, 0.965), (0.252, 0.91),
    (0.235, 0.84), (0.215, 0.77), (0.198, 0.70), (0.185, 0.63),
    (0.175, 0.56), (0.168, 0.49), (0.163, 0.42), (0.158, 0.35),
    (0.152, 0.29), (0.155, 0.23), (0.175, 0.175), (0.21, 0.125),
    (0.26, 0.08), (0.32, 0.045), (0.375, 0.022),
]


def _to_page(norm_x: float, norm_y: float) -> tuple[float, float]:
    return (MAP_L + norm_x * (MAP_R - MAP_L), MAP_T + norm_y * (MAP_B - MAP_T))


def _continent_polygon() -> Polygon:
    pts = [_to_page(nx, ny) for nx, ny in SA_OUTLINE]
    return Polygon(pts)


def _divider(y_base: float, amp: float, freq: float, phase: float) -> list[tuple[float, float]]:
    """Curva divisória de zonas, levemente ondulada, através da área do mapa."""
    pts = []
    x0, x1 = MAP_L - 30, MAP_R + 30
    n = 60
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        t = (x - MAP_L) / (MAP_R - MAP_L)
        y = MAP_T + (y_base + amp * math.sin(2 * math.pi * freq * t + phase)) * (MAP_B - MAP_T)
        pts.append((x, y))
    return pts


def _band_polygon(y_top_div, y_bot_div) -> Polygon:
    """Retângulo largo entre duas curvas divisórias (topo/baixo)."""
    xs = [p[0] for p in y_top_div] + [p[0] for p in y_bot_div]
    x0, x1 = min(xs), max(xs)
    coords = list(y_top_div) + list(reversed(y_bot_div))
    return Polygon(coords).buffer(0)


def _jaggedize(poly: Polygon, rng: random.Random, step: float = 7.0,
               amp: float = 1.1, stair_amp: float = 2.0) -> Polygon:
    """Densifica o contorno e injeta serrilhado + degraus (descontinuidades)."""
    coords = list(poly.exterior.coords)
    out: list[tuple[float, float]] = []
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(seg_len / step))
        # vetor normal (perpendicular)
        nx_, ny_ = -(y1 - y0) / max(seg_len, 1e-9), (x1 - x0) / max(seg_len, 1e-9)
        stair_sign = rng.choice([-1, 0, 1])
        stair_every = rng.randint(3, 7)
        for k in range(n):
            t = k / n
            px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            off = rng.uniform(-amp, amp)
            if stair_sign != 0 and k % stair_every == 0:
                off += stair_sign * stair_amp
            out.append((px + nx_ * off, py + ny_ * off))
    out.append(coords[-1])
    if len(out) >= 4:
        p = Polygon(out)
        if p.is_valid and p.area > 1:
            return p.buffer(0)
    return poly


def _make_specks(lines: list[list[tuple[float, float]]], rng: random.Random,
                 n: int = 46) -> list[Polygon]:
    """Pequenos polígonos-ruído distribuídos ao longo das fronteiras entre zonas."""
    segs: list[tuple[float, float, float, float, float]] = []  # (x0,y0,dx,dy,len)
    total = 0.0
    for line in lines:
        for i in range(len(line) - 1):
            x0, y0 = line[i]
            x1, y1 = line[i + 1]
            ln = math.hypot(x1 - x0, y1 - y0)
            if ln < 1e-6:
                continue
            segs.append((x0, y0, x1 - x0, y1 - y0, ln))
            total += ln
    specks = []
    if total <= 0:
        return specks
    for _ in range(n * 4):
        if len(specks) >= n:
            break
        target = rng.uniform(0, total)
        acc = 0.0
        for (x0, y0, dx, dy, ln) in segs:
            acc += ln
            if acc >= target:
                t = rng.random()
                px, py = x0 + dx * t, y0 + dy * t
                nx_, ny_ = -dy / ln, dx / ln
                px += nx_ * rng.uniform(-2.5, 2.5)
                py += ny_ * rng.uniform(-2.5, 2.5)
                r = rng.uniform(1.6, 4.2)
                kind = rng.choice(["square", "tri", "sliver"])
                if kind == "square":
                    sp = Polygon([(px - r, py - r), (px + r, py - r),
                                  (px + r, py + r), (px - r, py + r)])
                elif kind == "tri":
                    sp = Polygon([(px, py - r * 1.4), (px + r * 1.3, py + r),
                                  (px - r * 1.3, py + r)])
                else:
                    sp = Polygon([(px - r * 2, py), (px + r * 0.4, py - r * 0.5),
                                  (px + r * 0.4, py + r * 0.5)])
                if sp.is_valid:
                    specks.append(sp)
                break
    return specks


def _zone_bands(rng: random.Random) -> tuple[dict[str, Polygon], list[list[tuple[float, float]]]]:
    """Recorta faixas climáticas latitudinais contra o continente."""
    continent = _continent_polygon()
    dividers = [
        _divider(0.115, 0.012, 1.7, 0.4),   # tropical úmido / tropical
        _divider(0.300, 0.014, 1.3, 2.1),   # tropical / semiárido
        _divider(0.430, 0.010, 1.1, 4.0),   # semiárido / árido
        _divider(0.560, 0.013, 1.5, 1.2),   # árido / temperado
        _divider(0.735, 0.010, 1.2, 3.3),   # temperado / glacial
        _divider(0.870, 0.008, 1.0, 5.0),   # glacial / calota de gelo
    ]
    top_edge = _divider(-0.05, 0.0, 1.0, 0.0)
    bot_edge = _divider(1.05, 0.0, 1.0, 0.0)
    cuts = [top_edge] + dividers + [bot_edge]

    bands: dict[str, Polygon] = {}
    names = ["tropical_umido", "tropical", "semiarido", "arido",
             "temperado", "glacial", "calota_gelo"]
    for idx, name in enumerate(names):
        band = _band_polygon(cuts[idx], cuts[idx + 1])
        zone = band.intersection(continent)
        if zone.is_empty:
            continue
        # pode gerar MultiPolygon; mantém apenas partes relevantes
        parts = []
        if zone.geom_type == "Polygon":
            parts = [zone]
        else:
            parts = [g for g in zone.geoms if g.area > 60]
        for p in parts:
            bands.setdefault(name, Polygon())
            bands[name] = bands[name].union(_jaggedize(p, rng))
    return bands, dividers


def generate_sample_pdf(path: str, seed: int = 7) -> str:
    rng = random.Random(seed)
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    # fundo do oceano
    page.draw_rect(fitz.Rect(0, 0, PAGE_W, PAGE_H),
                   fill=ZONE_COLORS["oceano"], color=None)

    # ------------------------------------------------------------------ zonas
    bands, dividers = _zone_bands(rng)
    for name, poly in bands.items():
        if poly.is_empty:
            continue
        polys = [poly] if poly.geom_type == "Polygon" else [g for g in poly.geoms]
        for p in polys:
            pts = [fitz.Point(x, y) for x, y in p.exterior.coords]
            page.draw_polyline(pts, fill=ZONE_COLORS[name], color=None,
                               fill_opacity=1.0, closePath=True)
        # furos (ilhas/lagunas internas)
        if poly.geom_type == "Polygon":
            rings = [poly.exterior] + list(poly.interiors)
        else:
            rings = [r for g in poly.geoms for r in
                     ([g.exterior] + list(g.interiors))]
        for ring in rings[1:] if poly.geom_type == "Polygon" else rings:
            pass  # demonstração: sem furos internos complexos

    # calota de gelo andina extra sobre a cordilheira (oeste, latitudes médias)
    andes = Polygon([
        _to_page(0.185, 0.52), _to_page(0.205, 0.60), _to_page(0.225, 0.70),
        _to_page(0.245, 0.80), _to_page(0.265, 0.88), _to_page(0.245, 0.93),
        _to_page(0.215, 0.85), _to_page(0.195, 0.74), _to_page(0.175, 0.62),
        _to_page(0.170, 0.55),
    ])
    andes = _jaggedize(andes.intersection(_continent_polygon()), rng,
                       step=6.0, amp=1.0, stair_amp=1.8)
    if not andes.is_empty and andes.geom_type == "Polygon":
        page.draw_polyline([fitz.Point(x, y) for x, y in andes.exterior.coords],
                           fill=ZONE_COLORS["calota_gelo"], color=None, closePath=True)

    # specks de ruído (artefatos de vetorização) ao longo das fronteiras
    speck_lines = dividers + [list(andes.exterior.coords)]
    for sp in _make_specks(speck_lines, rng):
        page.draw_polyline([fitz.Point(x, y) for x, y in sp.exterior.coords],
                           fill=ZONE_COLORS["glacial"], color=None, closePath=True)

    # --------------------------------------------------------- linha de costa
    coast = [_to_page(nx, ny) for nx, ny in SA_OUTLINE]
    coast.append(coast[0])
    page.draw_polyline([fitz.Point(x, y) for x, y in coast],
                       color=(0.10, 0.19, 0.32), width=1.8, fill=None)

    # --------------------------------------------------------------- graticule
    grid_col = (0.55, 0.60, 0.66)
    for i in range(1, 10):
        x = MAP_L + (MAP_R - MAP_L) * i / 10
        page.draw_line(fitz.Point(x, MAP_T), fitz.Point(x, MAP_B),
                       color=grid_col, width=0.7, dashes="[2 3] 0")
    for j in range(1, 8):
        y = MAP_T + (MAP_B - MAP_T) * j / 8
        page.draw_line(fitz.Point(MAP_L, y), fitz.Point(MAP_R, y),
                       color=grid_col, width=0.7, dashes="[2 3] 0")
    lon_labels = ["80°O", "70°O", "60°O", "50°O", "40°O", "30°O"]
    lat_labels = ["10°N", "0°", "10°S", "20°S", "30°S", "40°S", "50°S"]
    for i, lab in enumerate(lon_labels):
        x = MAP_L + (MAP_R - MAP_L) * (i + 2) / 10
        page.insert_text(fitz.Point(x - 10, MAP_B + 14), lab,
                         fontsize=8.5, fontname="helv", color=(0.25, 0.28, 0.33))
    for j, lab in enumerate(lat_labels):
        y = MAP_T + (MAP_B - MAP_T) * (j + 1) / 8
        page.insert_text(fitz.Point(MAP_L - 42, y + 3), lab,
                         fontsize=8.5, fontname="helv", color=(0.25, 0.28, 0.33))

    # ----------------------------------------------------- pontos de amostragem
    samples = [
        ("AM-01", 0.38, 0.16), ("AM-02", 0.50, 0.22), ("AM-03", 0.55, 0.33),
        ("PA-04", 0.30, 0.38), ("PA-05", 0.44, 0.47), ("PA-06", 0.33, 0.58),
        ("GL-07", 0.30, 0.72), ("GL-08", 0.27, 0.86), ("AN-09", 0.21, 0.64),
        ("AN-10", 0.25, 0.52),
    ]
    for code, nx, ny in samples:
        px, py = _to_page(nx, ny)
        page.draw_circle(fitz.Point(px, py), 4.2, fill=(0.83, 0.16, 0.16),
                         color=(1, 1, 1), width=1.1)
        page.insert_text(fitz.Point(px + 7, py + 3.2), code,
                         fontsize=9, fontname="helv", color=(0.12, 0.12, 0.16))

    # --------------------------------------------------------------- moldura
    page.draw_rect(fitz.Rect(MAP_L, MAP_T, MAP_R, MAP_B),
                   color=(0.15, 0.18, 0.24), width=1.6, fill=None)

    # ------------------------------------------------------------ bloco título
    page.insert_text(fitz.Point(70, 62), "MAPA PALEOCLIMÁTICO — AMÉRICA DO SUL",
                     fontsize=21, fontname="hebo", color=(0.09, 0.13, 0.22))
    page.insert_text(
        fitz.Point(70, 84),
        "Último Máximo Glacial (~21 ka AP) · reconstrução ilustrativa vetorizada",
        fontsize=11, fontname="helv", color=(0.32, 0.36, 0.42))
    page.draw_line(fitz.Point(70, 96), fitz.Point(420, 96),
                   color=(0.09, 0.13, 0.22), width=1.2)

    # ----------------------------------------------------------------- legenda
    leg_x, leg_y, leg_w, leg_h = 880, 148, 232, 236
    page.draw_rect(fitz.Rect(leg_x, leg_y, leg_x + leg_w, leg_y + leg_h),
                   fill=(1, 1, 1), color=(0.15, 0.18, 0.24), width=1.2,
                   fill_opacity=0.94)
    page.insert_text(fitz.Point(leg_x + 12, leg_y + 20),
                     "Zonas climáticas (21 ka)", fontsize=10.5,
                     fontname="hebo", color=(0.09, 0.13, 0.22))
    legend_items = ["tropical_umido", "tropical", "semiarido", "arido",
                    "temperado", "glacial", "calota_gelo", "oceano"]
    for i, key in enumerate(legend_items):
        y = leg_y + 34 + i * 22
        page.draw_rect(fitz.Rect(leg_x + 12, y, leg_x + 30, y + 13),
                       fill=ZONE_COLORS[key], color=(0.2, 0.2, 0.25), width=0.5)
        page.insert_text(fitz.Point(leg_x + 38, y + 10.5), ZONE_LABELS[key],
                         fontsize=8.8, fontname="helv", color=(0.15, 0.17, 0.22))

    # ------------------------------------------------------------- barra escala
    sb_x, sb_y = 92, 742
    seg = 55
    for k in range(4):
        x0 = sb_x + k * seg
        page.draw_rect(fitz.Rect(x0, sb_y, x0 + seg, sb_y + 9),
                       fill=(0.1, 0.1, 0.15) if k % 2 == 0 else (1, 1, 1),
                       color=(0.1, 0.1, 0.15), width=0.8)
    for k, lab in enumerate(["0", "500", "1000", "1500", "2000 km"]):
        page.insert_text(fitz.Point(sb_x - 4 + k * seg - (4 if k else 0), sb_y - 6),
                         lab, fontsize=8.2, fontname="helv", color=(0.15, 0.17, 0.22))

    # ------------------------------------------------------------- flecha norte
    nx_, ny_ = 1088, 700
    p_n = fitz.Point(nx_, ny_ - 26)
    page.draw_polyline([fitz.Point(nx_ - 7, ny_), p_n, fitz.Point(nx_ + 7, ny_)],
                       fill=(0.09, 0.13, 0.22), color=(0.09, 0.13, 0.22),
                       width=1.0, closePath=True)
    page.insert_text(fitz.Point(nx_ - 3.5, ny_ + 14), "N", fontsize=11,
                     fontname="hebo", color=(0.09, 0.13, 0.22))

    # ------------------------------------------------------------------- fonte
    page.insert_text(
        fitz.Point(70, 812),
        "RefinaPaleo · Plataforma de Refinamento de Dados Paleoclimáticos — mapa de demonstração (PDF vetorial)",
        fontsize=8.5, fontname="helv", color=(0.42, 0.46, 0.52))

    doc.save(path, deflate=True)
    doc.close()
    return path


# ============================================================================
# Exemplo 2 — mapa de ARIDEZ interpolada (estilo kNN+IDW, campo rasterizado)
# Reproduz a estrutura do mapa real: gradiente com classes nítidas
# (Dry / Semi-arid / Humid), imagem embutida no PDF, costa vetorial,
# pontos de amostragem e tabela de contagem de pixels.
# ============================================================================

ARID_COLORS = {
    "oceano": (0.72, 0.84, 0.93),
    "dry": (0.80, 0.58, 0.26),
    "semi": (0.93, 0.82, 0.52),
    "humid": (0.22, 0.55, 0.40),
}


def generate_sample_gradient_pdf(path: str, seed: int = 11) -> str:
    """Mapa de aridez (115 Ma) com campo interpolado rasterizado embutido."""
    import cv2

    rng = np.random.default_rng(seed)
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)

    page.draw_rect(fitz.Rect(0, 0, PAGE_W, PAGE_H),
                   fill=(0.955, 0.96, 0.965), color=None)

    FL, FT, FR, FB = int(MAP_L), int(MAP_T), int(MAP_R), int(MAP_B)
    Wf, Hf = FR - FL, FB - FT

    # ---------------------------------------------------- campo interpolado
    # estações climáticas espalhadas com valor de umidade [0=Dry .. 1=Humid]
    # (no mapa real de 115 Ma a classe Humid domina: ~69% dos pixels)
    n_st = 26
    sx = rng.uniform(0.05, 0.95, n_st)
    sy = rng.uniform(0.05, 0.95, n_st)
    sv = np.clip(rng.normal(0.60, 0.24, n_st), 0.0, 1.0)

    yy, xx = np.mgrid[0:Hf, 0:Wf].astype(np.float32)
    u, v = xx / Wf, yy / Hf
    dist = np.sqrt((u[..., None] - sx) ** 2 + ((v[..., None] - sy) * 0.92) ** 2)
    wgt = 1.0 / (dist + 2e-3)                            # IDW power 1.0
    field = (wgt * sv).sum(-1) / wgt.sum(-1)

    # ruído de média resolução -> contornos de classe serrilhados pós-classif.
    small = rng.normal(0.0, 1.0, (16, 22)).astype(np.float32)
    noise = cv2.resize(small, (Wf, Hf), interpolation=cv2.INTER_CUBIC) * 0.048
    fn = field + noise

    # ------------------------------------------------- paleogeografia (terra)
    cont = _continent_polygon()
    cont_px = np.array([[(x - FL), (y - FT)] for x, y in cont.exterior.coords],
                       np.float32)
    land = np.zeros((Hf, Wf), np.uint8)
    cv2.fillPoly(land, [np.rint(cont_px).astype(np.int32)], 255)

    # ------------------------------------------------- classificação nítida
    cls = np.digitize(fn, [0.40, 0.56]).astype(np.uint8)   # 0 dry, 1 semi, 2 humid

    base = np.array([ARID_COLORS["dry"], ARID_COLORS["semi"],
                     ARID_COLORS["humid"]], np.float32)
    img = np.zeros((Hf, Wf, 3), np.float32)
    for c in range(3):
        m = cls == c
        if not m.any():
            continue
        vals = fn[m]
        lo, hi = np.percentile(vals, 4), np.percentile(vals, 96)
        shade = 0.84 + 0.26 * np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1)
        img[m] = base[c][None, :] * shade[:, None]
    img[land == 0] = np.array(ARID_COLORS["oceano"], np.float32)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # imagem embutida no PDF (campo climático raster; o restante é vetor)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("falha ao codificar o campo rasterizado")
    page.insert_image(fitz.Rect(FL, FT, FR, FB), stream=buf.tobytes())

    # --------------------------------------------------- linha de costa (vetor)
    coast = [_to_page(nx, ny) for nx, ny in SA_OUTLINE]
    coast.append(coast[0])
    page.draw_polyline([fitz.Point(x, y) for x, y in coast],
                       color=(0.10, 0.19, 0.32), width=1.8, fill=None)

    # ----------------------------------------------------------------- graticule
    grid_col = (0.42, 0.47, 0.54)
    for i in range(1, 10):
        x = MAP_L + (MAP_R - MAP_L) * i / 10
        page.draw_line(fitz.Point(x, MAP_T), fitz.Point(x, MAP_B),
                       color=grid_col, width=0.7, dashes="[2 3] 0")
    for j in range(1, 8):
        y = MAP_T + (MAP_B - MAP_T) * j / 8
        page.draw_line(fitz.Point(MAP_L, y), fitz.Point(MAP_R, y),
                       color=grid_col, width=0.7, dashes="[2 3] 0")
    lon_labels = ["80°O", "70°O", "60°O", "50°O", "40°O", "30°O"]
    lat_labels = ["10°N", "0°", "10°S", "20°S", "30°S", "40°S", "50°S"]
    for i, lab in enumerate(lon_labels):
        x = MAP_L + (MAP_R - MAP_L) * (i + 2) / 10
        page.insert_text(fitz.Point(x - 10, MAP_B + 14), lab,
                         fontsize=8.5, fontname="helv", color=(0.25, 0.28, 0.33))
    for j, lab in enumerate(lat_labels):
        y = MAP_T + (MAP_B - MAP_T) * (j + 1) / 8
        page.insert_text(fitz.Point(MAP_L - 42, y + 3), lab,
                         fontsize=8.5, fontname="helv", color=(0.25, 0.28, 0.33))

    # ----------------------------------------------------- pontos de amostragem
    samples = [
        ("S-01", 0.38, 0.16), ("S-02", 0.52, 0.24), ("S-03", 0.58, 0.36),
        ("S-04", 0.30, 0.36), ("S-05", 0.46, 0.46), ("S-06", 0.33, 0.58),
        ("S-07", 0.30, 0.72), ("S-08", 0.28, 0.85), ("S-09", 0.22, 0.63),
        ("S-10", 0.26, 0.51), ("S-11", 0.62, 0.20), ("S-12", 0.42, 0.66),
    ]
    for code, nx, ny in samples:
        px, py = _to_page(nx, ny)
        page.draw_circle(fitz.Point(px, py), 4.2, fill=(0.83, 0.16, 0.16),
                         color=(1, 1, 1), width=1.1)
        page.insert_text(fitz.Point(px + 7, py + 3.2), code,
                         fontsize=9, fontname="helv", color=(0.12, 0.12, 0.16))

    # ------------------------------------------------------------------ moldura
    page.draw_rect(fitz.Rect(MAP_L, MAP_T, MAP_R, MAP_B),
                   color=(0.15, 0.18, 0.24), width=1.6, fill=None)

    # ------------------------------------------------------------------- título
    page.insert_text(fitz.Point(70, 62), "MAPA PALEOCLIMÁTICO — ARIDEZ (115 MA)",
                     fontsize=21, fontname="hebo", color=(0.09, 0.13, 0.22))
    page.insert_text(
        fitz.Point(70, 84),
        "Cretáceo Médio · interpolação kNN+IDW (power 1.0) · gradiente nítido — reconstrução ilustrativa",
        fontsize=11, fontname="helv", color=(0.32, 0.36, 0.42))
    page.draw_line(fitz.Point(70, 96), fitz.Point(420, 96),
                   color=(0.09, 0.13, 0.22), width=1.2)

    # ------------------------------------------------------- legenda + colorbar
    leg_x, leg_y, leg_w, leg_h = 880, 148, 238, 158
    page.draw_rect(fitz.Rect(leg_x, leg_y, leg_x + leg_w, leg_y + leg_h),
                   fill=(1, 1, 1), color=(0.15, 0.18, 0.24), width=1.2,
                   fill_opacity=0.95)
    page.insert_text(fitz.Point(leg_x + 12, leg_y + 20),
                     "Categorias de aridez (115 Ma)", fontsize=10.5,
                     fontname="hebo", color=(0.09, 0.13, 0.22))
    items = [("humid", "Humid"), ("semi", "Semi-arid"), ("dry", "Dry")]
    for i, (key, lab) in enumerate(items):
        y = leg_y + 34 + i * 24
        page.draw_rect(fitz.Rect(leg_x + 12, y, leg_x + 34, y + 15),
                       fill=ARID_COLORS[key], color=(0.2, 0.2, 0.25), width=0.5)
        page.insert_text(fitz.Point(leg_x + 42, y + 11.5), lab,
                         fontsize=9.5, fontname="helv", color=(0.15, 0.17, 0.22))

    # ------------------------------------------- tabela de contagem de pixels
    tb_x, tb_y = 880, 322
    tb_w, tb_h = 238, 88
    page.draw_rect(fitz.Rect(tb_x, tb_y, tb_x + tb_w, tb_y + tb_h),
                   fill=(1, 1, 1), color=(0.15, 0.18, 0.24), width=1.2,
                   fill_opacity=0.95)
    page.insert_text(fitz.Point(tb_x + 12, tb_y + 18),
                     "Pixel counts", fontsize=10, fontname="hebo",
                     color=(0.09, 0.13, 0.22))
    page.insert_text(fitz.Point(tb_x + 12, tb_y + 36),
                     "Category    Dry  Semi-arid  Humid", fontsize=8,
                     fontname="helv", color=(0.25, 0.28, 0.33))
    page.insert_text(fitz.Point(tb_x + 12, tb_y + 52),
                     "0     136479    68797   449594", fontsize=8,
                     fontname="helv", color=(0.25, 0.28, 0.33))
    page.insert_text(fitz.Point(tb_x + 12, tb_y + 70),
                     "metodologia: kNN · IDW p=1.0 · sharp=18.0", fontsize=7.5,
                     fontname="helv", color=(0.45, 0.49, 0.55))

    # ------------------------------------------------------------- barra escala
    sb_x, sb_y = 92, 742
    seg = 55
    for k in range(4):
        x0 = sb_x + k * seg
        page.draw_rect(fitz.Rect(x0, sb_y, x0 + seg, sb_y + 9),
                       fill=(0.1, 0.1, 0.15) if k % 2 == 0 else (1, 1, 1),
                       color=(0.1, 0.1, 0.15), width=0.8)
    for k, lab in enumerate(["0", "500", "1000", "1500", "2000 km"]):
        page.insert_text(fitz.Point(sb_x - 4 + k * seg - (4 if k else 0), sb_y - 6),
                         lab, fontsize=8.2, fontname="helv", color=(0.15, 0.17, 0.22))

    # ------------------------------------------------------------- flecha norte
    nx_, ny_ = 1088, 700
    page.draw_polyline([fitz.Point(nx_ - 7, ny_), fitz.Point(nx_, ny_ - 26),
                        fitz.Point(nx_ + 7, ny_)],
                       fill=(0.09, 0.13, 0.22), color=(0.09, 0.13, 0.22),
                       width=1.0, closePath=True)
    page.insert_text(fitz.Point(nx_ - 3.5, ny_ + 14), "N", fontsize=11,
                     fontname="hebo", color=(0.09, 0.13, 0.22))

    page.insert_text(
        fitz.Point(70, 812),
        "RefinaPaleo · mapa de demonstração com campo rasterizado embutido (PDF híbrido: imagem + vetores)",
        fontsize=8.5, fontname="helv", color=(0.42, 0.46, 0.52))

    doc.save(path, deflate=True)
    doc.close()
    return path


if __name__ == "__main__":
    print(generate_sample_pdf("sample_paleomap.pdf"))
