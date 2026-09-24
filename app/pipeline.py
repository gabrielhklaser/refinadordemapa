"""
RefinaPaleo — Pipeline de Refinamento Visual de Mapas Paleoclimáticos.

Implementa a skill `refinamento-mapas-paleo`:

  1. Carregamento e desacoplamento de dados (PDF vetorizado -> camadas)
  2. Suavização morfológica / denoising (3 tecnologias distintas)
  3. Realce com rigor científico (geometria geográfica intocável)
  4. Geração de 3 intensidades por tecnologia (Mínima / Intermediária / Intensa)
  5. Conferência de confiabilidade (% de mudança de zonas, erro posicional)
  6. Exportação TIFF / PNG com metadados

Disciplina de máscaras
----------------------
* `zone_mask`   — preenchimentos grandes (zonas climáticas): ÚNICA região
                  elegível para refinamento estético.
* `protected`   — camadas vetoriais geográficas (linhas de costa, graticule,
                  molduras) + textuais (títulos, rótulos, legendas). Nunca é
                  alterada: restauração bit a bit ao final de cada versão.
* `obs_mask`    — pontos/anotações observacionais (marcadores com contorno,
                  amostras de escala, chaves de legenda). Também protegida.

Métricas de confiabilidade
--------------------------
* changed_pct — % de pixels de zona com ΔE (CIE76) > limiar (mudança estética
                confinada às fronteiras das zonas).
* shift_px    — deslocamento medido por correlação cruzada nas camadas
                protegidas (esperado: 0.0 px).
* psnr / ssim — similaridade global entre entrada e saída.
* erro vetorial/observacional — diferença máxima nas máscaras protegidas
                (garantido 0 px por restauração exata).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
from PIL.TiffImagePlugin import ImageFileDirectory_v2 as TiffInfo

# --------------------------------------------------------------------------
# Configuração geral
# --------------------------------------------------------------------------
THIN_STROKE_PT = 3.5        # traços <= esta espessura (pt) são camadas geográficas
SMALL_FILL_PT2 = 900.0      # preenchimento com bbox <= isto é anotação/ponto (pt²)
SPECK_MAX_PX2 = 800.0       # preenchimento pequeno SEM contorno <= isto é ruído de zona (px²)
PROTECT_DILATE_PX = 6       # raio de exclusão ao redor de camadas protegidas (px)
EDGE_DILATE_PX = 5          # meia-largura da faixa de fronteira elegível (px)
EDGE_GRAD_THR = 9.0         # limiar do gradiente (Lab) que define fronteira de zona
DE_THRESHOLD = 3.0          # ΔE CIE76 mínimo para considerar "mudança de zona"
MAX_PIXELS = 8_000_000      # teto de segurança para rasterização

STRENGTH_LABELS = {"minima": "Mínima", "intermediaria": "Intermediária", "intensa": "Intensa"}

# ----------------------------------------------------------------------------
# Tecnologias de refinamento (parâmetros por intensidade)
# ----------------------------------------------------------------------------
TECHNOLOGIES = [
    {
        "key": "morph",
        "name": "Suavização Morfológica Multiescala",
        "short": "MORF-MS",
        "desc": (
            "Operadores morfológicos de fechamento e abertura com elemento "
            "estruturante elíptico, aplicados no espaço de cor CIELAB. "
            "Regulariza serrilhados em degraus e elimina specks de vetorização "
            "nas fronteiras entre zonas climáticas."
        ),
        "strengths": {
            "minima":        {"r_close": 3, "r_open": 3, "median": 3},
            "intermediaria": {"r_close": 5, "r_open": 5, "median": 5},
            "intensa":       {"r_close": 9, "r_open": 9, "median": 7},
        },
    },
    {
        "key": "bilateral",
        "name": "Filtragem Bilateral Edge-Preserving",
        "short": "BILAT-EP",
        "desc": (
            "Filtragem bilateral iterativa: suprime ruído cromático e "
            "micro-oscilações de rasterização preservando descontinuidades de "
            "borda fortes. É a tecnologia de menor interferência na posição "
            "aparente das fronteiras."
        ),
        "strengths": {
            "minima":        {"d": 7, "sigma_color": 18, "sigma_space": 4, "iters": 1, "median": 3},
            "intermediaria": {"d": 9, "sigma_color": 32, "sigma_space": 6, "iters": 2, "median": 5},
            "intensa":       {"d": 11, "sigma_color": 48, "sigma_space": 8, "iters": 3, "median": 7},
        },
    },
    {
        "key": "diffusion",
        "name": "Difusão Anisotrópica (Perona–Malik)",
        "short": "DIFF-PM",
        "desc": (
            "Difusão condutiva adaptativa: fluir difusivo alto em gradientes "
            "fracos (ruído, serrilhado fino) e quase nulo em gradientes fortes "
            "(fronteiras de zonas), produzindo transições naturalmente suaves "
            "sem deslocar os limites regionais."
        ),
        "strengths": {
            "minima":        {"kappa": 10, "lam": 0.12, "iters": 12},
            "intermediaria": {"kappa": 16, "lam": 0.12, "iters": 26},
            "intensa":       {"kappa": 24, "lam": 0.12, "iters": 44},
        },
    },
]


def params_text(tech_key: str, p: dict) -> str:
    if tech_key == "morph":
        return (f"fechamento r={p['r_close']} px · abertura r={p['r_open']} px · "
                f"mediana {p['median']}×{p['median']} · kernel elíptico · CIELAB")
    if tech_key == "bilateral":
        return (f"d={p['d']} · σcor={p['sigma_color']} · σesp={p['sigma_space']} · "
                f"{p['iters']} iteração(ões) · mediana {p['median']}×{p['median']}")
    return f"λ={p['lam']} · κ={p['kappa']} · {p['iters']} iterações · espaço Lab"


# ----------------------------------------------------------------------------
# 1. Carregamento e desacoplamento de dados
# ----------------------------------------------------------------------------
def _flatten_bezier(p0, p1, p2, p3, n: int = 12) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)[:, None]
    p0, p1, p2, p3 = (np.asarray(v, np.float32) for v in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3


def _shape_points(items, zoom: float) -> np.ndarray:
    """Converte os itens de um desenho vetorial em polilinha (px da raster)."""
    pts: list[np.ndarray] = []
    for it in items:
        kind = it[0]
        if kind == "l":
            pts.append(np.asarray([[it[1].x, it[1].y], [it[2].x, it[2].y]], np.float32))
        elif kind == "c":
            bez = _flatten_bezier((it[1].x, it[1].y), (it[2].x, it[2].y),
                                  (it[3].x, it[3].y), (it[4].x, it[4].y))
            pts.append(bez)
        elif kind == "re":
            r = it[1]
            pts.append(np.asarray([[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]], np.float32))
        elif kind == "qu":
            q = it[1]
            for p in (q.ul, q.ur, q.lr, q.ll):
                pts.append(np.asarray([[p.x, p.y]], np.float32))
    if not pts:
        return np.zeros((0, 2), np.float32)
    poly = np.concatenate(pts, axis=0) * np.float32(zoom)
    return poly


@dataclass
class VectorLayers:
    """Resultado do desacoplamento das camadas do PDF."""
    zone_shapes: list = field(default_factory=list)        # preenchimentos grandes (zonas)
    obs_shapes: list = field(default_factory=list)         # pequenos COM contorno (observacionais)
    speck_shapes: list = field(default_factory=list)       # pequenos SEM contorno (ruído de zona)
    stroke_shapes: list = field(default_factory=list)      # traços (geográficos/anotação)
    text_rects: list = field(default_factory=list)         # caixas de texto
    image_rects: list = field(default_factory=list)        # imagens embutidas (bbox pt)
    field_rects: list = field(default_factory=list)        # imagens que dominam a página (campo climático)
    stats: dict = field(default_factory=dict)


def collect_vector_layers(page: fitz.Page) -> VectorLayers:
    """Classifica os desenhos vetoriais e imagens embutidas da página."""
    layers = VectorLayers()
    n_drawings = 0
    for d in page.get_drawings():
        n_drawings += 1
        fill = d.get("fill")
        stroke = d.get("color")
        width = float(d.get("width") or 1.0)
        rect = d["rect"]
        bbox_area_pt2 = max(rect.width, 0.0) * max(rect.height, 0.0)
        pts = _shape_points(d["items"], 1.0)  # coordenadas de página (pt)

        if fill is not None and stroke is None:
            if bbox_area_pt2 <= SMALL_FILL_PT2:
                layers.speck_shapes.append(pts)          # ruído de vetorização
            else:
                layers.zone_shapes.append(pts)           # zona climática
        elif fill is not None and stroke is not None:
            if bbox_area_pt2 <= SMALL_FILL_PT2:
                layers.obs_shapes.append((pts, width))   # ponto/marcador observacional
            else:
                layers.zone_shapes.append(pts)           # painel/caixa preenchida
                layers.stroke_shapes.append((pts, width))
        else:  # somente traço -> camada geográfica / anotação (protegida)
            layers.stroke_shapes.append((pts, width))

    for w in page.get_text("words"):
        x0, y0, x1, y1 = w[:4]
        layers.text_rects.append((x0, y0, x1, y1))

    # imagens embutidas: campo climático rasterizado dentro do PDF (ex.: mapas
    # de gradiente interpolado exportados do matplotlib com rasterização)
    page_area = page.rect.width * page.rect.height
    for info in page.get_image_info():
        bbox = info.get("bbox")
        if not bbox:
            continue
        layers.image_rects.append(tuple(bbox))
        area = max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0)
        if area > 0.08 * page_area:
            layers.field_rects.append(tuple(bbox))

    n_field_imgs = len(layers.field_rects)
    if layers.zone_shapes and n_field_imgs == 0:
        field_type = "vetorial (polígonos de zona)"
    elif n_field_imgs and not layers.zone_shapes:
        field_type = "imagem embutida (campo rasterizado)"
    elif n_field_imgs and layers.zone_shapes:
        field_type = "híbrido (vetorial + imagem)"
    else:
        field_type = "não identificado"

    layers.stats = {
        "drawings": n_drawings,
        "zone_fills": len(layers.zone_shapes),
        "obs_points": len(layers.obs_shapes),
        "specks": len(layers.speck_shapes),
        "geo_lines": len(layers.stroke_shapes),
        "text_words": len(layers.text_rects),
        "embedded_images": len(layers.image_rects),
        "field_images": n_field_imgs,
        "field_type": field_type,
    }
    return layers


def render_pdf_page(pdf_path: Path, dpi: int):
    """Rasteriza a página 1 do PDF no DPI solicitado (com teto de pixels)."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    max_zoom = (MAX_PIXELS / (page.rect.width * page.rect.height)) ** 0.5
    zoom = min(zoom, max_zoom)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False,
                          colorspace=fitz.csRGB)
    base = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy()
    n_pages = doc.page_count
    doc.close()
    return base, zoom, n_pages  # BGR


def rasterize_masks(layers: VectorLayers, width_px: int, height_px: int, zoom: float):
    """Rasteriza as máscaras de disciplina (zona / protegida / observacional)."""
    zone = np.zeros((height_px, width_px), np.uint8)
    protected = np.zeros((height_px, width_px), np.uint8)
    obs = np.zeros((height_px, width_px), np.uint8)

    def to_px(pts: np.ndarray) -> np.ndarray:
        return np.rint(pts * np.float32(zoom)).astype(np.int32)

    for pts in layers.zone_shapes:
        p = to_px(pts)
        if len(p) >= 3:
            cv2.fillPoly(zone, [p], 255)
    # campos climáticos embutidos como imagem no PDF também são região de zona
    for (x0, y0, x1, y1) in layers.field_rects:
        cv2.rectangle(zone,
                      (int(math.floor(x0 * zoom)), int(math.floor(y0 * zoom))),
                      (int(math.ceil(x1 * zoom)), int(math.ceil(y1 * zoom))), 255, -1)
    for pts, w in layers.stroke_shapes:
        p = to_px(pts)
        thick = max(1, int(round(w * zoom)) + 1)
        if len(p) >= 2:
            cv2.polylines(protected, [p], False, 255, thickness=thick)
    for pts, w in layers.obs_shapes:
        p = to_px(pts)
        thick = max(1, int(round(w * zoom)) + 1)
        if len(p) >= 3:
            cv2.fillPoly(obs, [p], 255)
            cv2.polylines(obs, [p], True, 255, thickness=thick)
        elif len(p) >= 2:
            cv2.polylines(obs, [p], False, 255, thickness=thick)
    pad = 1.5 * zoom
    for (x0, y0, x1, y1) in layers.text_rects:
        cv2.rectangle(protected,
                      (int(x0 * zoom - pad), int(y0 * zoom - pad)),
                      (int(x1 * zoom + pad), int(y1 * zoom + pad)), 255, -1)

    protected |= obs

    # preenchimentos pequenos sem contorno: ruído de zona se pequenos em raster
    area_px = cv2.connectedComponentsWithStats(zone, connectivity=8)[2][:, 4]

    masks = {
        "zone": zone,
        "protected": protected,
        "obs": obs,
    }
    return masks


def build_refine_masks(base_bgr: np.ndarray, masks: dict):
    """Região elegível para refinamento + faixa de fronteira das zonas."""
    zone = masks["zone"] > 0
    protected = masks["protected"] > 0

    protect_dil = cv2.dilate(protected.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (PROTECT_DILATE_PX * 2 + 1,) * 2)) > 0
    candidates = zone & ~protect_dil

    # gradientes no espaço Lab (suavizados) -> fronteiras das zonas
    lab = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab_blur = cv2.GaussianBlur(lab, (3, 3), 1.0)
    grad = np.zeros(lab_blur.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(lab_blur[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab_blur[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        grad += gx * gx + gy * gy
    grad = np.sqrt(grad)
    edges = grad > (EDGE_GRAD_THR * 4.0)

    ksz = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_DILATE_PX * 2 + 1,) * 2)
    band = cv2.dilate((edges & candidates).astype(np.uint8), ksz) > 0
    band &= candidates

    blend = cv2.GaussianBlur(band.astype(np.float32), (5, 5), 1.2)
    blend *= candidates.astype(np.float32)          # confinamento duro
    return {
        "candidates": candidates,
        "band": band,
        "blend": np.clip(blend, 0.0, 1.0)[..., None],   # HxWx1
        "zone": zone,
        "protected": protected,
        "obs": masks["obs"] > 0,
    }


# ----------------------------------------------------------------------------
# 2/3. Tecnologias de refinamento (opera no espaço Lab; entrada BGR uint8)
# ----------------------------------------------------------------------------
def _to_lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)


def _to_bgr(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def _median(lab: np.ndarray, k: int) -> np.ndarray:
    if k and k >= 3:
        return cv2.medianBlur(lab, k)
    return lab


def refine_morph(bgr: np.ndarray, p: dict) -> np.ndarray:
    """Tecnologia 1 — Suavização Morfológica Multiescala (fechamento→abertura)."""
    lab = _to_lab(bgr)
    r = max(p["r_close"], p["r_open"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    lab = cv2.morphologyEx(lab, cv2.MORPH_CLOSE, kernel)
    lab = cv2.morphologyEx(lab, cv2.MORPH_OPEN, kernel)
    r2 = min(p["r_close"], p["r_open"])
    if r2 != r:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r2 + 1, 2 * r2 + 1))
        lab = cv2.morphologyEx(lab, cv2.MORPH_OPEN, kernel2)
        lab = cv2.morphologyEx(lab, cv2.MORPH_CLOSE, kernel2)
    lab = _median(lab, p.get("median", 3))
    return _to_bgr(lab)


def refine_bilateral(bgr: np.ndarray, p: dict) -> np.ndarray:
    """Tecnologia 2 — Filtragem bilateral iterativa (preserva bordas)."""
    out = bgr
    for _ in range(p["iters"]):
        out = cv2.bilateralFilter(out, d=p["d"], sigmaColor=p["sigma_color"],
                                  sigmaSpace=p["sigma_space"])
    lab = _to_lab(out)
    lab = _median(lab, p.get("median", 3))
    return _to_bgr(lab)


def refine_diffusion(bgr: np.ndarray, p: dict) -> np.ndarray:
    """Tecnologia 3 — Difusão anisotrópica de Perona–Malik (espaço Lab)."""
    img = _to_lab(bgr).astype(np.float32) / 255.0
    kappa = float(p["kappa"]) / 255.0
    lam = float(p["lam"])
    for _ in range(p["iters"]):
        pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
        dn = pad[:-2, 1:-1] - img
        ds = pad[2:, 1:-1] - img
        dw = pad[1:-1, :-2] - img
        de = pad[1:-1, 2:] - img
        cn = np.exp(-((dn / kappa) ** 2))
        cs = np.exp(-((ds / kappa) ** 2))
        cw = np.exp(-((dw / kappa) ** 2))
        ce = np.exp(-((de / kappa) ** 2))
        img += lam * (cn * dn + cs * ds + cw * dw + ce * de)
    lab = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return _to_bgr(lab)


TECH_FUNCS = {"morph": refine_morph, "bilateral": refine_bilateral,
              "diffusion": refine_diffusion}


# ----------------------------------------------------------------------------
# Composição com disciplina de máscaras
# ----------------------------------------------------------------------------
def composite(base_bgr: np.ndarray, processed_bgr: np.ndarray,
              blend: np.ndarray, protected: np.ndarray) -> np.ndarray:
    """Mistura apenas na faixa elegível; restaura bit a bit as camadas protegidas."""
    bf = base_bgr.astype(np.float32)
    pf = processed_bgr.astype(np.float32)
    out = bf * (1.0 - blend) + pf * blend
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    out[protected] = base_bgr[protected]        # garantia exata (erro 0 px)
    return out


# ----------------------------------------------------------------------------
# 5. Conferência de confiabilidade
# ----------------------------------------------------------------------------
def _lab_true(lab_u8: np.ndarray) -> np.ndarray:
    lab = lab_u8.astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def _delta_e(a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
    la = _lab_true(cv2.cvtColor(a_bgr, cv2.COLOR_BGR2Lab))
    lb = _lab_true(cv2.cvtColor(b_bgr, cv2.COLOR_BGR2Lab))
    d = la - lb
    return np.sqrt(np.sum(d * d, axis=-1))


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    s_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    s_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    n = a.size
    cov_norm = 1.0  # janela gaussiana: média ponderada já suaviza
    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * s_ab * cov_norm + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (s_a + s_b + C2))
    return float(ssim_map.mean())


def _measure_shift(base_bgr: np.ndarray, out_bgr: np.ndarray,
                   protected: np.ndarray) -> float | None:
    """Deslocamento (px) das camadas protegidas via correlação cruzada."""
    prot = protected.astype(np.float32)
    if prot.sum() < 50:
        return None
    density = cv2.boxFilter(prot, -1, (129, 129), normalize=True)
    _, _, _, max_loc = cv2.minMaxLoc(density)
    cx, cy = max_loc
    h, w = base_bgr.shape[:2]
    half = 128
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    crop_b = cv2.cvtColor(base_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    crop_o = cv2.cvtColor(out_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if crop_b.shape[0] < 100 or crop_b.shape[1] < 100:
        return None
    cy2, cx2 = crop_b.shape[0] // 2, crop_b.shape[1] // 2
    r = 40
    patch = crop_b[cy2 - r:cy2 + r, cx2 - r:cx2 + r]
    if patch.size == 0:
        return None
    res = cv2.matchTemplate(crop_o, patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, peak = cv2.minMaxLoc(res)
    dx = (peak[0] + r) - cx2
    dy = (peak[1] + r) - cy2
    return float(np.hypot(dx, dy))


def reliability_report(base_bgr: np.ndarray, out_bgr: np.ndarray, ctx: dict) -> dict:
    """Calcula todas as métricas de confiabilidade de uma versão."""
    zone = ctx["zone"]
    protected = ctx["protected"]
    obs = ctx["obs"]

    diff = cv2.absdiff(base_bgr, out_bgr)
    max_diff_prot = int(diff[protected].max()) if protected.any() else 0
    max_diff_obs = int(diff[obs].max()) if obs.any() else 0

    de = _delta_e(base_bgr, out_bgr)
    changed = (de > DE_THRESHOLD) & zone
    zone_px = int(zone.sum())
    changed_px = int(changed.sum())
    changed_pct = 100.0 * changed_px / max(zone_px, 1)

    de_zone_max = float(de[zone].max()) if zone_px else 0.0
    de_outside = de[~zone & ~protected] if (~zone & ~protected).any() else np.array([0.0])

    mse = float(np.mean((base_bgr.astype(np.float64) - out_bgr.astype(np.float64)) ** 2))
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

    ga = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    go = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
    ssim = _ssim_gray(ga, go)

    shift = _measure_shift(base_bgr, out_bgr, protected)

    overlay = base_bgr.copy()
    if changed_px:
        m = changed[..., None]
        tint = np.zeros_like(overlay)
        tint[:, :, 2] = 255  # vermelho em BGR
        overlay = np.where(m, (overlay * 0.45 + tint * 0.55).astype(np.uint8), overlay)

    return {
        "changed_pct": round(changed_pct, 3),
        "changed_px": changed_px,
        "zone_px": zone_px,
        "delta_e_max_zone": round(de_zone_max, 2),
        "delta_e_max_outside_zones": round(float(de_outside.max()), 2),
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim, 5),
        "vector_max_diff_px": max_diff_prot,
        "obs_max_diff_px": max_diff_obs,
        "protected_px": int(protected.sum()),
        "obs_px": int(obs.sum()),
        "shift_px": shift,
        "overlay_bgr": overlay,
    }


# ----------------------------------------------------------------------------
# 6. Exportação
# ----------------------------------------------------------------------------
def _tiff_description(tech: dict, label: str, metrics: dict, dpi: int) -> str:
    shift = metrics.get("shift_px")
    return json.dumps({
        "software": "RefinaPaleo 1.0 — Plataforma de Refinamento de Dados Paleoclimáticos",
        "tecnologia": tech["name"],
        "intensidade": label,
        "mudanca_zonas_pct": metrics["changed_pct"],
        "erro_vetorial_px": metrics["vector_max_diff_px"],
        "erro_observacional_px": metrics["obs_max_diff_px"],
        "deslocamento_camadas_protegidas_px": shift,
        "dpi": dpi,
        "espaco_cor": "RGB (processamento em CIELAB)",
        "nota": "Camadas vetoriais geográficas e pontos observacionais restaurados bit a bit; "
                "refinamento confinado às fronteiras das zonas climáticas.",
    })  # ensure_ascii: TIFF armazena ASCII puro (evita perda de acentos)


def save_version_images(out_dir: Path, base_name: str, img_bgr: np.ndarray,
                        dpi: int, description: str) -> dict:
    """Exporta uma imagem em PNG e TIFF com metadados científicos."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    png_path = out_dir / f"{base_name}.png"
    tif_path = out_dir / f"{base_name}.tif"
    pil.save(png_path, dpi=(dpi, dpi), optimize=True)

    info = TiffInfo()
    info[270] = description          # ImageDescription
    info[296] = 2                    # unidade = polegada
    pil.save(tif_path, compression="tiff_adobe_deflate", dpi=(dpi, dpi), tiffinfo=info)
    return {"png": png_path.name, "tif": tif_path.name}


def save_jpeg(img_bgr: np.ndarray, path: Path, width: int | None = None, quality: int = 85):
    img = img_bgr
    if width and img.shape[1] > width:
        s = width / img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, quality=quality)


_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size: int):
    for fp in _FONT_PATHS:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def comparison_sheet(out_dir: Path, name: str, base_bgr: np.ndarray,
                     out_bgr: np.ndarray, overlay_bgr: np.ndarray,
                     tech_name: str, label: str, metrics: dict, width: int = 1150):
    """Folha de conferência: Original | Refinado | Mapa de alteração."""
    panels = [
        (base_bgr, "ORIGINAL — PDF de entrada (vetorizado)"),
        (out_bgr, f"REFINADO — {tech_name} · intensidade {label}"),
        (overlay_bgr, f"MAPA DE ALTERAÇÃO — ΔE (CIE76) > {DE_THRESHOLD:g} · "
                      f"{metrics['changed_pct']:.2f}% das zonas · erro vetorial "
                      f"{metrics['vector_max_diff_px']} px"),
    ]
    h0, w0 = base_bgr.shape[:2]
    s = width / w0
    ph = int(h0 * s)
    bar = 56
    gap = 10
    sheet_h = (bar + ph) * 3 + gap * 2 + 12
    sheet = np.full((sheet_h, width + 16, 3), 245, np.uint8)
    y = 6
    for img, title in panels:
        small = cv2.resize(img, (width, ph), interpolation=cv2.INTER_AREA)
        sheet[y:y + bar, :] = (18, 32, 54)[::-1]  # navy BGR
        pil = Image.fromarray(cv2.cvtColor(sheet[y:y + bar], cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((12, bar // 2 - 9), title, fill=(255, 255, 255), font=_font(20))
        sheet[y:y + bar] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        y += bar
        sheet[y:y + ph, 8:8 + width] = small
        y += ph + gap
    save_jpeg(sheet, out_dir / f"{name}.jpg", width=None, quality=86)


# ----------------------------------------------------------------------------
# Orquestração do job completo
# ----------------------------------------------------------------------------
def process_pdf(pdf_path: Path, out_dir: Path, dpi: int = 120,
                progress=None) -> dict:
    """Executa a skill completa sobre um PDF vetorizado.

    `progress(fase:str, pct:float)` é chamado conforme as etapas avançam.
    Retorna o payload de resultados (metadados + 9 versões + métricas).
    """
    progress = progress or (lambda fase, pct: None)
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Carregamento e desacoplamento ---------------------------------
    progress("Renderizando o PDF vetorial…", 5)
    base_bgr, zoom_eff, n_pages = render_pdf_page(pdf_path, dpi)
    dpi_eff = int(round(72.0 * zoom_eff))
    h, w = base_bgr.shape[:2]

    progress("Desacoplando camadas vetoriais…", 12)
    doc = fitz.open(pdf_path)
    page = doc[0]
    layers = collect_vector_layers(page)
    page_pt = (page.rect.width, page.rect.height)
    file_name = Path(pdf_path).name
    doc.close()

    masks = rasterize_masks(layers, w, h, zoom_eff)
    ctx = build_refine_masks(base_bgr, masks)

    # PNG original de referência
    rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(out_dir / "original.png", dpi=(dpi_eff, dpi_eff), optimize=True)
    save_jpeg(base_bgr, out_dir / "original_thumb.jpg", width=760)

    n_tech = len(TECHNOLOGIES)
    versions_payload = []
    total_steps = n_tech * 3
    step = 0
    max_changed = 0.0
    vector_errors = []

    for tech in TECHNOLOGIES:
        fn = TECH_FUNCS[tech["key"]]
        versions = []
        for skey in ("minima", "intermediaria", "intensa"):
            step += 1
            label = STRENGTH_LABELS[skey]
            params = tech["strengths"][skey]
            progress(f"Tecnologia {tech['short']} · intensidade {label}…",
                     15 + 80.0 * (step - 1) / total_steps)
            t0 = time.time()

            processed = fn(base_bgr, params)
            out_bgr = composite(base_bgr, processed, ctx["blend"], ctx["protected"])
            metrics = reliability_report(base_bgr, out_bgr, ctx)
            elapsed = time.time() - t0

            base_name = f"refinado_{tech['key']}_{skey}"
            desc = _tiff_description(tech, label, metrics, dpi_eff)
            files = save_version_images(out_dir, base_name, out_bgr, dpi_eff, desc)
            save_jpeg(out_bgr, out_dir / f"{base_name}_thumb.jpg", width=760)
            comparison_sheet(out_dir, f"comparativo_{tech['key']}_{skey}",
                             base_bgr, out_bgr, metrics.pop("overlay_bgr"),
                             tech["name"], label, metrics)

            m = {k: v for k, v in metrics.items()}
            if m["shift_px"] is not None:
                m["shift_px"] = round(m["shift_px"], 2)
            versions.append({
                "key": skey,
                "label": label,
                "params": params,
                "params_text": params_text(tech["key"], params),
                "metrics": m,
                "seconds": round(elapsed, 2),
                "files": {
                    "png": files["png"],
                    "tif": files["tif"],
                    "cmp": f"comparativo_{tech['key']}_{skey}.jpg",
                    "thumb": f"{base_name}_thumb.jpg",
                },
            })
            max_changed = max(max_changed, m["changed_pct"])
            vector_errors += [m["vector_max_diff_px"], m["obs_max_diff_px"]]
        versions_payload.append({
            "key": tech["key"], "name": tech["name"], "short": tech["short"],
            "desc": tech["desc"], "versions": versions,
        })

    payload = {
        "job_id": out_dir.name,
        "file_name": file_name,
        "pages": n_pages,
        "dpi": dpi_eff,
        "page_pt": [round(page_pt[0], 1), round(page_pt[1], 1)],
        "px": [w, h],
        "layers": layers.stats,
        "pipeline": {
            "espaco_cor": "CIELAB (transformado a partir do RGB de rasterização)",
            "faixa_refinamento_px": EDGE_DILATE_PX,
            "exclusao_protegida_px": PROTECT_DILATE_PX,
            "limiar_delta_e": DE_THRESHOLD,
        },
        "technologies": versions_payload,
        "summary": {
            "versions": total_steps,
            "max_changed_pct": round(max_changed, 3),
            "max_vector_error_px": max(vector_errors) if vector_errors else 0,
            "processing_seconds": round(time.time() - t_start, 2),
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("Concluído.", 100)
    return payload
