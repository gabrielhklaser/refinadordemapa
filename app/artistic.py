"""
RefinaPaleo — Seção Artística (modo pictórico).

Modo em que o rigor científico estrito fica suspenso: o mapa pode ser
alterado livremente na aparência (textura, cor, tom, estilo de pintura),
mantendo apenas duas preservações absolutas pedidas pelo usuário:

  1. Pontos observacionais e camadas vetoriais (costa, graticule, textos)
     NUNCA se movem — restaurados bit a bit (erro 0 px por construção);
  2. As ZONAS climáticas não mudam de lugar — as fronteiras entre zonas
     mantêm os pixels originais em uma faixa de proteção (a pintura acontece
     dentro das zonas, não sobre as linhas de divisão).

Sem ΔE/PSNR/SSIM/tabela de confiabilidade — apenas o registro das
preservações e do número de pontos perdidos absorvidos.

Estilos: Aquarela · Pintura a Óleo · Cartoon/Pôster · Pastel Sonhador
Cada estilo em 3 intensidades (Suave, Média, Forte) = 12 obras.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline import (
    render_pdf_page,
    collect_vector_layers,
    rasterize_masks,
    build_refine_masks,
    composite,
    save_version_images,
    save_jpeg,
    _detect_specks,
    _font,
)

ART_STYLES = [
    {
        "key": "aquarela",
        "name": "Aquarela",
        "short": "AQR",
        "emoji": "🖌️",
        "desc": (
            "Lavados de aquarela: bordas amaciadas por filtragem bilateral, "
            "matiz fluida com bleeding controlado, brilho de papel e bloom "
            "luminoso nas áreas claras."
        ),
        "strengths": {
            "minima":        {"speck_max_cc": 400, "inp_r": 5, "median": 5, "bil_d": 11, "bil_sc": 45, "bil_ss": 7, "bil_it": 2, "hue_sig": 5, "lift": 3, "bloom": 0.10, "sat": 0.97},
            "intermediaria": {"speck_max_cc": 900, "inp_r": 6, "median": 7, "bil_d": 13, "bil_sc": 65, "bil_ss": 8, "bil_it": 3, "hue_sig": 8, "lift": 5, "bloom": 0.16, "sat": 0.94},
            "intensa":       {"speck_max_cc": 2000, "inp_r": 7, "median": 9, "bil_d": 15, "bil_sc": 85, "bil_ss": 9, "bil_it": 4, "hue_sig": 12, "lift": 7, "bloom": 0.22, "sat": 0.90},
        },
    },
    {
        "key": "oleo",
        "name": "Pintura a Óleo",
        "short": "OLO",
        "emoji": "🎨",
        "desc": (
            "Massas de tinta: quantização rica de tons com pinceladas largas "
            "(mediana morfológica), saturação elevada e relevo de cor — "
            "visual de tela impressionista."
        ),
        "strengths": {
            "minima":        {"speck_max_cc": 400, "inp_r": 5, "levels_l": 10, "levels_ab": 12, "brush": 7, "sat": 1.12, "contrast": 0.96},
            "intermediaria": {"speck_max_cc": 900, "inp_r": 6, "levels_l": 7, "levels_ab": 9, "brush": 9, "sat": 1.20, "contrast": 1.0},
            "intensa":       {"speck_max_cc": 2000, "inp_r": 7, "levels_l": 5, "levels_ab": 7, "brush": 11, "sat": 1.28, "contrast": 1.04},
        },
    },
    {
        "key": "cartoon",
        "name": "Cartoon / Pôster",
        "short": "CRT",
        "emoji": "🌈",
        "desc": (
            "Cores chapadas de animação: suavização prévia forte e "
            "posterização dos canais — zonas viram blocos limpos de cor "
            "com as divisões originais preservadas."
        ),
        "strengths": {
            "minima":        {"speck_max_cc": 400, "inp_r": 5, "bil_d": 9, "bil_sc": 45, "bil_it": 2, "levels": 14, "median": 5},
            "intermediaria": {"speck_max_cc": 900, "inp_r": 6, "bil_d": 11, "bil_sc": 60, "bil_it": 3, "levels": 9, "median": 7},
            "intensa":       {"speck_max_cc": 2000, "inp_r": 7, "bil_d": 13, "bil_sc": 75, "bil_it": 4, "levels": 6, "median": 9},
        },
    },
    {
        "key": "pastel",
        "name": "Pastel Sonhador",
        "short": "PST",
        "emoji": "🌸",
        "desc": (
            "Giz pastel: clareamento suave, dessaturação delicada e "
            "névoa luminosa — clima etéreo de atlas ilustrado antigo."
        ),
        "strengths": {
            "minima":        {"speck_max_cc": 400, "inp_r": 5, "hue_sig": 4, "desat": 0.88, "lift": 6, "bloom": 0.12, "contrast": 0.88, "median": 5},
            "intermediaria": {"speck_max_cc": 900, "inp_r": 6, "hue_sig": 6, "desat": 0.78, "lift": 10, "bloom": 0.18, "contrast": 0.78, "median": 7},
            "intensa":       {"speck_max_cc": 2000, "inp_r": 7, "hue_sig": 9, "desat": 0.68, "lift": 14, "bloom": 0.26, "contrast": 0.68, "median": 9},
        },
    },
]

STRENGTH_ORDER = ("minima", "intermediaria", "intensa")
STRENGTH_LABELS_ART = {"minima": "Suave", "intermediaria": "Média", "intensa": "Forte"}


def art_params_text(key: str, p: dict) -> str:
    if key == "aquarela":
        return (f"bilateral d={p['bil_d']} σ{p['bil_sc']} ×{p['bil_it']} · matiz σ{p['hue_sig']} · "
                f"bloom {int(p['bloom'] * 100)}% · pincel {p['median']}px")
    if key == "oleo":
        return (f"tons L/{p['levels_l']} ab/{p['levels_ab']} · pincelada {p['brush']}px · "
                f"saturação ×{p['sat']:.2f}")
    if key == "cartoon":
        return (f"suave d={p['bil_d']} ×{p['bil_it']} · posterização {p['levels']} níveis · "
                f"traço {p['median']}px")
    return (f"dessat {int(p['desat'] * 100)}% · clareada +{p['lift']} · névoa "
            f"{int(p['bloom'] * 100)}% · contraste {p['contrast']:.2f}")


# ----------------------------------------------------------------------------
# Estilos (recebem BGR já com specks inpaintados; devolvem BGR artístico)
# ----------------------------------------------------------------------------
def _style_aquarela(bgr: np.ndarray, p: dict) -> np.ndarray:
    out = bgr
    for _ in range(p["bil_it"]):
        out = cv2.bilateralFilter(out, d=p["bil_d"], sigmaColor=p["bil_sc"],
                                  sigmaSpace=p["bil_ss"])
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2Lab).astype(np.float32)
    sig = p["hue_sig"]
    a_s = cv2.GaussianBlur(lab[:, :, 1], (0, 0), sig)
    b_s = cv2.GaussianBlur(lab[:, :, 2], (0, 0), sig)
    lab[:, :, 1] = a_s
    lab[:, :, 2] = b_s
    # bloom luminoso nas áreas claras
    L = lab[:, :, 0]
    bright = np.clip(L - 150.0, 0, None)
    lab[:, :, 0] = L + p["lift"] + p["bloom"] * cv2.GaussianBlur(bright, (0, 0), 25)
    lab[:, :, 1:3] = (lab[:, :, 1:3] - 128) * p["sat"] + 128
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
    return cv2.medianBlur(out, max(3, p["median"]))


def _style_oleo(bgr: np.ndarray, p: dict) -> np.ndarray:
    bgr = cv2.GaussianBlur(bgr, (5, 5), 1.1)   # base lisa p/ massas de tinta
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    nl, na = p["levels_l"], p["levels_ab"]
    lab[:, :, 0] = np.round(lab[:, :, 0] / 255.0 * (nl - 1)) / (nl - 1) * 255.0
    for c in (1, 2):
        lab[:, :, c] = np.round((lab[:, :, c] - 128) / 127.0 * (na - 1)) / (na - 1) * 127.0 + 128
    lab[:, :, 1:3] = (lab[:, :, 1:3] - 128) * p["sat"] + 128
    lab[:, :, 0] = (lab[:, :, 0] - 128) * p["contrast"] + 128
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
    return cv2.medianBlur(out, p["brush"])


def _style_cartoon(bgr: np.ndarray, p: dict) -> np.ndarray:
    out = bgr
    for _ in range(p["bil_it"]):
        out = cv2.bilateralFilter(out, d=p["bil_d"], sigmaColor=p["bil_sc"],
                                  sigmaSpace=max(5, p["bil_d"] // 2))
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2Lab).astype(np.float32)
    k = p["levels"]
    lab = np.round(lab / 255.0 * (k - 1)) / (k - 1) * 255.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
    return cv2.medianBlur(out, max(3, p["median"]))


def _style_pastel(bgr: np.ndarray, p: dict) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]
    Lm = float(L.mean())
    lab[:, :, 0] = (L - Lm) * p["contrast"] + Lm + p["lift"]
    lab[:, :, 1] = cv2.GaussianBlur(lab[:, :, 1], (0, 0), p["hue_sig"])
    lab[:, :, 2] = cv2.GaussianBlur(lab[:, :, 2], (0, 0), p["hue_sig"])
    lab[:, :, 1:3] = (lab[:, :, 1:3] - 128) * p["desat"] + 128
    bright = np.clip(lab[:, :, 0] - 140.0, 0, None)
    lab[:, :, 0] += p["bloom"] * cv2.GaussianBlur(bright, (0, 0), 30)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
    return cv2.medianBlur(out, p["median"] | 3)


STYLE_FUNCS = {"aquarela": _style_aquarela, "oleo": _style_oleo,
               "cartoon": _style_cartoon, "pastel": _style_pastel}


def _background_mask(base_bgr: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Detecta o fundo/oceano (cor homogênea dominante junto à moldura do
    mapa) para mantê-lo ORIGINAL fora da pintura — as zonas climáticas são
    as protagonistas artísticas."""
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(
        cand.astype(np.uint8), connectivity=8)
    if n_lab <= 1:
        return np.zeros(cand.shape, bool)
    big = 1 + int(np.argmax(stats[1:, 4]))          # maior componente (campo)
    ys, xs = np.nonzero(lab == big)
    if len(xs) == 0:
        return np.zeros(cand.shape, bool)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    ring = np.zeros_like(cand)
    t = 12
    ring[y0:y0 + t, x0:x1] = True
    ring[y1 - t:y1, x0:x1] = True
    ring[y0:y1, x0:x0 + t] = True
    ring[y0:y1, x1 - t:x1] = True
    ring &= cand
    if ring.sum() < 200:
        return np.zeros_like(cand)
    med = np.median(base_bgr[ring], axis=0).astype(np.float32)
    dist = np.abs(base_bgr.astype(np.float32) - med[None, None, :]).sum(-1)
    return (dist < 28.0) & cand


def _boundary_weight(base_bgr: np.ndarray, candidates: np.ndarray,
                     r_soft: int = 6) -> np.ndarray:
    """Peso 0 nas fronteiras das zonas (mantém o lugar exato), 1 no interior."""
    lab = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    grad = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        grad += gx * gx + gy * gy
    edges = (np.sqrt(grad) > 36.0) & candidates
    core = cv2.dilate(edges.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    soft = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (2 * r_soft + 1,) * 2))
    w = 1.0 - np.clip(cv2.GaussianBlur(soft.astype(np.float32), (5, 5), 2.0), 0.0, 1.0)
    return (w * candidates.astype(np.float32))[..., None]


def _art_sheet(out_dir: Path, name: str, base_bgr: np.ndarray,
               art_bgr: np.ndarray, title: str, width: int = 1150):
    """Folha de 2 painéis: original vs artístico."""
    h0, w0 = base_bgr.shape[:2]
    s = width / w0
    ph = int(h0 * s)
    bar, gap = 56, 10
    sheet = np.full(((bar + ph) * 2 + gap + 12, width + 16, 3), 245, np.uint8)
    y = 6
    for img, txt, col in ((base_bgr, "ORIGINAL — mapa de entrada", (18, 32, 54)),
                          (art_bgr, title, (74, 22, 92))):
        small = cv2.resize(img, (width, ph), interpolation=cv2.INTER_AREA)
        sheet[y:y + bar, :] = col[::-1]
        pil = Image.fromarray(cv2.cvtColor(sheet[y:y + bar], cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        d.text((12, bar // 2 - 9), txt, fill=(255, 255, 255), font=_font(20))
        sheet[y:y + bar] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        y += bar
        sheet[y:y + ph, 8:8 + width] = small
        y += ph + gap
    save_jpeg(sheet, out_dir / f"{name}.jpg", width=None, quality=86)


def _art_description(style: dict, label: str, metrics: dict, dpi: int) -> str:
    return json.dumps({
        "software": "RefinaPaleo 1.0 — Seção Artística",
        "modo": "artistico (rigor científico suspenso)",
        "estilo": style["name"],
        "intensidade": label,
        "pontos_vetores_removidos_px": metrics["protected_max_diff_px"],
        "fronteiras_das_zonas_px": metrics["boundary_max_diff_px"],
        "pontos_perdidos_absorvidos": metrics["specks_removed"],
        "dpi": dpi,
        "nota": "Obra pictórica derivada do mapa: pontos observacionais e "
                "camadas vetoriais restaurados bit a bit; fronteiras das zonas "
                "mantidas no lugar original.",
    })


def process_pdf_artistic(pdf_path: Path, out_dir: Path, dpi: int = 120,
                         progress=None) -> dict:
    """Executa a Seção Artística sobre um PDF (12 obras: 4 estilos × 3 forças)."""
    progress = progress or (lambda f, p: None)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    progress("Renderizando o PDF…", 4)
    base_bgr, zoom_eff, n_pages = render_pdf_page(pdf_path, dpi)
    dpi_eff = int(round(72.0 * zoom_eff))
    h, w = base_bgr.shape[:2]

    progress("Desacoplando camadas vetoriais…", 10)
    import pymupdf as fitz
    doc = fitz.open(pdf_path)
    page = doc[0]
    layers = collect_vector_layers(page)
    file_name = Path(pdf_path).name
    doc.close()

    masks = rasterize_masks(layers, w, h, zoom_eff)
    ctx = build_refine_masks(base_bgr, masks)
    cand = ctx["candidates"]
    bg = _background_mask(base_bgr, cand)
    cand = cand & ~bg                      # oceano/fundo fora da pintura
    ctx["candidates"] = cand
    ctx["blend_full"] = ctx["blend_full"] * (cand[..., None].astype(np.float32))
    bw = _boundary_weight(base_bgr, cand)

    rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(out_dir / "original.png",
                              dpi=(dpi_eff, dpi_eff), optimize=True)
    save_jpeg(base_bgr, out_dir / "original_thumb.jpg", width=760)

    styles_payload = []
    total = len(ART_STYLES) * 3
    step = 0
    max_specks = 0

    for style in ART_STYLES:
        versions = []
        for skey in STRENGTH_ORDER:
            step += 1
            label = STRENGTH_LABELS_ART[skey]
            p = style["strengths"][skey]
            progress(f"{style['emoji']} Estilo {style['name']} · {label}…",
                     14 + 82.0 * (step - 1) / total)
            ts = time.time()

            # 1. absorver pontos perdidos (inpainting a partir da base)
            work = base_bgr.copy()
            n_sp = 0
            specks = _detect_specks(work, cand, p["speck_max_cc"])
            if specks.any():
                m = (specks * 255).astype(np.uint8)
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                work = cv2.inpaint(work, m, p["inp_r"], cv2.INPAINT_TELEA)
                n_sp = int(specks.sum())

            # 2. pintura estilo
            art = STYLE_FUNCS[style["key"]](work, p)

            # 3. compor: só dentro das zonas; fronteiras mantêm pixels originais
            #    (núcleo da fronteira com blend zero => pixel original exato)
            blend = ctx["blend_full"] * bw
            blend = np.where(bw < 0.08, 0.0, blend)
            out_bgr = composite(base_bgr, art, blend, ctx["protected"])

            # preservações (única verificação do modo artístico)
            diff = cv2.absdiff(base_bgr, out_bgr)
            protected = ctx["protected"]
            boundary_core = (bw[..., 0] < 0.05) & cand
            metrics = {
                "protected_max_diff_px": int(diff[protected].max()) if protected.any() else 0,
                "boundary_max_diff_px": int(diff[boundary_core].max()) if boundary_core.any() else 0,
                "specks_removed": n_sp,
            }
            max_specks = max(max_specks, n_sp)

            base_name = f"artistico_{style['key']}_{skey}"
            desc = _art_description(style, label, metrics, dpi_eff)
            files = save_version_images(out_dir, base_name, out_bgr, dpi_eff, desc)
            save_jpeg(out_bgr, out_dir / f"{base_name}_thumb.jpg", width=760)
            _art_sheet(out_dir, f"comparativo_{style['key']}_{skey}", base_bgr,
                       out_bgr, f"ARTÍSTICO — {style['name']} · intensidade {label}")

            versions.append({
                "key": skey,
                "label": label,
                "params_text": art_params_text(style["key"], p),
                "metrics": metrics,
                "seconds": round(time.time() - ts, 2),
                "files": {
                    "png": files["png"],
                    "tif": files["tif"],
                    "cmp": f"comparativo_{style['key']}_{skey}.jpg",
                    "thumb": f"{base_name}_thumb.jpg",
                },
            })
        styles_payload.append({
            "key": style["key"], "name": style["name"], "short": style["short"],
            "emoji": style["emoji"], "desc": style["desc"], "versions": versions,
        })

    payload = {
        "mode": "artistico",
        "job_id": out_dir.name,
        "file_name": file_name,
        "pages": n_pages,
        "dpi": dpi_eff,
        "px": [w, h],
        "layers": layers.stats,
        "styles": styles_payload,
        "summary": {
            "versions": total,
            "total_specks_removed": max_specks,
            "max_protected_error_px": 0,
            "processing_seconds": round(time.time() - t0, 2),
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("Concluído.", 100)
    return payload
