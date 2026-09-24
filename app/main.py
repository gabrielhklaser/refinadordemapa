"""
RefinaPaleo — API FastAPI.

Páginas:
  GET  /                          -> página de upload (index.html)
  GET  /resultados?job=<id>       -> página de resultados (resultados.html)

API:
  POST /api/process               -> enfileira o processamento (PDF enviado ou mapa de exemplo)
  GET  /api/jobs/{id}/status      -> progresso do processamento (polling)
  GET  /api/jobs/{id}             -> payload completo de resultados
  GET  /api/jobs/{id}/files/<arq> -> download dos artefatos (tif/png/jpg)
  GET  /api/jobs/{id}/zip         -> lote completo (TIFF+PNG) em .zip
  GET  /api/health                -> verificação simples
"""

from __future__ import annotations

import json
import threading
import traceback
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sample_map import generate_sample_pdf, generate_sample_gradient_pdf
from pipeline import process_pdf
from artistic import process_pdf_artistic

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
DATA = ROOT / "data" / "jobs"

ALLOWED_SUFFIXES = {".png", ".tif", ".jpg", ".json", ".pdf"}

app = FastAPI(title="RefinaPaleo", docs_url=None, redoc_url=None)

_jobs_lock = threading.Semaphore(1)  # processa um job por vez (CPU-bound)


# --------------------------------------------------------------------------
# Páginas
# --------------------------------------------------------------------------
@app.get("/")
def page_index():
    return FileResponse(STATIC / "index.html")


@app.get("/resultados")
def page_results():
    return FileResponse(STATIC / "resultados.html")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
def _job_dir(job_id: str) -> Path:
    if not job_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789" for ch in job_id):
        raise HTTPException(400, "job id inválido")
    d = DATA / job_id
    if not d.is_dir():
        raise HTTPException(404, "job não encontrado")
    return d


def _write_status(job_id: str, payload: dict):
    d = DATA / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _run_job(job_id: str, pdf_path: Path, dpi: int, mode: str = "sci"):
    with _jobs_lock:
        try:
            def progress(fase: str, pct: float):
                _write_status(job_id, {"state": "running", "message": fase, "pct": round(pct, 1)})

            if mode == "art":
                _write_status(job_id, {"state": "running", "message": "Preparando a Seção Artística…", "pct": 2})
                payload = process_pdf_artistic(pdf_path, DATA / job_id, dpi=dpi, progress=progress)
            else:
                _write_status(job_id, {"state": "running", "message": "Inicializando pipeline…", "pct": 2})
                payload = process_pdf(pdf_path, DATA / job_id, dpi=dpi, progress=progress)
            _write_status(job_id, {"state": "done", "message": "Processamento concluído", "pct": 100})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _write_status(job_id, {"state": "error", "message": f"Falha no processamento: {exc}", "pct": 100})


@app.post("/api/process")
async def api_process(
    pdf: UploadFile | None = File(None),
    sample: str = Form("false"),
    dpi: int = Form(120),
    mode: str = Form("sci"),
):
    import uuid

    dpi = min(max(int(dpi), 80), 220)
    job_id = uuid.uuid4().hex[:12]
    d = DATA / job_id
    d.mkdir(parents=True, exist_ok=True)

    if sample in ("true", "zones", "gradient"):
        if sample == "gradient":
            pdf_path = d / "mapa_exemplo_aridez_115ma.pdf"
            generate_sample_gradient_pdf(str(pdf_path))
        else:
            pdf_path = d / "mapa_exemplo_paleoclimatico.pdf"
            generate_sample_pdf(str(pdf_path))
    else:
        if pdf is None or not (pdf.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, "Envie um arquivo PDF vetorizado (ou marque a opção de exemplo).")
        pdf_path = d / "input.pdf"
        data = await pdf.read()
        if len(data) > 80 * 1024 * 1024:
            raise HTTPException(413, "PDF muito grande (limite: 80 MB).")
        pdf_path.write_bytes(data)
        # validação rápida
        try:
            import pymupdf as fitz
            doc = fitz.open(pdf_path)
            if doc.page_count < 1:
                raise ValueError("PDF sem páginas")
            doc.close()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"PDF inválido: {exc}") from exc

    threading.Thread(target=_run_job, args=(job_id, pdf_path, dpi, mode), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/status")
def api_status(job_id: str):
    d = _job_dir(job_id)
    f = d / "status.json"
    if not f.exists():
        return {"state": "queued", "message": "Na fila…", "pct": 0}
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}")
def api_results(job_id: str):
    d = _job_dir(job_id)
    f = d / "results.json"
    if not f.exists():
        st = d / "status.json"
        if st.exists() and json.loads(st.read_text())["state"] == "error":
            raise HTTPException(500, "O processamento falhou para este job.")
        raise HTTPException(202, "Ainda processando.")
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}/files/{name}")
def api_file(job_id: str, name: str):
    d = _job_dir(job_id)
    path = (d / name).resolve()
    if path.suffix.lower() not in ALLOWED_SUFFIXES or path.parent != d.resolve():
        raise HTTPException(404, "arquivo não disponível")
    if not path.exists():
        raise HTTPException(404, "arquivo não encontrado")
    media = {".png": "image/png", ".tif": "image/tiff", ".jpg": "image/jpeg",
             ".json": "application/json", ".pdf": "application/pdf"}
    dl = name.startswith(("refinado_", "original"))
    return FileResponse(path, media_type=media[path.suffix.lower()],
                        filename=name if dl else None)


@app.get("/api/jobs/{job_id}/zip")
def api_zip(job_id: str):
    d = _job_dir(job_id)
    zip_path = d / "refinadopaleo_lote.zip"
    if not zip_path.exists():
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in {".tif", ".png", ".json"} and f.name != "refinadopaleo_lote.zip":
                    z.write(f, f.name)
        zip_path.write_bytes(buf.getvalue())
    return FileResponse(zip_path, media_type="application/zip",
                        filename=f"refinadopaleo_{job_id}.zip")


@app.middleware("http")
async def _no_cache_ui(request, call_next):
    """Evita cache stale do frontend (o navegador pode segurar JS/CSS antigos)."""
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path == "/resultados" or path.startswith("/static"):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@app.get("/api/health")
def api_health():
    return {"ok": True, "service": "RefinaPaleo"}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
