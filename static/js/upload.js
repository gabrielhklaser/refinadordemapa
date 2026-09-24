/* RefinaPaleo — página de upload */
(function () {
  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const fileNameEl = $("dz-file-name");
  const dzFile = $("dz-file");
  const btnProcess = $("btn-process");
  const btnSample = $("btn-sample");
  const dpiSel = $("dpi");
  const sampleTypeSel = $("sample-type");
  const progressCard = $("progress-card");
  const progressMsg = $("progress-msg");
  const progressPct = $("progress-pct");
  const progressBar = $("progress-bar");
  const errorBox = $("error-box");
  const toast = $("toast");

  let selectedFile = null;
  let busy = false;
  let pollTimer = null;

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("on");
    setTimeout(() => toast.classList.remove("on"), 3800);
  }

  function setError(msg) {
    if (!msg) { errorBox.classList.remove("on"); return; }
    errorBox.textContent = "⚠️ " + msg;
    errorBox.classList.add("on");
  }

  function setFile(file) {
    if (!file) return;
    if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
      setError("O arquivo precisa ser um PDF vetorizado.");
      selectedFile = null;
      btnProcess.disabled = true;
      return;
    }
    setError("");
    selectedFile = file;
    fileNameEl.textContent = file.name + " (" + (file.size / 1024 / 1024).toFixed(2) + " MB)";
    dzFile.style.display = "flex";
    btnProcess.disabled = false;
  }

  dropzone.addEventListener("click", () => { if (!busy) fileInput.click(); });
  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !busy) { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (e) => {
    if (busy) return;
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  function setProgress(pct, msg) {
    progressPct.textContent = Math.round(pct) + "%";
    progressBar.style.width = pct + "%";
    if (msg) progressMsg.textContent = msg;
  }

  function poll(jobId) {
    pollTimer = setInterval(async () => {
      try {
        const r = await fetch(`/api/jobs/${jobId}/status`);
        const st = await r.json();
        setProgress(st.pct || 0, st.message);
        if (st.state === "done") {
          clearInterval(pollTimer);
          window.location.href = `/resultados?job=${jobId}`;
        } else if (st.state === "error") {
          clearInterval(pollTimer);
          fail(st.message || "Falha no processamento.");
        }
      } catch (_) { /* segue tentando */ }
    }, 1200);
  }

  function fail(msg) {
    busy = false;
    progressCard.classList.remove("on");
    btnProcess.disabled = !selectedFile;
    btnSample.disabled = false;
    setError(msg);
  }

  async function submit(withSample) {
    if (busy) return;
    busy = true;
    setError("");
    btnProcess.disabled = true;
    btnSample.disabled = true;
    progressCard.classList.add("on");
    setProgress(1, "Enviando mapa…");

    const fd = new FormData();
    fd.append("dpi", dpiSel.value);
    if (withSample) fd.append("sample", "true");
    else fd.append("pdf", selectedFile);

    try {
      const r = await fetch("/api/process", { method: "POST", body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || "Falha ao iniciar o processamento.");
      }
      const { job_id } = await r.json();
      setProgress(2, "Na fila de processamento…");
      poll(job_id);
    } catch (e) {
      fail(e.message);
    }
  }

  btnProcess.addEventListener("click", () => {
    if (!selectedFile) { setError("Selecione um PDF vetorizado primeiro."); return; }
    submit(false);
  });
  btnSample.addEventListener("click", () => {
    selectedFile = null;
    dzFile.style.display = "none";
    submit(true);
  });
})();
