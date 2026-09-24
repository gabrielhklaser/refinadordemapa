/* RefinaPaleo — página de resultados */
(function () {
  const $ = (id) => document.getElementById(id);
  const job = new URLSearchParams(window.location.search).get("job");
  const toast = $("toast");

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("on");
    setTimeout(() => toast.classList.remove("on"), 3600);
  }

  const fileURL = (name) => `/api/jobs/${job}/files/${encodeURIComponent(name)}`;

  function fmt(n, d = 2) {
    if (n === null || n === undefined || n === "inf") return "—";
    return Number(n).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  function chgBadge(pct) {
    if (pct < 1.5) return "badge low";
    if (pct < 5) return "badge mid";
    return "badge";
  }

  if (!job) {
    $("loading").textContent = "Job não informado — volte e processe um mapa.";
    return;
  }

  fetch(`/api/jobs/${job}`)
    .then((r) => {
      if (r.status === 202) throw new Error("Ainda processando — aguarde e recarregue a página.");
      if (!r.ok) throw new Error("Não foi possível carregar este job (" + r.status + ").");
      return r.json();
    })
    .then(render)
    .catch((e) => {
      $("loading").style.display = "none";
      const eb = $("error-box");
      eb.textContent = "⚠️ " + e.message;
      eb.classList.add("on");
    });

  function render(data) {
    $("loading").style.display = "none";
    $("content").hidden = false;
    window.document.title = `Resultados — ${data.file_name} · RefinaPaleo`;

    /* ---------- toolbar ---------- */
    $("chip-file").textContent = `📄 ${data.file_name}`;
    $("chip-px").textContent = `🖼️ ${data.px[0]}×${data.px[1]} px · ${data.dpi} dpi${data.pages > 1 ? ` · página 1/${data.pages}` : ""}`;
    $("btn-zip").href = `/api/jobs/${job}/zip`;

    /* ---------- resumo global ---------- */
    $("st-vers").textContent = data.summary.versions;
    $("st-err").textContent = `${data.summary.max_vector_error_px} px`;
    const shifts = data.technologies.flatMap(t => t.versions.map(v => v.metrics.shift_px)).filter(v => v !== null && v !== undefined);
    $("st-shift").textContent = shifts.length ? `${Math.max(...shifts).toFixed(1)} px` : "0 px";
    $("st-chg").textContent = `${fmt(data.summary.max_changed_pct)}%`;

    /* ---------- original ---------- */
    $("orig-img").src = fileURL("original.png");
    const L = data.layers;
    const kvRows = [
      ["Desenhos vetoriais", L.drawings],
      ["Preenchimentos de zona", L.zone_fills],
      ["Linhas geográficas", L.geo_lines],
      ["Pontos observacionais", L.obs_points],
      ["Specks de ruído detectados", L.specks],
      ["Palavras de texto", L.text_words],
    ];
    if (L.embedded_images !== undefined) kvRows.push(["Imagens embutidas", L.embedded_images]);
    kvRows.push(["Tipo de campo climático", L.field_type || "—"]);
    $("layers-kv").innerHTML = kvRows
      .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    $("orig-chips").innerHTML = `
      <span class="chip">Espaço de processamento: ${data.pipeline.espaco_cor}</span>
      <span class="chip">Faixa de refinamento: ±${data.pipeline.faixa_refinamento_px} px</span>
      <span class="chip">Margem protegida: ${data.pipeline.exclusao_protegida_px} px</span>
      <span class="chip">ΔE limiar: ${data.pipeline.limiar_delta_e}</span>`;

    /* ---------- tecnologias ---------- */
    const list = $("tech-list");
    list.innerHTML = "";
    const tableRows = [];

    data.technologies.forEach((tech, ti) => {
      const block = document.createElement("div");
      block.className = "tech-block";
      block.innerHTML = `
        <div class="tech-head">
          <span class="tech-tag">${tech.short}</span>
          <div>
            <h3>${tech.name}</h3>
            <p>${tech.desc}</p>
          </div>
        </div>
        <div class="vers-grid"></div>`;
      const grid = block.querySelector(".vers-grid");

      tech.versions.forEach((v) => {
        const m = v.metrics;
        const card = document.createElement("div");
        card.className = "vers-card";
        card.innerHTML = `
          <img loading="lazy" src="${fileURL(v.files.thumb)}" alt="${tech.name} — ${v.label}">
          <div class="vers-body">
            <div class="vers-title"><h4>${v.label}</h4>
              <span class="badge ${chgBadge(m.changed_pct)}">zonas +${fmt(m.changed_pct)}%</span></div>
            <div class="vers-params">${v.params_text}</div>
            <div class="vers-metrics">
              <span class="badge">erro vetorial ${m.vector_max_diff_px} px</span>
              <span class="badge">obs. ${m.obs_max_diff_px} px</span>
              <span class="badge">PSNR ${fmt(m.psnr_db, 1)} dB</span>
            </div>
            <div class="vers-actions">
              <a class="btn btn-ghost btn-small" href="${fileURL(v.files.tif)}" download="RefinaPaleo_${tech.key}_${v.key}.tif">⬇ TIF</a>
              <a class="btn btn-primary btn-small" href="${fileURL(v.files.png)}" download="RefinaPaleo_${tech.key}_${v.key}.png">⬇ PNG</a>
            </div>
            <div class="mini-note">Processado em ${fmt(v.seconds, 1)} s · clique no cartão p/ comparativo</div>
          </div>`;
        card.addEventListener("click", (e) => {
          if (e.target.closest("a")) return;
          openModal(tech, v);
        });
        grid.appendChild(card);

        tableRows.push(`
          ${v.key === "minima" ? `<tr class="tech-sep"><td colspan="10">${tech.short} — ${tech.name}</td></tr>` : ""}
          <tr>
            <td class="tech-cell"><b>${tech.short}</b><span>${ti + 1}ª tecnologia</span></td>
            <td>${v.label}</td>
            <td style="font-size:.76rem;color:var(--ink-faint);font-family:var(--mono);">${v.params_text}</td>
            <td class="num">${fmt(m.changed_pct, 3)}%</td>
            <td class="num"><span class="ok-check">${m.vector_max_diff_px} px</span></td>
            <td class="num"><span class="ok-check">${m.obs_max_diff_px} px</span></td>
            <td class="num">${m.shift_px === null || m.shift_px === undefined ? "—" : fmt(m.shift_px, 1) + " px"}</td>
            <td class="num">${fmt(m.psnr_db, 1)}</td>
            <td class="num">${fmt(m.ssim, 4)}</td>
            <td><span class="ok-check">✓ Aprovada</span></td>
          </tr>`);
      });
      list.appendChild(block);
    });

    $("tbl-body").innerHTML = tableRows.join("");

    /* ---------- modal ---------- */
    const modal = $("modal");
    $("modal-close").addEventListener("click", () => modal.classList.remove("on"));
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.remove("on"); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.classList.remove("on"); });

    function openModal(tech, v) {
      $("modal-title").textContent = `${tech.name} — intensidade ${v.label}`;
      $("modal-img").src = fileURL(v.files.cmp);
      const m = v.metrics;
      $("modal-metrics").innerHTML = `
        <span class="badge ${chgBadge(m.changed_pct)}">mudança nas zonas: ${fmt(m.changed_pct, 3)}%</span>
        <span class="badge">ΔE máx. nas zonas: ${fmt(m.delta_e_max_zone, 1)}</span>
        <span class="badge ok">erro vetorial: ${m.vector_max_diff_px} px</span>
        <span class="badge ok">erro observacional: ${m.obs_max_diff_px} px</span>
        <span class="badge">deslocamento: ${m.shift_px === null || m.shift_px === undefined ? "—" : fmt(m.shift_px, 1) + " px"}</span>
        <span class="badge">PSNR: ${fmt(m.psnr_db, 1)} dB</span>
        <span class="badge">SSIM: ${fmt(m.ssim, 4)}</span>`;
      $("modal-actions").innerHTML = `
        <a class="btn btn-ghost" href="${fileURL(v.files.tif)}" download="RefinaPaleo_${tech.key}_${v.key}.tif">⬇ Baixar TIFF</a>
        <a class="btn btn-primary" href="${fileURL(v.files.png)}" download="RefinaPaleo_${tech.key}_${v.key}.png">⬇ Baixar PNG</a>
        <a class="btn btn-ghost" href="${fileURL(v.files.cmp)}" download>⬇ Folha de conferência</a>`;
      modal.classList.add("on");
    }
  }
})();
