// ---- state ----
const files = { expr: null, meta: null, script: null };
let metaColumns = [];

const $ = (s) => document.querySelector(s);
const el = (id) => document.getElementById(id);

// ---- drop zones ----
function wireDrop(zoneId, key, onFile) {
  const zone = el(zoneId);
  const input = zone.querySelector("input");
  const setFile = (f) => {
    if (!f) return;
    files[key] = f;
    zone.classList.add("filled");
    zone.querySelector(".drop-file").textContent = f.name;
    if (onFile) onFile(f);
    refreshRunBtn();
  };
  zone.addEventListener("click", (e) => { if (e.target !== input) input.click(); });
  input.addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragover", "dragenter"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("dragover"); }));
  zone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
}

wireDrop("drop-expr", "expr");
wireDrop("drop-meta", "meta", readMetaHeader);
wireDrop("drop-script", "script");

function refreshRunBtn() {
  el("btn-run").disabled = !files.expr;
}

// ---- parse metadata header to populate mapping dropdowns ----
function readMetaHeader(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const firstLine = String(reader.result).split(/\r?\n/)[0];
    metaColumns = firstLine.split(",").map((c) => c.trim().replace(/^"|"$/g, ""));
    fillMapping(metaColumns);
  };
  reader.readAsText(file.slice(0, 4096));
}

function guess(cols, names) {
  return cols.find((c) => names.includes(c.toLowerCase())) || "";
}

function fillMapping(cols) {
  el("mapping").hidden = false;
  const opts = (withNone) =>
    (withNone ? '<option value="">— none —</option>' : "") +
    cols.map((c) => `<option value="${c}">${c}</option>`).join("");
  el("sel-label").innerHTML = opts(false);
  el("sel-group").innerHTML = opts(true);
  el("sel-batch").innerHTML = opts(true);
  el("sel-label").value = guess(cols, ["diagnosis", "label", "outcome", "class", "target", "y"]) || cols[0];
  el("sel-group").value = guess(cols, ["patient_id", "subject", "subject_id", "donor", "patient", "pid"]);
  el("sel-batch").value = guess(cols, ["plate", "batch", "site", "run", "lane"]);
}

// ---- views ----
function show(view) {
  ["input-view", "loading-view", "result-view"].forEach((v) => (el(v).hidden = v !== view));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

let pollTimer = null;
function startLoader() {
  show("loading-view");
  el("loader-caption").textContent = "Starting…";
  el("loader-bar-fill").style.width = "3%";
  el("loader-steps").textContent = "";
}
function setProgress(stage, pct) {
  if (stage) el("loader-caption").textContent = stage;
  if (pct != null) {
    el("loader-bar-fill").style.width = Math.max(3, pct) + "%";
    el("loader-steps").textContent = Math.round(pct) + "%";
  }
}
function stopLoader() { clearInterval(pollTimer); pollTimer = null; }

// ---- run ----
el("btn-run").addEventListener("click", runAnalyze);
el("btn-sample").addEventListener("click", () => runSample(0));
el("btn-sample-clean").addEventListener("click", () => runSample(1));
el("btn-back").addEventListener("click", () => show("input-view"));

async function runAnalyze() {
  el("err").hidden = true;
  const fd = new FormData();
  fd.append("expression", files.expr);
  if (files.meta) fd.append("metadata", files.meta);
  if (files.script) fd.append("script", files.script);
  fd.append("label_col", el("sel-label").value || "");
  fd.append("group_col", el("sel-group").value || "");
  fd.append("batch_col", el("sel-batch").value || "");
  fd.append("model", el("sel-model").value);
  fd.append("nested", el("chk-nested").checked ? "true" : "false");
  await post("/api/analyze", fd);
}

async function runSample(clean) {
  el("err").hidden = true;
  await post("/api/sample?clean=" + clean, new FormData());
}

async function post(url, fd) {
  startLoader();
  try {
    const r = await fetch(url, { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok || data.error || !data.job_id) {
      stopLoader();
      fail(data.error || "Something went wrong.");
      return;
    }
    pollProgress(data.job_id);
  } catch (e) {
    stopLoader();
    fail("Network error: " + e.message);
  }
}

function pollProgress(jid) {
  stopLoader();
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch("/api/progress/" + jid);
      const p = await r.json();
      if (!r.ok || p.error) { stopLoader(); fail(p.error || "Audit failed."); return; }
      setProgress(p.stage, p.pct);
      if (p.status === "done") { stopLoader(); render(p.result); }
    } catch (e) {
      stopLoader();
      fail("Lost connection to the server: " + e.message);
    }
  }, 500);
}

function fail(msg) {
  show("input-view");
  const e = el("err");
  e.textContent = msg;
  e.hidden = false;
}

// ---- render ----
function fmt(x, d = 2) { return (x === null || x === undefined || isNaN(x)) ? "—" : Number(x).toFixed(d); }
const isReg = (data) => data.task === "regression";

function render(data) {
  show("result-view");
  renderMeta(data);
  renderVerdict(data);
  renderLadder(data);
  renderModel(data);
  renderCodeAudit(data);
  renderDataAudit(data);
  renderReport(data);
  document.querySelectorAll("#result-view .card").forEach((c, i) => {
    c.classList.remove("reveal"); void c.offsetWidth;
    c.style.animationDelay = i * 60 + "ms"; c.classList.add("reveal");
  });
}

function renderMeta(data) {
  const inp = data.input || {};
  const bits = [`${inp.n}×${inp.p}`, `task: ${data.task}`];
  if (inp.group_col) bits.push(`group: ${inp.group_col}`);
  if (inp.batch_col) bits.push(`batch: ${inp.batch_col}`);
  if (inp.sample) bits.push(`sample: ${inp.sample}`);
  el("result-meta").textContent = bits.join("  ·  ");
}

function headline(data) {
  // returns {naive, honest, name, unit}
  const naive = data.cv_ladder.naive[0];
  if (isReg(data)) {
    const h = data.model.metrics ? data.model.metrics.headline : data.model.auc;
    return { naive, honest: h, name: data.model.name || "model", unit: "R²" };
  }
  const honest = data.model.auc !== undefined ? data.model.auc
                 : (data.model.metrics ? data.model.metrics.headline : naive);
  return { naive, honest, name: data.model.name, unit: "AUC" };
}

function renderVerdict(data) {
  const h = headline(data);
  const infl = (data.inflation !== undefined) ? data.inflation : (h.naive - h.honest);
  const big = Math.abs(infl) >= 0.05;
  const good = !big;
  el("verdict").innerHTML = `
    <div class="aucs">
      <div class="auc-block">
        <div class="auc-num leak">${fmt(h.naive)}</div>
        <div class="auc-label">Naive ${h.unit}</div>
      </div>
      <div class="arrow">→</div>
      <div class="auc-block">
        <div class="auc-num trust">${fmt(h.honest)}</div>
        <div class="auc-label">Honest ${h.unit}</div>
      </div>
    </div>
    <div class="badge ${good ? "good" : "bad"}">
      ${infl >= 0 ? "+" : ""}${fmt(infl)} ${good ? "— no meaningful inflation" : "inflation — artifact, not biology"}
    </div>
    <div class="sub">${good
      ? "The reported score holds up under a leakage-safe, group-aware re-analysis."
      : `Leakage inflated the reported ${h.unit} by ${fmt(Math.abs(infl))}. The honest number is what the data supports.`}
      Model: <strong>${h.name}</strong>.</div>`;
}

const LADDER_NAMES = {
  naive: "Naive (preprocessing + selection on full data)",
  honest_randomsplit: "Honest preprocessing (fit inside folds)",
  groupkfold: "Subject-safe (GroupKFold on subject)",
  groupkfold_batchcorrected: "Fully honest (+ in-fold batch centering)",
};
function renderLadder(data) {
  const rungs = Object.entries(data.cv_ladder);
  const isRegT = isReg(data);
  const max = isRegT ? Math.max(1, ...rungs.map(([, v]) => v[0])) : 1;
  el("ladder").innerHTML = rungs.map(([k, v], i) => {
    const [m, s] = v;
    const pct = Math.max(0, Math.min(100, (m / max) * 100));
    const color = i === 0 ? "var(--leak)" :
      i === rungs.length - 1 ? "var(--trust)" : "var(--amber)";
    return `<div class="rung">
      <div class="rung-top">
        <span class="rung-name">${LADDER_NAMES[k] || k}</span>
        <span class="rung-val">${fmt(m)} ± ${fmt(s)}</span>
      </div>
      <div class="rung-bar"><div class="rung-fill" data-w="${pct}"
        style="background:${color}"></div></div>
    </div>`;
  }).join("");
  requestAnimationFrame(() =>
    document.querySelectorAll(".rung-fill").forEach((f) => (f.style.width = f.dataset.w + "%")));
}

function renderModel(data) {
  const m = data.model;
  const metrics = m.metrics || {};
  let cards;
  if (isReg(data)) {
    cards = `
      <div class="metric"><div class="m-val">${fmt(metrics.r2)}</div><div class="m-lab">R²</div></div>
      <div class="metric"><div class="m-val">${fmt(metrics.rmse)}</div><div class="m-lab">RMSE</div></div>
      <div class="metric"><div class="m-val">${fmt(metrics.mae)}</div><div class="m-lab">MAE</div></div>`;
  } else {
    const auc = m.auc !== undefined ? m.auc : metrics.auc_ovr_macro;
    const brier = m.brier !== undefined ? m.brier : metrics.brier;
    cards = `
      <div class="metric"><div class="m-val">${fmt(auc)}</div><div class="m-lab">Honest AUC</div></div>
      <div class="metric"><div class="m-val">${fmt(brier)}</div><div class="m-lab">Brier score</div>
        <div class="m-note">0.25 = uninformative · lower is better</div></div>`;
  }

  let cand = "";
  const cands = m.candidates || {};
  const keys = Object.keys(cands);
  if (keys.length > 1) {
    cand = `<div class="cand">` + keys.map((name) => {
      const c = cands[name];
      const win = name === m.name;
      const pct = Math.max(2, Math.min(100, (c.auc || 0) * 100));
      return `<div class="cand-row">
        <span class="cand-name ${win ? "win" : ""}">${name}${win ? " ✓" : ""}</span>
        <span class="cand-track"><span class="cand-fill" style="width:${pct}%;
          background:${win ? "var(--trust)" : "rgba(255,255,255,0.25)"}"></span></span>
        <span class="cand-val">${fmt(c.auc)}</span>
      </div>`;
    }).join("") + `</div>`;
  }

  let note = "";
  if (data.model_selection) {
    const ms = data.model_selection;
    note += `<div class="model-note">Even choosing the better model is checked: nested selection
      AUC <strong>${fmt(ms.nested_selection_auc)}</strong> (selection optimism ${fmt(ms.selection_optimism)}).</div>`;
  }
  if (data.tuning) {
    const t = data.tuning;
    const fl = t.flat_gridsearch_auc !== undefined ? t.flat_gridsearch_auc : t.flat_gridsearch;
    const ne = t.nested_cv_auc !== undefined ? t.nested_cv_auc : t.nested_cv;
    note += `<div class="model-note">Hyperparameter honesty: flat GridSearch ${fmt(fl)} →
      nested CV <strong>${fmt(ne)}</strong> (tuning optimism ${fmt(t.tuning_optimism)}).</div>`;
  }

  el("model-body").innerHTML = `<div class="metric-row">${cards}</div>${cand}${note}`;
}

function renderCodeAudit(data) {
  const box = el("code-audit"), tag = el("code-tag");
  if (!data.code_audit) {
    box.innerHTML = `<div class="finding-msg">No analysis script uploaded — upload a
      .py/.ipynb to statically audit the code for leakage.</div>`;
    tag.className = "tag"; tag.textContent = "n/a";
    return;
  }
  const ca = data.code_audit;
  const crit = (ca.findings || []).filter((f) => f.severity === "critical");
  if (crit.length === 0) {
    box.innerHTML = `<div class="audit-empty">✓ No leakage patterns found in the code.</div>`;
    tag.className = "tag good"; tag.textContent = "clean";
  } else {
    tag.className = "tag bad"; tag.textContent = crit.length + " leak" + (crit.length > 1 ? "s" : "");
    box.innerHTML = crit.map((f) => `
      <div class="finding">
        <span class="sev-dot critical"></span>
        <div class="finding-body">
          <div class="finding-title">${labelFor(f.check)} ${f.line ? `<span class="dim">· line ${f.line}</span>` : ""}</div>
          <div class="finding-msg">${f.message}</div>
          ${f.code ? `<div class="finding-code">${escapeHtml(f.code)}</div>` : ""}
        </div>
      </div>`).join("");
  }
}

function renderDataAudit(data) {
  const box = el("data-audit"), tag = el("data-tag");
  const da = data.data_audit;
  const shown = (da.findings || []).filter((f) => f.severity === "critical" || f.severity === "warning");
  tag.className = "tag " + (da.n_critical ? "bad" : "good");
  tag.textContent = da.n_critical ? `${da.n_critical} critical` : "clean";
  if (shown.length === 0) {
    box.innerHTML = `<div class="audit-empty">✓ No critical or warning findings.</div>`;
    return;
  }
  box.innerHTML = shown.map((f) => `
    <div class="finding">
      <span class="sev-dot ${f.severity}"></span>
      <div class="finding-body">
        <div class="finding-title">${labelFor(f.check)}
          <span class="dim">· ${f.severity}</span></div>
        <div class="finding-msg">${f.message}</div>
      </div>
    </div>`).join("");
}

const CHECK_LABELS = {
  leaky_preprocessing: "Preprocessing before split",
  leaky_feature_selection: "Feature selection before split",
  leaky_resampling: "Resampling before split",
  subject_leakage: "Subject / repeated-measures leakage",
  batch_confound: "Batch confounded with outcome",
  dimensionality: "High-dimensional small-n regime",
  class_balance: "Class balance",
  duplicate_rows: "Duplicate rows",
  no_split: "No train/test split",
};
function labelFor(c) { return CHECK_LABELS[c] || c; }

function renderReport(data) {
  el("report-md").innerHTML = mdToHtml(data.report_md || "");
  el("btn-download").onclick = () => {
    const blob = new Blob([data.report_md || ""], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "trust_report.md";
    a.click();
  };
}

// ---- tiny markdown renderer (headings, tables, bold, code, hr) ----
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function inline(s) {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}
function mdToHtml(md) {
  const lines = md.split("\n");
  let html = "", i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\|(.+)\|\s*$/.test(line) && i + 1 < lines.length && /^\|[-:\s|]+\|\s*$/.test(lines[i + 1])) {
      const rows = [];
      while (i < lines.length && /^\|(.+)\|\s*$/.test(lines[i])) { rows.push(lines[i]); i++; }
      const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(rows[0]);
      const body = rows.slice(2).map(cells);
      html += "<table><thead><tr>" + head.map((h) => `<th>${inline(h)}</th>`).join("") +
        "</tr></thead><tbody>" +
        body.map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>";
      continue;
    }
    if (/^#{1,6}\s/.test(line)) {
      const lvl = line.match(/^#+/)[0].length;
      html += `<h${lvl}>${inline(line.replace(/^#+\s/, ""))}</h${lvl}>`;
    } else if (/^---\s*$/.test(line)) {
      html += "<hr/>";
    } else if (/^\s*-\s+/.test(line)) {
      html += `<div>• ${inline(line.replace(/^\s*-\s+/, ""))}</div>`;
    } else if (line.trim() === "") {
      html += "";
    } else {
      html += `<p>${inline(line)}</p>`;
    }
    i++;
  }
  return html;
}
