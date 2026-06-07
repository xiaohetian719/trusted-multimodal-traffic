// Traffic AI Platform - Frontend Application

const API = "/api";
let currentTask = null;
let pollTimer = null;

// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================
//  Upload
// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================

const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const fileHint = document.getElementById("fileHint");
const runOptions = document.getElementById("runOptions");
const btnRun = document.getElementById("btnRun");

uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("click", (e) => e.stopPropagation());

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag-over");
});
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  fileHint.textContent = `已选择: ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;

  const formData = new FormData();
  formData.append("video", file);

  try {
    fileHint.textContent += " - 上传中...";
    const resp = await fetch(`${API}/upload`, { method: "POST", body: formData });
    const data = await resp.json();
    if (data.error) {
      fileHint.textContent = `上传失败: ${data.error}`;
      return;
    }
    currentTask = data;
    fileHint.textContent = `✅ 上传成功! ID: ${data.task_id}`;
    runOptions.style.display = "flex";
    btnRun.disabled = false;
    updateStatus("ready", "视频已就绪");
  } catch (err) {
    fileHint.textContent = `上传失败: ${err.message}`;
  }
}

// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================
//  Pipeline
// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================

async function startPipeline() {
  if (!currentTask) return;
  btnRun.disabled = true;
  btnRun.textContent = "启动中...";

  const maxSecs = document.getElementById("maxSecs").value || null;

  try {
    const resp = await fetch(`${API}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: currentTask.task_id, max_secs: maxSecs ? parseFloat(maxSecs) : null }),
    });
    const data = await resp.json();
    if (data.error) {
      alert(data.error);
      btnRun.disabled = false;
      btnRun.textContent = "▶ 开始分析";
      return;
    }
    // Show progress
    document.getElementById("uploadPanel").style.display = "none";
    document.getElementById("progressPanel").style.display = "block";
    document.getElementById("resultsPanel").style.display = "none";
    updateStatus("running", "分析中...");
    pollProgress();
  } catch (err) {
    alert(`启动失败: ${err.message}`);
    btnRun.disabled = false;
    btnRun.textContent = "▶ 开始分析";
  }
}

async function pollProgress() {
  if (!currentTask) return;
  try {
    const resp = await fetch(`${API}/status/${currentTask.task_id}`);
    const task = await resp.json();
    currentTask = task;

    document.getElementById("progressBar").style.width = task.progress + "%";
    document.getElementById("progressPct").textContent = task.progress + "%";
    document.getElementById("progressMsg").textContent = task.message;
    document.getElementById("progressElapsed").textContent = `已用: ${task.elapsed}s`;

    if (task.status === "done") {
      updateStatus("ready", "分析完成");
      document.getElementById("progressBar").style.width = "100%";
      document.getElementById("progressPct").textContent = "100%";
      setTimeout(() => loadResults(), 500);
      return;
    }
    if (task.status === "error") {
      updateStatus("error", "出错");
      document.getElementById("progressMsg").textContent = "错误: " + (task.error || "未知错误");
      document.getElementById("progressBar").style.background = "#f87171";
      return;
    }
  } catch (err) {
    console.error("Poll error:", err);
  }
  pollTimer = setTimeout(pollProgress, 2000);
}

function updateStatus(state, text) {
  const el = document.getElementById("headerStatus");
  el.className = "header-status " + (state === "running" ? "running" : "");
  el.innerHTML = `<span class="dot"></span> ${text}`;
}

// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================
//  Results
// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================

async function loadResults() {
  if (!currentTask) return;
  document.getElementById("progressPanel").style.display = "none";

  try {
    const resp = await fetch(`${API}/result/${currentTask.task_id}`);
    const data = await resp.json();
    displayResults(data);
  } catch (err) {
    alert(`加载结果失败: ${err.message}`);
  }
}

function displayResults(data) {
  const panel = document.getElementById("resultsPanel");
  panel.style.display = "block";

  // Stats
  document.getElementById("resultStats").innerHTML = `
    <div class="stat-card"><div class="val">${data.frames_processed || 0}</div><div class="lbl">处理帧数</div></div>
    <div class="stat-card"><div class="val">${data.vehicles_detected || 0}</div><div class="lbl">检测车辆</div></div>
    <div class="stat-card"><div class="val" style="color:${data.collision_risks > 0 ? '#f87171' : '#4ade80'}">${data.collision_risks || 0}</div><div class="lbl">碰撞风险</div></div>
    <div class="stat-card"><div class="val">${data.elapsed || 0}s</div><div class="lbl">总用时</div></div>
  `;

  // Video
  const videoSrc = `/output/${data.annotated_video}`;
  document.getElementById("resultVideo").src = videoSrc;
  document.getElementById("downloadVideo").href = videoSrc;

  // Report
  const reportEl = document.getElementById("reportContent");
  if (data.report_text) {
    reportEl.innerHTML = formatMarkdown(data.report_text);
  } else {
    reportEl.textContent = "报告未生成（Ollama 可能未运行，但事实数据已包含在下方）";
  }
  document.getElementById("downloadReport").href = `/output/${data.report_file}`;

  // Eval cards
  const evalMetrics = data.eval_metrics || {};
  const summary = evalMetrics.summary || {};
  const stats = summary.metrics_statistics || {};
  let evalHtml = "";
  if (summary.average_score) {
    const s = summary.average_score;
    const cls = s >= 85 ? "score-high" : s >= 60 ? "score-mid" : "score-low";
    evalHtml += `<div class="eval-card ${cls}"><div class="val">${s.toFixed(1)}</div><div class="lbl">综合评分</div></div>`;
  }
  for (const [name, info] of Object.entries(stats)) {
    const v = info.mean || 0;
    const cls = v >= 80 ? "score-high" : v >= 50 ? "score-mid" : "score-low";
    evalHtml += `<div class="eval-card ${cls}"><div class="val">${v.toFixed(1)}</div><div class="lbl">${name}</div></div>`;
  }
  document.getElementById("evalCards").innerHTML = evalHtml;

  // Eval HTML report
  if (data.eval_dir) {
    document.getElementById("evalHtmlFrame").src = `/output/${data.eval_dir}/evaluation_report.html`;
  }

  // Charts
  const charts = data.eval_charts || [];
  const chartsGrid = document.getElementById("chartsGrid");
  chartsGrid.innerHTML = charts.map(c =>
    `<div class="chart-card"><img src="${c}" alt="chart" loading="lazy"></div>`
  ).join("");

  // Scroll to results
  panel.scrollIntoView({ behavior: "smooth" });
}

// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================
//  Tabs
// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
  });
});

// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================
//  Simple Markdown formatter
// ============================================================
//  Video error handler
// ============================================================

function onVideoError() {
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoFallback').style.display = 'block';
}

// ============================================================

function formatMarkdown(text) {
  return text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^- (.+)$/gm, "• $1<br>")
    .replace(/\n/g, "<br>");
}
