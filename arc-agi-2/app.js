const D = window.ARC_DATA;
const T = D.training, E = D.evaluation, G = D.gemini;
const REPO = "https://github.com/algilani2008/arc-agi-2";

function renderKpis() {
  document.getElementById("libCount").textContent = T.programs;
  const k = [
    ["مقسم التدريب (pass@2)", PF.pct(T.accuracy), `${PF.fmt(T.solved)} من ${PF.fmt(T.tasks)} مهمة`],
    ["مقسم التقييم (pass@2)", PF.pct(E.accuracy), `${PF.fmt(E.solved)} من ${E.tasks} مهمة — الأقرب للاختبار المخفي`],
    ["برامج Gemini صحيحة", `${PF.fmt(G.testCorrect)}/${G.attempted}`, `${PF.pct(G.testCorrect / G.attempted * 100)} من مهام التقييم المجرّبة`],
    ["برامج DSL", PF.fmt(T.programs), `يجري على ${PF.fmt(T.tasks)} مهمة في ${Math.round(T.runtime)} ثانية`],
  ];
  document.getElementById("kpis").innerHTML = k.map(([l, v, d], i) =>
    `<div class="card kpi"><p class="l">${l}</p><p class="v ${i === 1 ? "bad" : i === 2 ? "good" : ""}">${v}</p><p class="d">${d}</p></div>`).join("");
}

let famChart, splitChart;
function familyGroup(name) {
  const base = name.split(/[:>+(]/)[0];
  return { geom: "Geometry", crop_nonbg: "Crop / Scale", scale_up: "Crop / Scale", scale_down: "Crop / Scale", pattern: "Pattern", grid: "Grid", shape: "Shape" }[base] || "Other";
}
function renderResults() {
  const fam = T.byFamily;
  const colorOf = { Geometry: "--c1", "Crop / Scale": "--c4", Pattern: "--c3", Grid: "--c5", Shape: "--c2", Other: "--ink-3" };
  document.getElementById("famHint").textContent = `${T.solved} مهمة محلولة موزعة على ${fam.length} برنامجاً. اللون حسب العائلة.`;
  if (famChart) famChart.destroy();
  famChart = PF.chart(document.getElementById("famChart"), {
    type: "bar",
    data: { labels: fam.map(f => f[0]), datasets: [{ label: "مهام محلولة", data: fam.map(f => f[1]), backgroundColor: fam.map(f => PF.css(colorOf[familyGroup(f[0])])), borderRadius: 3, maxBarThickness: 16 }] },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false }, tooltip: { callbacks: { title: c => c[0].label, label: c => ` ${c.parsed.x} مهمة — ${familyGroup(c.label)}` } } },
      scales: { x: { grid: { color: PF.css("--line") }, ticks: { precision: 0 } }, y: { position: "right", grid: { display: false }, ticks: { autoSkip: false, font: { family: PF.css("--mono").replace(/"/g, ""), size: 10.5 } } } },
    },
  });

  if (splitChart) splitChart.destroy();
  splitChart = PF.chart(document.getElementById("splitChart"), {
    type: "bar",
    data: {
      labels: [`التدريب (${PF.fmt(T.tasks)})`, `التقييم (${E.tasks})`],
      datasets: [
        { label: "برنامج طابق أمثلة التدريب %", data: [T.fit / T.tasks * 100, E.fit / E.tasks * 100], backgroundColor: PF.css("--c1"), borderRadius: 4 },
        { label: "حل صحيح pass@2 %", data: [T.accuracy, E.accuracy], backgroundColor: PF.css("--accent"), borderRadius: 4 },
      ],
    },
    options: { scales: { y: { beginAtZero: true, suggestedMax: 10, ticks: { callback: v => v + "%" }, grid: { color: PF.css("--line") } }, x: { grid: { display: false } } },
      plugins: { tooltip: { callbacks: { label: c => ` ${c.dataset.label.replace(" %", "")}: ${c.parsed.y.toFixed(1)}%` } } } },
  });

  document.getElementById("fitsList").innerHTML = T.fits.slice(0, 8).map(([n, c]) => `<span class="fam-badge">${PF.esc(n)}</span> <span class="mono">× ${c}</span>`).join("<br>");
}

let gemFilter = "";
function renderGemini() {
  document.getElementById("gemSub").textContent = `النموذج ${G.model} على ${G.attempted} مهمة من مقسم التقييم — نفس المقسم اللي يحل منه الـ DSL صفراً.`;
  const failed = G.tasks.filter(t => t.status !== "verified").length;
  document.getElementById("gemKpis").innerHTML = [
    ["متحقق على أمثلة التدريب", `${G.verified}/${G.attempted}`, `فشل ${failed} مهام`],
    ["صحيح على مثال الاختبار", `${PF.fmt(G.testCorrect)}/${G.attempted}`, PF.pct(G.testCorrect / G.attempted * 100)],
    ["التوكنز (مدخل / مخرج)", `${G.tokensIn.replace(/,\d{3}$/, "K")} / ${G.tokensOut.replace(/,\d{3}$/, "K")}`, "كل التشغيلات"],
  ].map(([l, v, d]) => `<div class="card kpi"><p class="l">${l}</p><p class="v" style="font-size: 22px">${v}</p><p class="d">${d}</p></div>`).join("");

  const rows = G.tasks.filter(t => !gemFilter || t.status === gemFilter);
  document.querySelector("#gemTable tbody").innerHTML = rows.map(t => `
    <tr>
      <td class="mono">${t.files.length
        ? t.files.map(f => `<a href="${REPO}/blob/main/gemini/results/programs/${encodeURIComponent(f)}.py" target="_blank" rel="noopener">${PF.esc(f)}</a>`).join("<br>")
        : PF.esc(t.id)}</td>
      <td class="${t.status === "verified" ? "status-ok" : "status-bad"}">${t.status === "verified" ? "✓ متحقق" : "✗ فشل"}</td>
      <td class="n">${t.test ? "✓" : "—"}</td>
      <td class="n">${t.rounds}</td>
      <td class="n">${t.lines || "—"}</td>
    </tr>`).join("");
}
document.querySelectorAll("#gemFilter button").forEach(b => b.addEventListener("click", () => {
  gemFilter = b.dataset.f;
  document.querySelectorAll("#gemFilter button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  renderGemini();
}));

document.addEventListener("DOMContentLoaded", () => {
  renderKpis();
  renderResults();
  renderGemini();
  PF.onTheme.push(renderResults);
});
