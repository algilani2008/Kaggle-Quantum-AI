const R = window.DEMO_REPORT;
const f4 = (v) => Number(v).toFixed(4);
const TASK_AR = { binary: "تصنيف ثنائي", multiclass: "تصنيف متعدد الفئات", regression: "انحدار" };

function renderDemo() {
  const date = new Date(R.run_at).toLocaleDateString("ar-SA-u-nu-latn-ca-gregory", { day: "numeric", month: "long", year: "numeric" });
  document.getElementById("demoSub").innerHTML = `نتيجة تشغيل فعلي للمحرك بتاريخ ${date} على بيانات صناعية (${R.features.input_shape[0].toLocaleString("en-US")} صف تدريب) — الأرقام من <code>report.json</code> كما خرجت.`;
  document.getElementById("metricName").textContent = R.metric;
  const fe = R.features;
  document.getElementById("demoKpis").innerHTML = [
    ["المهمة", TASK_AR[R.task] || R.task, `الهدف: ${R.target}`],
    ["الميزات", `${fe.input_shape[1]} → ${fe.final_shape[1]}`, `${fe.cross_features.length} ميزة متقاطعة`],
    [`أفضل تجميع (${R.metric})`, f4(R.final_cv_score), `الجولة ${R.selected_round}`],
    ["مدة التشغيل", `${Math.round(R.runtime_sec)}s`, "5 طيّات × جولتين"],
  ].map(([l, v, d]) => `<div class="card kpi"><p class="l">${l}</p><p class="v" style="font-size: 22px">${PF.esc(v)}</p><p class="d">${PF.esc(d)}</p></div>`).join("");

  document.getElementById("featFlow").innerHTML = `<span class="n">${fe.input_shape[1]}</span> عمود ← ترميز ${fe.categorical_encoded.length} فئوية + ${fe.row_stats.length} إحصاءات صف + ${fe.cross_features.length} متقاطعة − ${fe.dropped_constant.length} ثابت ← <span class="n">${fe.final_shape[1]}</span> ميزة`;
  document.getElementById("featTags").innerHTML = `<p class="note">أكثر الأعمدة معلوماتية (تُبنى منها التقاطعات):</p>` + fe.top_numeric_by_mutual_info.map(c => `<span class="tag">${PF.esc(c)}</span>`).join("") +
    `<p class="note" style="margin-top: 6px">أمثلة تقاطعات:</p>` + fe.cross_features.slice(0, 6).map(c => `<span class="tag">${PF.esc(c)}</span>`).join("") + ` <span class="note">… +${fe.cross_features.length - 6}</span>`;

  const ea = R.error_analysis;
  document.getElementById("errShare").innerHTML = `أسوأ 10% من الصفوف مسؤولة عن <b>${Math.round(ea.worst10_error_share * 100)}%</b> من الخطأ. أكثر الميزات اختلافاً فيها: ${ea.top_shifted_features.slice(0, 3).map(([n, s]) => `<span class="tag">${PF.esc(n)}</span> (${s} SD)`).join(" ")}`;
  const actAr = (a) => a
    .replace(/^Errors concentrated.*?: increased regularisation/, "الأخطاء متركزة ← رفع التنظيم (regularisation)")
    .replace(/^Worst rows form a distinct region \(max shift ([\d.]+) SD\): upweighting hard examples x(\d+)/, "الصفوف الصعبة منطقة مميزة (انحراف $1 SD) ← وزن الأمثلة الصعبة ×$2")
    .replace(/^Lowered learning rate x([\d.]+) and raised trees x([\d.]+)/, "خفض معدل التعلم ×$1 ورفع عدد الأشجار ×$2");
  document.getElementById("errActions").innerHTML = ea.actions.map(a => `<li style="font-size: 13.5px">${PF.esc(actAr(a))}</li>`).join("");

  const beat = R.beat_baseline;
  document.getElementById("gate").innerHTML = `
    <div class="card"><p class="note">أفضل تجميع</p><p class="big">${f4(R.final_cv_score)}</p></div>
    <div class="vs">${beat ? "أعلى من" : "أقل من"}</div>
    <div class="card"><p class="note">خط الأساس</p><p class="big">${f4(R.baseline)}</p><p class="note">${R.baseline_source.includes("default") ? "LightGBM افتراضي (أول تشغيل)" : "أفضل تشغيل سابق"}</p></div>`;
  document.getElementById("gateNote").innerHTML = beat
    ? `✅ تفوّق على خط الأساس، فكُتب <code>submission.csv</code> وحُدّث خط الأساس في <code>state.json</code>.`
    : `🛑 <b>لم يتفوّق، فلم يُكتب أي تقديم.</b> على بيانات صناعية بسيطة كهذه، نموذج LightGBM واحد يكفي والتجميع لا يضيف شيئاً — والبوابة منعت تقديماً أسوأ تلقائياً. هذا بالضبط سبب وجودها: في التشغيل اليومي ما يُستبدل أفضل تقديم بنتيجة أضعف.`;

  document.getElementById("runLog").textContent = R.log.join("\n");
}

let scoreChart;
function renderScores() {
  const r1 = R.round1.base_scores, r2 = R.round2 ? R.round2.base_scores : null;
  const labels = ["XGBoost", "LightGBM", "CatBoost", "التجميع"];
  const v1 = [r1.xgboost, r1.lightgbm, r1.catboost, R.round1.stack_score];
  const v2 = r2 ? [r2.xgboost, r2.lightgbm, r2.catboost, R.round2.stack_score] : null;
  const all = v1.concat(v2 || [], [R.baseline]);
  const pad = (Math.max(...all) - Math.min(...all)) * 0.6 || 0.001;
  if (scoreChart) scoreChart.destroy();
  scoreChart = PF.chart(document.getElementById("scoreChart"), {
    data: {
      labels,
      datasets: [
        { type: "bar", label: "الجولة 1", data: v1, backgroundColor: PF.css("--accent"), borderRadius: 4 },
        ...(v2 ? [{ type: "bar", label: "الجولة 2 (بعد تحليل الأخطاء)", data: v2, backgroundColor: PF.css("--c1"), borderRadius: 4 }] : []),
        { type: "line", label: "خط الأساس", data: labels.map(() => R.baseline), borderColor: PF.css("--c2"), borderDash: [6, 4], pointRadius: 0, borderWidth: 2 },
      ],
    },
    options: {
      plugins: { tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${f4(c.parsed.y)}` } } },
      scales: { y: { min: Math.min(...all) - pad, max: Math.max(...all) + pad, ticks: { callback: v => Number(v).toFixed(4) }, grid: { color: PF.css("--line") } }, x: { grid: { display: false } } },
    },
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderDemo();
  renderScores();
  PF.onTheme.push(renderScores);
});
