/* Shared helpers: theme toggle, number formatting, Chart.js defaults that follow the CSS theme */
(function () {
  const root = document.documentElement;
  try { const t = localStorage.getItem("pf_theme"); if (t) root.dataset.theme = t; } catch (e) {}

  window.PF = {
    css(name) { return getComputedStyle(root).getPropertyValue(name).trim(); },
    fmt(n, d = 0) { return Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }); },
    pct(n, d = 1, sign = false) { const s = (sign && n > 0 ? "+" : "") + Number(n).toFixed(d) + "%"; return s; },
    esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); },
    charts: [],
    palette() { return ["--c1", "--c2", "--c3", "--c4", "--c5"].map(v => PF.css(v)); },
    applyChartDefaults() {
      if (!window.Chart) return;
      Chart.defaults.font.family = PF.css("--body").replace(/"/g, "") || "system-ui";
      Chart.defaults.font.size = 12;
      Chart.defaults.locale = "en-US";
      Chart.defaults.color = PF.css("--ink-2");
      Chart.defaults.borderColor = PF.css("--line");
      Chart.defaults.plugins.legend.labels.boxWidth = 12;
      Chart.defaults.plugins.tooltip.rtl = true;
      Chart.defaults.plugins.tooltip.textDirection = "rtl";
      Chart.defaults.maintainAspectRatio = false;
      Chart.defaults.animation.duration = 400;
    },
    chart(el, config) {
      const c = new Chart(el, config);
      PF.charts.push({ c, config });
      return c;
    },
    onTheme: [],
    toggleTheme() {
      const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("pf_theme", root.dataset.theme); } catch (e) {}
      PF.applyChartDefaults();
      PF.onTheme.forEach(fn => fn());
    },
  };

  document.addEventListener("DOMContentLoaded", () => {
    PF.applyChartDefaults();
    document.querySelectorAll("[data-theme-toggle]").forEach(b => b.addEventListener("click", PF.toggleTheme));
  });
})();
