/* ReGE project page: theme toggle, result charts, results table and copy buttons. */
(function () {
  "use strict";

  // ---------- results of the paper ----------

  var BENCHES = ["tb2", "spider", "aw", "bcp"];
  var BENCH_NAME = { tb2: "Terminal-Bench 2.0", spider: "Spider2-Lite", aw: "AndroidWorld", bcp: "BrowseComp-Plus", avg: "Average" };
  var METHODS = [
    { name: "Control", label: "Control", kind: "ctl" },
    { name: "Self-Consistency", label: "SC", kind: "base" },
    { name: "Best-of-N", label: "Best-of-N", kind: "base" },
    { name: "Reflexion", label: "Reflexion", kind: "base" },
    { name: "AggAgent", label: "AggAgent", kind: "base" },
    { name: "ReGE", label: "ReGE", kind: "ours" }
  ];

  // Table 2: per benchmark, [task success (%), cost relative to Control (x)] for each method in METHODS order;
  // null = unavailable or inapplicable.
  var MAIN = {
    gpt: { model: "gpt-5.6-luna", awInstances: 342, cells: {
      tb2: [[41.6, 1.00], null, [54.0, 4.17], [46.1, 2.71], [46.1, 5.83], [56.2, 5.25]],
      spider: [[59.3, 1.00], [64.4, 4.00], [63.7, 4.32], [64.4, 1.50], [55.6, 11.71], [66.7, 5.36]],
      aw: [[53.8, 1.00], null, [57.9, 4.02], [55.3, 2.30], [51.5, 4.92], [58.8, 5.06]],
      bcp: [[64.1, 1.00], [74.0, 4.00], [74.0, 4.01], [73.3, 2.09], [76.1, 4.74], [76.6, 4.52]],
      avg: [[54.7, 1.00], null, [62.4, 4.13], [59.8, 2.15], [57.3, 6.80], [64.6, 5.05]]
    }},
    claude: { model: "claude-sonnet-5", awInstances: 114, cells: {
      tb2: [[40.4, 1.00], null, [38.2, 4.06], [48.3, 2.62], [46.1, 6.67], [50.6, 5.26]],
      spider: [[57.8, 1.00], [60.0, 4.00], [61.5, 4.16], [59.3, 1.48], [60.7, 8.75], [63.7, 5.04]],
      aw: [[61.4, 1.00], null, [64.9, 4.01], [61.4, 2.07], [65.8, 5.09], [67.5, 5.02]],
      bcp: [[74.3, 1.00], [84.6, 4.00], [84.7, 4.02], [84.5, 2.14], [86.3, 4.72], [87.5, 4.39]],
      avg: [[58.5, 1.00], null, [62.3, 4.06], [63.4, 2.08], [64.7, 6.31], [67.3, 4.93]]
    }}
  };

  // Table 3: [method, success (%), construction, execution, total (thousand tokens per query), construction saved (%)]
  var COST = {
    gpt: [["AggAgent", 75.3, 232.9, 44.8, 277.7, null], ["ReGE-Syn", 75.2, 4.2, 229.8, 234.1, 98.2]],
    claude: [["AggAgent", 87.0, 211.7, 134.8, 346.6, null], ["ReGE-Syn", 86.2, 9.6, 214.2, 223.8, 95.5]]
  };

  var MINUS = "−";
  var SVGNS = "http://www.w3.org/2000/svg";
  var state = { line: "gpt" };

  function signed(v, digits) {
    return (v < 0 ? MINUS : "+") + Math.abs(v).toFixed(digits);
  }

  // ---------- theme ----------

  var root = document.documentElement;
  var darkQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function currentTheme() {
    return root.getAttribute("data-theme") || (darkQuery && darkQuery.matches ? "dark" : "light");
  }
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("rege-theme", next); } catch (e) { /* storage unavailable */ }
    });
  }

  var nav = document.querySelector(".nav");
  function onScroll() {
    if (nav) nav.classList.toggle("scrolled", window.scrollY > 8);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---------- svg helpers ----------

  function svgEl(tag, attrs, parent) {
    var node = document.createElementNS(SVGNS, tag);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function svgText(parent, x, y, str, cls, anchor) {
    var node = svgEl("text", { x: x, y: y, "class": cls }, parent);
    if (anchor) node.setAttribute("text-anchor", anchor);
    node.textContent = str;
    return node;
  }

  var measureCtx = document.createElement("canvas").getContext("2d");
  function textWidth(str, px, weight) {
    measureCtx.font = (weight || 400) + " " + px + "px Inter, ui-sans-serif, system-ui, sans-serif";
    return measureCtx.measureText(str).width;
  }

  // Horizontal bar from baseline x0 to data end x1, rounded only at the data end.
  function barPath(x0, x1, y, h, radius) {
    var len = Math.abs(x1 - x0);
    if (len < 0.5) return null;
    var r = Math.min(radius, len, h / 2);
    var dir = x1 > x0 ? 1 : -1;
    var xe = x1 - dir * r;
    return "M" + x0 + "," + y + "H" + xe + "Q" + x1 + "," + y + " " + x1 + "," + (y + r) +
      "V" + (y + h - r) + "Q" + x1 + "," + (y + h) + " " + xe + "," + (y + h) + "H" + x0 + "Z";
  }

  // ---------- tooltip ----------

  var tooltip = document.getElementById("tooltip");
  function fillTooltip(title, sub, pairs) {
    tooltip.textContent = "";
    var t = document.createElement("div");
    t.className = "tt-title";
    t.textContent = title;
    tooltip.appendChild(t);
    if (sub) {
      var s = document.createElement("div");
      s.className = "tt-sub";
      s.textContent = sub;
      tooltip.appendChild(s);
    }
    pairs.forEach(function (pair) {
      var row = document.createElement("div");
      row.className = "tt-row";
      var label = document.createElement("span");
      label.textContent = pair[0];
      var value = document.createElement("b");
      value.textContent = pair[1];
      row.appendChild(label);
      row.appendChild(value);
      tooltip.appendChild(row);
    });
  }
  function placeTooltip(x, y) {
    tooltip.hidden = false;
    var w = tooltip.offsetWidth, h = tooltip.offsetHeight;
    var left = x + 16, top = y + 16;
    if (left + w > window.innerWidth - 8) left = x - w - 16;
    if (top + h > window.innerHeight - 8) top = y - h - 16;
    tooltip.style.left = Math.max(8, left) + "px";
    tooltip.style.top = Math.max(8, top) + "px";
  }
  function hideTooltip() {
    tooltip.hidden = true;
  }
  function attachTooltip(node, content) {
    node.addEventListener("pointerenter", function (e) { content(); placeTooltip(e.clientX, e.clientY); });
    node.addEventListener("pointermove", function (e) { placeTooltip(e.clientX, e.clientY); });
    node.addEventListener("pointerleave", hideTooltip);
    node.addEventListener("focus", function () {
      content();
      var box = node.getBoundingClientRect();
      placeTooltip(box.left + box.width * 0.55, box.top + box.height * 0.5);
    });
    node.addEventListener("blur", hideTooltip);
  }

  // ---------- task success, one panel per benchmark ----------

  var ACC_DOMAIN = [0, 100];
  var ACC_TICKS = [0, 25, 50, 75, 100];

  function renderAcc(container, bench) {
    var line = MAIN[state.line];
    var cells = line.cells[bench];
    var W = Math.max(260, Math.floor(container.clientWidth));
    var labelW = Math.ceil(Math.max.apply(null, METHODS.map(function (m) { return textWidth(m.label, 12.5, 650); }))) + 14;
    var valueW = 40, rowH = 27, barH = 14, axisH = 20, top = 2;
    var x0 = labelW, x1 = W - valueW - 6;
    var scale = function (v) { return x0 + (v - ACC_DOMAIN[0]) / (ACC_DOMAIN[1] - ACC_DOMAIN[0]) * (x1 - x0); };
    var rowsBottom = top + METHODS.length * rowH;
    var H = rowsBottom + axisH;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "group",
      "aria-label": "Task success on " + BENCH_NAME[bench] + " with " + line.model });

    var compact = x1 - x0 < 200;
    (compact ? [0, 50, 100] : ACC_TICKS).forEach(function (t) {
      var x = Math.round(scale(t)) + 0.5;
      svgEl("line", { x1: x, x2: x, y1: top, y2: rowsBottom, "class": t === 0 ? "zero" : "grid" }, svg);
      svgText(svg, x, rowsBottom + 15, t + (t === 100 ? "%" : ""), "tick", "middle");
    });

    var control = cells[0][0];
    METHODS.forEach(function (m, i) {
      var cell = cells[i];
      var y = top + i * rowH;
      var g = svgEl("g", { "class": "row", tabindex: 0, role: "img",
        "aria-label": m.name + ": " + (cell ? cell[0].toFixed(1) + "% task success" : "not reported") }, svg);
      svgEl("rect", { x: -6, y: y, width: W + 12, height: rowH, rx: 7, "class": "hover" }, g);
      svgText(g, 0, y + rowH / 2 + 4.5, m.label, "row-label" + (m.kind === "ours" ? " ours" : ""));
      if (!cell) {
        svgText(g, x0 + 6, y + rowH / 2 + 4, "not reported", "na");
        svgText(g, W, y + rowH / 2 + 4.5, "—", "value", "end");
        return;
      }
      var d = barPath(Math.round(scale(0)), scale(cell[0]), y + (rowH - barH) / 2, barH, 4);
      if (d) svgEl("path", { d: d, "class": "bar " + m.kind }, g);
      svgText(g, W, y + rowH / 2 + 4.5, cell[0].toFixed(1), "value" + (m.kind === "ours" ? " strong" : ""), "end");
      attachTooltip(g, function () {
        var pairs = [["Task success", cell[0].toFixed(1) + "%"]];
        if (i > 0) pairs.push(["vs Control", signed(cell[0] - control, 1) + " pp"]);
        pairs.push(["Cost vs Control", cell[1].toFixed(2) + "×"]);
        fillTooltip(m.name, BENCH_NAME[bench] + " · " + line.model, pairs);
      });
    });
    var xc = Math.round(scale(control)) + 0.5;
    svgEl("line", { x1: xc, x2: xc, y1: top, y2: rowsBottom, "class": "ref", "aria-hidden": "true" }, svg);

    container.textContent = "";
    container.appendChild(svg);
  }

  // ---------- construction and execution tokens ----------

  var COST_DOMAIN = [0, 360];
  var COST_TICKS = [0, 100, 200, 300];

  function renderCost(container, key) {
    var rows = COST[key];
    var model = MAIN[key].model;
    var W = Math.max(260, Math.floor(container.clientWidth));
    var labelW = Math.ceil(Math.max.apply(null, rows.map(function (r) { return textWidth(r[0], 12.5, 650); }))) + 14;
    var valueW = 52, rowH = 34, barH = 16, axisH = 20, top = 2;
    var x0 = labelW, x1 = W - valueW - 6;
    var scale = function (v) { return x0 + (v - COST_DOMAIN[0]) / (COST_DOMAIN[1] - COST_DOMAIN[0]) * (x1 - x0); };
    var rowsBottom = top + rows.length * rowH;
    var H = rowsBottom + axisH;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "group",
      "aria-label": "Construction and execution tokens per query on matched BrowseComp-Plus queries with " + model });

    var compact = x1 - x0 < 200;
    (compact ? [0, 200] : COST_TICKS).forEach(function (t) {
      var x = Math.round(scale(t)) + 0.5;
      svgEl("line", { x1: x, x2: x, y1: top, y2: rowsBottom, "class": t === 0 ? "zero" : "grid" }, svg);
      svgText(svg, x, rowsBottom + 15, t === 0 ? "0" : t + "k", "tick", "middle");
    });

    rows.forEach(function (r, i) {
      var ours = r[0] !== "AggAgent";
      var y = top + i * rowH;
      var by = y + (rowH - barH) / 2;
      var g = svgEl("g", { "class": "row", tabindex: 0, role: "img",
        "aria-label": r[0] + ": " + r[2].toFixed(1) + "k construction and " + r[3].toFixed(1) + "k execution tokens, " +
          r[4].toFixed(1) + "k in total, " + r[1].toFixed(1) + "% success" }, svg);
      svgEl("rect", { x: -6, y: y, width: W + 12, height: rowH, rx: 7, "class": "hover" }, g);
      svgText(g, 0, y + rowH / 2 + 4.5, r[0], "row-label" + (ours ? " ours" : ""));
      var xs = Math.round(scale(0)), xb = scale(r[2]), xt = scale(r[2] + r[3]);
      if (xb - xs >= 0.5) svgEl("rect", { x: xs, y: by, width: Math.max(1, xb - xs), height: barH, "class": "bar build" }, g);
      var d = barPath(xb, xt, by, barH, 4);
      if (d) svgEl("path", { d: d, "class": "bar exec" }, g);
      svgText(g, W, y + rowH / 2 + 4.5, r[4].toFixed(1) + "k", "value" + (ours ? " strong" : ""), "end");
      attachTooltip(g, function () {
        var pairs = [["Success", r[1].toFixed(1) + "%"], ["Construction", r[2].toFixed(1) + "k"]];
        if (r[5] !== null) pairs.push(["Construction saved", r[5].toFixed(1) + "%"]);
        pairs.push(["Execution", r[3].toFixed(1) + "k"], ["Total", r[4].toFixed(1) + "k"]);
        fillTooltip(r[0], "BrowseComp-Plus · " + model, pairs);
      });
    });
    container.textContent = "";
    container.appendChild(svg);
  }

  var charts = Array.prototype.slice.call(document.querySelectorAll(".chart[data-chart]"));
  function renderCharts(kind) {
    charts.forEach(function (c) {
      var type = c.getAttribute("data-chart");
      if (kind && type !== kind) return;
      if (type === "acc") renderAcc(c, c.getAttribute("data-bench"));
      else renderCost(c, c.getAttribute("data-line"));
    });
  }
  renderCharts();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { renderCharts(); });
  var lastWidths = charts.map(function (c) { return c.clientWidth; });
  var pending = false;
  window.addEventListener("resize", function () {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(function () {
      pending = false;
      var widths = charts.map(function (c) { return c.clientWidth; });
      if (widths.some(function (w, i) { return w !== lastWidths[i]; })) {
        lastWidths = widths;
        hideTooltip();
        renderCharts();
      }
    });
  });

  // ---------- results table ----------

  var body = document.getElementById("results-body");
  var meta = document.getElementById("table-meta");
  var COLS = BENCHES.concat(["avg"]);

  function renderTable() {
    var line = MAIN[state.line];
    var best = {}, second = {};
    COLS.forEach(function (col) {
      var values = line.cells[col].filter(Boolean).map(function (c) { return c[0]; });
      best[col] = Math.max.apply(null, values);
      second[col] = Math.max.apply(null, values.filter(function (v) { return v < best[col]; }));
    });
    body.textContent = "";
    METHODS.forEach(function (m, i) {
      var tr = document.createElement("tr");
      tr.className = m.kind === "ours" ? "ours sep" : m.kind === "ctl" ? "ctl" : "";
      var name = document.createElement("td");
      var who = document.createElement("span");
      who.className = "who";
      who.appendChild(document.createElement("i"));
      who.appendChild(document.createTextNode(m.kind === "ours" ? "ReGE (ours)" : m.name));
      name.appendChild(who);
      tr.appendChild(name);
      COLS.forEach(function (col) {
        var cell = line.cells[col][i];
        var td = document.createElement("td");
        if (col === "avg") td.className = "avg";
        if (!cell) {
          td.textContent = "—";
        } else {
          var acc = document.createElement("span");
          acc.className = "acc";
          acc.textContent = cell[0].toFixed(1);
          var cost = document.createElement("span");
          cost.className = "cost";
          cost.textContent = cell[1].toFixed(2) + "×";
          td.appendChild(acc);
          td.appendChild(cost);
          if (cell[0] === best[col]) td.className += (td.className ? " " : "") + "best";
          else if (cell[0] === second[col]) td.className += (td.className ? " " : "") + "second";
        }
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    meta.textContent = line.model + ": 89 Terminal-Bench 2.0 tasks, 135 Spider2-Lite instances, " + line.awInstances +
      " AndroidWorld instances and 830 BrowseComp-Plus questions. Each cell gives task success (%) and cost relative to Control (×).";
  }

  // ---------- model toggles (chart and table stay in sync) ----------

  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab[data-line]"));
  var awCounts = Array.prototype.slice.call(document.querySelectorAll(".aw-n"));

  function setLine(key) {
    state.line = key;
    tabs.forEach(function (t) {
      var on = t.getAttribute("data-line") === key;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
    });
    awCounts.forEach(function (n) { n.textContent = "· " + MAIN[key].awInstances + " instances"; });
    hideTooltip();
    renderCharts("acc");
    if (body) renderTable();
  }

  tabs.forEach(function (t) {
    t.addEventListener("click", function () { setLine(t.getAttribute("data-line")); });
    t.addEventListener("keydown", function (e) {
      var step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!step) return;
      e.preventDefault();
      var group = Array.prototype.slice.call(t.parentNode.querySelectorAll(".tab[data-line]"));
      var next = group[(group.indexOf(t) + step + group.length) % group.length];
      next.focus();
      setLine(next.getAttribute("data-line"));
    });
  });
  setLine(state.line);

  // ---------- copy buttons ----------

  function copyText(str) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(str);
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = str;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try { document.execCommand("copy") ? resolve() : reject(new Error("copy failed")); }
      catch (e) { reject(e); }
      finally { document.body.removeChild(area); }
    });
  }
  Array.prototype.forEach.call(document.querySelectorAll(".copy[data-copy]"), function (button) {
    var label = button.querySelector("span");
    button.addEventListener("click", function () {
      var source = document.getElementById(button.getAttribute("data-copy"));
      copyText(source.textContent).then(function () {
        label.textContent = "Copied";
      }, function () {
        label.textContent = "Press Ctrl+C";
      }).then(function () {
        window.setTimeout(function () { label.textContent = "Copy"; }, 1600);
      });
    });
  });
})();
