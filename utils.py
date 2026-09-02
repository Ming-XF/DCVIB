"""共享图表渲染工具：SVG 折线对比卡，供 adv_eval.py 与 rebuild_tune_html.py 共用。

绘制规范（dataviz）：2px 折线、≥8px 圆点带 2px 面环、发丝实线网格、图例恒在
（≥2 系列）、≤4 系列末端直标（引线连接，纵向避碰）、十字准线 + tooltip、
浅/深双模式。颜色为 8 色固定顺序分类调色板（不循环复用），经
validate_palette.js 校验：相邻对 CVD ΔE≥8、正常视觉 ΔE≥15；黄/品红/青三色在
浅色面上对比 <3:1，靠图例文字 + 汇总表 + tooltip 兜底。

调用方用 plot_bounds() 把数据坐标换算成像素坐标，再交 curve_card() 渲染。
"""

import html
import json
import math

CURVE_COLORS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
CURVE_DASHES = ("", "7 5", "2 4", "1 3")  # >8 系列时颜色复用，靠线型兜底

_W, _H = 860, 380            # viewBox 尺寸
_MARGIN_L, _MARGIN_T, _MARGIN_R, _MARGIN_B = 56, 18, 16, 44
_GUTTER = 170                # ≤4 系列时末端直标标签的右侧留白（引线连接）


def plot_bounds(n_series):
    """返回 (x0, x1, y0, y1, gutter)：与 curve_card 相同的绘图区几何。

    调用方用它把数据坐标换算成像素坐标（x 传列位置、y 传 0~1 范围内的值）。
    """
    gutter = _GUTTER if n_series <= 4 else 0
    x0, x1 = _MARGIN_L, _W - _MARGIN_R - gutter
    y0, y1 = _MARGIN_T, _H - _MARGIN_B
    return x0, x1, y0, y1, gutter


def curve_css():
    """曲线卡样式：调色板 CSS 变量（浅色 + 深色两套）与卡片/图例/tooltip 样式。"""
    light_vars = "\n".join(f"  --series-{i + 1}: {c};" for i, c in enumerate(CURVE_COLORS["light"]))
    dark_vars = "\n".join(f"    --series-{i + 1}: {c};" for i, c in enumerate(CURVE_COLORS["dark"]))
    dark_block = f"""    color-scheme: dark;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255, 255, 255, .10);
{dark_vars}"""
    return f"""<style>
.viz-root {{
  color-scheme: light;
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px 10px;
  position: relative;
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11, 11, 11, .10);
{light_vars}
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
{dark_block}
  }}
}}
:root[data-theme="dark"] .viz-root {{
{dark_block}
}}
.viz-card {{ margin-top: 1.6em; }}
.viz-root h3 {{ font-size: 1.02em; margin: 0 0 2px; }}
.viz-root .viz-sub {{ color: var(--text-secondary); font-size: .82em; margin-bottom: 8px; }}
.viz-root svg {{ display: block; width: 100%; height: auto; }}
.viz-root .legend {{ display: flex; flex-wrap: wrap; gap: 4px 18px; margin-top: 4px; }}
.viz-root .legend-item {{
  display: inline-flex; align-items: center; gap: 6px;
  font-size: .82em; color: var(--text-secondary); cursor: pointer;
  border: none; background: none; padding: 2px 0; font-family: inherit;
}}
.viz-root .legend-item .key {{ display: inline-block; width: 18px; height: 2px; border-radius: 1px; }}
.viz-root .legend-item.off {{ opacity: .35; }}
.viz-root g.series.off {{ display: none; }}
.viz-root .tooltip {{
  position: absolute; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 6px; padding: 7px 10px; font-size: 12px; pointer-events: none;
  box-shadow: 0 2px 6px rgba(0, 0, 0, .18); z-index: 10;
}}
.viz-root .tooltip .tip-title {{ color: var(--muted); font-size: 11px; margin-bottom: 4px; }}
.viz-root .tooltip .tip-row {{ display: flex; align-items: center; gap: 6px; white-space: nowrap; }}
.viz-root .tooltip .tip-key {{ display: inline-block; width: 14px; height: 2px; flex: none; }}
.viz-root .tooltip .tip-val {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.viz-root .tooltip .tip-label {{ color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; max-width: 150px; }}
.viz-root .hover-capture:focus {{ outline: 1px solid var(--muted); outline-offset: -1px; }}
</style>"""


def _nice_step(span, target=6):
    """把 y 轴跨度取整到 {1,2,5}×10^k 的刻度步长（约 target 个刻度）。"""
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def curve_card(norm_id, title, subtitle, xs, x_labels, series, y_min=0.0, y_max=1.0,
               x_sublabels=None, col_titles=None):
    """渲染单张 SVG 折线对比卡（含图例、十字准线、tooltip 与交互数据）。

    xs: 各列 x 像素坐标（调用方经 plot_bounds 换算）；x_labels: 各列刻度标签；
    x_sublabels: 各列第二行小标签（可 None，如 clean 标记）。
    series: [(label, values, stds, texts)]，values/std 与 xs 等长；value 为 None
      表示该列缺失（折线断开、不画点/误差棒）；stds 为 None 或 0 不画误差棒；
      texts 为各列 tooltip 数值字符串（None 显示 '-'）。
    col_titles: 各列 tooltip 标题（默认 x_labels）。y 范围 [y_min, y_max]。
    """
    color_var = lambda i: f"var(--series-{i % 8 + 1})"
    x0, x1, y0, y1, _ = plot_bounds(len(series))
    col_titles = col_titles or list(x_labels)

    def yp(v):
        return y0 + (y_max - v) * (y1 - y0) / (y_max - y_min)

    # 轴基线与网格（发丝实线网格、刻度文字 muted）
    parts = [
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" stroke="var(--baseline)"/>',
        f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="var(--baseline)"/>',
    ]
    step = _nice_step(y_max - y_min)
    t = math.ceil(y_min / step - 1e-9) * step
    while t <= y_max + 1e-9:
        y = yp(t)
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>'
            f'<text x="{x0 - 6:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{t:g}</text>'
        )
        t += step
    for x, label, sub in zip(xs, x_labels, x_sublabels or [None] * len(xs)):
        parts.append(
            f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y1 + 4:.1f}" stroke="var(--baseline)"/>'
            f'<text x="{x:.1f}" y="{y1 + 18:.1f}" text-anchor="middle" font-size="11" fill="var(--muted)">{label}</text>'
        )
        if sub:
            parts.append(
                f'<text x="{x:.1f}" y="{y1 + 31:.1f}" text-anchor="middle" font-size="9" fill="var(--muted)">{sub}</text>'
            )

    # 系列折线（None 处断开分段）+ 误差棒 + 圆点
    series_groups, end_pts = [], []
    for i, (label, vals, stds, _) in enumerate(series):
        dash = CURVE_DASHES[(i // 8) % len(CURVE_DASHES)] if i >= 8 else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        color = color_var(i)
        group = []
        seg = []
        for x, v in zip(xs, vals):
            if v is None:
                if len(seg) >= 2:
                    group.append(
                        f'<path d="M ' + " ".join(f"{px:.1f},{yp(pv):.1f}" for px, pv in seg)
                        + f'" fill="none" stroke="{color}" stroke-width="2"'
                        + f' stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
                    )
                seg = []
            else:
                seg.append((x, v))
        if len(seg) >= 2:
            group.append(
                f'<path d="M ' + " ".join(f"{px:.1f},{yp(pv):.1f}" for px, pv in seg)
                + f'" fill="none" stroke="{color}" stroke-width="2"'
                + f' stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
            )
        for x, v, s in zip(xs, vals, stds):
            if v is None:
                continue
            if s is not None and s > 1e-12:
                group.append(
                    f'<line x1="{x:.1f}" y1="{yp(max(y_min, v + s)):.1f}" x2="{x:.1f}"'
                    f' y2="{yp(min(y_max, v - s)):.1f}" stroke="{color}" stroke-width="1.5" stroke-opacity="0.9"/>'
                )
        for x, v in zip(xs, vals):
            if v is None:
                continue
            group.append(
                f'<circle cx="{x:.1f}" cy="{yp(v):.1f}" r="4" fill="{color}" stroke="var(--surface-1)" stroke-width="2"/>'
            )
        series_groups.append(f'<g class="series" data-series="{i}">' + "".join(group) + "</g>")
        last = next(((x, v) for x, v in zip(reversed(xs), reversed(vals)) if v is not None), None)
        if last is not None:
            end_pts.append((i, yp(last[1]), label))

    # 末端直标（≤4 系列；纵向避碰后仍越界或标签过宽则整体放弃，交由图例承载）
    end_labels = []
    if len(series) <= 4 and end_pts:
        ordered = sorted(end_pts, key=lambda t: t[1])
        ys = []
        for _, y, _ in ordered:
            if ys:
                y = max(y, ys[-1] + 15)
            ys.append(y)
        max_w = max(len(l) for _, _, l in end_pts) * 6.4 + 4
        gutter = plot_bounds(len(series))[4]
        if ys[-1] <= y1 - 2 and max_w <= gutter - 14:
            label_y = {i: ly for (i, _, _), ly in zip(ordered, ys)}
            for i, py, label in end_pts:
                color = color_var(i)
                ly = label_y[i]
                end_labels.append(
                    f'<line x1="{x1:.1f}" y1="{py:.1f}" x2="{x1 + 8:.1f}" y2="{ly:.1f}" stroke="{color}" stroke-width="1"/>'
                    f'<text x="{x1 + 12:.1f}" y="{ly + 4:.1f}" font-size="11" fill="var(--text-secondary)">{html.escape(label)}</text>'
                )

    # 图例（≥2 系列恒在；单击显隐对应系列）
    legend = ""
    if len(series) >= 2:
        items = "".join(
            f'<button type="button" class="legend-item" data-series="{i}">'
            f'<span class="key" style="background:{color_var(i)}"></span>'
            f'<span>{html.escape(label)}</span></button>'
            for i, (label, _, _, _) in enumerate(series)
        )
        legend = f'<div class="legend">{items}</div>'

    # 交互数据（tooltip 数值、列位置；标签经 textContent 注入）
    series_data = []
    for i, (label, _, _, texts) in enumerate(series):
        texts_out = [t if t is not None else "-" for t in (texts or [None] * len(xs))]
        series_data.append({"label": label, "texts": texts_out, "color": color_var(i)})
    data_json = json.dumps(
        {"xs": [round(x, 1) for x in xs], "titles": list(col_titles), "series": series_data},
        ensure_ascii=False,
    ).replace("</", "<\\/")

    return f"""
<div class="viz-card">
<div class="viz-root">
<h3>{title}</h3>
<div class="viz-sub">{subtitle}</div>
<svg viewBox="0 0 {_W} {_H}" role="img" aria-label="{title}">
<clipPath id="advclip-{norm_id}"><rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}"/></clipPath>
{''.join(parts)}
<g clip-path="url(#advclip-{norm_id})">
{''.join(series_groups)}
</g>
{''.join(end_labels)}
<line class="crosshair" x1="{xs[0]:.1f}" x2="{xs[0]:.1f}" y1="{y0:.1f}" y2="{y1:.1f}" stroke="var(--muted)" stroke-opacity="0"/>
<rect class="hover-capture" x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" fill="transparent" tabindex="0" aria-label="移动光标或按左右方向键查看各系列数值"/>
</svg>
{legend}
<div class="tooltip" hidden></div>
<script type="application/json" class="viz-data">{data_json}</script>
</div>
</div>"""


def curve_script():
    """全部曲线卡共用的交互脚本：十字准线吸附最近列 + 单 tooltip 列出所有系列。"""
    return """<script>
(function () {
  document.querySelectorAll(".viz-card").forEach(function (card) {
    var dataEl = card.querySelector("script.viz-data");
    if (!dataEl) return;
    var data = JSON.parse(dataEl.textContent);
    var svg = card.querySelector("svg");
    var cap = svg.querySelector(".hover-capture");
    var hair = svg.querySelector(".crosshair");
    var tip = card.querySelector(".tooltip");
    var groups = Array.prototype.slice.call(svg.querySelectorAll("g.series"));
    var viewW = svg.viewBox.baseVal.width;
    var col = 0;

    function visible(i) { return !groups[i] || !groups[i].classList.contains("off"); }

    function showTooltip(j, clientX, clientY) {
      hair.setAttribute("x1", data.xs[j]);
      hair.setAttribute("x2", data.xs[j]);
      hair.setAttribute("stroke-opacity", "1");
      tip.textContent = "";
      var title = document.createElement("div");
      title.className = "tip-title";
      title.textContent = data.titles[j];
      tip.appendChild(title);
      data.series.forEach(function (s, i) {
        if (!visible(i)) return;
        var row = document.createElement("div");
        row.className = "tip-row";
        var key = document.createElement("span");
        key.className = "tip-key";
        key.style.background = s.color;
        var val = document.createElement("span");
        val.className = "tip-val";
        val.textContent = s.texts[j];
        var lab = document.createElement("span");
        lab.className = "tip-label";
        lab.textContent = s.label;
        row.appendChild(key);
        row.appendChild(val);
        row.appendChild(lab);
        tip.appendChild(row);
      });
      var root = card.querySelector(".viz-root");
      var rootRect = root.getBoundingClientRect();
      var left = clientX - rootRect.left + 14;
      var top = clientY - rootRect.top + 14;
      if (left + 240 > rootRect.width) left = clientX - rootRect.left - 240;
      if (top + 120 > rootRect.height) top = clientY - rootRect.top - 120;
      tip.style.left = Math.max(4, left) + "px";
      tip.style.top = Math.max(4, top) + "px";
      tip.hidden = false;
    }

    function hideTooltip() { tip.hidden = true; hair.setAttribute("stroke-opacity", "0"); }

    function nearestCol(clientX) {
      var rect = svg.getBoundingClientRect();
      var x = (clientX - rect.left) * (viewW / rect.width);
      var best = 0, bestD = Infinity;
      data.xs.forEach(function (v, j) {
        var d = Math.abs(v - x);
        if (d < bestD) { bestD = d; best = j; }
      });
      return best;
    }

    function tooltipAt(j) {
      var rect = svg.getBoundingClientRect();
      showTooltip(j, rect.left + data.xs[j] * (rect.width / viewW), rect.top + rect.height * 0.2);
    }

    cap.addEventListener("pointermove", function (e) {
      col = nearestCol(e.clientX);
      showTooltip(col, e.clientX, e.clientY);
    });
    cap.addEventListener("pointerleave", hideTooltip);
    cap.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") col = Math.max(0, col - 1);
      else if (e.key === "ArrowRight") col = Math.min(data.xs.length - 1, col + 1);
      else return;
      e.preventDefault();
      tooltipAt(col);
    });
    cap.addEventListener("focus", function () { tooltipAt(col); });
    cap.addEventListener("blur", hideTooltip);

    card.querySelectorAll(".legend-item").forEach(function (item) {
      item.addEventListener("click", function () {
        var i = Number(item.dataset.series);
        var off = groups[i].classList.toggle("off");
        item.classList.toggle("off", off);
      });
    });
  });
})();
</script>"""
