"""Regenerate every figure and the README results table from real outputs of the repo.

    python docs/make_figures.py         # reads bench/results/*.json, compiles a kernel for the hero

Writes docs/assets/banner.svg, bench.svg, ir_to_wgsl.svg and rewrites the README block between
<!-- results:start --> and <!-- results:end -->. PNG copies: node docs/render_png.mjs (resvg).
"""
from __future__ import annotations

import html
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelforge import zoo  # noqa: E402
from kernelforge.compiler import compile_group, form_for  # noqa: E402
from kernelforge.compute import compute_def  # noqa: E402
from kernelforge.fusion import fuse  # noqa: E402

ACCENT = "#A3E635"
DIM = "#4d7c0f"
GREY = "#8b949e"
FG = "#e6edf3"
BG = "#0d1117"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
A = ROOT / "docs" / "assets"


def esc(s):
    return html.escape(str(s), quote=True)


def load():
    res = json.loads((ROOT / "bench/results/rtx2060.json").read_text())
    tun = json.loads((ROOT / "bench/results/rtx2060_tuning.json").read_text())
    return res, tun


# ----------------------------------------------------------------------------- banner

def banner(tun):
    # motif: the autotuner's measured landscape for one kernel (every candidate it timed, sorted)
    key = next((k for k in tun if k.startswith("matvec q4_0 4864")), next(iter(tun)))
    times = sorted(r["meas_s"] for r in tun[key]["log"] if r["meas_s"])
    best = times[0]
    speed = [best / t for t in times]
    x0, x1, base, top = 880, 1236, 320, 70
    w = (x1 - x0) / len(speed)
    bars = []
    for i, s in enumerate(speed):
        h = (base - top) * s
        col = ACCENT if i == 0 else DIM
        op = 1.0 if i == 0 else 0.35 + 0.6 * s
        bars.append(f'<rect x="{x0 + i * w:.1f}" y="{base - h:.1f}" width="{max(w - 2, 1):.1f}" height="{h:.1f}" '
                    f'fill="{col}" opacity="{op:.2f}"/>')
    grid = "".join(f'<path d="M{x} 0V400" stroke="#161b22"/>' for x in range(0, 1281, 40)) + \
        "".join(f'<path d="M0 {y}H1280" stroke="#161b22"/>' for y in range(0, 401, 40))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 400" width="1280" height="400">
<rect width="1280" height="400" fill="{BG}"/>
<g stroke-width="1">{grid}</g>
<text x="64" y="190" font-family="{MONO}" font-weight="700" font-size="88" fill="{FG}">kernelforge</text>
<text x="68" y="240" font-family="{MONO}" font-size="24" fill="{GREY}">a tiny ML compiler: tensor graph &#8594;</text>
<text x="68" y="274" font-family="{MONO}" font-size="24" fill="{GREY}">fused, tiled loop nests &#8594; WGSL &#183; Metal &#183; C</text>
<rect x="68" y="300" width="56" height="6" fill="{ACCENT}"/>
{"".join(bars)}
<text x="{x0}" y="348" font-family="{MONO}" font-size="13" fill="{GREY}">autotuner: {len(times)} schedules timed,</text>
<text x="{x0}" y="368" font-family="{MONO}" font-size="13" fill="{GREY}">{esc(key)}, RTX 2060</text>
</svg>
'''
    (A / "banner.svg").write_text(svg, encoding="utf-8")


# ----------------------------------------------------------------------------- bench chart

def bench_chart(res):
    rows = [r for r in res["rows"] if "hand" in r]
    gem = [r for r in res["rows"] if "hand" not in r]
    W, rowh, left = 1000, 26, 330
    H = 90 + rowh * (len(rows) + len(gem)) + 70
    scale_max = max(max(r["gen_tuned"]["value"] / r["hand"]["value"], r["gen_default"]["value"] / r["hand"]["value"])
                    for r in rows)
    smax = max(2.0, math.ceil(scale_max * 2) / 2)
    px = lambda v: left + (W - left - 120) * v / smax  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="{BG}" rx="8"/>',
           f'<text x="24" y="34" font-family="{MONO}" font-size="17" font-weight="700" fill="{FG}">generated / hand-written throughput, RTX 2060 (Dawn D3D12, median of {res["meta"]["runs"]})</text>',
           f'<text x="24" y="58" font-family="{MONO}" font-size="13" fill="{GREY}">'
           f'<tspan fill="{ACCENT}">&#9632;</tspan> generated, autotuned   <tspan fill="{DIM}">&#9632;</tspan> generated, default schedule   '
           f'| = hand-written (1.0x)   label: tuned absolute   M1 column: &#9203; not measured</text>']
    y = 80
    step = 0.5 if smax <= 4 else (1 if smax <= 8 else 2)
    for i in range(0, int(smax / step) + 1):
        v2 = i * step * 2
        xv = px(v2 / 2)
        out.append(f'<path d="M{xv:.1f} 70V{70 + rowh * len(rows) + 4}" stroke="#21262d"/>'
                   f'<text x="{xv:.1f}" y="{78 + rowh * len(rows) + 12}" font-family="{MONO}" font-size="11" fill="{GREY}" text-anchor="middle">{v2 / 2:g}x</text>')
    out.append(f'<path d="M{px(1):.1f} 70V{70 + rowh * len(rows) + 4}" stroke="{FG}" stroke-dasharray="3 3"/>')
    for r in rows:
        h = r["hand"]["value"]
        d, t = r["gen_default"]["value"] / h, r["gen_tuned"]["value"] / h
        lab = f'{r["name"]} {r["dtype"]}'
        out.append(f'<text x="{left - 10}" y="{y + 15}" font-family="{MONO}" font-size="12" fill="{FG}" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<rect x="{left}" y="{y + 3}" width="{px(min(d, smax)) - left:.1f}" height="7" fill="{DIM}"/>')
        out.append(f'<rect x="{left}" y="{y + 11}" width="{px(min(t, smax)) - left:.1f}" height="9" fill="{ACCENT}"/>')
        out.append(f'<text x="{px(min(max(t, d), smax)) + 6:.1f}" y="{y + 18}" font-family="{MONO}" font-size="11" fill="{GREY}">'
                   f'{t:.2f}x  ({r["gen_tuned"]["value"]:.0f} {r["metric"]})</text>')
        y += rowh
    y += 40
    if gem:
        out.append(f'<text x="24" y="{y}" font-family="{MONO}" font-size="13" fill="{GREY}">GEMM (no hand-written GPU GEMM exists in tinyinfer): tuned vs default schedule, GFLOP/s</text>')
        y += 12
        gmax = max(r["gen_tuned"]["value"] for r in gem)
        for r in gem:
            gx = lambda v: left + (W - left - 160) * v / gmax  # noqa: E731
            out.append(f'<text x="{left - 10}" y="{y + 15}" font-family="{MONO}" font-size="12" fill="{FG}" text-anchor="end">{esc(r["name"])} {esc(r["dtype"])} {"x".join(map(str, r["shape"]))}</text>')
            out.append(f'<rect x="{left}" y="{y + 3}" width="{gx(r["gen_default"]["value"]) - left:.1f}" height="7" fill="{DIM}"/>')
            out.append(f'<rect x="{left}" y="{y + 11}" width="{gx(r["gen_tuned"]["value"]) - left:.1f}" height="9" fill="{ACCENT}"/>')
            out.append(f'<text x="{gx(r["gen_tuned"]["value"]) + 6:.1f}" y="{y + 18}" font-family="{MONO}" font-size="11" fill="{GREY}">'
                       f'{r["gen_tuned"]["value"]:.0f} (default {r["gen_default"]["value"]:.0f})</text>')
            y += rowh
    out.append(f'<text x="24" y="{H - 14}" font-family="{MONO}" font-size="11" fill="{GREY}">measured {res["meta"]["date"]} on {esc(res["meta"]["adapter"].get("device", "?"))}; regenerate: python bench/run_bench.py &amp;&amp; python docs/make_figures.py</text>')
    out.append("</svg>")
    (A / "bench.svg").write_text("\n".join(out), encoding="utf-8")


# ----------------------------------------------------------------------------- IR -> WGSL hero

def ir_to_wgsl(tun):
    g = zoo.ffn_gate(4864, 896, "q4_0")
    G = fuse(g)[1]
    cd = compute_def(G, form_for(G))
    key = "ffn-gate+silu*mul q4_0 4864x896"
    kn = tun.get(key, {}).get("best")
    ck = compile_group(G, __import__("kernelforge.target", fromlist=["x"]).RTX2060_WGSL, kn)
    left = ["# tensor graph (fused group)"]
    left += [f"%{n.name} = {n.op}({', '.join('%' + i.name for i in n.inputs)})" for n in G.nodes]
    left += ["", "# compute definition"] + [ln.strip() for ln in str(cd).splitlines()[1:]]
    left += ["", "# schedule (autotuned knobs)"] + [f"{k} = {v}" for k, v in (kn or {}).items()]
    wrapped = []
    for ln in left:
        while len(ln) > 52:
            wrapped.append(ln[:52])
            ln = "    " + ln[52:]
        wrapped.append(ln)
    src = [ln for ln in ck.source.splitlines() if ln.strip()]
    right = src[:46]
    right = [ln if len(ln) <= 92 else ln[:89] + "..." for ln in right]
    lh = 15
    H = 70 + lh * max(len(wrapped), len(right)) + 30
    W = 1500
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" rx="10" fill="{BG}" stroke="#30363d"/>',
           '<circle cx="22" cy="20" r="6" fill="#ff5f56"/><circle cx="42" cy="20" r="6" fill="#ffbd2e"/><circle cx="62" cy="20" r="6" fill="#27c93f"/>',
           f'<text x="90" y="25" font-family="{MONO}" font-size="13" fill="{GREY}">python -m kernelforge explain ffn_q4_0   (real output, Qwen2.5-0.5B FFN shape 4864x896, Q4_0)</text>',
           f'<path d="M520 40V{H - 10}" stroke="#30363d"/>']
    for i, ln in enumerate(wrapped):
        col = ACCENT if ln.startswith("#") else FG
        out.append(f'<text x="20" y="{62 + i * lh}" font-family="{MONO}" font-size="12" fill="{col}" xml:space="preserve">{esc(ln)}</text>')
    for i, ln in enumerate(right):
        col = GREY if ln.lstrip().startswith("//") else FG
        out.append(f'<text x="540" y="{62 + i * lh}" font-family="{MONO}" font-size="12" fill="{col}" xml:space="preserve">{esc(ln)}</text>')
    if len(src) > len(right):
        out.append(f'<text x="540" y="{62 + len(right) * lh}" font-family="{MONO}" font-size="12" fill="{GREY}">... {len(src) - len(right)} more lines (full kernel: python -m kernelforge explain ffn_q4_0)</text>')
    out.append("</svg>")
    (A / "ir_to_wgsl.svg").write_text("\n".join(out), encoding="utf-8")


# ----------------------------------------------------------------------------- README table

def table(res, tun):
    lines = ["| family | case | dtype | metric | hand-written | generated (default) | generated (tuned) | tuned / hand | M1 (tuned / hand) |",
             "|---|---|---|---|---:|---:|---:|---:|:---:|"]
    for r in res["rows"]:
        h = r.get("hand")
        hv = f'{h["value"]:.1f}' if h else "n/a"
        ratio = f'**{r["gen_tuned"]["value"] / h["value"]:.2f}x**' if h else f'{r["gen_tuned"]["value"] / r["gen_default"]["value"]:.2f}x vs default'
        lines.append(f'| {r["family"]} | {r["name"]} `{"x".join(map(str, r["shape"]))}` | {r["dtype"]} | {r["metric"]} | {hv} | '
                     f'{r["gen_default"]["value"]:.1f} | {r["gen_tuned"]["value"]:.1f} | {ratio} | ⏳ |')
    errs = [v["rel_err"] for r in res["rows"] for k, v in r.items() if isinstance(v, dict) and "rel_err" in v]
    m = res["meta"]
    lines.append("")
    lines.append(f'Measured {m["date"]} on {m["adapter"].get("device")} ({m["adapter"].get("description")}), Windows 10, '
                 f'Ryzen 5 3600 host; {m["webgpu"]}; {m["timer"]}; median of {m["runs"]} runs. '
                 f'Every output was checked against NumPy in the same run (worst max-relative error {max(errs):.1e}).')
    return "\n".join(lines)


def tuning_table(tun):
    lines = ["| kernel | template | space | timed | tuning wall time | default | tuned | best knobs |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for k, v in tun.items():
        lines.append(f'| {k} | {v["template"]} | {v["space"]} | {v["measured"]} | {v["seconds"]:.0f} s | '
                     f'{v["default_s"] * 1e6:.1f} us | {v["best_s"] * 1e6:.1f} us | `{json.dumps(v["best"], separators=(",", ":"))}` |')
    return "\n".join(lines)


def main():
    res, tun = load()
    banner(tun)
    bench_chart(res)
    ir_to_wgsl(tun)
    rd = ROOT / "README.md"
    if rd.exists():
        s = rd.read_text(encoding="utf-8")
        for tag, body in (("results", table(res, tun)), ("tuning", tuning_table(tun))):
            s = re.sub(rf"<!-- {tag}:start -->.*?<!-- {tag}:end -->",
                       f"<!-- {tag}:start -->\n{body}\n<!-- {tag}:end -->", s, flags=re.S)
        ln = ROOT / "bench/results/lanes_rtx2060.json"
        if ln.exists():
            L = json.loads(ln.read_text())
            Ts = [k for k in L[0] if k.startswith("T")]
            head = "| dtype | tinyinfer port (T = 32) | " + " | ".join(f"T = {k[1:]}" for k in Ts) + " |"
            sep = "|---|---:|" + "---:|" * len(Ts)
            body = "\n".join(f'| {r["dtype"]} | {r["hand"]:.0f} | ' + " | ".join(
                (f'**{r[k]:.0f}**' if r[k] == min(r[t] for t in Ts) else f'{r[k]:.0f}') for k in Ts) + " |" for r in L)
            s = re.sub(r"<!-- lanes:start -->.*?<!-- lanes:end -->",
                       f"<!-- lanes:start -->\n{head}\n{sep}\n{body}\n<!-- lanes:end -->", s, flags=re.S)
        cm = ROOT / "bench/results/costmodel_rtx2060.json"
        if cm.exists():
            c = json.loads(cm.read_text())
            body = "\n".join(f'| {x["kernel"]} | {x["n"]} | {x["spearman"]:.2f} | {x["best_rank_by_model"]} | '
                             f'{x["topk_regret"]:.2f}x |' for x in c)
            s = re.sub(r"<!-- costmodel:start -->.*?<!-- costmodel:end -->",
                       "<!-- costmodel:start -->\n| kernel | schedules timed | Spearman(pred, meas) | model rank of true best | top-16 pick vs true best |\n|---|---:|---:|---:|---:|\n"
                       + body + "\n<!-- costmodel:end -->", s, flags=re.S)
        rd.write_text(s, encoding="utf-8")
    print("figures + README tables regenerated")


if __name__ == "__main__":
    main()
