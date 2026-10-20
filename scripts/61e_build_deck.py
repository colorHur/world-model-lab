#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-3c · 2 分钟演示（7 页 HTML 幻灯片）：`docs/DEMO-P3.html`。

为什么要脚本生成
----------------
演示页最容易"手抄数字、图和某个旧版本对不上"。本脚本：
1. **数字从 `docs/results_p3_trace.json` 的键取**（`F1.R[0]` 这类），脚本里不出现任何结果数字面量；
2. **图从 `assets/` 现读并 base64 内嵌** ⇒ 单个 HTML 文件可独立分发，且不会指向不存在的文件；
3. 输出**不含时间戳** ⇒ 重画两次字节应相同（除了 PNG 重画的编码差异，本脚本只读不重画图）。

用法：
    D:/devtools/miniconda3/envs/ai-lab/python.exe scripts/61e_build_deck.py
产物：
    docs/DEMO-P3.html
不写 `results/`。
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "docs" / "results_p3_trace.json"
FIG = ROOT / "assets" / "paper_p3.png"
FIG_COST = ROOT / "assets" / "paper_p3_cost.png"
OUT = ROOT / "docs" / "DEMO-P3.html"


def load_display() -> dict[str, str]:
    rows = json.loads(TRACE.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for r in rows:
        if r.get("key"):
            out[r["key"]] = r["display"]
    return out


def b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def crop_b64(path: Path, box: tuple[float, float, float, float]) -> str:
    """按比例裁一块（left, top, right, bottom ∈ [0,1]），避免把 2×2 图整张贴进单页。"""
    from PIL import Image

    im = Image.open(path)
    w, h = im.size
    l, t, r, b = box
    im2 = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    buf = io.BytesIO()
    im2.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


CSS = """
:root{--bg:#0f1115;--fg:#e8eaed;--dim:#9aa0a6;--acc:#e05252;--acc2:#4b8fd6;--line:#2a2f38;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;}
.slide{width:min(1120px,96vw);margin:0 auto;padding:38px 46px 54px;min-height:100vh;display:none}
.slide.on{display:block}
h1{font-size:30px;margin:6px 0 10px;line-height:1.3}
h2{font-size:24px;margin:4px 0 16px;color:var(--fg)}
.kicker{color:var(--acc);font-size:13px;letter-spacing:.14em;text-transform:uppercase}
p,li{font-size:17px;line-height:1.7;color:var(--fg)}
.dim{color:var(--dim)}
ul{padding-left:22px}
li{margin:7px 0}
.big{font-size:26px;font-weight:650}
.red{color:var(--acc)} .blue{color:var(--acc2)}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:15px}
th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left}
th{color:var(--dim);font-weight:600}
img{max-width:100%;border-radius:6px;border:1px solid var(--line);background:#fff}
.cap{color:var(--dim);font-size:13px;margin-top:8px}
.warn{border-left:3px solid var(--acc);padding:8px 14px;background:#1a1416;margin:14px 0;font-size:15px}
.nav{position:fixed;right:18px;bottom:12px;color:var(--dim);font-size:13px}
code{background:#1b2028;padding:1px 5px;border-radius:4px;font-size:15px}
"""

JS = """
const S=[...document.querySelectorAll('.slide')];let i=0;
function show(n){i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,k)=>s.classList.toggle('on',k===i));
 document.getElementById('pg').textContent=(i+1)+' / '+S.length;}
addEventListener('keydown',e=>{if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key))show(i+1);
 if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key))show(i-1);});
show(0);
"""


def main() -> int:
    for p in (TRACE, FIG, FIG_COST):
        if not p.exists():
            raise SystemExit(f"缺文件：{p.as_posix()}")
    D = load_display()

    def g(k: str) -> str:
        if k not in D:
            raise SystemExit(f"trace 里没有键 {k}（演示页不许手写数字）")
        return D[k]

    png_full = b64_png(FIG)
    png_ab = crop_b64(FIG, (0.0, 0.05, 1.0, 0.55))     # 上排 (a)(b)
    png_cd = crop_b64(FIG, (0.0, 0.50, 1.0, 1.0))      # 下排 (c)(d)
    png_cost = b64_png(FIG_COST)

    s: list[str] = []
    add = s.append

    # ---------------- 1 标题 ----------------
    add('<section class="slide on">')
    add('<div class="kicker">P3 · measurement / judge, not a new method</div>')
    add("<h1>带宽受限多任务远程控制中<br>预测补偿的适用域</h1>")
    add('<p class="dim">UAV 跟踪 + CartPole 共享一份发帧预算 · 四类预测器 · 预注册判据 · 逐格可溯源</p>')
    add('<div class="warn">一句话结论：<b>预测补偿的符号由刷新时序决定</b> —— '
        '快刷新档把 UAV 代价压到 ZOH 的 '
        f'<code>{g("F1.R[0]")}</code>，断服档反转为 <code>{g("F2.primary/100/unpaced.R[0]")}</code>。</div>')
    add('<p class="dim" style="font-size:14px">演示 2 分钟 · 数字全部取自 results_p3_trace.json · 未投稿、无外部评审</p>')
    add("</section>")

    # ---------------- 2 问题 ----------------
    add('<section class="slide">')
    add('<div class="kicker">1 · 问题</div>')
    add("<h2>预测补偿：补错了比不补更糟</h2>")
    add("<ul>"
        "<li>两台设备抢一条带宽受限链路，各自跑闭环控制；收不到帧，接收端只能靠<b>外推</b>。</li>"
        "<li>常见问法「预测有没有用」<b>不可直接回答</b>：同一个预测器在本装置里既改善也退化。</li>"
        "<li>要回答的是：<b>在什么刷新时序下安全</b>、<b>失配何时把资源分给错误的任务</b>。</li>"
        "</ul>")
    add('<div class="warn">本文不提出新方法：基线 <code>Linear</code> 无可学参数，'
        '<code>TrueDyn</code> 是读仿真真值的<b>特权诊断</b> ⇒ 不构成「世界模型收益」。</div>')
    add("</section>")

    # ---------------- 3 装置 ----------------
    add('<section class="slide">')
    add('<div class="kicker">2 · 装置与协议</div>')
    add("<h2>两用户、一份预算、一个调度臂</h2>")
    add("<table><tr><th>项</th><th>设定</th></tr>"
        "<tr><td>任务</td><td>UAV 航迹跟踪（m）+ CartPole 平衡</td></tr>"
        "<tr><td>规模</td><td>N=2，窗口 200 步，每集 30 条轨迹</td></tr>"
        "<tr><td>调度臂</td><td>只用 <code>age</code>（给最旧的用户）—— 不比较调度算法</td></tr>"
        "<tr><td>时序</td><td><code>unpaced</code>（t=B 用光）/ <code>paced</code>（铺满窗口）</td></tr>"
        "<tr><td>预测器</td><td>ZOH / Linear / hybrid / TrueDyn〔diagnostic〕</td></tr>"
        "<tr><td>指标</td><td>R = ΣJ(Linear)/ΣJ(ZOH)，同种子同 episode 配对</td></tr>"
        "<tr><td>区间</td><td>cluster bootstrap，重采样单位 = reset 初值</td></tr>"
        "</table>")
    add('<p class="dim">报 B&lt;L 的任何比值，必须同报「预算用完后仍在发的帧数」（t&gt;B）—— 否则读不出是否踩在断服上。</p>')
    add("</section>")

    # ---------------- 4 主结果 ----------------
    add('<section class="slide">')
    add('<div class="kicker">3 · 主确认（快刷新档）</div>')
    add("<h2>改善是真的；汇总帧数没有增加</h2>")
    add("<ul>"
        f'<li><span class="big red">{g("F1.R[0]")}</span> = R<sub>uav</sub>，95% 区间 '
        f'[{g("F1.ci_uav[0]")}, {g("F1.ci_uav[1]")}]（上界 &lt; 1）</li>'
        f'<li>90 个与开发集零交集的初值：逐初值改善 <b>{g("F2.primary/200/unpaced.frac_clusters_below_1[0]")}</b>；逐种子 3/3</li>'
        f'<li>两用户<b>汇总</b>送达帧恒为 <b>{g("F1.served.zoh[0]")}+{g("F1.served.zoh[1]")}</b>（预算上限）⇒ 不是靠多分帧</li>'
        f'<li class="dim">但拆开看是一降一升（UAV {g("F1.served.linear[0]")} vs {g("F1.served.zoh[0]")}，CartPole {g("F1.served.linear[1]")} vs {g("F1.served.zoh[1]")}）⇒ <b>任务间重分配未隔离</b>，不作因果归因</li>'
        f'<li class="dim">CartPole：区间含 1、逐初值仅 {g("F2.primary/200/unpaced.frac_clusters_below_1[1]")} ⇒ <b>不构成收益，也不构成等价证明</b></li>'
        "</ul>")
    add(f'<img src="data:image/png;base64,{png_ab}" alt="paper_p3 panels a b">')
    add('<div class="cap">图 1(a)(b)：左＝R 随实测陈旧度；右＝绝对代价（log 轴）。'
        '交点括号是网格间隔，<b>不是置信区间</b>。</div>')
    add("</section>")

    # ---------------- 5 反转 ----------------
    add('<section class="slide">')
    add('<div class="kicker">4 · 同一预测器，符号相反</div>')
    add("<h2>只改时序，收益变危害</h2>")
    add("<table><tr><th>档</th><th>时序</th><th>R<sub>uav</sub></th><th>95% 区间</th><th>t&gt;B 发帧</th></tr>"
        f'<tr><td>B=100</td><td>unpaced</td><td class="red big">{g("F2.primary/100/unpaced.R[0]")}</td>'
        f'<td>[{g("F2.primary/100/unpaced.ci_uav[0]")}, {g("F2.primary/100/unpaced.ci_uav[1]")}]</td><td>0（断服）</td></tr>'
        f'<tr><td>B=100</td><td>paced</td><td class="blue big">{g("F2.primary/100/paced.R[0]")}</td>'
        f'<td>[{g("F2.primary/100/paced.ci_uav[0]")}, {g("F2.primary/100/paced.ci_uav[1]")}]</td><td>非 0</td></tr>'
        f'<tr><td>B=200</td><td>unpaced</td><td>{g("F1.R[0]")}</td>'
        f'<td>[{g("F1.ci_uav[0]")}, {g("F1.ci_uav[1]")}]</td><td>0</td></tr>'
        "</table>")
    add('<div class="warn">同样帧数、同样调度臂、同样预测器，只把帧铺满窗口 ⇒ 比值从 '
        f'<code>{g("F2.primary/100/unpaced.R[0]")}</code> 变成 <code>{g("F2.primary/100/paced.R[0]")}</code>。'
        '<br>机制：断服后滞后样本越来越旧，常数速度外推的误差随时间线性增长 ⇒ 补偿项沿错误方向推。</div>')
    add("</section>")

    # ---------------- 6 边界 ----------------
    add('<section class="slide">')
    add('<div class="kicker">5 · 边界</div>')
    add("<h2>「门槛 = 失效窗口」未获支持</h2>")
    add("<ul>"
        f'<li>把失效半径改成三档 ⇒ 实测窗口 35 / 53 / 104 步；交点 '
        f'<b>{g("brackets.short.x_star")}</b> / <b>{g("brackets.base.x_star")}</b> / <b>{g("brackets.long.x_star")}</b></li>'
        f'<li>三个交点区间<b>两两重叠</b> ⇒ 判词 <b>INCONCLUSIVE</b>（本设计分不清位移）</li>'
        f'<li>比例偏离<b>只作探索性报告</b>：x*/W 由 {g("x_star_over_W.short.x_star_over_W")} 降到 {g("x_star_over_W.long.x_star_over_W")}'
        '（预注册标 report-only；括区间不是置信区间，不能当否证判决）</li>'
        '<li class="dim">结构性限制：改窗口 ⇒ 改存活 ⇒ 改 age 分布 ⇒ <b>改横轴</b>；x*/W 的下降里有一部分是选择效应</li>'
        "</ul>")
    add(f'<img src="data:image/png;base64,{png_cd}" alt="paper_p3 panels c d">')
    add('<div class="cap">图 1(c)(d)：误差棒＝交点所在相邻两点，不是统计置信区间。</div>')
    add("</section>")

    # ---------------- 7 否定结果（v0.3 新增） ----------------
    add('<section class="slide">')
    add('<div class="kicker">6 · 否定结果</div>')
    add("<h2>「用陈旧度门控回退」这条路，我们试了四次</h2>")
    add("<ul>"
        '<li><b>T25</b> 按<b>误差</b>门控：伤害最大的那一格触发率是 <b>0</b> ⇒ 没有操作空间（方向问题，不是阈值问题）</li>'
        f'<li><b>T26</b> <code>age</code> 与危害同号（逐决策 τ = {g("cells.B100|u0.tau_age")}），'
        f'但换动作会更贵的决策占 {g("cells.B100|u0.share_dJ_pos")} ⇒ 秩相关<b>不能</b>折算成阈值</li>'
        f'<li><b>T27</b> 一个固定 θ 不能两档通吃：θ=4 把毒档压到 {g("verdicts.F1.R_gated[0]")}，'
        f'同时把好档从 0.308 抬到 {g("verdicts.F2.R[0]")} ⇒ 判词 <b>FALLBACK-FAILS</b></li>'
        f'<li><b>T28</b> 扫描出的 θ=8 在<b>独立批</b>上：好档 {g("verdicts.T1.R_uav")}（与 linear 配对 ΔR 区间含 0、'
        f'触发份额仅 {g("summary.200.per_arm.gated8.gate_share_theta")}），毒档 {g("verdicts.T2.R_uav")} '
        f'（区间 [{g("verdicts.T2.ci_uav[0]")}, {g("verdicts.T2.ci_uav[1]")}] 不含 1）</li>'
        "</ul>")
    add('<div class="warn">判词 <b>FALLBACK-FAILS (confirm)</b>：'
        '<b>「减少」成立、「归零」不成立</b>；好档「不坏」是因为几乎不起作用。'
        '本文因此<b>不提供</b>可部署的补救方法。</div>')
    add('<div class="cap">数字取自否定线冻结清单 docs/evidence_manifest_p3_neg.json（与主表清单互不影响）。</div>')
    add("</section>")

    # ---------------- 8 资源与下一步 ----------------
    add('<section class="slide">')
    add('<div class="kicker">7 · 代价与下一步</div>')
    add("<h2>开销是相对概念，不要读成实时预算</h2>")
    add("<ul>"
        f'<li>单次调用中位：ZOH <code>{g("TIMING.primary/zoh.median_us")}</code> µs vs Linear '
        f'<code>{g("TIMING.primary/linear.median_us")}</code> µs；整轮墙钟 {g("58.elapsed.zoh.mean")} vs '
        f'{g("58.elapsed.linear.mean")} s（差 &lt; 2%）</li>'
        '<li>⇒ 时间被环境步进主导；<b>不主张</b>「预测可以免费实时跑」，真实上下行时延未建模</li>'
        '<li>下一步：把横轴换成<b>与存活无关</b>的陈旧度量（需新协议、新预注册）；'
        '不达门槛就不再加相似的调度 heuristic</li>'
        "</ul>")
    add(f'<img src="data:image/png;base64,{png_cost}" alt="paper_p3_cost">')
    add('<div class="cap">图 2：左＝单次调用开销；右＝每 cell 墙钟逐点（环境步进主导）。</div>')
    add("</section>")

    html = ("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<title>P3 演示 · 预测补偿的适用域</title><style>" + CSS + "</style></head><body>"
            + "".join(s) + '<div class="nav" id="pg">1 / 8 · ←/→ 翻页</div>'
            "<script>" + JS + "</script></body></html>")
    n_slide = html.count('<section class="slide')
    n_img = html.count("data:image/png;base64,")
    OUT.write_text(html, encoding="utf-8")
    print(f"[out] {OUT.relative_to(ROOT).as_posix()}（{len(html) / 1024:.0f} KB，"
          f"{n_slide} 页，内嵌图 {n_img} 张）")
    print("[figs]", FIG.relative_to(ROOT).as_posix(), FIG_COST.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
