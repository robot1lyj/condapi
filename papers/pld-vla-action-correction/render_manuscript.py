"""Render the paper draft and audit as a self-contained reader (no project runtime)."""
from pathlib import Path
import html
import json
import re
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent
draft = (ROOT / "drafts/manuscript-v0.1.md").read_text()
audit = json.loads((ROOT / "qa_logs/defensive-writing-audit.json").read_text())
evidence = json.loads((ROOT / "sources/manuscript-evidence-v0.1.json").read_text())
equations = [
    "aᵢⱼ = uᵢⱼ × { s⁺ᵢⱼ  当 uᵢⱼ ≥ 0；s⁻ᵢⱼ  当 uᵢⱼ < 0 }",
    "Aᵢ[k] = Bᵢ[k] + wᵢ[k] Mᵢ (ρ ⊙ aᵢ)",
    "yᵢ = rᵢ + (1 − dᵢᵗᵉʳᵐ) γₛ^Δtᵢ × [ minⱼ Q̄ⱼ(xᵢ₊₁, a′ᵢ₊₁) − α cᵢ₊₁ log πθ(u′ᵢ₊₁ | xᵢ₊₁) ]",
    "Lπ = E[xᵢ : cᵢ = 1, uᵢ ∼ πθ] [ α log πθ(uᵢ | xᵢ) − minⱼ Qⱼ(xᵢ, aᵢ) ]",
]
fragments = {}

def hold(content):
    key = "PAPERINSERT" + str(len(fragments)) + "END"
    fragments[key] = content
    return key

def display_math(match):
    tex = match.group(1).strip()
    index = int(re.search(r"\\tag\{(\d+)\}", tex)[1]) - 1
    return "\n\n" + hold(
        '<div class="equation"><div>' + html.escape(equations[index]) +
        '</div><details><summary>公式 ' + str(index+1) +
        ' · 查看 LaTeX</summary><pre>' + html.escape(tex) + '</pre></details></div>'
    ) + "\n\n"

draft = re.sub(r"\\\[(.*?)\\\]", display_math, draft, flags=re.S)

def inline_math(match):
    text = match.group(1)
    substitutions = {
        r"\mathbb{R}^{H\times D}": "ℝᴴˣᴰ", r"\mathrm{term}": "term",
        r"\tanh": "tanh ", r"\in": " ∈ ", r"\rho": "ρ",
        r"\Delta t_i": "Δtᵢ", r"\gamma_s": "γₛ", r"\mathrm{rad}": "rad",
        r"\,": " ", r"\{": "{", r"\}": "}",
        "s_i^+": "sᵢ⁺", "s_i^-": "sᵢ⁻", "t_{i+1}": "tᵢ₊₁",
        "a'_{i+1}": "a′ᵢ₊₁", "d_i^{term}": "dᵢᵗᵉʳᵐ",
        "_i": "ᵢ", "^7": "⁷",
    }
    for before, after in substitutions.items():
        text = text.replace(before, after)
    return hold('<span class="math">' + html.escape(text) + '</span>')

draft = re.sub(r"\$([^$\n]+)\$", inline_math, draft)
md = MarkdownIt("commonmark", {"html": False}).enable("table")
tokens = md.parse(draft)
nav = []
for i, token in enumerate(tokens):
    if token.type == "heading_open" and token.tag == "h2":
        ident = "section-" + str(len(nav))
        token.attrSet("id", ident)
        nav.append((ident, tokens[i+1].content))
body = md.renderer.render(tokens, md.options, {})
for key, content in fragments.items():
    body = body.replace("<p>"+key+"</p>", content).replace(key, content)
body = body.replace("<table>", '<div class="table-scroll"><table>').replace("</table>", "</table></div>")
body = re.sub(r'<h1>.*?</h1>', '', body, count=1)

architecture = '''<figure class="architecture"><svg viewBox="0 0 920 386" role="img" aria-label="VLA参考计划、局部残差、RTC执行和奖励学习闭环">
<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="#6f8990" stroke-width="1.2"/></marker></defs>
<text x="28" y="28" class="svg-label">连续执行</text>
<g fill="none" stroke="#6f8990" stroke-width="1.6" marker-end="url(#arr)">
<path d="M266 100 H335"/><path d="M585 100 H650"/><path d="M770 143 V230"/>
<path d="M650 280 H585"/><path d="M335 280 H266"/><path d="M148 230 V193 H455 V143"/>
</g>
<g class="svg-box"><rect x="28" y="55" width="238" height="90" rx="12"/><rect x="335" y="55" width="250" height="90" rx="12"/><rect x="650" y="55" width="238" height="90" rx="12"/></g>
<g class="svg-box green"><rect x="650" y="232" width="238" height="92" rx="12"/><rect x="335" y="232" width="250" height="92" rx="12"/><rect x="28" y="232" width="238" height="92" rx="12"/></g>
<g text-anchor="middle"><text x="147" y="91" class="svg-title">冻结 VLA</text><text x="147" y="118" class="svg-small">多视角观测 → 参考计划</text>
<text x="460" y="91" class="svg-title">局部残差专家</text><text x="460" y="118" class="svg-small">入口 / 方向可行域 / 包络</text>
<text x="769" y="91" class="svg-title">原 RTC 执行</text><text x="769" y="118" class="svg-small">已承诺前缀 / 实发命令</text>
<text x="769" y="267" class="svg-title">来源与结果核验</text><text x="769" y="295" class="svg-small">排队完成 → 多帧证据</text>
<text x="460" y="267" class="svg-title">版本化 Replay</text><text x="460" y="295" class="svg-small">真实区间 / 单次终止奖励</text>
<text x="147" y="267" class="svg-title">SAC 更新</text><text x="147" y="295" class="svg-small">下一轮专家快照</text></g>
<text x="28" y="365" class="svg-label">后续研究 → 表征与动作架构 / 下游价值 / 专家经验回灌</text>
</svg><figcaption>框架示意。图中实线对应当前接口；底部列出后续论文扩展方向。</figcaption></figure>'''
body = re.sub(r'<pre><code class="language-text">观测.*?</code></pre>', architecture, body, count=1, flags=re.S)
cards=[]
for row in audit["records"]:
    path=next(f["path"] for f in audit["files"] if f["key"]==row["source"])
    cards.append('<article class="audit-card" data-search="'+html.escape(" ".join(str(x) for x in row.values()),quote=True)+'">'
      '<div class="audit-meta">'+f'{row["id"]:02} / {row["source"]} · L{row["line"]}'+'<span>'+html.escape(row["why"])+'</span></div>'
      '<div class="before"><small>原文</small>'+html.escape(row["original"])+'</div>'
      '<div class="after"><small>改写 / 安放方式</small>'+html.escape(row["rewrite"])+'</div>'
      '<a class="source" href="../'+html.escape(path)+'">查看原材料 ↗</a></article>')
nav_html=''.join('<a href="#'+i+'">'+html.escape(t)+'</a>' for i,t in nav)

page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLA 动作修正与强化学习 · 论文初稿</title><style>
:root{--bg:#f5f4f0;--paper:#fffefb;--ink:#20363c;--muted:#61747a;--line:#dce3de;--green:#197563;--soft:#edf4ef;--blue:#2b5d9a;--gold:#a86429}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:30px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.85 system-ui,"Noto Sans CJK SC","Microsoft Yahei",sans-serif}
a{color:var(--green);text-underline-offset:4px}button,input{font:inherit}button{cursor:pointer;background:transparent;color:var(--ink);border:1px solid var(--line);padding:7px 15px;border-radius:6px}button:hover,button[aria-pressed=true]{color:var(--green);border-color:var(--green);background:var(--soft)}
a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid var(--green);outline-offset:3px}
.mast{max-width:1400px;margin:auto;padding:23px 40px;display:flex;justify-content:space-between;font-size:12px;letter-spacing:.1em;border-bottom:1px solid var(--line)}.mast a{text-decoration:none}
.shell{max-width:1400px;margin:auto;display:grid;grid-template-columns:215px minmax(0,1fr);gap:48px;padding:45px 40px}
aside{position:sticky;top:30px;align-self:start;max-height:92vh;overflow:auto;font-size:12px}aside strong{display:block;font-size:13px;margin-bottom:15px}aside a{display:block;color:var(--muted);text-decoration:none;padding:7px 0;line-height:1.6}aside .top-link{margin-top:18px;color:var(--green)}
main{min-width:0;max-width:970px}.eyebrow{font:600 11px/1.5 ui-monospace,monospace;letter-spacing:.16em;color:var(--green)}.hero h1{font:600 clamp(32px,3.8vw,52px)/1.35 system-ui;letter-spacing:-.04em;margin:20px 0}.hero p{max-width:760px;color:var(--muted);font-size:16px}
.rule{height:3px;background:var(--green);width:56px;margin:30px 0}.roadmap{display:grid;grid-template-columns:repeat(3,1fr);border-block:1px solid var(--line);margin:25px 0 35px}.roadmap>div{padding:18px 12px 18px 0}.roadmap small{display:block;color:var(--muted);font-size:12px}.roadmap b{font-size:14px}.roadmap em{font-style:normal;color:var(--green);margin-right:7px}
.demo,.paper,.audit{background:var(--paper);border:1px solid var(--line);border-radius:12px;margin:28px 0;padding:30px}.demo h2{font-size:22px;margin:0 0 9px}.demo>p,.demo figcaption{color:var(--muted);font-size:13px}.demo-control{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.demo svg{width:100%;display:block;min-width:580px}.svg-scroll{overflow:auto}
.step-copy{background:var(--soft);border-radius:8px;min-height:94px;padding:15px 18px;margin-top:14px}.step-copy strong{display:block;color:var(--green);font-size:15px}.step-copy p{margin:4px 0;font-size:13px}
.paper{padding:35px 42px}.paper>p:first-child{font-size:12px;color:var(--muted)}.paper h2{font-size:25px;margin:50px 0 18px;padding-top:15px;border-top:1px solid var(--line);letter-spacing:-.02em}.paper h3{font-size:18px;margin:30px 0 12px}.paper p{margin:15px 0}.paper strong{font-weight:650}.paper li{margin:6px 0}
code{font:13px/1.5 ui-monospace,monospace;background:var(--soft);padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre{white-space:pre;overflow:auto;padding:17px;background:var(--soft);font:12px/1.65 ui-monospace,monospace;border-radius:7px}pre code{padding:0;background:transparent}
.table-scroll{overflow:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.7;min-width:590px}td,th{padding:12px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--soft);font-weight:650}td:first-child{font-weight:600}
.math{font-family:Cambria,"Times New Roman",serif;white-space:nowrap;padding-inline:2px}.equation{padding:20px;margin:22px 0;background:#f1f5f6;border-left:3px solid var(--blue);font:18px/1.9 Cambria,"Times New Roman",serif;overflow:auto}.equation>div{min-width:500px}.equation summary{font:11px/1.5 system-ui;color:var(--muted);cursor:pointer;margin-top:10px}.equation pre{font-size:11px}
.architecture{margin:22px 0}.architecture svg{width:100%;min-width:570px}.architecture{overflow:auto}.architecture figcaption{font-size:12px;color:var(--muted)}.svg-box rect{fill:#edf2f8;stroke:#cbd7e4}.svg-box.green rect{fill:#edf4ef;stroke:#bfd7cc}.svg-title{font:600 19px system-ui;fill:#20363c}.svg-small{font:14px system-ui;fill:#61747a}.svg-label{font:13px system-ui;fill:#197563}
.audit h2{margin:0;font-size:26px}.audit>p{font-size:14px;color:var(--muted)}.audit-tools{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}.audit-tools input{background:white;border:1px solid var(--line);border-radius:6px;padding:9px 12px;flex:1;min-width:180px}.audit-tools span{font-size:12px;color:var(--muted)}
.audit-card{border-top:1px solid var(--line);padding:22px 0}.audit-meta{display:flex;gap:15px;justify-content:space-between;flex-wrap:wrap;font:11px/1.6 ui-monospace,monospace;color:var(--muted);margin-bottom:10px}.audit-meta span{font-family:system-ui}.before,.after{font-size:14px;padding:10px 14px}.before{color:var(--muted);background:#f6f3ee}.after{color:var(--ink);background:var(--soft);border-left:3px solid var(--green)}.audit-card small{display:block;font-size:10px;letter-spacing:.08em;margin-bottom:3px}.source{font-size:11px}.audit-card[hidden]{display:none}
.bottom{font-size:12px;color:var(--muted);padding:12px 0 35px}.read-tools{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;margin-bottom:24px}
@media(max-width:1050px){.shell{grid-template-columns:170px minmax(0,1fr);gap:25px;padding:30px 24px}.paper{padding:25px}.demo,.audit{padding:24px}}
@media(max-width:760px){.mast{padding:18px 20px}.shell{display:block;padding:24px 16px}aside{position:static;max-height:none;border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:25px}aside nav{display:none}aside .top-link{display:inline-block;margin:0 15px 0 0}.hero h1{font-size:32px}.roadmap{grid-template-columns:1fr}.roadmap>div{padding:10px 0}.paper,.demo,.audit{padding:19px}.paper h2{font-size:22px}.paper{font-size:15px}.hero p{font-size:14px}}
@media print{aside,.mast,.read-tools,.demo-control,.audit-tools{display:none}.shell{display:block;padding:0}.paper{border:0}.hero h1{font-size:30px}.audit{break-before:page}.paper h2{break-after:avoid}.audit-card{break-inside:avoid}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body>
<header class="mast"><span>VLA / RESEARCH PAPER</span><a href="../README.md">论文工作区 ↗</a></header>
<div class="shell"><aside><strong>中文初稿 · v0.1</strong><a class="top-link" href="#rtc-rl">RTC × RL 时序图</a><a class="top-link" href="#manuscript">阅读正文</a><a class="top-link" href="#audit">78 处逐句改写</a><nav>__NAV__</nav></aside><main>
<header class="hero"><span class="eyebrow">LOCAL CORRECTION → LEARNING → GENERALIST</span>
<h1>面向持续执行 VLA 的<br>局部动作修正与强化学习</h1><div class="rule"></div>
<p>研究目标：让现场的局部尝试成为可重复的动作能力，并进一步沉淀到 VLA。当前先写扎实 RTC 与残差学习的衔接，后续 RL 与架构实验继续形成论文的核心贡献。</p>
<div class="roadmap"><div><em>01</em><b>RTC × RL 衔接</b><small>第一部分 · 方法与系统证据</small></div><div><em>02</em><b>局部专家学习</b><small>算法基线 · 策略实验结构</small></div><div><em>03</em><b>架构与能力回灌</b><small>随实验形成贡献 · 保留扩展</small></div></div>
<div class="read-tools"><a href="manuscript-v0.1.md">完整 Markdown ↓</a><a href="../qa_logs/defensive-writing-audit.md">逐句改写表 ↓</a><a href="../sources/manuscript-evidence-v0.1.json">固定证据索引 ↓</a></div></header>
<section class="demo" id="rtc-rl"><span class="eyebrow">PART I / EXECUTION INTERFACE</span><h2>停止新增修正之后，队列还在执行</h2>
<p>RTC 和实时 VLA 强化学习已有先例。这里展示我们的具体衔接：停止生成、排队完成与结果结算各自有明确时刻。以下为教学时序，非实测轨迹。</p>
<div class="demo-control"><button data-stage="0" aria-pressed="true">① 生成修正</button><button data-stage="1" aria-pressed="false">② 等待尾部</button><button data-stage="2" aria-pressed="false">③ 核验结算</button></div>
<div class="svg-scroll"><svg id="handoff" viewBox="0 0 850 264" role="img" aria-label="生成修正、等待已承诺尾部和核验结果的时间轴">
<line x1="135" y1="40" x2="815" y2="40" stroke="#c0ceca"/><g font-size="12" fill="#61747a"><text x="135" y="25">局部入口</text><text x="378" y="25">停止新增</text><text x="591" y="25">队列尾部完成</text><text x="750" y="25">结果结算</text>
<text x="10" y="93">专家输出</text><text x="10" y="143">实际执行</text><text x="10" y="193">学习记录</text></g>
<g fill="#2b5d9a"><rect x="135" y="68" width="262" height="36" rx="5"/><rect x="135" y="118" width="474" height="36" rx="5"/></g>
<g fill="#e9eee9"><rect x="399" y="68" width="416" height="36" rx="5"/><rect x="611" y="118" width="204" height="36" rx="5"/></g>
<g fill="#197563"><rect x="135" y="168" width="262" height="36" rx="5"/><rect x="399" y="168" width="210" height="36" rx="5"/><rect x="611" y="168" width="204" height="36" rx="5"/></g>
<g text-anchor="middle" font-size="13" fill="white"><text x="266" y="91">有界残差</text><text x="372" y="141">已有队列＋承诺修正</text><text x="266" y="191">决策转移</text><text x="504" y="191">等待转移</text><text x="713" y="191">终态证据 → 奖励</text></g>
<g fill="#20363c" font-size="13" text-anchor="middle"><text x="607" y="91">零新增残差</text><text x="713" y="141">Pi 继续执行</text></g>
<path id="cursor" d="M265 43 V220" stroke="#ac662b" stroke-width="2" stroke-dasharray="4 4"/><text x="135" y="248" font-size="12" fill="#61747a">排队期间继续传播价值；actor 与温度跳过该状态的更新。</text></svg></div>
<div class="step-copy" aria-live="polite"><strong id="stage-title"></strong><p id="stage-copy"></p></div></section>
<article class="paper" id="manuscript">__BODY__</article>
<section class="audit" id="audit"><span class="eyebrow">EDITING / CLAIM → EVIDENCE</span><h2>78 处表达，逐处重组</h2><p>范围：原研究稿 30 处、实验记录 24 处、当前 HTML 24 处。保留事实与测量条件，把预防性辩解改为机制、定义和实验职责。历史档案保留原文。</p>
<div class="audit-tools"><input id="audit-search" type="search" aria-label="搜索原文、改写或修改原因" placeholder="搜索：不能证明 / 奖励 / RTC…"><span id="audit-count">78 / 78 处</span></div>__AUDIT__</section>
<footer class="bottom">独立设计 · 2026-10-13 · 实现快照 22fcebf，归档部署 e21431b。系统核验与策略结果分开呈现。<br><a href="../experiments/20261013-rtc-rl/">原始实验札记</a> · <a href="../literature-review.md">文献调研</a> · <a href="../05_style_guide.md">写作规范</a></footer>
</main></div><script>
const stages=[
["生成修正：只编辑尚未承诺的后缀","局部专家根据当前观测、参考计划和队列状态输出残差；正负方向分别计算可行幅度。",265],
["停止新增：已承诺的修正继续完成","进入 draining 后，新增残差为零。原 RTC 前缀保持，Pi 继续执行；这段真实等待也进入 replay。",500],
["完成排队：让结果回到对应尝试","残差尾部完成并留出观测余量后核验结果。终止奖励发放一次，价值沿此前转移传播。",745]
];
function stage(i){document.getElementById("stage-title").textContent=stages[i][0];document.getElementById("stage-copy").textContent=stages[i][1];document.getElementById("cursor").setAttribute("d","M"+stages[i][2]+" 43 V220");document.querySelectorAll("[data-stage]").forEach(b=>b.setAttribute("aria-pressed",String(Number(b.dataset.stage)===i)));}
document.querySelectorAll("[data-stage]").forEach(b=>b.addEventListener("click",()=>stage(Number(b.dataset.stage))));stage(0);
document.getElementById("audit-search").addEventListener("input",function(){const q=this.value.toLowerCase().trim();let n=0;document.querySelectorAll(".audit-card").forEach(card=>{card.hidden=!card.dataset.search.toLowerCase().includes(q);if(!card.hidden)n++;});document.getElementById("audit-count").textContent=n+" / 78 处";});
</script></body></html>'''
page=page.replace("__NAV__",nav_html).replace("__BODY__",body).replace("__AUDIT__","".join(cards))
(ROOT/"drafts/manuscript-v0.1.html").write_text(page)
print(json.dumps({"html_bytes":len(page.encode()),"audit_count":len(cards),"headings":len(nav)},ensure_ascii=False))
