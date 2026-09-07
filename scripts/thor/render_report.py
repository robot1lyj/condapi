"""Render a self-contained Chinese dashboard from measured experiment JSON."""

# Chinese UI copy intentionally uses Chinese punctuation.
# ruff: noqa: RUF001

import argparse
import html
import json
from pathlib import Path
import re


def latency_chart(experiments):
    """Fastest eight complete calls; original measurements remain in the table."""
    measured = sorted(
        (row for row in experiments if row.get("p50_ms") is not None and row.get("p95_ms") is not None),
        key=lambda row: row["p50_ms"],
    )[:8]
    if not measured:
        return ""
    limit = max(120, max(row["p95_ms"] for row in measured) * 1.15)
    scale, left = 620 / limit, 180
    height = 52 + len(measured) * 42
    bars = []
    for index, row in enumerate(measured):
        y = 32 + index * 42
        label = html.escape(row["name"].split(" · ")[0], quote=True)
        p50, p95 = row["p50_ms"], row["p95_ms"]
        color = "#6057e7" if index == 0 else "#a6a2e8"
        bars.append(
            f'<text x="12" y="{y + 16}" fill="#172139">{label}</text>'
            f'<rect x="{left}" y="{y}" width="{p50 * scale:.2f}" height="24" rx="5" fill="{color}"/>'
            f'<line x1="{left + p95 * scale:.2f}" x2="{left + p95 * scale:.2f}" '
            f'y1="{y - 2}" y2="{y + 26}" stroke="#172139" stroke-width="2"/>'
            f'<text x="815" y="{y + 16}" fill="#172139">{p50:.2f} / {p95:.2f} ms</text>'
        )
    target = left + 100 * scale
    return (
        '<section class="panel"><h2>延迟速览 · 最快八组</h2>'
        '<p class="sub">条形为完整调用 P50，黑线为 P95；虚线是 100ms 目标。只展示已完成实测，不表示任务精度通过。</p>'
        '<div class="tablewrap">'
        f'<svg role="img" aria-label="最快配置完整推理延迟与100毫秒目标比较" viewBox="0 0 1010 {height}" '
        'style="width:100%;min-width:700px;font:13px system-ui,sans-serif">'
        f'<line x1="{target:.2f}" x2="{target:.2f}" y1="20" y2="{height - 12}" '
        'stroke="#c35935" stroke-dasharray="4 4"/>'
        f'<text x="{target:.2f}" y="14" text-anchor="middle" fill="#c35935">100 ms</text>'
        + "".join(bars)
        + "</svg></div></section>"
    )


def render(data: dict) -> str:
    def esc(value):
        return html.escape(str(value), quote=True)

    def metric(row, key, unit=""):
        value = row.get(key)
        if key == "max_abs_error" and value is not None:
            return f"{float(value):.6g}{unit}"
        return "—" if value is None else f"{float(value):.3f}{unit}"

    rows = []
    for row in data.get("experiments", []):
        status = row.get("status", "待测试")
        # Missing metrics must stay missing; never substitute theoretical or community numbers.
        rows.append(
            f"<tr><td><strong>{esc(row['name'])}</strong><small>{esc(row.get('scope', ''))}</small></td>"
            f"<td>{esc(row['weights'])}</td><td>{esc(row['compute'])}</td>"
            f'<td><span class="badge">{esc(status)}</span></td>'
            f"<td>{metric(row, 'p50_ms')}</td><td>{metric(row, 'p95_ms')}</td>"
            f"<td>{metric(row, 'max_abs_error')}</td><td>{esc(row.get('note', ''))}</td></tr>"
        )
    facts = "".join(
        f'<article class="fact"><div>{esc(item["label"])}</div><strong>{esc(item["value"])}</strong>'
        f"<small>{esc(item.get('detail', ''))}</small></article>"
        for item in data.get("facts", [])
    )
    timeline = "".join(
        f'<li><span class="dot"></span><strong>{esc(item["title"])}</strong><p>{esc(item["detail"])}</p></li>'
        for item in data.get("stages", [])
    )
    notices = "".join(f"<li>{esc(item)}</li>" for item in data.get("limitations", []))
    detail_tables = []
    for table in data.get("detail_tables", []):
        columns = "".join(f"<th>{esc(value)}</th>" for value in table["columns"])
        cells = "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in table["rows"])
        detail_tables.append(
            f'<section class="panel"><h2>{esc(table["title"])}</h2><p class="sub">{esc(table.get("description", ""))}</p>'
            f'<div class="tablewrap"><table><thead><tr>{columns}</tr></thead><tbody>{cells}</tbody></table></div></section>'
        )
    recommendations = "".join(
        f"<li><strong>{esc(item['title'])}</strong><p>{esc(item['detail'])}</p></li>"
        for item in data.get("recommendations", [])
    )
    sources = "".join(
        f'<li><a href="{esc(item["url"])}" rel="noreferrer">{esc(item["title"])}</a>：{esc(item["note"])}</li>'
        for item in data.get("sources", [])
        if item["url"].startswith("https://")
    )
    next_plan = (
        f'<section class="panel"><h2>下一轮推理测试方案</h2><ol>{recommendations}</ol>'
        f"<h2>社区依据（不混入本机实测数字）</h2><ul>{sources}</ul></section>"
        if recommendations or sources
        else ""
    )
    source = esc(json.dumps(data, ensure_ascii=False, indent=2))
    template = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thor · Pi0.5 推理实验室</title><style>
:root{--bg:#f4f6fb;--ink:#172139;--muted:#66738b;--line:#e4e8f1;--accent:#6057e7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 system-ui,"Noto Sans CJK SC",sans-serif}
header{background:#172139;color:white;padding:42px max(6vw,24px) 56px;position:relative;overflow:hidden}
header:after{content:"π";position:absolute;right:7%;top:-60px;font:280px Georgia;color:#ffffff08}
.eyebrow{font-size:12px;letter-spacing:3px;color:#b2acf9}h1{font-size:36px;margin:12px 0 8px;letter-spacing:-1px}
header p{color:#bcc6d9;margin:0;max-width:820px}.meta{font-size:12px;color:#9eabc2;margin-top:18px}
main{max-width:1400px;margin:-22px auto 48px;padding:0 28px;position:relative}.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.fact,.panel{background:white;border:1px solid var(--line);border-radius:16px;box-shadow:0 5px 22px #18223805}
.fact{padding:22px}.fact>div{color:var(--muted);font-size:13px}.fact strong{display:block;font-size:25px;margin:6px 0}
small{display:block;color:var(--muted);font-size:12px}.panel{margin-top:22px;padding:26px}h2{font-size:19px;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 22px;font-size:13px}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:950px;text-align:left}
th{background:#f7f8fc;color:var(--muted);font-size:12px;font-weight:500}th,td{padding:15px 12px;border-bottom:1px solid var(--line)}td{font-size:13px}
td:last-child{max-width:290px}.badge{white-space:nowrap;border-radius:20px;background:#eeecff;color:#5b52c9;padding:4px 10px;font-size:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.timeline{list-style:none;padding:0;margin:22px 0 0}.timeline li{position:relative;padding:0 0 22px 25px;border-left:1px solid var(--line);margin-left:5px}
.timeline p{font-size:13px;color:var(--muted);margin:6px 0 0}.dot{position:absolute;width:9px;height:9px;border-radius:50%;background:var(--accent);left:-5px;top:8px}
.notice{background:#fffbef;border:1px solid #f2e2af;padding:18px;border-radius:12px;font-size:13px}.notice ul{padding-left:20px;margin:8px 0}
details{margin-top:22px}summary{cursor:pointer;color:var(--accent)}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fc;padding:16px;border-radius:10px}
footer{color:var(--muted);font-size:12px;margin-top:20px;text-align:center}@media(max-width:800px){.facts{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr;gap:0}h1{font-size:28px}main{padding:0 14px}.panel{padding:18px}}
</style><header><div class="eyebrow">EDGE INFERENCE / 可复现实测</div><h1>Thor · Pi0.5 推理实验室</h1>
<p>原始 JAX 权重 → 精度对照 → 真实样本离线回放。把可运行、数值一致与任务有效分开验收。</p>
<div class="meta">更新于 UPDATE · 基础模型 pi05_base · 不连接机械臂执行动作</div></header>
<main><section class="facts">FACTS</section>LATENCY_CHART<section class="panel"><h2>精度配置对照</h2>
<p class="sub">参数存储精度不等于计算精度。延迟以预热后完整 policy 调用为准，首次编译单独记录；“—”表示尚无实测数据。</p>
<div class="tablewrap"><table><thead><tr><th>配置</th><th>权重</th><th>计算</th><th>状态</th><th>P50 / ms</th><th>P95 / ms</th><th>最大绝对误差</th><th>说明</th></tr></thead><tbody>ROWS</tbody></table></div></section>
DETAIL_TABLES
<div class="grid"><section class="panel"><h2>执行进度与证据</h2><ul class="timeline">TIMELINE</ul></section>
<section class="panel"><h2>结果解释边界</h2><p class="sub">性能数据只回答“多快”，不自动回答“动作是否正确”。</p>
<div class="notice"><strong>当前限制</strong><ul>NOTICES</ul></div>
<details><summary>展开原始记录 JSON</summary><pre>SOURCE</pre></details></section></div>
NEXT_PLAN
<footer>本页为离线静态报告，无外部 CDN、无遥测。每次采集新结果后重新生成，不会自动冒充实时状态。</footer></main></html>"""
    replacements = {
        "UPDATE": esc(data["updated_at"]),
        "FACTS": facts,
        "LATENCY_CHART": latency_chart(data.get("experiments", [])),
        "ROWS": "".join(rows),
        "TIMELINE": timeline,
        "NOTICES": notices,
        "SOURCE": source,
        "DETAIL_TABLES": "".join(detail_tables),
        "NEXT_PLAN": next_plan,
    }
    # A single substitution avoids interpreting placeholder words inside data.
    return re.sub("|".join(replacements), lambda match: replacements[match[0]], template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")


if __name__ == "__main__":
    main()
