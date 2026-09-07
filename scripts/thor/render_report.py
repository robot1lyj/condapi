"""Render a self-contained Chinese dashboard from measured experiment JSON."""

# Chinese UI copy intentionally uses Chinese punctuation.
# ruff: noqa: RUF001

import argparse
import html
import json
from pathlib import Path


def render(data: dict) -> str:
    def esc(value):
        return html.escape(str(value), quote=True)

    def metric(row, key, unit=""):
        value = row.get(key)
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
    source = esc(json.dumps(data, ensure_ascii=False, indent=2))
    return (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
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
<main><section class="facts">FACTS</section><section class="panel"><h2>精度配置对照</h2>
<p class="sub">参数存储精度不等于计算精度。延迟以预热后完整 policy 调用为准，首次编译单独记录；“—”表示尚无实测数据。</p>
<div class="tablewrap"><table><thead><tr><th>配置</th><th>权重</th><th>计算</th><th>状态</th><th>P50 / ms</th><th>P95 / ms</th><th>最大绝对误差</th><th>说明</th></tr></thead><tbody>ROWS</tbody></table></div></section>
<div class="grid"><section class="panel"><h2>执行进度与证据</h2><ul class="timeline">TIMELINE</ul></section>
<section class="panel"><h2>结果解释边界</h2><p class="sub">性能数据只回答“多快”，不自动回答“动作是否正确”。</p>
<div class="notice"><strong>当前限制</strong><ul>NOTICES</ul></div>
<details><summary>展开原始记录 JSON</summary><pre>SOURCE</pre></details></section></div>
<footer>本页为离线静态报告，无外部 CDN、无遥测。每次采集新结果后重新生成，不会自动冒充实时状态。</footer></main></html>""".replace(
            "UPDATE", esc(data["updated_at"])
        )
        .replace("FACTS", facts)
        .replace("ROWS", "".join(rows))
        .replace("TIMELINE", timeline)
        .replace("NOTICES", notices)
        .replace("SOURCE", source)
    )


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
