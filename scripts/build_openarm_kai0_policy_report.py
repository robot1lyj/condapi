"""Build a self-contained HTML report for the completed OpenArm KAI0 K-Policy run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
from typing import Any


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_text_atomic(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _report_payload(
    *,
    metrics_path: pathlib.Path,
    selection_path: pathlib.Path,
    deployment_path: pathlib.Path,
    hq_audit_path: pathlib.Path,
    site_selection_path: pathlib.Path,
    k_data_report_path: pathlib.Path,
    k_data_audit_path: pathlib.Path,
) -> dict[str, Any]:
    selection = _load_json(selection_path)
    deployment = _load_json(deployment_path)
    hq_audit = _load_json(hq_audit_path)
    site_selection = _load_json(site_selection_path)
    k_data = _load_json(k_data_report_path)
    k_data_audit = _load_json(k_data_audit_path)
    metrics = _load_jsonl(metrics_path)
    if not metrics:
        raise ValueError(f"Training metrics are empty: {metrics_path}")
    if not selection.get("candidates"):
        raise ValueError("Policy selection has no checkpoint candidates")
    if not hq_audit.get("passed") or not k_data_audit.get("passed"):
        raise ValueError("Cannot publish K-Policy report before HQ and K-Data audits pass")
    if deployment.get("checkpoint") != selection.get("selected_checkpoint"):
        raise ValueError("Deployment checkpoint does not match the selected checkpoint")
    return {
        "schema_version": "openarm_kai0_policy_report_v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "training_metrics": metrics,
        "selection": selection,
        "deployment": deployment,
        "hq_audit": hq_audit,
        "site_selection": site_selection,
        "k_data": k_data,
        "k_data_audit": k_data_audit,
    }


def _render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenArm K-Policy 训练报告</title>
<style>
:root{{--ink:#17202a;--muted:#66727f;--line:#dce2e8;--panel:#fff;--bg:#f4f6f8;--blue:#1769aa;--green:#16835f;--amber:#a96b08;--red:#b63e3e}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Arial,"Noto Sans SC",sans-serif;letter-spacing:0}}
main{{max-width:1240px;margin:0 auto;padding:28px 22px 48px}} header{{padding:4px 0 22px;border-bottom:1px solid var(--line)}}
h1{{font-size:28px;margin:0 0 8px}} h2{{font-size:19px;margin:0 0 16px}} p{{margin:0;color:var(--muted)}}
.chips{{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}} .chip,.badge{{border:1px solid var(--line);background:#fff;padding:4px 8px;border-radius:6px;font-size:12px}}
.badge.good{{color:var(--green);border-color:#a8d9c8;background:#f1fbf7}} .badge.pick{{color:var(--blue);border-color:#acd0ed;background:#f2f8fd}}
section{{padding:24px 0;border-bottom:1px solid var(--line)}} .summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;min-height:112px;box-shadow:0 2px 7px rgba(20,32,45,.04)}}
.label{{font-size:12px;color:var(--muted);margin-bottom:7px}} .value{{font-size:20px;font-weight:700;overflow-wrap:anywhere}} .detail{{font-size:12px;color:var(--muted);margin-top:7px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .panel{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px}}
canvas{{display:block;width:100%;height:300px;aspect-ratio:16/7}} table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line)}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{font-size:12px;color:var(--muted);background:#f8fafb}}
tr.selected{{background:#edf7ff}} .scroll{{overflow:auto;border-radius:8px}} .gates{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}
.gate{{background:#fff;border:1px solid var(--line);border-left:4px solid var(--green);border-radius:6px;padding:9px 11px;overflow-wrap:anywhere}} .gate.fail{{border-left-color:var(--red)}}
footer{{padding-top:20px;color:var(--muted);font-size:12px}} @media(max-width:900px){{.summary,.grid2{{grid-template-columns:1fr 1fr}}}} @media(max-width:620px){{main{{padding:18px 12px 36px}}.summary,.grid2,.gates{{grid-template-columns:1fr}}h1{{font-size:23px}}}}
</style>
</head>
<body><main>
<header><h1>OpenArm K-Policy 训练报告</h1><p>路线 K: Stage Advantage 二值 AWBC, 从 pi0.5 base 训练、双域离线选模并部署。</p><div class="chips"><span class="chip" id="generated"></span><span class="chip">absolute_advantage · stage top-30%</span><span class="chip">global batch 128 · 80k</span></div></header>
<section><div class="summary">
<div class="card"><div class="label">选中 checkpoint</div><div class="value" id="selected-step"></div><div class="detail" id="selected-score"></div></div>
<div class="card"><div class="label">正式 K-Data</div><div class="value" id="data-count"></div><div class="detail" id="source-count"></div></div>
<div class="card"><div class="label">Site Stage 路径</div><div class="value" id="site-mode"></div><div class="detail" id="site-checkpoint"></div></div>
<div class="card"><div class="label">gpu25 服务</div><div class="value" id="endpoint"></div><div class="detail" id="deploy-prompt"></div></div>
</div></section>
<section><h2>训练曲线</h2><div class="grid2"><div class="panel"><canvas id="loss-chart"></canvas></div><div class="panel"><canvas id="grad-chart"></canvas></div></div></section>
<section><h2>16 checkpoint 双域比较</h2><div class="grid2"><div class="panel"><canvas id="site-chart"></canvas></div><div class="panel"><canvas id="hq-chart"></canvas></div></div><div class="scroll" style="margin-top:16px"><table><thead><tr><th>step</th><th>Site MAE</th><th>Site critical</th><th>HQ MAE</th><th>HQ critical</th><th>HQ gap</th><th>rank</th></tr></thead><tbody id="candidate-rows"></tbody></table></div></section>
<section><h2>数据与标签分布</h2><div class="scroll"><table><thead><tr><th>来源</th><th>episodes</th><th>stage 0 positive</th><th>stage 0 ratio</th><th>stage 1 positive</th><th>stage 1 ratio</th></tr></thead><tbody id="source-rows"></tbody></table></div></section>
<section><h2>质量闸门</h2><div class="gates" id="gates"></div></section>
<footer id="footer"></footer>
</main><script id="report-data" type="application/json">{data}</script>
<script>
const D=JSON.parse(document.getElementById('report-data').textContent); const $=id=>document.getElementById(id); const f=(v,n=4)=>Number(v).toFixed(n);
$('generated').textContent='生成 '+new Date(D.generated_at).toLocaleString(); $('selected-step').textContent=D.selection.selected_step;
const pick=D.selection.candidates.find(x=>x.step===D.selection.selected_step); $('selected-score').textContent='weighted rank '+f(pick.rank_score,2)+' · Site critical '+f(pick.site_critical_mae);
$('data-count').textContent=D.k_data.total_episodes+' episodes'; $('source-count').textContent=Object.entries(D.k_data.materialized_episode_counts).map(([k,v])=>k+' '+v).join(' · ');
$('site-mode').textContent=D.site_selection.mode; $('site-checkpoint').textContent=(D.site_selection.checkpoint||'').split('/').slice(-2).join('/');
$('endpoint').textContent=D.deployment.host+':'+D.deployment.port; $('deploy-prompt').textContent=D.deployment.prompt;
function chart(id,rows,series,title){{const c=$(id),ctx=c.getContext('2d');function draw(){{const dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const pad={{l:48,r:16,t:30,b:35}},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;ctx.font='12px Arial';ctx.fillStyle='#17202a';ctx.fillText(title,pad.l,17);const xs=rows.map(r=>+r.step),ys=rows.flatMap(r=>series.map(s=>+r[s.key])).filter(Number.isFinite);if(!xs.length||!ys.length)return;let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(ymax===ymin)ymax=ymin+1;ctx.strokeStyle='#dce2e8';ctx.fillStyle='#66727f';for(let i=0;i<=4;i++){{let y=pad.t+ph*i/4,v=ymax-(ymax-ymin)*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillText(v.toPrecision(3),4,y+4)}}series.forEach(s=>{{ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();rows.forEach((r,i)=>{{const x=pad.l+(+r.step-xmin)/(xmax-xmin||1)*pw,y=pad.t+(ymax-+r[s.key])/(ymax-ymin)*ph;i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});ctx.stroke()}});ctx.fillStyle='#66727f';ctx.fillText(xmin,pad.l,h-10);ctx.fillText(xmax,w-pad.r-38,h-10);series.forEach((s,i)=>{{ctx.fillStyle=s.color;ctx.fillRect(pad.l+i*130,h-22,10,3);ctx.fillText(s.label,pad.l+15+i*130,h-17)}})}}new ResizeObserver(draw).observe(c);draw()}}
const M=D.training_metrics; chart('loss-chart',M,[{{key:'loss',label:'loss',color:'#1769aa'}}],'Policy loss');chart('grad-chart',M,[{{key:'grad_norm',label:'grad norm',color:'#a96b08'}}],'Gradient norm');
const C=D.selection.candidates;chart('site-chart',C,[{{key:'site_mae',label:'MAE',color:'#1769aa'}},{{key:'site_critical_mae',label:'critical',color:'#b63e3e'}}],'Site holdout');chart('hq-chart',C,[{{key:'hq_mae',label:'MAE',color:'#16835f'}},{{key:'hq_critical_mae',label:'critical',color:'#a96b08'}}],'HQ holdout');
C.forEach(r=>{{const tr=document.createElement('tr');if(r.step===D.selection.selected_step)tr.className='selected';tr.innerHTML=`<td>${{r.step}} ${{r.step===D.selection.selected_step?'<span class="badge pick">选中</span>':''}}</td><td>${{f(r.site_mae)}}</td><td>${{f(r.site_critical_mae)}}</td><td>${{f(r.hq_mae)}}</td><td>${{f(r.hq_critical_mae)}}</td><td>${{f(r.hq_gap_ratio,2)}}</td><td>${{f(r.rank_score,2)}}</td>`;$('candidate-rows').appendChild(tr)}});
const counts=D.k_data.materialized_label_counts;Object.entries(D.k_data.materialized_episode_counts).forEach(([kind,eps])=>{{const a=counts[kind],tr=document.createElement('tr');tr.innerHTML=`<td>${{kind}}</td><td>${{eps}}</td><td>${{a['0'].positive}}</td><td>${{f(a['0'].positive_ratio,3)}}</td><td>${{a['1'].positive}}</td><td>${{f(a['1'].positive_ratio,3)}}</td>`;$('source-rows').appendChild(tr)}});
const gates={{'HQ Stage final':D.hq_audit.passed,'K-Data loader':D.k_data_audit.passed,'Site score selected':!!D.site_selection.mode,'Deployment checkpoint match':D.deployment.checkpoint===D.selection.selected_checkpoint,'Forced positive prompt':D.deployment.prompt==='Fold the T-shirt properly, Advantage: positive'}};Object.entries(gates).forEach(([k,v])=>{{const e=document.createElement('div');e.className='gate '+(v?'':'fail');e.textContent=(v?'PASS · ':'FAIL · ')+k;$('gates').appendChild(e)}});
$('footer').textContent='报告范围: K-Policy 训练、离线双域选模与部署。数据合同: OpenArm 16D degrees, gripper 0=open / -66=closed。';
</script></body></html>"""


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    payload = _report_payload(
        metrics_path=args.metrics,
        selection_path=args.selection,
        deployment_path=args.deployment,
        hq_audit_path=args.hq_audit,
        site_selection_path=args.site_selection,
        k_data_report_path=args.k_data_report,
        k_data_audit_path=args.k_data_audit,
    )
    payload_path = args.output.parent / "report.json"
    _write_text_atomic(payload_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_text_atomic(args.output, _render_html(payload))
    return {
        "output": str(args.output),
        "payload": str(payload_path),
        "selected_step": payload["selection"]["selected_step"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=pathlib.Path, required=True)
    parser.add_argument("--selection", type=pathlib.Path, required=True)
    parser.add_argument("--deployment", type=pathlib.Path, required=True)
    parser.add_argument("--hq-audit", type=pathlib.Path, required=True)
    parser.add_argument("--site-selection", type=pathlib.Path, required=True)
    parser.add_argument("--k-data-report", type=pathlib.Path, required=True)
    parser.add_argument("--k-data-audit", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
