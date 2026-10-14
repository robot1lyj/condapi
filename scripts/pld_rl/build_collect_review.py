"""Build the reusable suggestion-only review UI for an immutable collect batch."""

# ruff: noqa: RUF001

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.review_policy import automatic_reward

TEMPLATE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PLD-Grasp · 奖励复核</title><style>
body{font:15px system-ui;margin:0;background:#f1f5f9;color:#172033}header,main{max-width:1300px;margin:auto;padding:20px}header{position:sticky;top:0;background:#f1f5f9f2;z-index:1}h1{font-size:23px;margin:0 0 8px}button,input,select{font:inherit;padding:7px;margin:3px;border:1px solid #bbc7d8;border-radius:6px}button{cursor:pointer;background:white}article{background:white;border:1px solid #dbe3ef;border-radius:12px;padding:18px;margin:0 0 20px}.images{display:flex;overflow:auto;gap:10px}.images figure{margin:8px 0;min-width:240px;max-width:320px}.images img{width:100%;border-radius:6px}figcaption,.muted{font-size:13px;color:#52627b}label{display:inline-block;margin:6px}h2{font-size:18px}#count{font-weight:600}a{color:#1d4ed8}textarea{width:95%;font:inherit;padding:6px}.error{color:#b91c1c}</style>
<header><h1>PLD-Grasp · 仅复核无建议片段</h1><div>有建议的结果直接作为自动奖励，包括弱证据；人工修改优先。抓起持住=1，明确失败=0，整轮C/N仅用于评估。</div>
<div><input id="search" placeholder="搜索 E01 / L / 编号"><select id="filter"><option value="pending">尚无建议、待复核</option><option value="unresolved">所有无建议片段</option><option value="done">人工已复核</option><option value="automatic">自动采用的建议（可选查看）</option></select><button id="prev">上一页</button><button id="next">下一页</button><span id="count"></span></div>
<div><input id="reviewer" value="用户复核" aria-label="审核人"><span id="save">正在读取复核记录…</span> · <a href="/api/rewards" target="_blank">查看奖励与来源</a></div><div class="muted" id="batch"></div></header><main id="cards"></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),key='pld-review:'+D.identity,legacyKey='pld-collect13:'+D.identity;
let saved={},page=0;
const labels={'':'未确认',held:'抓起并持住（1）',empty:'抬升后空手（0）',not_lifted:'有抓取意图但未抬起（0）',no_closure:'有抓取意图但未闭合（0）',dropped:'抓住后提前丢失（0）',exclude:'排除，不训练'};
function el(t,text){const e=document.createElement(t);if(text!==undefined)e.textContent=text;return e}
function valid(c){const s=saved[c.candidate_id];return s?.reward_valid&&s.status}
function matches(){const q=document.getElementById('search').value.toLowerCase(),f=document.getElementById('filter').value;return D.cards.filter(c=>c.candidate_id.toLowerCase().includes(q)&&(f==='pending'&&!c.automatic&&!valid(c)||f==='unresolved'&&!c.automatic||f==='done'&&saved[c.candidate_id]?.status||f==='automatic'&&c.automatic))}
function counts(){const auto=D.cards.filter(c=>c.automatic&&!saved[c.candidate_id]?.status).length,human=D.cards.filter(valid).length,pending=D.cards.filter(c=>!c.automatic&&!valid(c)).length;document.getElementById('count').textContent=`自动建议 ${auto} · 人工已复核 ${human} · 待复核 ${pending} · 第${page+1}页`}
async function send(changes){const r=await fetch('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({identity:D.identity,changes})});const x=await r.json();if(!r.ok)throw Error(x.error);Object.assign(saved,x.reviews);localStorage.setItem(key,JSON.stringify(saved));return x}
function render(){const list=matches();page=Math.max(0,Math.min(page,Math.ceil(list.length/6)-1));counts();const root=document.getElementById('cards');root.replaceChildren();if(!list.length)root.append(el('p','当前队列没有需要复核的片段。'));
 for(const c of list.slice(page*6,page*6+6)){
  const s=saved[c.candidate_id]||{},a=el('article');a.append(el('h2',`${c.candidate_id} · ${c.arm} · 窗口 ${c.outcome.window_id}`));
  a.append(el('p',`建议：${c.suggested?labels[c.suggested]:'无可用建议，请人工确认'}；${c.confidence||'无模型判定'}。${c.reason||''}`));
  a.append(el('p',`终点 ${c.outcome.reason}，最大抬升 ${(100*c.outcome.max_lift_m).toFixed(1)}cm；${c.flags.join(' / ')||c.pending_reason||'自动采用模型建议'}。`));
  const images=el('div');images.className='images';for(const f of c.images){const fig=el('figure'),im=el('img');im.src=f.path;im.loading='lazy';im.alt=c.candidate_id+' '+f.caption;fig.append(im,el('figcaption',f.caption));images.append(fig)}a.append(images);
  const link=el('a','打开完整窗口证据图');link.href=c.montage;link.target='_blank';a.append(link);
  const sel=el('select');sel.setAttribute('aria-label','抓取结果');for(const [v,t]of Object.entries(labels)){const o=el('option',t);o.value=v;sel.append(o)}sel.value=s.status||'';
  const intent=el('input');intent.type='checkbox';intent.checked=!!s.grasp_intent_confirmed;const il=el('label');il.append(intent,document.createTextNode('确认是抓积木，而非调整箱子'));
  const loss=el('input');loss.type='checkbox';loss.checked=!!s.held_then_lost_confirmed;const ll=el('label');ll.append(loss,document.createTextNode('确认先持物、后提前丢失'));
  const note=el('textarea');note.rows=1;note.placeholder='复核说明';note.value=s.note||'';
  const error=el('p',s.error||'');error.className='error';
  async function update(){const v={status:sel.value,grasp_intent_confirmed:intent.checked,held_then_lost_confirmed:loss.checked,note:note.value,reviewer:document.getElementById('reviewer').value};saved[c.candidate_id]=v;localStorage.setItem(key,JSON.stringify(saved));const msg=document.getElementById('save');msg.textContent='正在保存…';try{await send({[c.candidate_id]:v});error.textContent=saved[c.candidate_id].error||'';msg.textContent='已保存到数据集文件';counts()}catch(e){msg.textContent='保存失败：'+e.message}}
  for(const e of [sel,intent,loss,note])e.onchange=update;a.append(el('div'),sel,il,ll,note,error);root.append(a);
 }}
for(const id of ['search','filter'])document.getElementById(id).oninput=()=>{page=0;render()};document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};
(async()=>{try{const r=await fetch('/api/reviews');if(!r.ok)throw Error('复核服务未启动');const x=await r.json();if(x.identity!==D.identity)throw Error('证据版本不匹配');saved=x.reviews;const old={...JSON.parse(localStorage.getItem(legacyKey)||'{}'),...JSON.parse(localStorage.getItem(key)||'{}')},migrate={};for(const [id,s]of Object.entries(old))if(s.status&&!saved[id]&&D.cards.some(c=>c.candidate_id===id))migrate[id]=s;if(Object.keys(migrate).length)await send(migrate);document.getElementById('save').textContent='已保存到数据集文件；旧复核记录已保留';document.getElementById('batch').textContent=D.batch;render()}catch(e){document.getElementById('save').textContent=e.message}})();
</script></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--curator", default="curator-v2")
    p.add_argument("--vision", default="vision-v1")
    p.add_argument("--output", default="review.html")
    args = p.parse_args()
    root = args.dataset
    target = root / args.output
    if target.exists():
        p.error("write a new view file or archive the previous view before rebuilding")
    candidates_path = root / args.curator / "candidates.json"
    candidates = json.loads(candidates_path.read_text())
    judged = {c["candidate_id"]: c for c in json.loads((root / args.vision / "candidates.json").read_text())}
    cards = []
    for c in candidates:
        if not c.get("outcome"):
            continue
        cid, proof, images = c["candidate_id"], c["outcome"], []
        context = c["frames"][:1]
        end = [f for f in c["frames"] if f["row"] in proof["fresh_rows"]]
        for f in context + end[-3:]:
            for view in (("top", c["arm"]) if f in context else (c["arm"],)):
                if view in f["views"]:
                    images.append({"path": args.curator + "/" + f["views"][view]["image"], "caption":  # noqa: PERF401
                                   f"{view} row={f['row']} t={f['since_close_s']:+.2f}s dz={100*f['lift_from_closure_m']:.1f}cm"})
        result = judged.get(cid, {})
        automatic, reason = automatic_reward(c, result)
        cards.append({"candidate_id": cid, "candidate_sha256": c["candidate_sha256"], "episode_id": c["episode_id"],
                      "arm": c["arm"], "outcome": proof, "images": images, "montage": args.curator + "/" + c["montage"],
                      "automatic": bool(automatic), "pending_reason": reason,
                      "suggested": result.get("suggested_status") or {"success": "held", "failure": "empty"}.get(result.get("suggested_label")),
                      "confidence": result.get("confidence"), "reason": " ".join(result.get("reasons", [])),
                      "flags": result.get("audit_flags", [])})
    identity = hashlib.sha256(candidates_path.read_bytes()).hexdigest()
    data = json.dumps({"identity": identity, "cards": cards, "batch": root.name}, ensure_ascii=False).replace("<", "\\u003c")
    target.write_text(TEMPLATE.replace("__DATA__", data))
    print(json.dumps({"page": str(target), "cards": len(cards), "automatic": sum(c["automatic"] for c in cards),
                      "no_suggestion": sum(not c["automatic"] for c in cards)}))


if __name__ == "__main__":
    main()
