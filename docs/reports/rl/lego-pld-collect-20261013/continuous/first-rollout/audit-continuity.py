"""Recorded command continuity/scope audit; no training or reward publication."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import av
import numpy as np
from PIL import Image, ImageDraw
from yam_pld_rl.native_recording import read_episode
from yam_pld_rl.curator import native_replies
from yam_pld_rl.collector import validate_collector_trace
from yam_pld_rl.block_contract import decode
from yam_pld_rl.kinematics import YamKinematics

out = Path(__file__).resolve().parent
folder = out/'raw-v2/session_20261014_110716_0aa69a/episode_000001'
rows, events, _ = read_episode(folder)
origins, canceled = native_replies(rows, events)
traces = sorted([r['reply']['pld'] for r in origins.values() if r['reply'].get('pld')
                 and r['reply'].get('discarded') is False and not r['reply'].get('error')],
                key=lambda p: (p['epoch'], p['start_tick']))
m = json.loads(Path('/home/wuyan-lyj/condapi-data/rl/pld-continuous-20261014-r1/bundle-r3/manifest.json').read_text())
c = m['contract']
fk = YamKinematics(Path.cwd()/'third_party/yam-fk-model/yam_linear4310.xml')
groups = defaultdict(list)
decode_checks = 0
boundary = []
for p in traces:
    if not p['gate']['active']:
        continue
    validate_collector_trace(p, m)
    decoded = decode(p['physical_base'], p['action'], p['arm'], p['delay_steps'], c,
                     context=p['trajectory_context'])
    np.testing.assert_array_equal(decoded, p['physical_actions'])
    groups[p['window_id']].append(p)
    decode_checks += 1
    d = p['delay_steps']; off = 0 if p['arm']=='left' else 7
    if p['gate'].get('apply_residual') and p['trajectory_context']['reason'] is None and p['trajectory_projection']['reason'] != 'bridge_outside_grasp_workspace':
        q=np.asarray(p['physical_actions'])[:,off:off+6]
        boundary.append(float(abs((q[d]-q[d-1])-(q[d-1]-q[d-2])).max()))
t=np.array([r['time'] for r in rows]); t-=t[0]
actual=np.array([r['submitted_action'] for r in rows]); base=actual.copy()
valid=np.zeros(len(rows),bool); delta=np.zeros_like(actual); ids=np.full(len(rows),-1)
arm_masks={a:np.zeros(len(rows),bool) for a in ('left','right')}
scope_errors=[]
for i,r in enumerate(rows):
    sel=r['details'].get('policy_selection') or {}; token=sel.get('request')
    if not token: continue
    p=origins[(token['epoch'],str(token['request_id']))]['reply']['pld'];k=sel['model_index']
    np.testing.assert_allclose(actual[i],p['physical_actions'][k],atol=1e-6,rtol=0)
    base[i]=p['physical_base'][k];delta[i]=actual[i]-base[i];valid[i]=True;ids[i]=p['request_id']
    if np.max(abs(delta[i]))<=1e-6: continue
    a=p['arm'];off=0 if a=='left' else 7;arm_masks[a][i]=True
    xyz=fk.position(actual[i,off:off+6]);lo,hi=c['grasp_z_range_m'][a]
    if xyz[0]>.40+1e-7 or xyz[2]<lo-1e-7 or xyz[2]>hi+.12+1e-7:
        scope_errors.append(i)
def quant(v):
    return None if not len(v) else dict(zip(('p50','p95','max'),map(float,np.quantile(v,[.5,.95,1]))))
summary=[]
for wid,ps in groups.items():
    ticks=[r['tick'] for r in rows]
    arm=ps[0]['arm']; off=0 if arm=='left' else 7
    selected=[i for i,r in enumerate(rows) if r['epoch']==ps[0]['epoch'] and any(
        r['tick']>=p['start_tick'] and r['tick']<p['start_tick']+50 and str(ids[i])==str(p['request_id'])
        for p in ps)]
    selected=[i for i in selected if arm_masks[arm][i]]
    nz=[p for p in ps if p.get('residual_applied')]
    summary.append({'window_id':wid,'arm':arm,'start_s':(ps[0]['start_tick']-rows[0]['tick'])/30,
                    'direct_effect_start_s':float(t[selected[0]]) if selected else None,
                    'direct_effect_end_s':float(t[selected[-1]]) if selected else None,
                    'correcting_proposals':sum(p['gate'].get('apply_residual') is True for p in ps),
                    'nonzero_proposals':len(nz),'direct_corrected_ticks':len(selected),
                    'no_correction_reasons':dict(Counter(p['trajectory_context']['reason'] for p in ps if not p['residual_applied'])),
                    'stop_reasons':[p['gate'].get('stop_reason') for p in ps if p['gate'].get('stop_reason')]})
timelines=[]
for arm,off in (('left',0),('right',7)):
    mask=arm_masks[arm]
    dist=np.linalg.norm(fk.positions(actual[mask,off:off+6])-fk.positions(base[mask,off:off+6]),axis=1)*1000
    vel=np.diff(actual[:,off:off+6],axis=0);acc=np.diff(actual[:,off:off+6],n=2,axis=0)
    continuous=(valid[:-1]&valid[1:])
    continuous2=valid[:-2]&valid[1:-1]&valid[2:]
    timelines.append({'arm':arm,'direct_nonzero_ticks':int(mask.sum()),'direct_fk_mm':quant(dist),
                       'joint_step_max_rad':float(abs(vel[continuous]).max()),
                       'command_second_difference_rms_rad':float(np.sqrt(np.mean(acc[continuous2]**2)))})
result={'episode_id':rows[0]['episode_id'],'duration_s':float(t[-1]),'active_trace_decoder_checks':decode_checks,
        'scope_violations_actual_corrected_commands':scope_errors,'first_editable_velocity_discontinuity_rad':quant(boundary),
        'windows':summary,'windows_by_arm':dict(Counter(x['arm'] for x in summary)),
        'nonzero_windows_by_arm':dict(Counter(x['arm'] for x in summary if x['direct_corrected_ticks'])),
        'actual_direct_effects':timelines,'training_executed':False,'rewards_published':False,
        'counterfactual_pure_pi_run':False,'automated_scope_is_kinematic_not_visual_semantics':True}
(out/'continuity-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
np.savez_compressed(out/'continuity-signals.npz',t=t,actual=actual,base=base,delta=delta,valid=valid)
# Whole-episode top overview plus before/end scenes, using recorded frame indices.
target=[int(np.argmin(abs(t-s))) for s in np.linspace(0,t[-1],24)]
canvas=Image.new('RGB',(4*320,6*262),'white');draw=ImageDraw.Draw(canvas)
jobs=defaultdict(dict)
for j,i in enumerate(target): jobs[rows[i]['videos']['top']][rows[i]['video_indices'][0]]=(j,i)
for video,picks in jobs.items():
    with av.open(video) as container:
        container.streams.video[0].codec_context.thread_count=1
        for k,frame in enumerate(container.decode(video=0)):
            if k not in picks: continue
            j,i=picks[k];x=(j%4)*320;y=(j//4)*262
            canvas.paste(frame.to_image().resize((320,240)),(x,y));draw.text((x+5,y+242),f'{t[i]:.2f}s',fill='black')
canvas.save(out/'top-timeline.jpg',quality=92)
print(json.dumps({k:v for k,v in result.items() if k!='windows'},indent=2))
