from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
import av
from yam_pld_rl.native_recording import read_episode
from yam_pld_rl.curator import native_replies,reconstruct_intervals
from yam_pld_rl.kinematics import YamKinematics
from yam_pld_rl.block_contract import decode
from yam_pld_rl.collector import validate_collector_trace
out=Path(__file__).resolve().parent
rows,events,_=read_episode(next((out/'raw-v2').glob('*/episode_000002')))
reply,cancelled=native_replies(rows,events)
traces=sorted([x['reply']['pld'] for x in reply.values() if x['reply'].get('pld') and not x['reply'].get('discarded')],key=lambda p:p['start_tick'])
manifest=json.loads(Path('/home/wuyan-lyj/condapi-data/rl/pld-wide008-20261013-r1/bundle/manifest.json').read_text());c=manifest['contract']
fk=YamKinematics(Path.cwd()/'third_party/yam-fk-model/yam_linear4310.xml')
for p in traces:
 if p['gate']['active']:
  validate_collector_trace(p,manifest)
  np.testing.assert_array_equal(decode(p['physical_base'],p['action'],p['arm'],p['delay_steps'],c),p['physical_actions'])
t=np.array([r['time'] for r in rows]);t-=t[0]
actual=np.asarray([r['submitted_action'] for r in rows]);pi=actual.copy();delta=np.zeros_like(actual)
valid=np.zeros(len(rows),bool);ids=np.full(len(rows),-1);window=np.full(len(rows),'',dtype=object)
for i,row in enumerate(rows):
 sel=row['details'].get('policy_selection') or {};token=sel.get('request')
 if not token:continue
 origin=reply[(token['epoch'],str(token['request_id']))]['reply'];p=origin['pld'];k=sel['model_index']
 np.testing.assert_allclose(actual[i],p['physical_actions'][k],atol=1e-6,rtol=0)
 pi[i]=p['physical_base'][k];delta[i]=actual[i]-pi[i];valid[i]=True;ids[i]=p['request_id'];window[i]=p.get('window_id','')
mask=(np.abs(delta[:,7:13]).max(1)>1e-9)
correcting=[p for p in traces if p['gate'].get('apply_residual') is True]
physical=np.array([np.asarray(p['action'])*np.asarray(c['radius'][7:]) for p in correcting])
centers=np.array([p['actor_physical_center'] for p in correcting])
pairs=[(a,b) for a,b in zip(correcting[:-1],correcting[1:]) if a['window_id']==b['window_id']]
changes=[]
for a,b in pairs:
 da=np.asarray(a['action'])*np.asarray(c['radius'][7:]);db=np.asarray(b['action'])*np.asarray(c['radius'][7:])
 # Meaningful signs only, so a numerical zero doesn't count as a reversal.
 flipped=(da[:6]*db[:6]<0)&(np.abs(da[:6])>.003)&(np.abs(db[:6])>.003)
 changes.append({'from_request':a['request_id'],'to_request':b['request_id'],'period_s':(b['start_tick']-a['start_tick'])/30,'max_joint_peak_change_rad':float(abs(db[:6]-da[:6]).max()),'meaningful_joint_sign_flips':int(flipped.sum())})
max_change=max(changes,key=lambda x:x['max_joint_peak_change_rad'])
spans=[]
for a,b in zip(np.flatnonzero(np.diff(np.r_[False,mask,False]))[::2],np.flatnonzero(np.diff(np.r_[False,mask,False]))[1::2]):
 spans.append({'start_s':float(t[a]),'end_s':float(t[b-1]),'nonzero_ticks':int(b-a),'request_id':int(ids[a]),'peak_fk_mm':float(np.linalg.norm(fk.positions(actual[a:b,7:13])-fk.positions(pi[a:b,7:13]),axis=1).max()*1000)})
windows=[]
for w in dict.fromkeys(p['window_id'] for p in correcting):
 ps=[p for p in traces if p.get('window_id')==w]
 first,last=ps[0],ps[-1]
 lo=first['start_tick']+10;hi=last['start_tick']+10
 wmask=np.array([r['tick']>=lo and r['tick']<hi for r in rows])
 windows.append({'window':w,'first_command_s':float(t[[r['tick'] for r in rows].index(lo)]),'last_command_s':float(t[[r['tick'] for r in rows].index(hi)]),'correcting_blocks':sum(p['gate'].get('apply_residual') is True for p in ps),'direct_corrected_ticks':int((wmask&mask).sum()),'ticks':int(wmask.sum())})
stepmask=valid[1:]&valid[:-1]
accmask=valid[2:]&valid[1:-1]&valid[:-2]
def signal(q):
 vel=np.diff(q[:,7:13],axis=0)*30;acc=np.diff(q[:,7:13],n=2,axis=0)*900
 return {'velocity_rms_rad_s':float(np.sqrt(np.mean(vel[stepmask]**2))),'max_step_rad':float(np.abs(vel[stepmask]).max()/30),'acceleration_rms_rad_s2':float(np.sqrt(np.mean(acc[accmask]**2))),'max_abs_second_difference_rad':float(np.abs(acc[accmask]).max()/900)}
result={'episode_id':rows[0]['episode_id'],'duration_s':float(t[-1]),'correcting_blocks':len(correcting),'windows':windows,'sample_refresh_s_quantiles':np.quantile([x['period_s'] for x in changes],[0,.5,1]).tolist(),'adjacent_pairs':len(changes),'pairs_with_sign_flip':sum(x['meaningful_joint_sign_flips']>0 for x in changes),'max_peak_change':max_change,'joint_peak_rms_rad':float(np.sqrt(np.mean(physical[:,:6]**2))),'actor_center_rms_rad':float(np.sqrt(np.mean(centers[:,:6]**2))),'pulses':spans,'mean_delta_zero_ticks_between_pulses':float(np.mean([(spans[i+1]['start_s']-spans[i]['end_s'])*30-1 for i in range(len(spans)-1) if spans[i+1]['start_s']-spans[i]['end_s']<.5])),'signals':{'actual':signal(actual),'recorded_pi_reference':signal(pi),'residual_only':signal(delta)},'pairs':changes,'policy_origin_and_decoder_exact':True,'independent_counterfactual_pi_trajectory':False,'hardware_fault_excluded_from_diagnosis_by_user':True}
(out/'continuity-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
np.savez_compressed(out/'continuity-signals.npz',t=t,actual=actual,pi=pi,delta=delta,mask=mask,ids=ids,valid=valid)
# Timestamped frames from recorded video, no inferred success labels.
wanted=[int(np.argmin(abs(t-s))) for s in (1.8,2.0,2.13,2.33,2.47,2.67,2.8,3.0,3.13,3.33,3.47,3.67)]
canvas=Image.new('RGB',(4*320,3*260),'white');draw=ImageDraw.Draw(canvas)
lookup={rows[i]['video_indices'][2]:j for j,i in enumerate(wanted)}
with av.open(rows[0]['videos']['right']) as container:
 for k,frame in enumerate(container.decode(video=0)):
  if k in lookup:
   j=lookup[k];im=frame.to_image().resize((320,240));x0=(j%4)*320;y0=(j//4)*260;canvas.paste(im,(x0,y0));draw.text((x0+5,y0+242),f'{t[wanted[j]]:.3f}s / request {ids[wanted[j]]}',fill='black')
canvas.save(out/'right-wrist-sequence.jpg',quality=92)
print(json.dumps({k:v for k,v in result.items() if k not in ('pulses','pairs')},indent=2))
