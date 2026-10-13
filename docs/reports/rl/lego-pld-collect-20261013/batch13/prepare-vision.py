"""Stage causally scoped frozen-model inputs, never approves a reward."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
SRC = OUT / 'curator-v2'
DST = OUT / 'vision-v1'
DST.mkdir(exist_ok=False)
for name in ('images', 'montages', 'reference-images', 'run-scripts'):
    (DST / name).mkdir()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


audit = json.loads((SRC / 'audit.json').read_text())
pending = {(r['window'][1], r['window'][-1])
           for episode in audit for r in episode['association_exclusions']
           if r['reason'].startswith("'E")}
candidates = json.loads((SRC / 'candidates.json').read_text())
prepared, deferred, jobs, copied = [], [], {}, set()
for c in candidates:
    proof = c.get('outcome')
    if not proof or (c['episode_id'], proof['window_id']) not in pending:
        continue
    fresh = proof['fresh_rows']
    if not proof['kinematic_complete'] or not fresh:
        deferred.append({'candidate_id': c['candidate_id'], 'reason': 'missing_complete_fresh_endpoint'}); continue
    lifted = proof['lifted_rows']
    eligible = [f for f in c['frames'] if f['row'] in lifted and c['arm'] in f['views']]
    unique = {(f['views'][c['arm']]['source_video'], f['views'][c['arm']]['video_frame_index']) for f in eligible}
    if fresh[-1] not in lifted or len(unique) < 2:
        deferred.append({'candidate_id': c['candidate_id'], 'reason': 'endpoint_needs_nonlift_or_short_evidence_review',
                         'outcome': proof, 'montage': c.get('montage')}); continue
    assert eligible[-1]['row'] == fresh[-1]
    # Retain entry context and only post-drain lifted snapshots for VLM result evidence.
    context = [f for f in c['frames'] if f['tick'] <= c['close_tick'] and c['arm'] in f['views']]
    selected = ([context[0]] if context else []) + eligible
    rec = {**c, 'frames': selected, 'model_evidence_scope': 'preclosure_context_and_postdrain_lifted_endpoint',
           'original_candidate_sha256': c['candidate_sha256'], 'approved': False}
    prepared.append(rec)
    for f in selected:
        for view, ref in f['views'].items():
            name = ref['image']
            if name not in copied:
                shutil.copyfile(SRC / name, DST / name); copied.add(name)
            if view == c['arm']:
                jobs[ref['key']] = {'id': ref['key'], 'image': name, 'episode_number': c['episode_number'],
                                    'episode_id': c['episode_id'], 'view': view, 'sha256': sha(DST / name)}
    shutil.copyfile(SRC / c['montage'], DST / c['montage'])
old_refs = Path('/home/wuyan-lyj/condapi-data/rl/lego-grasp-reclassified14-20261011-v3r3/reference-labels.json')
references = json.loads(old_refs.read_text())
for ref in references:
    source = Path(ref['image']); name = f"reference-images/REF{ref['number']:03d}.jpg"
    shutil.copyfile(source, DST / name)
    ref.update(original_episode_number=ref['episode_number'], episode_number=1000+ref['episode_number'],
               image=str(DST / name), source_image=str(source), source_sha256=sha(source),
               dataset_namespace='old_unmodified_pi14_20261011')
write(DST / 'reference-labels-staged.json', references)
write(DST / 'image-jobs.json', list(jobs.values()))
write(DST / 'candidates-prepared.json', prepared)
write(DST / 'deferred-endpoints.json', deferred)
for script in ('review_grasps_vlm.py', 'probe_dinov3.py'):
    shutil.copyfile(ROOT / 'scripts/parts_rl' / script, DST / 'run-scripts' / script)
vlm_script = DST / 'run-scripts/review_grasps_vlm.py'
text = vlm_script.read_text()
old = 'Normal release over a bin after carrying is still success.'
assert old in text
text = text.replace(old, 'This is a local grasp-and-hold goal. A success requires the brick to STILL be held in the LAST lifted image. Earlier contact or earlier holding followed by an empty last image is failure. All lifted images are after the queued corrections drained; do not credit later Pi recovery.')
vlm_script.write_text(text)
write(DST / 'preparation.json', {'source_curator_manifest_sha256': sha(SRC / 'manifest.json'),
    'pending_windows': len(pending), 'vlm_candidates': len(prepared), 'deferred_endpoints': len(deferred),
    'source_reference_labels_sha256': sha(old_refs), 'references': len(references),
    'reference_namespace_note': 'Old raw-Pi14 and new collect13 are different episode UUID sets; offset preserves identity separation for legacy numeric reference exclusion.',
    'endpoint_prompt_adaptation': 'Local held-at-latest-frame goal replaces historical normal-release success clause; proposals still require review.',
    'automatic_tracking': False, 'training_executed': False, 'approved_rewards': 0,
    'excluded_candidate_reasons': dict(Counter(r['reason'] for r in deferred))})
print(json.dumps({'candidates': len(prepared), 'deferred': len(deferred), 'images': len(jobs)}), flush=True)
