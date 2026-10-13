"""Read-only CPU audit of a hash-verified collect snapshot; never trains."""
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path

import av
import numpy as np

from yam_pld_rl.curator import native_replies
from yam_pld_rl.kinematics import YamKinematics
from yam_pld_rl.native_recording import read_episode

OUT = Path(__file__).resolve().parent
ROOT = Path.cwd()
SOURCE = json.loads((OUT / 'source-before.json').read_text())
RECEIPT = json.loads((OUT / 'copy-receipt.json').read_text())
assert RECEIPT['pre_post_source_hashes_equal'] and RECEIPT['local_hashes_match']
RAW = OUT / 'raw' / Path(SOURCE['source_root']).name
DEST = OUT / 'batch-audit-v1'
DEST.mkdir(exist_ok=False)
fk = YamKinematics(ROOT / 'third_party/yam-fk-model/yam_linear4310.xml')


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def percentiles(values):
    return dict(zip(('p50', 'p95', 'p99', 'max'), map(float, np.percentile(values, (50, 95, 99, 100))))) if values else None


def error_runs(times, mask):
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [{'start_row': int(a), 'end_row': int(b-1), 'start_s': float(times[a]),
             'end_s': float(times[b-1]), 'duration_s': float(times[b-1]-times[a])}
            for a, b in zip(changes[::2], changes[1::2]) if times[b-1]-times[a] >= .5]


reports = []
for number, entry in enumerate(SOURCE['episodes'], 1):
    folder = RAW / entry['path']
    manifest = json.loads((folder / 'manifest.json').read_text())
    rows, events, hashes = read_episode(folder)
    replies_by_key, cancelled = native_replies(rows, events)
    replies = [r['reply'] for r in replies_by_key.values()]
    traces = [r['pld'] for r in replies if r.get('pld')]
    t = np.array([r['time'] for r in rows]); t -= t[0]
    q = np.array([r['measured_state'] for r in rows])
    a = np.array([r['submitted_action'] for r in rows])
    valid = np.array([r['details'].get('source') == 'policy' and r['submitted_valid'] and r['measured_state_valid'] for r in rows])
    tracked = {}
    for arm, off in (('left', 0), ('right', 7)):
        err = np.abs(q[:, off:off+6] - a[:, off:off+6])
        tracked[arm] = {
            'p50_error_rad': np.median(err[valid], axis=0).tolist(),
            'p95_error_rad': np.percentile(err[valid], 95, axis=0).tolist(),
            'measured_range_rad': np.ptp(q[:, off:off+6], axis=0).tolist(),
            'target_range_rad': np.ptp(a[:, off:off+6], axis=0).tolist(),
            'error_over_0_3rad_runs': [error_runs(t, valid & (err[:, j] > .3)) for j in range(6)],
            'note': 'Dynamic tracking error is not itself a hardware-fault diagnosis.'}
    missing, mismatches, corrections = [], [], []
    phys_by_arm, base_by_arm = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        sel = row['details'].get('policy_selection') or {}
        token = sel.get('request') or {}
        if not token:
            continue
        origin = replies_by_key.get((token['epoch'], str(token['request_id'])))
        if origin is None:
            missing.append(i); continue
        p = origin['reply'].get('pld')
        if not p:
            continue
        k = sel['model_index']
        phys = np.asarray(p['physical_actions'][k]); base = np.asarray(p['physical_base'][k])
        diff = float(np.max(np.abs(a[i] - phys)))
        if diff > 1e-6:
            mismatches.append({'row': i, 'max_abs_difference': diff})
        delta = phys - base
        if not np.any(np.abs(delta) > 1e-9):
            continue
        arm = p['arm']; off = 0 if arm == 'left' else 7
        corrections.append({'row': i, 'arm': arm, 'window_id': p['window_id'],
                            'max_joint_rad': float(np.max(np.abs(delta[off:off+6]))),
                            'gripper_delta': float(delta[off+6])})
        phys_by_arm[arm].append(phys[off:off+6]); base_by_arm[arm].append(base[off:off+6])
    fk_max = 0.
    for arm in phys_by_arm:
        offsets = fk.positions(phys_by_arm[arm]) - fk.positions(base_by_arm[arm])
        fk_max = max(fk_max, float(np.linalg.norm(offsets, axis=1).max()))
    video_stats = []
    for segment in manifest['segments']:
        for view in ('top', 'left', 'right'):
            video = folder / segment['path'] / (view + '.mp4')
            count, previous = 0, None
            with av.open(str(video)) as container:
                container.streams.video[0].codec_context.thread_count = 1
                for index, frame in enumerate(container.decode(video=0)):
                    stamp = float(frame.pts * frame.time_base)
                    assert previous is None or stamp > previous, str(video)
                    previous = stamp; count += 1
            assert count == segment['video_frames'][view], str(video)
            col = ('top', 'left', 'right').index(view)
            indexes = [r['video_indices'][col] for r in rows if r['videos'][view] == str(video)]
            assert all(0 <= i < count for i in indexes), str(video)
            video_stats.append({'path': str(video.relative_to(RAW)), 'frames': count,
                                'pts_monotonic': True, 'indices_in_range': True})
    # Save recorded initial/final scene only, without inventing a global label.
    for tag, row in (('start', rows[0]), ('end', rows[-1])):
        for view, col in (('top', 0), ('left', 1), ('right', 2)):
            wanted = row['video_indices'][col]
            with av.open(row['videos'][view]) as container:
                container.streams.video[0].codec_context.thread_count = 1
                for index, frame in enumerate(container.decode(video=0)):
                    if index == wanted:
                        frame.to_image().convert('RGB').save(DEST / f'E{number:02d}-{tag}-{view}.jpg', quality=94)
                        break
                else:
                    raise ValueError('missing endpoint frame')
    accepted = [r for r in replies if not r.get('discarded') and not r.get('error')
                and (r['token']['epoch'], str(r['token']['request_id'])) not in cancelled]
    global_events = [e for e in events if e.get('event') == 'evaluation']
    report = {'number': number, 'episode_path': entry['path'], 'episode_id': manifest['episode_id'],
              'rows': len(rows), 'duration_s': float(t[-1]), 'manifest_pld': manifest.get('pld'),
              'mode_counts': dict(Counter(p['mode'] for p in traces)),
              'contract_sha256': sorted({p['contract_sha256'] for p in traces}),
              'snapshot_sha256': sorted({p['snapshot_sha256'] for p in traces}),
              'delay_steps': dict(Counter(p['delay_steps'] for p in traces)),
              'human_rows': sum(r['is_intervention'] for r in rows),
              'invalid_observation_rows': sum(not r['observation_valid'] for r in rows),
              'invalid_submission_rows': sum(not r['submitted_valid'] for r in rows),
              'source_counts': dict(Counter(r['details'].get('source') for r in rows)),
              'replies': len(replies), 'accepted_replies': len(accepted),
              'discarded_replies': [{'token': r['token'], 'error': r.get('error'), 'slack': r.get('rtc_slack_ticks')}
                                    for r in replies if r.get('discarded') or r.get('error')],
              'cancelled_requests': sorted(cancelled),
              'server_ms': percentiles([r['server_timing']['infer_ms'] for r in replies if r.get('server_timing')]),
              'worker_ms': percentiles([r['worker_elapsed_ms'] for r in replies if r.get('worker_elapsed_ms')]),
              'tracking': tracked, 'missing_action_origins': missing, 'submission_mismatches': mismatches,
              'correction_rows': len(corrections), 'correction_rows_by_arm': dict(Counter(c['arm'] for c in corrections)),
              'max_joint_correction_rad': max((c['max_joint_rad'] for c in corrections), default=0.),
              'max_gripper_correction': max((abs(c['gripper_delta']) for c in corrections), default=0.),
              'max_direct_fk_correction_m': fk_max, 'videos': video_stats,
              'raw_evaluation_events': global_events, 'sources_sha256': hashes,
              'training_executed': False, 'reward_labels_published': 0, 'robot_success_rate': None}
    if manifest['episode_id'] == 'b05fdb3e4ef1449a89a0fd607819184e':
        report['reviewed_global_evaluation'] = {'C': 9, 'N': 10, 'progress': .9,
            'source': str(ROOT / 'docs/reports/rl/lego-pld-collect-20261013/e05-fast-d10/end-state-audit.json'),
            'raw_evaluation_overridden_only_in_derived_data': True, 'used_by_SAC': False}
    write(DEST / f'E{number:02d}.json', report)
    reports.append({k: v for k, v in report.items() if k not in ('sources_sha256', 'raw_evaluation_events', 'videos')})
    print(json.dumps({'episode': number, 'rows': len(rows), 'accepted': len(accepted),
                      'correction_rows': len(corrections), 'videos_passed': len(video_stats)}, ensure_ascii=False), flush=True)
    del rows, events, replies_by_key, replies, accepted, traces, origin, p
    gc.collect()
write(DEST / 'summary.json', {'episodes': reports, 'total_rows': sum(r['rows'] for r in reports),
    'total_duration_s': sum(r['duration_s'] for r in reports), 'training_executed': False,
    'reward_labels_published': 0, 'robot_success_rate': None})
