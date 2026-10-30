"""Recorded-input RTC guard check without a server, network port or robot client."""

import argparse
import dataclasses
import json
from pathlib import Path

from benchmark_pi05 import digest, read_observation
import numpy as np
from rtc_policy import TrainedRtcInference, reject_large_joint_step
from rtc_trt_policy import RtcTensorRTAdapter, create_rtc_transform_policy
from openpi.shared import normalize
from openpi.training import config

JOINTS = list(range(6)) + list(range(7, 13))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'checkpoint', 'cases', 'jax-reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads((args.engine / 'engine_report.json').read_text())
    export = json.loads((Path(report['source_export']) / 'export_report.json').read_text())
    manifest = json.loads((args.checkpoint / 'rtc_manifest.json').read_text())
    cases = json.loads(args.cases.read_text())
    refs = json.loads((args.jax_reference / 'reference_manifest.json').read_text())
    if refs['cases_sha256'] != digest(args.cases) or cases['norm_stats_sha256'] != manifest['norm_stats_sha256']:
        raise ValueError('Checkpoint/cases/reference identity mismatch')
    train = config.get_config('pi05_yam')
    train = dataclasses.replace(train, model=dataclasses.replace(
        train.model, dtype=export['compute_dtype'], rtc_training_max_delay=manifest['max_delay_steps']))
    stats = normalize.deserialize_json((args.checkpoint / 'assets/yam/norm_stats.json').read_text())
    quantiles = train.data.create(train.assets_dirs, train.model).use_quantile_norm
    if not quantiles:
        raise ValueError('Quantile norm required')
    backend = RtcTensorRTAdapter(args.engine, max_delay=manifest['max_delay_steps'], allow_experimental=True)
    backend.enable_cuda_graph()
    serving = TrainedRtcInference(create_rtc_transform_policy(train, stats), stats,
                                 max_delay=manifest['max_delay_steps'], use_quantiles=True,
                                 text_bucket=export['text_bucket'], num_steps=export['contract']['steps'],
                                 sampler=backend, max_joint_step_rad=0.2)
    noise = np.random.default_rng(refs['noise_seed']).standard_normal((1, 50, 32)).astype(np.float32)
    args.output.mkdir(parents=True)
    rows = []
    for index, (case, ref) in enumerate(zip(cases['cases'], refs['cases'], strict=True)):
        obs = read_observation(args.cases.parent / case['sample'])
        prefix = np.load(args.cases.parent / case['committed_actions'], allow_pickle=False)
        delay = case['delay_steps']
        rtc = {key: case[key] for key in ('delay_steps', 'observation_policy_tick',
                                        'target_start_tick', 'committed_start_tick')}
        rtc['committed_actions'] = prefix
        actual = np.asarray(serving.infer_rtc(obs, rtc, noise=noise)['actions'], dtype=np.float32)
        with np.load(args.jax_reference / ref['reference'], allow_pickle=False) as arrays:
            expected = np.asarray(arrays['physical'], dtype=np.float32)
        if actual.shape != (50, 14) or expected.shape != (50, 14) or not np.isfinite(expected).all():
            raise ValueError('Invalid physical actions')
        if not np.array_equal(actual[:delay], prefix) or not np.array_equal(expected[:delay], prefix):
            raise ValueError('RTC prefix changed')
        reject_large_joint_step(expected, obs['observation.state'], delay, max_step_rad=0.2)
        def step_max(actions):
            tail = actions[delay - 1:, JOINTS] if delay else np.vstack(
                (np.asarray(obs['observation.state'])[JOINTS], actions[:, JOINTS]))
            return float(np.max(np.abs(np.diff(tail, axis=0))))
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        rows.append(dict(sample=case['sample'], delay_steps=delay,
                         jax_max_joint_step_rad=step_max(expected),
                         trt_max_joint_step_rad=step_max(actual),
                         joint_max_abs_rad=float(difference[:, JOINTS].max()),
                         gripper_max_abs=float(difference[:, [6, 13]].max()), prefix_exact=True))
        np.savez_compressed(args.output / f'case-{index:03d}.npz', jax=expected, trt=actual)
    receipt = dict(status='offline_guard_passed_no_service', cases=rows, guard_rad_per_tick=0.2,
                   engine_sha256=report['engine_sha256'], cases_sha256=digest(args.cases),
                   norm_stats_sha256=manifest['norm_stats_sha256'])
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
