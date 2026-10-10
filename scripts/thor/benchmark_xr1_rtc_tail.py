"""Focused native XR-1 latency repeat with Python GC observation, no GC changes.

Import the frozen main probe from the same directory. No model math, prefix,
precision or scheduling changes; full-path samples increased to 60.
"""

import gc
import time

import numpy as np
import torch

import benchmark_xr1_rtc as probe


def observe_total(model, prepared, d, repeats):
    times, preprocess_times, model_times, events = [], [], [], []
    gc_events = []
    started = {}

    def on_gc(phase, info):
        generation = info["generation"]
        if phase == "start":
            started[generation] = time.perf_counter()
        elif generation in started:
            gc_events.append({"generation": generation,
                              "duration_ms": (time.perf_counter() - started.pop(generation)) * 1000})

    gc.callbacks.append(on_gc)
    try:
        for index in range(60):
            _, _, preprocess, _ = prepared[index % len(prepared)]
            torch.manual_seed(index + 42)
            torch.cuda.synchronize()
            old_gc_count = len(gc_events)
            start = time.perf_counter()
            batch = preprocess()
            torch.cuda.synchronize()
            pre_end = time.perf_counter()
            output = model(dict(batch, prefix_length=d)).float().cpu().numpy()
            end = time.perf_counter()
            if not np.isfinite(output).all():
                raise RuntimeError("Non-finite output")
            pre_ms = (pre_end - start) * 1000
            model_ms = (end - pre_end) * 1000
            total_ms = (end - start) * 1000
            preprocess_times.append(pre_ms)
            model_times.append(model_ms)
            times.append(total_ms)
            events.append({"sample": index, "fixture": prepared[index % len(prepared)][0],
                           "total_ms": total_ms, "preprocess_h2d_ms": pre_ms,
                           "forward_d2h_ms": model_ms, "gc": gc_events[old_gc_count:]})
    finally:
        gc.callbacks.remove(on_gc)
    return {"local_preprocess_forward_d2h": probe.summary(times),
            "preprocess_h2d": probe.summary(preprocess_times),
            "forward_d2h": probe.summary(model_times), "full_path_samples": events,
            "gc_observed": gc_events, "gc_enabled": gc.isenabled()}


if __name__ == "__main__":
    probe.total_measure = observe_total
    probe.main()
