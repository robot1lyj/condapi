from build_suite_report import attach_front_runner


def test_highlight_is_measured_latency_not_accuracy_approval():
    data = {
        "facts": [{}, {}],
        "experiments": [
            {"name": "failed", "p50_ms": None},
            {"name": "slow", "p50_ms": 177, "p95_ms": 178, "max_abs_error": 0.02},
            {"name": "fast", "p50_ms": 99, "p95_ms": 101, "max_abs_error": 0.03},
        ],
    }
    attach_front_runner(data)
    assert data["latency_front_runner"]["name"] == "fast"
    assert data["latency_front_runner"]["accuracy_approved"] is False
    assert data["latency_front_runner"]["p95_ms"] == 101
    assert data["facts"][1]["value"] == "99.00 ms"
