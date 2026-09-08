"""Student-readable report view; complete experimental evidence remains in JSON."""

# ruff: noqa: RUF001
import argparse
import html
import json
import math
from pathlib import Path
import re

# Seven purposeful contrasts, rather than merely the fastest seven runs.
CONFIGURATIONS = {
    "A": ("JAX · 原始参考", "全部误差的比较基准", "FP32", "FP32", "jax", "float32"),
    "B": ("JAX · 只改计算精度", "保留 FP32 权重，只降低主计算精度", "FP32", "BF16 为主", "jax", "bfloat16"),
    "C": ("JAX · 权重也用 BF16", "进一步降低内存中的权重精度", "BF16", "BF16 为主", "jax", "bfloat16"),
    "D": ("PyTorch · FP32 对照", "检查转换后是否接近原 JAX 输出", "FP32", "FP32", "pytorch", "float32"),
    "I": ("PyTorch · 编译加速", "简单的非 TensorRT 备用路线", "BF16 + FP32", "BF16 + FP32", "pytorch", "bfloat16"),
    "V": ("TensorRT · 时间缓存版", "固定时间预计算 + GPU 执行图", "BF16 + FP32", "BF16 + FP32", "tensorrt", "bfloat16"),
    "W": (
        "TensorRT · 填充优化版",
        "在时间缓存版上去掉无效文本填充",
        "BF16 + FP32",
        "BF16 + FP32",
        "tensorrt",
        "bfloat16",
    ),
}


def number(value, *, latency=False):
    """Missing stays missing; tiny nonzero errors must not round to zero."""
    if value is None:
        return "—"
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Report metrics must be finite and non-negative")
    if latency:
        return f"{value:,.2f}"
    if 0 < value < 1e-8:
        return f"{value:.3g}"
    decimals = 8 if value < 1e-4 else 6
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".") if value else "0"


def select_configurations(data):
    records = {}
    for record in data.get("measured_records", []):
        parts = record["run_id"].split("-")
        mode = parts[1] if len(parts) > 2 else ""
        if mode not in CONFIGURATIONS:
            continue
        if mode in records:
            raise ValueError(f"Ambiguous duplicate configuration: {mode}")
        title, purpose, weights, compute, backend, dtype = CONFIGURATIONS[mode]
        if (
            record["status"] != "measured_not_accuracy_approved"
            or record.get("backend", "jax") != backend
            or record["compute_dtype"] != dtype
            or not record["suite"]["base_model_only"]
        ):
            raise ValueError(f"Record does not match curated label: {mode}")
        if mode in ("A", "B", "C", "D"):
            expected = "bfloat16" if mode == "C" else "float32"
            actual = {key.removeprefix("torch.") for key in record["loaded_param_dtypes"]}
            if actual != {expected}:
                raise ValueError(f"Weight precision does not match label: {mode}")
        if mode == "I" and (
            not record.get("compiled")
            or not record.get("batch_vision")
            or set(record["loaded_param_dtypes"]) != {"torch.float32", "torch.bfloat16"}
        ):
            raise ValueError("I requires compiled mixed-precision PyTorch with batched vision")
        if mode in ("V", "W"):
            engine = record["engine_report"]
            if (
                not record.get("time_modulation_cache")
                or not record.get("engine_cuda_graph")
                or record.get("text_bucket", 200) != (80 if mode == "W" else 200)
                or engine["quantization"] is not None
                or engine["tf32"]
                or not engine["strongly_typed"]
            ):
                raise ValueError(f"Engine does not match curated configuration: {mode}")
        records[mode] = {
            "mode": mode,
            "title": title,
            "purpose": purpose,
            "weights": weights,
            "compute": compute,
            "record": record,
        }
    comparisons = {
        row["candidate"]: row for row in [*data.get("comparisons", []), *data.get("additional_comparisons", [])]
    }
    reference = records.get("A", {}).get("record", {}).get("run_id")
    selected = []
    for mode in CONFIGURATIONS:
        if mode not in records:
            continue
        row = records[mode]
        comparison = comparisons.get(row["record"]["run_id"])
        if comparison and comparison["reference"] != reference:
            raise ValueError("Displayed error must use the original JAX A reference")
        row["error"] = comparison.get("physical_dataset_units", {}) if comparison else {}
        selected.append(row)
    return selected


def render(data: dict) -> str:
    selected = select_configurations(data)

    def esc(value):
        return html.escape(str(value), quote=True)

    candidate = next((row for row in selected if row["mode"] == "W"), None)
    metrics = candidate["record"] if candidate else {}
    error = candidate["error"] if candidate else {}
    rows, evidence = [], []
    for row in selected:
        mode, record = row["mode"], row["record"]
        badge = '<span class="badge">当前候选</span>' if mode == "W" else ""
        row_class = ' class="candidate"' if mode == "W" else ""
        mae = "基准" if mode == "A" else number(row["error"].get("mae"))
        maximum = "基准" if mode == "A" else number(row["error"].get("max_abs"))
        weight_style = "fp32" if row["weights"] == "FP32" else "mixed"
        compute_style = "fp32" if row["compute"] == "FP32" else "mixed"
        rows.append(
            f'<tr{row_class} data-config="{mode}"><th scope="row"><div class="row-title">'
            f"{esc(row['title'])} {badge}</div><small>{esc(row['purpose'])}</small></th>"
            f'<td><span class="precision {weight_style}">{esc(row["weights"])}</span></td>'
            f'<td><span class="precision {compute_style}">{esc(row["compute"])}</span></td>'
            f'<td class="numeric">{number(record.get("p50_ms"), latency=True)}</td>'
            f'<td class="numeric">{number(record.get("p95_ms"), latency=True)}</td>'
            f'<td class="numeric">{mae}</td><td class="numeric">{maximum}</td></tr>'
        )
        evidence.append(f"<li><strong>{esc(row['title'])}</strong><code>{esc(record['run_id'])}</code></li>")
    first = selected[0]["record"] if selected else {}
    samples = first.get("measurements", [])
    sample_count, repeats = len(samples), first.get("repeats", 0)
    if selected and any(
        row["record"]["suite_sha256"] != first["suite_sha256"]
        or row["record"]["repeats"] != repeats
        or row["record"]["noise_sha256"] != first["noise_sha256"]
        for row in selected
    ):
        raise ValueError("Curated comparisons require the same suite, repeats and noise")
    all_records = data.get("measured_records", [])
    total_calls = sum(len(record["measurements"]) * record["repeats"] for record in all_records)
    candidate_note = (
        "TensorRT · BF16 + FP32 · 时间缓存 + GPU 执行图 + 无效填充消除"
        if candidate
        else "尚无最终候选实测；不以其他配置或社区数字代替。"
    )
    measured_latency = metrics.get("p50_ms")
    if measured_latency is None:
        performance_note = "尚无当前候选的完整延迟数据。"
    elif measured_latency <= 100:
        performance_note = "典型耗时达到 100ms 目标，仍需结合 P95 判断。"
    elif measured_latency <= 110:
        performance_note = "典型耗时接近 100ms，尚未低于 100ms。"
    else:
        performance_note = "典型耗时尚未达到 100ms 目标。"
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    replacements = {
        "UPDATE": esc(data["updated_at"]),
        "CONFIG_COUNT": str(len(selected)),
        "CANDIDATE_NOTE": esc(candidate_note),
        "PERFORMANCE_NOTE": performance_note,
        "P50": number(metrics.get("p50_ms"), latency=True),
        "P95": number(metrics.get("p95_ms"), latency=True),
        "MAE": number(error.get("mae")),
        "ROWS": "".join(rows) or '<tr><td colspan="7">待测试：暂无符合条件的实测记录。</td></tr>',
        "EPISODES": str(len({sample["episode"] for sample in samples})),
        "SAMPLES": str(sample_count),
        "REPEATS": str(repeats),
        "CALLS": str(sample_count * repeats),
        "TOTAL_CONFIGS": str(len(all_records)),
        "TOTAL_CALLS": str(total_calls),
        "EVIDENCE": "".join(evidence),
    }
    # Replace explicit tokens once; evidence content cannot become a placeholder.
    return re.sub(r"\{\{([A-Z_0-9]+)\}\}", lambda match: replacements[match[1]], template)


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
