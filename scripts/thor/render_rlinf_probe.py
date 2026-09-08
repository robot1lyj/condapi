"""Render the bounded RLinf study separately from the seven deployment contrasts."""

# ruff: noqa: RUF001
import argparse
import html
import json
from pathlib import Path

from scripts.thor.render_report import number
from scripts.thor.render_report import select_configurations

RUNS = (
    ("pi05-rlinf-f32-20260908-r1", "RLinf 原版 · FP32", "float32", "upstream"),
    ("pi05-rlinf-f32-tanh-20260908-r1", "RLinf · 仅对齐 GELU · FP32", "float32", "tanh"),
    ("pi05-rlinf-bf16-20260908-r1", "RLinf 原版 · BF16", "bfloat16", "upstream"),
    ("pi05-rlinf-bf16-tanh-20260908-r1", "RLinf · 仅对齐 GELU · BF16", "bfloat16", "tanh"),
)


def render(evidence, status):
    baseline = {row["mode"]: row for row in select_configurations(status)}
    rows = []

    def row(title, weights, compute, record, error, kind):
        return (
            f'<tr><th scope="row">{html.escape(title)}<small>{html.escape(kind)}</small></th>'
            f"<td>{html.escape(weights)}</td><td>{html.escape(compute)}</td>"
            f'<td class="numeric">{number(record["p50_ms"], latency=True)}</td>'
            f'<td class="numeric">{number(error["mae"])}</td>'
            f'<td class="numeric">{number(error["max_abs"])}</td></tr>'
        )

    rows.append(
        row(
            "现有 OpenPI PyTorch · FP32",
            "FP32",
            "FP32",
            baseline["D"]["record"],
            baseline["D"]["error"],
            "历史对照 · 未编译",
        )
    )
    for run_id, title, dtype, gelu in RUNS:
        record = json.loads((evidence / run_id / "result.json").read_text())
        if (
            record["status"] != "measured_not_accuracy_approved"
            or record["compute_dtype"] != dtype
            or record.get("gelu", "upstream") != gelu
            or set(record["loaded_param_dtypes"]) != {f"torch.{dtype}"}
            or record["repeats"] != 3
            or record["action_shape"] != [9, 3, 50, 14]
        ):
            raise ValueError("Probe configuration mismatch")
        for key in (
            "suite_sha256",
            "norm_stats_sha256",
            "noise_sha256",
            "checkpoint_metadata_sha256",
            "steps",
            "horizon",
        ):
            if record[key] != baseline["A"]["record"][key]:
                raise ValueError(f"Probe comparison mismatch: {key}")
        comparison = next(c for c in record["comparisons"] if c["reference"] == baseline["A"]["record"]["run_id"])
        rows.append(
            row(
                title,
                "FP32" if dtype == "float32" else "BF16",
                "FP32" if dtype == "float32" else "BF16 + 内部 FP32",
                record,
                comparison["physical_dataset_units"],
                "本轮小测试 · 未编译；非完整 RLinf 训练框架",
            )
        )
    rows.append(
        row(
            "现有 TensorRT · 104ms 候选",
            "BF16 + FP32",
            "BF16 + FP32",
            baseline["W"]["record"],
            baseline["W"]["error"],
            "历史部署候选 · 已优化",
        )
    )
    mapping = json.loads((evidence / "pi05-rlinf-mapping-20260908-r1.json").read_text())
    if not mapping["all_tensors_bit_equal"] or mapping["tensor_count"] != 667:
        raise ValueError("Weight equivalence conclusion requires the complete mapping audit")
    style = Path(__file__).with_name("report_template.html").read_text().split("<style>")[1].split("</style>")[0]
    body = Path(__file__).with_name("rlinf_probe_template.html").read_text()
    return body.replace("{{STYLE}}", style).replace("{{ROWS}}", "".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.evidence, json.loads(args.status.read_text())), encoding="utf-8")


if __name__ == "__main__":
    main()
