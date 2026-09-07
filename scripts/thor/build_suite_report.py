"""Build the Chinese report from three completed local precision suites."""

# ruff: noqa: RUF001
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from compare_suites import compare
from compare_suites import read_run
from compare_suites import telemetry_metrics
from render_report import render

JOINTS = [*[f"左关节{i}" for i in range(1, 7)], "左夹爪", *[f"右关节{i}" for i in range(1, 7)], "右夹爪"]


def build(run_paths, logs, plan):
    records = [read_run(path)[0] for path in run_paths]
    sample_count = len(records[0]["measurements"])
    episode_count = len({sample["episode"] for sample in records[0]["measurements"]})
    repeat_count = records[0]["repeats"]
    formal_calls = sum(len(r["measurements"]) * r["repeats"] for r in records)
    warmup_calls = sum(len(m["warmup_ms"]) for r in records for m in r["measurements"])
    expected = [
        ("checkpoint", "float32", "float32"),
        ("checkpoint", "bfloat16", "float32"),
        ("bfloat16", "bfloat16", "bfloat16"),
    ]
    for record, (params, compute, actual) in zip(records, expected, strict=True):
        if (record["params_dtype"], record["compute_dtype"]) != (params, compute) or record["loaded_param_dtypes"] != {
            actual: 51
        }:
            raise ValueError("Report requires verified A/B/C modes in this order")
    comparisons = [compare(run_paths[0], p) for p in run_paths[1:]]
    storage_comparison = compare(run_paths[1], run_paths[2])
    telemetry = [telemetry_metrics(logs / f"{record['run_id']}.tegrastats") for record in records]
    for record in records:
        exit_record = json.loads((logs / f"{record['run_id']}.exit.json").read_text())
        if exit_record["exit_code"] != 0:
            raise ValueError("Host runner did not complete successfully")
    experiments = []
    for index, record in enumerate(records):
        experiments.append(
            {
                "name": ["A · 原精度参考", "B · 仅 BF16 计算", "C · BF16 加载与计算"][index],
                "weights": ["FP32", "FP32", "原盘 FP32 → 内存 BF16"][index],
                "compute": ["FP32 / highest", "BF16", "BF16"][index],
                "status": "已实测 · 非任务精度验收",
                "scope": f"{episode_count} 轨迹 / {sample_count} 输入 / 每输入 {repeat_count} 次",
                "p50_ms": record["p50_ms"],
                "p95_ms": record["p95_ms"],
                "max_abs_error": 0 if index == 0 else comparisons[index - 1]["physical_dataset_units"]["max_abs"],
                "note": "误差相对 A，采用数据集原始单位；原始 checkpoint 从未改写。",
            }
        )
    sample_rows = []
    for index, sample in enumerate(records[0]["measurements"]):
        phase = {"early": "早期", "middle": "中期", "late": "后期"}[sample["phase"]]
        sample_rows.append(
            [
                f"{sample['episode']} / {phase} / 帧 {sample['frame']}",
                *[f"{r['measurements'][index]['p50_ms']:.2f}" for r in records],
                *[f"{c['per_sample'][index]['normalized']['max_abs']:.6f}" for c in comparisons],
            ]
        )
    drift_rows = []
    for label, comparison in zip(
        ("B − A（计算精度）", "C − A（计算与参数）", "C − B（参数加载精度）"),
        [*comparisons, storage_comparison],
        strict=True,
    ):
        physical = comparison["physical_dataset_units"]
        normalized = comparison["normalized_active_14d"]
        sample, step, dim = physical["worst_index"]
        drift_rows.append(
            [
                label,
                f"{comparison['speedup_p50']:.2f}×",
                *[f"{physical[k]:.6f}" for k in ("mae", "rmse", "p95_abs", "max_abs")],
                f"{normalized['mae']:.6f}",
                f"{normalized['max_abs']:.6f}",
                f"{records[0]['measurements'][sample]['sample']} / 动作步 {step} / {JOINTS[dim]}",
            ]
        )
    joint_rows = [
        [
            name,
            *[f"{c['physical_dataset_units']['per_dimension_max_abs'][index]:.6f}" for c in comparisons],
            *[f"{c['normalized_active_14d']['per_dimension_mae'][index]:.6f}" for c in comparisons],
        ]
        for index, name in enumerate(JOINTS)
    ]
    resource_rows = []
    for label, record, stats in zip("ABC", records, telemetry, strict=True):
        resource_rows.append(
            [
                label,
                f"{record['load_s']:.2f} s",
                f"{record['measurements'][0]['warmup_ms'][0] / 1000:.2f} s",
                f"{record['loaded_param_bytes'] / 1024**3:.2f} GiB",
                f"{stats['system_ram_peak_mib'] / 1024:.2f} GiB",
                f"{stats['gpu_temperature_max_c']:.1f} °C",
                f"{stats['gpu_clock_min_mhz']}–{stats['gpu_clock_max_mhz']} MHz",
                f"{stats['input_power_max_w']:.1f} W",
                f"{record['repeat_max_abs_difference']:.6f}",
            ]
        )
    return {
        "updated_at": datetime.datetime.now(datetime.UTC)
        .astimezone(datetime.timezone(datetime.timedelta(hours=8)))
        .isoformat(),
        "facts": [
            {"label": "完整模型模式", "value": "3 / 3 完成", "detail": "A、B、C 均在 NVIDIA Thor GPU 上执行"},
            {
                "label": "真实录像输入",
                "value": f"{episode_count} 轨迹 · {sample_count} 组",
                "detail": "三路 RGB + 14D 状态 + 文本；来源见 suite 清单",
            },
            {
                "label": "稳定推理调用",
                "value": f"{formal_calls} 次",
                "detail": f"每配置 {sample_count} × {repeat_count}；另有 {warmup_calls} 次预热调用",
            },
            {"label": "性能模式", "value": "仅推理 MAXN", "detail": "结束恢复 120W；记录实际频率与温度"},
        ],
        "experiments": experiments,
        "detail_tables": [
            {
                "title": "误差分解：计算与权重分别带来什么变化",
                "description": f"比较每个输入的第一份稳定输出（{sample_count} 个独立输入，不把 {repeat_count} 次重复冒充更多样本）。归一化误差仅统计有效 14D；物理输出保持数据集原始单位，不宣称弧度或毫米。动作步从 0 开始。",
                "columns": [
                    "比较",
                    "P50 加速",
                    "MAE",
                    "RMSE",
                    "P95 绝对误差",
                    "最大绝对误差",
                    "归一化 MAE",
                    "归一化最大误差",
                    "原单位最大误差位置",
                ],
                "rows": drift_rows,
            },
            {
                "title": "逐录像 / 阶段实测",
                "description": "耗时为完整 policy 调用 P50（ms），包括预处理、同步 GPU 执行、主机输出和少量证据拷贝；不是异步 dispatch 时间。",
                "columns": [
                    "录像 / 阶段 / 帧",
                    "A / ms",
                    "B / ms",
                    "C / ms",
                    "B−A 归一化最大误差",
                    "C−A 归一化最大误差",
                ],
                "rows": sample_rows,
            },
            {
                "title": "双臂逐维差异",
                "description": "关节和夹爪分别列出，避免总体平均值掩盖单维漂移。当前基础模型未微调，不能把动作相似度当成任务成功率。",
                "columns": ["维度", "B−A 原单位最大误差", "C−A 原单位最大误差", "B−A 归一化 MAE", "C−A 归一化 MAE"],
                "rows": joint_rows,
            },
            {
                "title": "加载、首次编译与资源",
                "description": "系统 RAM 包含操作系统和远程桌面，不等于模型显存；温度/功率为 1 秒采样最大值，覆盖加载、预热和正式测试；频率范围不等价于硬件降频原因诊断。",
                "columns": [
                    "配置",
                    "模型加载",
                    "首次调用（含编译）",
                    "实际参数占用",
                    "系统 RAM 峰值",
                    "GPU 最高温",
                    "GPU 实际频率",
                    "输入功率采样峰值",
                    "固定噪声重复差异",
                ],
                "rows": resource_rows,
            },
        ],
        "stages": [
            {
                "title": "原生 JAX 容器实测通过",
                "detail": "NGC JAX 26.05 ARM64；保留 NVIDIA JAX/Flax/Orbax 栈，兼容新版 Orbax 元数据。CPU PyTorch 仅满足类型/IO 依赖，推理由 JAX GPU 完成。",
            },
            {
                "title": "固定输入与精度审计",
                "detail": f"原盘 checkpoint 保持 FP32，无 LoRA；A/B 加载 51 个 FP32 叶子，C 加载 51 个 BF16 叶子。种子 {records[0]['seed']}、固定同一噪声、{records[0]['steps']} 次去噪、horizon 50。",
            },
            {
                "title": "真实本地离线回放完成",
                "detail": f"从 {episode_count} 条本地录像选取 {sample_count} 个真实阶段输入；三路图像与低维数据校验时间戳。容器断网，没有服务器实时传输，也未连接机械臂。",
            },
            {
                "title": "原始证据保留",
                "detail": "每次运行独立目录，含 result.json、动作 NPY、归一化动作、noise.npy、逐样本记录；宿主日志和 tegrastats 单独留存。失败的首轮 Orbax 加载日志也保留。",
            },
        ],
        "limitations": [
            "模型为未针对 YAM 微调的 pi05_base；没有实机闭环、成功率或安全动作验收。",
            "本次 norm 来自同一组回放录像，仅为可复现精度/性能对照；未来必须绑定微调 checkpoint 自己的 norm。",
            "A 是本机新版 JAX 的数值参考，不代表已验证与服务器训练版 JAX 完全相同。",
            f"{episode_count} 条轨迹、{sample_count} 个状态、一个固定噪声种子；不足以评价泛化、动态闭环和所有极端输入。",
            "BF16 会改变数值；目前未制定任务级容差，不因输出有限或推理更快就默认批准部署。",
            "首次编译曾出现 CUDA autotune 计时警告；正式耗时使用预热后的同步主机计时，不采用编译器内部测量值。",
        ],
        "recommendations": plan.get("recommendations", []),
        "sources": plan.get("sources", []),
        "measured_records": records,
        "comparisons": comparisons,
        "storage_comparison": storage_comparison,
        "telemetry": telemetry,
        "host_manifests": {
            record["run_id"]: json.loads(path.read_text())
            if path.exists()
            else {"status": "not_captured_by_manual_launcher"}
            for record in records
            for path in [logs / f"{record['run_id']}.manifest.json"]
        },
    }


def attach_conversion(data, path):
    audit = json.loads(path.read_text())
    if audit.get("load_precision") != "float32" or audit.get("output_precision") != "float32":
        raise ValueError("The conversion progress row requires an FP32 artifact")
    data["conversion_evidence"] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "audit": audit,
    }
    data["detail_tables"].insert(
        0,
        {
            "title": "100 ms 加速主线 · 权重转换进度",
            "description": "FP32 转换已经完成；跨框架动作误差与延迟见实测表。首次权重加载检查数值相等（不区分正负零），不替代后续推理对照。",
            "columns": ["阶段", "实测结果", "下一步"],
            "rows": [
                [
                    "原始 JAX → PyTorch FP32",
                    f"{audit['mapped_tensor_count']} 张量 / {audit['mapped_element_count']:,} 参数",
                    "同输入 / 同噪声动作对照",
                ],
                ["LoRA", "当前基础模型不含 LoRA；转换器拒绝丢弃适配器", "微调后独立验证合并路径"],
                ["加速目标", "完整调用约 100 ms 或以下", "三相机 / H50 / 去噪 10，不缩减合同"],
            ],
        },
    )


def attach_additional_runs(data, reference, run_paths, logs):
    rows = []
    for path in run_paths:
        record = read_run(path)[0]
        comparison = compare(reference, path, cross_backend=True)
        run_id = record["run_id"]
        exit_record = json.loads((logs / f"{run_id}.exit.json").read_text())
        if exit_record["exit_code"] != 0:
            raise ValueError("Additional backend did not finish successfully")
        stats = telemetry_metrics(logs / f"{run_id}.tegrastats")
        name = f"{run_id.split('-')[1]} · {record['backend']} / {record['compute_dtype']}"
        detail = (
            f"compile={record['compiled']} / attention={record['attention']} / "
            f"合批相机={record.get('batch_vision', False)} / CUDA graph={record.get('cuda_graph', False)} / "
            f"分段编译={record.get('compiled_graph_parts', False)} / "
            f"掩码={record.get('attention_mask', 'float32')} / TF32 关闭"
        )
        if record.get("cuda_graph"):
            graph_error = max(m["cuda_graph_vs_eager_max_abs"] for m in record["measurements"])
            detail += f" / 图重放与非图旧循环最大差={graph_error:.6g}（同算子实现，计时外逐输入验证）"
        error = comparison["physical_dataset_units"]
        normalized = comparison["normalized_active_14d"]
        data["experiments"].append(
            {
                "name": name,
                "weights": record["params_dtype"],
                "compute": record["compute_dtype"],
                "status": "已实测 · 非任务精度验收",
                "scope": f"{len(record['measurements'])} 输入 / 每输入 {record['repeats']} 次",
                "p50_ms": record["p50_ms"],
                "p95_ms": record["p95_ms"],
                "max_abs_error": error["max_abs"],
                "note": detail,
            }
        )
        rows.append(
            [
                name,
                detail,
                f"{record['p50_ms']:.2f}",
                f"{record['p95_ms']:.2f}",
                f"{error['mae']:.8f}",
                f"{error['max_abs']:.8f}",
                f"{normalized['max_abs']:.8f}",
                f"{record['repeat_max_abs_difference']:.8f}",
                f"{stats['gpu_temperature_max_c']:.1f} °C",
            ]
        )
        data.setdefault("additional_comparisons", []).append(comparison)
        data.setdefault("additional_telemetry", {})[run_id] = stats
        data["measured_records"].append(record)
        data["host_manifests"][run_id] = json.loads((logs / f"{run_id}.manifest.json").read_text())
    if rows:
        data["detail_tables"].insert(
            0,
            {
                "title": "跨框架实测 · 保持三相机 / H50 / 去噪 10",
                "description": "统一对照原 JAX FP32（A）。数据单位未做硬件校准，数值误差不等于任务成功率。所有数值来自完成的本地回放，不使用社区宣传延迟。",
                "columns": [
                    "配置",
                    "实现",
                    "P50 ms",
                    "P95 ms",
                    "动作 MAE",
                    "动作最大误差",
                    "归一化 14D 最大误差",
                    "重复最大变化",
                    "GPU 最高温度",
                ],
                "rows": rows,
            },
        )
        data["facts"][0].update(
            value=f"{len(data['experiments'])} 组已实测", detail="原生 JAX 与新增后端；精度资格分别说明"
        )
        count = sum(len(record["measurements"]) * record["repeats"] for record in data["measured_records"])
        data["facts"][2].update(value=f"{count} 次", detail="正式调用总数；编译和预热另行记录")


def attach_failed_runs(data, run_ids, logs):
    for run_id in run_ids:
        exit_record = json.loads((logs / f"{run_id}.exit.json").read_text())
        if exit_record["exit_code"] == 0:
            raise ValueError("Successful runs must not be listed as failed")
        manifest = json.loads((logs / f"{run_id}.manifest.json").read_text())
        log_path = logs / f"{run_id}.log"
        error = next(
            (line for line in reversed(log_path.read_text().splitlines()) if "Error:" in line),
            "See raw log for failure details",
        )
        data.setdefault("failed_runs", []).append(
            {
                "run_id": run_id,
                "exit": exit_record,
                "manifest": manifest,
                "error": error,
                "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            }
        )
        data["experiments"].append(
            {
                "name": run_id,
                "weights": manifest["checkpoint"],
                "compute": "见实际命令",
                "status": "失败 · 无有效延迟",
                "scope": "不计入完成的推理调用",
                "p50_ms": None,
                "p95_ms": None,
                "max_abs_error": None,
                "note": error,
            }
        )


def attach_front_runner(data):
    """Highlight measured latency without turning it into an accuracy approval."""
    measured = [row for row in data["experiments"] if row.get("p50_ms") is not None]
    if not measured:
        return
    best = min(measured, key=lambda row: row["p50_ms"])
    data["latency_front_runner"] = {
        "name": best["name"],
        "p50_ms": best["p50_ms"],
        "p95_ms": best["p95_ms"],
        "max_abs_error_vs_jax_fp32": best["max_abs_error"],
        "target_ms": 100,
        "accuracy_approved": False,
    }
    data["facts"][1] = {
        "label": "当前最快完整调用 · P50",
        "value": f"{best['p50_ms']:.2f} ms",
        "detail": f"{best['name']}；P95 {best['p95_ms']:.2f} ms；目标约 100 ms，任务精度尚未验收",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--conversion-audit", type=Path)
    parser.add_argument("--additional-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--failed-runs", nargs="*", default=[])
    args = parser.parse_args()
    data = build(args.runs, args.logs, json.loads(args.plan.read_text()))
    if args.conversion_audit:
        attach_conversion(data, args.conversion_audit)
    attach_additional_runs(data, args.runs[0], args.additional_runs, args.logs)
    attach_failed_runs(data, args.failed_runs, args.logs)
    attach_front_runner(data)
    args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    args.html.write_text(render(data))


if __name__ == "__main__":
    main()
