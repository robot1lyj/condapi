"""Render a compact Chinese report from completed realtime-vla JSON evidence."""

# ruff: noqa: RUF001
import argparse
import html
import json
from pathlib import Path

RUNS = [
    ("原版 · BF16 / text200", "realtime-vla-upstream-20260911-r2"),
    ("位置修正 · BF16 / text200", "realtime-vla-position-20260911-r1"),
    ("位置＋时间广播 · BF16 / text200", "realtime-vla-time-position-20260911-r1"),
    ("位置＋时间广播 · BF16 / text80", "realtime-vla-text80-20260911-r1"),
]


def render(evidence):
    rows = []
    for label, run in RUNS:
        record = json.loads((evidence / f"{run}.result.json").read_text())
        if not record["finite"] or len(record["measurements"]) != 9:
            raise ValueError("Incomplete or nonfinite result")
        error = record["error_dataset_units"]
        cells = [
            label,
            f"{record['p50_ms']:.2f}",
            f"{record['p95_ms']:.2f}",
            f"{error['mae']:.6f}",
            f"{error['p95_abs']:.6f}",
            f"{error['max_abs']:.6f}",
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
    control = json.loads((evidence / "pi05-W-20260911-realtime-control.result.json").read_text())
    audit = json.loads((evidence / "array-audit.json").read_text())
    if not audit["W_exact_vs_20260908"] or not audit["time_broadcast_exact_vs_position_only"]:
        raise ValueError("Report statements require exact saved-array comparisons")
    control_error = audit["W_vs_JAX"]
    rows.insert(
        0,
        f'<tr class="best"><td>现有 TensorRT · BF16＋FP32 / text80</td><td>{control["p50_ms"]:.2f}</td>'
        f"<td>{control['p95_ms']:.2f}</td><td>{control_error['mae']:.6f}</td>"
        f"<td>{control_error['p95_abs']:.6f}</td><td>{control_error['max_abs']:.6f}</td></tr>",
    )
    return (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Thor · realtime-vla 实测</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#19313e;font:15px/1.8 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:40px 24px}a{color:#087b70}h1{font-size:32px;margin:8px 0}
.tag{letter-spacing:2px;color:#087b70;font-weight:700}.hero{background:#173743;color:white;padding:24px 28px;border-radius:16px;margin:24px 0}
.hero h2{margin:0 0 8px}.hero p{margin:0;color:#d0e7e7}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card,.tablebox{background:white;border:1px solid #d8e3e8;border-radius:12px;padding:20px;margin:20px 0}
.cards .card{margin:0}.tablebox{overflow:auto}table{width:100%;border-collapse:collapse;min-width:850px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:13px 12px;border-bottom:1px solid #e4ebef}th:first-child,td:first-child{text-align:left}
th{font-size:13px;color:#516874}.best{background:#eaf6f0;font-weight:650}h2{font-size:21px}h3{margin-top:0}
small,.muted{color:#57717e}code{overflow-wrap:anywhere}li{margin:7px 0}footer{font-size:13px;color:#57717e}
@media(max-width:700px){.cards{grid-template-columns:1fr}main{padding:24px 16px}h1{font-size:26px}}
</style><main><a href="index.html">← 推理总览</a><p class="tag">THOR / PI0.5 / 2026-09-11</p>
<h1>realtime-vla：真实权重与录像回放</h1>
<p class="muted">三相机 · H50 · 10步去噪 · 14维动作 · 9个真实输入 × 每组20次 · 仅Thor离线测试</p>
<div class="hero"><h2>本轮不替换现有 TensorRT 路线</h2>
<p>原版约135 ms；修正位置编码后平均误差改善，短文本版约124 ms，仍未优于现有路线。没有使用FP8，也没有任务成功率结论。</p></div>
<div class="tablebox"><table><thead><tr><th>配置与实际精度</th><th>P50 / ms</th><th>P95 / ms</th><th>平均误差 MAE</th><th>95%绝对误差</th><th>最大误差</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table>
<p><small>误差统一相对原JAX FP32，使用数据集动作单位。时间包含图像预处理、分词、GPU推理、取回动作与YAM动作还原；不含相机采集和网络。TensorRT为本日复测，输出与原W逐值一致。</small></p></div>
<div class="cards"><div class="card"><h3>怎么读这些指标？</h3><p>P50是典型耗时；P95表示95%的调用不比它慢。MAE是所有动作点平均偏差；最大误差是最不一致的一个点。数值越小通常越接近参考，但不能直接换算成抓取成功率。</p></div>
<div class="card"><h3>这次保持了什么？</h3><p>同一基础JAX权重、录像、噪声与归一化统计。每个输入逐token验证与原OpenPI相同，64–70个有效token均保留。三相机、H50和10步没有减少。MAXN与锁频仅在测试期间启用。</p></div></div>
<div class="card"><h2>原版和修正版分开看</h2><ol>
<li>固定上游 <code>b86a942a073ea241f9bd6916a705f81906f4638b</code>。完整读取Pi0.5转换、推理、测试，以及依赖的Pi0内核；不运行其随机/未加载权重的测速作为正式成绩。</li>
<li>上游forward不返回动作，包装器读取最终diffusion_noise缓冲区；沿用YAM输入/输出变换，不套用示例机器人合同。</li>
<li>原版动作位置从“有效前缀长度−1”开始；独立修正版改为“有效前缀长度”，与原JAX公式一致。平均误差从0.005956降至0.002517。</li>
<li>时间MLP首层只写首行，增加显式广播作诊断。本批次广播修正版与仅位置修正版的动作完全相同，不能声称它带来精度提升。</li>
<li>text80只用于本批短输入；超过80个有效token明确拒绝，不允许静默截断。生产服务自动回退尚未实现。</li>
</ol></div>
<div class="card"><h2>证据与限制</h2><p>基础模型尚未针对YAM微调；9个输入不是180个独立场景。全量微调checkpoint需要重新转换和验证。BF16内核重排与舍入不是FP32逐位等价。</p>
<p>首轮 <code>realtime-vla-upstream-20260911-r1</code> 因本地包装漏做图像归一化被排除；原始产物保留，不计入本表或有效复现次数。</p>
<p><a href="evidence/20260911/realtime-vla/">原始JSON与日志</a> · <a href="../../reference/thor/10_acceleration_execution.md">复现细节</a> · <a href="https://github.com/dexmal/realtime-vla">上游项目</a></p></div>
<footer>四组有效Dexmal配置720次＋TensorRT对照180次。未操作3588，未部署生产服务，原始权重和引擎保留。</footer></main></html>"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.evidence))


if __name__ == "__main__":
    main()
