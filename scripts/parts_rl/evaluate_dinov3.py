"""Compare frozen features to reference exemplars, then audit separate episodes."""
# ruff: noqa: RUF001

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np


def predict(features, jobs, labels):
    """No evaluation labels or fitted parameters enter this classifier."""
    reference = [i for i, job in enumerate(jobs) if job["split"] == "reference"]
    evaluation = [i for i, job in enumerate(jobs) if job["split"] == "evaluation"]
    ref_episodes = {jobs[i]["episode_number"] for i in reference}
    eval_episodes = {jobs[i]["episode_number"] for i in evaluation}
    if ref_episodes & eval_episodes:
        raise ValueError("reference and evaluation episodes overlap")
    if {jobs[i]["sha256"] for i in reference} & {jobs[i]["sha256"] for i in evaluation}:
        raise ValueError("reference and evaluation images overlap")
    if set(labels) != {jobs[i]["id"] for i in reference}:
        raise ValueError("only the complete reference labels may enter prediction")
    if features.ndim != 2 or len(features) != len(jobs) or not np.isfinite(features).all():
        raise ValueError("invalid feature matrix")
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    if (norm <= 0).any():
        raise ValueError("zero feature vector")
    normalized = features / norm
    groups = {}
    for label in ["held_at_snapshot", "empty_at_snapshot"]:
        groups[label] = [
            i for i in reference if labels[jobs[i]["id"]] == label and jobs[i]["roi_measurement"]["geometry_valid"]
        ]
        if not groups[label]:
            raise ValueError("both reference classes require valid exemplars")
    results = []
    for index in evaluation:
        nearest = {}
        for label, indices in groups.items():
            similarities = normalized[indices] @ normalized[index]
            selected = int(np.argmax(similarities))
            nearest[label] = {"id": jobs[indices[selected]]["id"], "similarity": float(similarities[selected])}
        margin = nearest["held_at_snapshot"]["similarity"] - nearest["empty_at_snapshot"]["similarity"]
        label = "held_at_snapshot" if margin >= 0.02 else "empty_at_snapshot" if margin <= -0.02 else "uncertain"
        if not jobs[index]["roi_measurement"]["geometry_valid"]:
            label = "uncertain"
        results.append(
            {"id": jobs[index]["id"], "label": label, "margin": margin, "nearest": nearest, "approved_reward": None}
        )
    return results


def metrics(predictions, gold):
    result = dict.fromkeys(["tp", "tn", "fp", "fn", "unknown_held", "unknown_empty", "audit_uncertain"], 0)
    for p in predictions:
        g = gold[p["id"]]
        if g == "uncertain":
            result["audit_uncertain"] += 1
        elif p["label"] == "uncertain":
            result["unknown_held" if g == "held_at_snapshot" else "unknown_empty"] += 1
        elif g == "held_at_snapshot":
            result["tp" if p["label"] == g else "fn"] += 1
        else:
            result["tn" if p["label"] == g else "fp"] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    jobs_path = root / "feature-jobs.json"
    jobs = json.loads(jobs_path.read_text())
    run = json.loads((root / "dinov3/run.json").read_text())
    if hashlib.sha256(jobs_path.read_bytes()).hexdigest() != run["jobs_sha256"]:
        raise ValueError("feature job identity mismatch")
    labels = {
        r["candidate_id"]: r["label"] for r in json.loads((root / "reference-labels.json").read_text())["records"]
    }
    with np.load(root / "dinov3/features.npz") as saved:
        if saved["ids"].tolist() != [j["id"] for j in jobs]:
            raise ValueError("feature order mismatch")
        predictions = {name: predict(saved[name], jobs, labels) for name in ["roi", "cls"]}
    # Persist predictions before reading the evaluation audit.
    (root / "predictions.json").write_text(json.dumps(predictions, indent=2, allow_nan=False) + "\n")
    audit = json.loads((root / "evaluation-audit.json").read_text())
    audit_by_id = {r["candidate_id"]: r for r in audit["records"]}
    gold = {cid: r["label"] for cid, r in audit_by_id.items()}
    if set(gold) != {p["id"] for p in predictions["roi"]}:
        raise ValueError("evaluation audit must cover every evaluation image")
    comparison = {name: metrics(rows, gold) for name, rows in predictions.items()}
    cls_predictions = {p["id"]: p for p in predictions["cls"]}
    agreement = [
        {**p, "label": p["label"] if p["label"] == cls_predictions[p["id"]]["label"] else "uncertain"}
        for p in predictions["roi"]
    ]
    comparison["agreement_review_exploratory"] = metrics(agreement, gold)
    for field in ["threshold_met", "shadow_tolerant_threshold_met"]:
        rows = []
        for j in jobs:
            if j["split"] != "evaluation":
                continue
            value = j["roi_measurement"].get(field)
            label = "uncertain" if value is None else "held_at_snapshot" if value else "empty_at_snapshot"
            rows.append({"id": j["id"], "label": label})
        comparison[field] = metrics(rows, gold)
    summary = {
        "schema": "parts_dinov3_exemplar_audit_v1",
        "reference_images": 20,
        "evaluation_images": 20,
        "reference_episode_numbers": sorted({j["episode_number"] for j in jobs if j["split"] == "reference"}),
        "evaluation_episode_numbers": sorted({j["episode_number"] for j in jobs if j["split"] == "evaluation"}),
        "metrics": comparison,
        "run": run,
        "margin_threshold": 0.02,
        "reviewer": audit["reviewer"],
        "audit_uncertain_excluded_from_confusion": True,
        "agreement_review_gate_exploratory": True,
        "evaluation_audit_sha256": hashlib.sha256((root / "evaluation-audit.json").read_bytes()).hexdigest(),
        "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "automatic_tracking": False,
        "gradient_updates": 0,
        "training_executed": False,
        "automatic_reward_ready": False,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    names = {"held_at_snapshot": "当前持物", "empty_at_snapshot": "当前空手", "uncertain": "不确定"}
    jobs_by_id = {j["id"]: j for j in jobs}
    cls_by_id = {p["id"]: p for p in predictions["cls"]}
    cards = []
    for p in predictions["roi"]:
        j = jobs_by_id[p["id"]]
        cls = cls_by_id[p["id"]]
        neighbor_images = "".join(
            f'<figure><img src="{jobs_by_id[n["id"]]["crop"]}"><figcaption>'
            f"{names[label]}参考 {n['id']} · 相似度{n['similarity']:.3f}</figcaption></figure>"
            for label, n in p["nearest"].items()
        )
        m = j["roi_measurement"]
        percent = m.get("shadow_tolerant_color_fraction")
        percent_text = "未知" if percent is None else f"{percent:.1%}"
        cards.append(
            f"<article><h2>{p['id']} · 局部DINO：{names[p['label']]}</h2>"
            f"<p>目视审核：{names[gold[p['id']]]}；整图特征：{names[cls['label']]}；"
            f"复核标记：{p['label'] == 'uncertain' or p['label'] != cls['label']}；"
            f"局部差值 {p['margin']:+.3f}（不是概率）；宽松彩色占比 {percent_text}；"
            f"夹指几何有效：{m['geometry_valid']}</p>"
            f"<p>{html.escape(audit_by_id[p['id']]['reason'])}</p>"
            f'<img class="source" src="sam3-eval/roi/{p["id"]}.jpg" alt="原图与尖端区域">'
            f'<div class="references"><figure><img src="{j["crop"]}"><figcaption>本例送入DINO的裁剪</figcaption></figure>'
            + neighbor_images
            + f'</div><video controls preload="none" src="{j["clip"]}"></video></article>'
        )
    table_rows = "".join(
        f"<tr><td>{html.escape(k)}</td>"
        + "".join(
            f"<td>{m[field]}</td>"
            for field in ["tp", "unknown_held", "fn", "fp", "tn", "unknown_empty", "audit_uncertain"]
        )
        + "</tr>"
        for k, m in comparison.items()
    )
    (root / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>DINOv3抓取诊断 · 新20例</title>'
        "<style>body{font:16px system-ui;background:#eef1f5;margin:24px;color:#172333}article{background:white;padding:20px;margin:22px 0;border-radius:12px}"
        ".source{width:100%;max-width:1280px}.references{display:flex;flex-wrap:wrap}figure{margin:12px}figure img{width:224px}video{width:100%;max-width:960px}"
        "table{border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #ccc}</style>"
        "<h1>SAM3尖端定位 + 冻结DINOv3 · 新20例诊断</h1>"
        "<p>此前20例为参考；本页20例来自另外5集录像。局部特征只比较两指尖端连接带对应的DINO patch。"
        "ROI为主方法，CLS为预先声明的整图特征对照；两者相似度差门槛均为±0.02。"
        "另列分支冲突转复核的探索性统计，这是观察结果后提出的保护规则。没有模型训练、自动跟踪或奖励批准。</p>"
        "<p>审核针对采样时刻，不等同于完整抓取结果；模糊例不计入混淆矩阵。参考标签与本页目视审核来自助手，尚非用户验收。"
        "小批次没有覆盖白色积木、不同照明及独立现场分布。</p>"
        "<table><tr><th>方法</th><th>持物检出</th><th>持物未知</th><th>持物漏判</th><th>空手误报</th><th>空手排除</th><th>空手未知</th><th>审核未知</th></tr>"
        + table_rows
        + "</table>"
        + "".join(cards)
        + "</html>"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
