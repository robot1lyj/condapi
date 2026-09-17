"""Compare suspect published episode arrays with local source Parquet, read-only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path("/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting")
published = Path(
    "/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train"
)
out = Path(
    "/home/wuyan/lyj/YAM/training-runs/control/data-audit-20260917/source_comparison.json"
)
manifest = {
    x["source_episode_index"]: x
    for x in map(json.loads, (root / "manifests/train.jsonl").read_text().splitlines())
}
if out.exists():
    raise FileExistsError(out)
rows = []
for eid, sid in [(3227, 93591), (3306, 96098), (1086, 30941)]:
    m = manifest[sid]
    p = root / f"train/data/source-file-{m['data']['file_index']:03d}.parquet"
    t = pq.read_table(p, filters=[("episode_index", "=", sid)])
    dst = pq.read_table(
        published / f"data/chunk-{eid // 1000:03d}/file-{eid % 1000:03d}.parquet"
    )
    row = {
        "episode": eid,
        "source_episode": sid,
        "source_path": str(p),
        "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "source_rows": len(t),
        "published_rows": len(dst),
        "columns": {},
    }
    for key in ["observation.state", "action"]:
        a = np.asarray(t[key].to_pylist())
        b = np.asarray(dst[key].to_pylist())
        row["columns"][key] = {
            "equal": bool(np.array_equal(a, b)),
            "max_abs_difference": float(np.max(np.abs(a - b))),
        }
    rows.append(row)
out.write_text(json.dumps(rows, indent=2) + "\n")
print(out)
