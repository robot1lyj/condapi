#!/usr/bin/env bash
# Thor HOST only. One-shot orchestration; native runner owns model checkpoints.
set -Eeuo pipefail
backend='' task_root=/home/wuyan-lyj/thor/flora/data/tshirt-fineart-20261103 container=flora-annotation
while (($#)); do
  case "$1" in
    --backend) backend="$2"; shift 2 ;;
    --task-root) task_root="$2"; shift 2 ;;
    --container) container="$2"; shift 2 ;;
    -h|--help) echo 'Usage: run_tshirt_annotation_pipeline.sh --backend native|gdn [--task-root PATH] [--container flora-annotation]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$backend" == native || "$backend" == gdn ]] || { echo 'Explicit --backend native|gdn required' >&2; exit 2; }
[[ "$task_root" == /home/wuyan-lyj/thor/flora/data/tshirt-fineart-20261103 && "$container" == flora-annotation ]] || {
  echo 'This host script is restricted to the named Thor task/container.' >&2; exit 2;
}
[[ "$(uname -m)" == aarch64 && -d "$task_root/logs" ]] || { echo 'Requires the prepared Thor host task root.' >&2; exit 2; }
flora_host=/home/wuyan-lyj/thor/flora
container_root=/data/tshirt-fineart-20261103
result_host="$flora_host/results/tshirt-fineart-20261103"
result_container=/results/tshirt-fineart-20261103
tool_relative=tools/automatic-v3
tools="$container_root/$tool_relative"
runner="$tools/run_flora_tshirt_batch.py"
selector_model=/data/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
logs="$task_root/logs"
# A task-level orchestration lock is separate from either GPU lock.
exec 8>"$logs/.annotation-pipeline.lock"
flock -n -x 8 || { echo 'Another pipeline owns this task; existing process untouched.' >&2; exit 75; }
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
stage=INITIALIZING
for suffix in exit state.json; do
  old="$logs/pipeline-$backend.$suffix"
  [[ ! -e "$old" ]] || mv -n "$old" "$old.previous-$run_id"
done
state() {
  stage="$1"
  python3 - "$logs/pipeline-$backend.state.json" "$backend" "$stage" "${2-}" "$run_id" <<'PY'
import datetime,json,os,pathlib,sys
path=pathlib.Path(sys.argv[1]); code=int(sys.argv[4]) if sys.argv[4] else None
value={"backend":sys.argv[2],"stage":sys.argv[3],"run_id":sys.argv[5],
 "status":"RUNNING" if code is None else "COMPLETED_RELEASE_PENDING" if code==0 else "FAILED_NOT_EPISODE_DAMAGE",
 "exit_code":int(sys.argv[4]) if sys.argv[4] else None,"updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "training_ready":False,"semantic_resolution":"AUTOMATIC_WITH_EXPLICIT_PENDING_QUEUE","scope":"this Thor task only; no training/server/3588 operations"}
tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");os.replace(tmp,path)
PY
}
finish() {
  code=$?
  trap - EXIT
  if ((code==0)); then state COMPLETED_AUTOMATIC_ANNOTATION_RELEASE_PENDING "$code" || true
  else state "FAILED_AT_$stage" "$code" || true; fi
  printf '%s\n' "$code" > "$logs/pipeline-$backend.exit.tmp"
  mv "$logs/pipeline-$backend.exit.tmp" "$logs/pipeline-$backend.exit"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
state VALIDATING_BINDING
# Freeze tool/backend identity once. GDN selection records actual experiment
# evidence; it is not a user approval mechanism or a claim of bitwise parity.
python3 - "$task_root" "$backend" "$tool_relative" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);backend=sys.argv[2];tools=root/sys.argv[3]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def child(name):
 p=(root/name).resolve(); assert p.is_relative_to(root.resolve()),"Receipt escapes task root";return p
def receipt(item):
 p=child(item["path"]);assert p.is_file() and sha(p)==item["sha256"],str(p)+" receipt changed";return p
names=("run_flora_tshirt_batch.py","prepare_flora_windows.py","run_fineart_auto_batch.py",
       "prepare_fineart_subtask_review.py","auto_fineart_timeline.py","select_fineart_candidates.py",
       "materialize_fineart_language.py","tshirt-fineart-v1.json","run_tshirt_annotation_pipeline.sh")
binding={"backend":backend,"tool_relative":sys.argv[3],"tools_sha256":{n:sha(tools/n) for n in names},
 "runner_sha256":sha(tools/"run_flora_tshirt_batch.py"),
 "windows_tool_sha256":sha(tools/"prepare_flora_windows.py"),"task_root":str(root)}
if backend=="gdn":
 p=root/"backend-selection.json";s=json.loads(p.read_text())
 assert s.get("validated_selection") is True and s.get("backend")=="gdn" and s.get("status"),"GDN selection missing"
 assert s.get("overlay_root")=="gdn-overlay-candidate-02" and s.get("selection_reason"),"Selection must explain actual evidence"
 assert int(receipt(s["build_exit"]).read_text().strip())==0,"Candidate build failed"
 assert int(receipt(s["import_exit"]).read_text().strip())==0,"Candidate import failed"
 kernel=json.loads(receipt(s["kernel_report"]).read_text())
 assert kernel.get("status")=="DRAFT_KERNEL_NUMERIC_CHECKS_PASS_NOT_FLORA_PARITY","Kernel receipt is not the numeric draft pass"
 assert kernel.get("cases") and all(x.get("draft_pass") is True for x in kernel["cases"]),"Kernel cases incomplete"
 assert s.get("comparison_reports"),"Actual comparison report receipts required"
 for item in s["comparison_reports"]: receipt(item)
 assert (child(s["overlay_root"])/"site").is_dir() and (child(s["overlay_root"])/"build-tools").is_dir()
 binding["selection_sha256"]=sha(p)
p=root/"logs"/f"pipeline-{backend}.binding.json"
if p.exists(): assert json.loads(p.read_text())==binding,"Pipeline identity changed; preserve old evidence and choose a new version"
else:
 with p.open("x") as out: json.dump(binding,out,indent=2);out.write("\n")
PY

run_batch() {
  local label="$1" manifest="$2" expected="$3"
  local output_host="$result_host/$label" output_container="$result_container/$label"
  state "FREEZING_$label"
  # Results were created by the inference container; use that same owner for
  # binding files rather than changing ownership of existing result trees.
  docker exec -i "$container" python - "$container_root" "$backend" "$manifest" "$output_container" "$expected" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);backend=sys.argv[2];manifest=root/sys.argv[3];output=pathlib.Path(sys.argv[4]);expected=int(sys.argv[5])
pipeline_binding=json.loads((root/"logs"/f"pipeline-{backend}.binding.json").read_text())
tools=root/pipeline_binding["tool_relative"]
for name,expected_sha in pipeline_binding["tools_sha256"].items():
 assert hashlib.sha256((tools/name).read_bytes()).hexdigest()==expected_sha,"Tool/policy changed during pipeline: "+name
if backend=="gdn": assert hashlib.sha256((root/"backend-selection.json").read_bytes()).hexdigest()==pipeline_binding["selection_sha256"],"Backend selection changed"
raw=manifest.read_bytes();jobs=[json.loads(s) for s in raw.splitlines() if s.strip()]
assert jobs and len({j["id"] for j in jobs})==len(jobs),"Empty/duplicate job manifest"
if expected: assert len(jobs)==expected,"Pilot must contain all 23 windows"
if not expected:
 accepted={(r["pool"],r["source_episode_index"]) for line in (root/"dataset/audit.jsonl").read_text().splitlines() if line.strip()
           for r in [json.loads(line)] if r["pool"] in ("train","dev") and r["status"]=="PREPARED_REVIEW_PENDING"}
 lineage=json.loads(manifest.with_name("lineage.json").read_text())
 assert {(w["pool"],w["source_episode_index"]) for w in lineage["windows"]}==accepted,"Train/dev parent coverage differs"
 assert len(lineage["windows"])==len(jobs),"Lineage/job count mismatch"
 assert all(w["pool"] in ("train","dev") for w in lineage["windows"]),"Reserve must stay download-only"
for j in jobs:
 assert j.get("profile")=="yam" and j.get("instruction")=="fold and stack the t-shirts","Native profile/instruction changed"
 assert all(j[k].startswith("/data/tshirt-fineart-20261103/annotation/") for k in ("video","left_wrist","right_wrist")),"Foreign video path"
binding={"manifest_relative_path":sys.argv[3],"manifest_sha256":hashlib.sha256(raw).hexdigest(),"jobs":len(jobs),
         "pipeline_binding":pipeline_binding}
if output.exists() and any(output.iterdir()) and not (output/"pipeline-binding.json").is_file():
 raise RuntimeError("Existing unbound results preserved; use a new version")
output.mkdir(parents=True,exist_ok=True)
p=output/"pipeline-binding.json"
if p.exists(): assert json.loads(p.read_text())==binding,"Frozen results manifest/tool/backend changed"
else:
 with p.open("x") as stream:json.dump(binding,stream,indent=2);stream.write("\n")
p=output/"input-manifest.jsonl"
if p.exists(): assert p.read_bytes()==raw,"Frozen manifest changed"
else:
 with p.open("xb") as stream:stream.write(raw)
PY
  state "WAITING_GPU_$label"
  # Read-only FD: never create/replace/truncate the existing viewer GPU lock.
  exec 9<"$flora_host/catalog/gpu-worker.lock"
  flock -x 9
  state "ANNOTATING_$label"
  local code=0
  if [[ "$backend" == native ]]; then
    docker exec -w /workspace/Flora "$container" env -u PYTHONPATH python "$runner" \
      --flora-root /workspace/Flora --input-jsonl "$output_container/input-manifest.jsonl" \
      --output-dir "$output_container" --mode session-cache --lock-file /data/.flora-annotation-gpu.lock || code=$?
  else
    local overlay="$container_root/gdn-overlay-candidate-02"
    docker exec -w /workspace/Flora -e PYTHONPATH="$overlay/site:$overlay/build-tools" \
      -e TRITON_CACHE_DIR="$overlay/triton-cache" -e TORCH_EXTENSIONS_DIR="$overlay/torch-extensions" \
      "$container" python "$runner" --flora-root /workspace/Flora \
      --input-jsonl "$output_container/input-manifest.jsonl" --output-dir "$output_container" \
      --mode session-cache --lock-file /data/.flora-annotation-gpu.lock || code=$?
  fi
  flock -u 9
  exec 9<&-
  ((code==0)) || return "$code"
}

record_normalized_batch() {
  docker exec -i "$container" python - "$container_root" "$backend" "$1" "$2" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);backend,label=sys.argv[2:4];output=pathlib.Path(sys.argv[4])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert output.resolve().is_relative_to((root/"auto").resolve())
value={"output_root":str(output),"summary_sha256":sha(output/"summary.json"),
       "pipeline_binding_sha256":sha(root/"logs"/f"pipeline-{backend}.binding.json"),"training_ready":False}
p=root/"auto"/f"{label}.pipeline-result.json"
if p.exists(): assert json.loads(p.read_text())==value,"Completed batch pointer changed"
else:
 with p.open("x") as stream: json.dump(value,stream,indent=2);stream.write("\n")
PY
}

normalize_batch() {
  local label="$1" lineage="$2" output="$container_root/auto/$1"
  state "NORMALIZING_$label"
  # A successful retry is durable across host wrapper restarts.
  if docker exec "$container" test -f "$container_root/auto/$label.pipeline-result.json"; then
    output=$(docker exec -i "$container" python - "$container_root" "$backend" "$label" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);backend,label=sys.argv[2:]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((root/"auto"/f"{label}.pipeline-result.json").read_text());p=pathlib.Path(v["output_root"])
assert p.resolve().is_relative_to((root/"auto").resolve())
assert sha(p/"summary.json")==v["summary_sha256"],"Completed batch summary changed"
assert sha(root/"logs"/f"pipeline-{backend}.binding.json")==v["pipeline_binding_sha256"],"Pipeline binding changed"
print(p)
PY
)
  fi
  # Revalidate completed runs; summary existence alone is not a resume receipt.
  if docker exec "$container" test -f "$output/summary.json"; then
    docker exec -i "$container" python - "$output" "$container_root/$lineage" "$container_root/dataset/audit.jsonl" "$tools" "$result_container/$label" <<'PY'
import hashlib,json,pathlib,sys
output,lineage,audit,tools,raw=map(pathlib.Path,sys.argv[1:])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
s=load(output/"summary.json");o=load(output/"ownership.json")
assert s["training_ready"] is False and s["status"] in ("LANGUAGE_COLUMNS_MATERIALIZED_NOT_DATASET_RELEASED","ANNOTATION_VIEW_NOT_MATERIALIZED_NOT_TRAINING_READY")
assert o["raw_results"]==str(raw) and o["unknown_mode"]=="native-empty-anchor-fallback"
assert sha(lineage)==o["lineage"]["sha256"]==sha(pathlib.Path(o["lineage"]["path"])),"Lineage snapshot changed"
assert sha(tools/"tshirt-fineart-v1.json")==o["policy"]["sha256"]==sha(pathlib.Path(o["policy"]["path"])),"Policy changed"
for name,receipt in o["tools"].items():
 assert sha(tools/name)==receipt["sha256"]==sha(pathlib.Path(receipt["path"])),"Frozen tool changed"
for receipt in s["artifacts"].values():
 assert sha(pathlib.Path(receipt["path"]))==receipt["sha256"],"Normalization artifact changed"
previous=pathlib.Path(o["parent_audit"]["path"])
assert sha(previous)==o["parent_audit"]["sha256"],"Frozen parent audit changed"
def records(path):
 rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
 value={(r["pool"],r["source_episode_index"]):r for r in rows}
 assert len(value)==len(rows),"Duplicate parent audit records"
 return value
old,current=records(previous),records(audit)
for window in load(lineage)["windows"]:
 key=(window["pool"],window["source_episode_index"])
 assert current[key]==old[key],"Covered source parent metadata changed"
if "materialization_manifest" in s["artifacts"]:
 manifest=pathlib.Path(s["artifacts"]["materialization_manifest"]["path"])
 for parent in load(manifest)["parents"]:
  receipt=parent["output_parquet"]
  path=pathlib.Path(receipt["path"])
  path=path if path.is_absolute() else manifest.parent/path
  assert sha(path)==receipt["sha256"],"Derived parent Parquet changed"
print(json.dumps({"resume":"VERIFIED","pending_parent_count":s["pending_parent_count"],"training_ready":False}))
PY
    record_normalized_batch "$label" "$output"
    return
  fi
  # Explicit failed runs are retained; a retry always receives a new directory.
  if docker exec "$container" test -d "$output"; then
    docker exec "$container" test -f "$output/failure.json" || {
      echo "Existing uncompleted normalization retained: $output; inspect its owner before retry." >&2; return 75;
    }
    output="$output-attempt-$run_id"
  fi
  # Host lock coordinates with the viewer; selector also owns its container lock.
  exec 9<"$flora_host/catalog/gpu-worker.lock"
  flock -x 9
  local code=0
  local -a runtime=(env -u PYTHONPATH)
  if [[ "$backend" == gdn ]]; then
    local overlay="$container_root/gdn-overlay-candidate-02"
    runtime=(env "PYTHONPATH=$overlay/site:$overlay/build-tools" "TRITON_CACHE_DIR=$overlay/triton-cache" "TORCH_EXTENSIONS_DIR=$overlay/torch-extensions")
  fi
  docker exec -w /workspace/Flora "$container" "${runtime[@]}" python "$tools/run_fineart_auto_batch.py" \
    --lineage "$container_root/$lineage" --parent-audit "$container_root/dataset/audit.jsonl" \
    --raw-results "$result_container/$label" --policy "$tools/tshirt-fineart-v1.json" \
    --model-dir "$selector_model" --flora-root /workspace/Flora --tools-dir "$tools" \
    --output-root "$output" --unknown-mode native-empty-anchor-fallback || code=$?
  flock -u 9
  exec 9<&-
  ((code==0)) || return "$code"
  record_normalized_batch "$label" "$output"
}

run_batch "pilot-$backend-v1" annotation/pilot-v2/jobs.jsonl 23
normalize_batch "pilot-$backend-v1" annotation/pilot-v2/lineage.json
state WAITING_DOWNLOAD_COMPLETION
# Deliberately no GPU locks while waiting, and no retries of failed native work.
while [[ ! -s "$logs/prepare-all.exit" ]]; do sleep 15; done
download_code=$(<"$logs/prepare-all.exit")
[[ "$download_code" =~ ^[0-9]+$ && "$download_code" -le 255 ]] || { echo 'Malformed download exit receipt' >&2; exit 65; }
if ((download_code!=0)); then echo "Download/preparation failed: $download_code; dataset remains pending." >&2; exit "$download_code"; fi
state VALIDATING_DATASET_COMPLETION
python3 - "$task_root" "$backend" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);value=json.loads((root/"dataset/completion.json").read_text())
assert value.get("status")=="PREPARATION_COMPLETE_NOT_TRAINING_READY" and value.get("processed")==value.get("expected")==411,"Parent preparation incomplete"
assert value.get("training_ready") is False,"Unexpected training-ready claim"
binding=json.loads((root/"logs"/f"pipeline-{sys.argv[2]}.binding.json").read_text())
assert hashlib.sha256((root/binding["tool_relative"]/"prepare_flora_windows.py").read_bytes()).hexdigest()==binding["windows_tool_sha256"],"Window tool changed during pipeline"
PY
state PREPARING_TRAIN_DEV_WINDOWS
# CPU-only work; reserve is intentionally not in --pools.
docker exec -w /workspace/Flora -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 "$container" \
  nice -n 10 ionice -c 2 -n 7 env -u PYTHONPATH python "$tools/prepare_flora_windows.py" \
  --root "$container_root/dataset" --flora-root /workspace/Flora \
  --output "$container_root/annotation/train-dev-v1" --pools train dev
run_batch "train-dev-$backend-v1" annotation/train-dev-v1/jobs.jsonl 0
normalize_batch "train-dev-$backend-v1" annotation/train-dev-v1/lineage.json
state AUTOMATIC_ANNOTATION_FINISHED_RELEASE_PENDING
