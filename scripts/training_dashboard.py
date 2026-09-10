"""Read-only, local training dashboard; standard library only, no model/token usage.

Serve configured JSONL, CSV or Trainer-state sources. Optionally mirror one file over SSH.
Never starts training, edits checkpoints, or exposes the repository as a file server.
"""

import argparse
from collections import deque
import contextlib
import csv
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import parse_qs
from urllib.parse import urlsplit

HTML = Path(__file__).with_name("training_dashboard.html")
MAX_BYTES = 32 * 1024 * 1024
MAX_ROWS = 20_000


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


ALIASES = {
    "step": ("step", "global_step", "steps", "trainer/global_step"),
    "loss": ("loss", "train/loss", "train_loss"),
    "val_loss": ("val_loss", "eval_loss", "eval/loss", "validation/loss"),
    "learning_rate": ("learning_rate", "lr", "train/lr"),
}


def flatten(value, prefix="", depth=0):
    if depth > 8:
        raise ValueError("metric nesting exceeds eight levels")
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(flatten(item, name, depth + 1))
        else:
            result[name] = item
    return result


def normalize(source, field_map=None, *, csv_values=False):
    """Names may be mapped, but units and numeric precision are never guessed."""
    source = dict(source)
    envelope = source.pop("metrics", None)
    values = flatten(source)
    if isinstance(envelope, dict):
        values.update(flatten(envelope))
    if csv_values:
        for key, value in values.items():
            with contextlib.suppress(TypeError, ValueError):
                values[key] = float(value)
    row = {key: value for key, value in values.items() if number(value)}
    for target, names in ALIASES.items():
        for name in names:
            if name in row:
                row.setdefault(target, row[name])
                break
    for target, name in (field_map or {}).items():
        # Explicit mappings override aliases, including an absent/invalid source.
        row.pop(target, None)
        if number(values.get(name)):
            row[target] = values[name]
    for key in ("schema_version", "timestamp", "wall_time", "global_step", "steps", "trainer/global_step"):
        row.pop(key, None)
    bad = any(isinstance(value, float) and not math.isfinite(value) for value in values.values())
    return row, bad


def read_metrics(path, batch_size=None, *, source_format="auto", field_map=None):
    """One run per source. Same-step train/eval events merge; rollback discards future."""
    if not path.exists():
        return {"rows": [], "modified_at": None, "invalid_lines": 0, "trimmed": False}
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("指标文件超过 32 MiB; 请归档旧日志或提高经过审核的读取上限")
    source_format = (
        source_format
        if source_format != "auto"
        else {".csv": "csv", ".json": "trainer-state"}.get(path.suffix, "jsonl")
    )
    invalid = 0
    if source_format == "trainer-state":
        # Rewritten JSON must be complete; a partial snapshot returns HTTP 503.
        state = json.loads(raw)
        if not isinstance(state, dict) or not isinstance(state.get("log_history"), list):
            raise ValueError("expected a Trainer state object with log_history")
        sources = state["log_history"]
    elif source_format == "csv":
        complete = b"".join(line for line in raw.splitlines(keepends=True) if line.endswith(b"\n"))
        sources = list(csv.DictReader(io.StringIO(complete.decode("utf-8-sig"))))
    else:
        sources = []
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue
            try:
                sources.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                invalid += 1
    rows = deque(maxlen=MAX_ROWS)
    count = 0
    for source in sources:
        if not isinstance(source, dict):
            invalid += 1
            continue
        row, bad = normalize(source, field_map, csv_values=source_format == "csv")
        invalid += int(bad)
        step = row.get("step")
        if not number(step) or step < 0 or int(step) != step:
            invalid += int(not bad)
            continue
        row["step"] = int(step)
        seconds = row.get("step_seconds", row.get("seconds"))
        if seconds is not None and seconds > 0:
            row["step_seconds"] = seconds
            if batch_size is not None:
                row.setdefault("samples_per_second", batch_size / seconds)
        if rows and rows[-1]["step"] == step:
            rows[-1].update(row)
        else:
            while rows and rows[-1]["step"] >= step:
                rows.pop()
            rows.append(row)
        count += 1
    return {
        "rows": list(rows),
        "modified_at": path.stat().st_mtime,
        "invalid_lines": invalid,
        "trimmed": count > MAX_ROWS,
    }


def combine_history(current, history):
    """Prepend parent-run rows and let the resumed run own overlapping steps."""
    current_rows = current["rows"]
    cutoff = current_rows[0]["step"] if current_rows else None
    by_step = {}
    for source in history:
        for row in source["rows"]:
            if cutoff is None or row["step"] < cutoff:
                by_step[row["step"]] = row
    for row in current_rows:
        by_step[row["step"]] = row
    rows = [by_step[step] for step in sorted(by_step)]
    return {
        **current,
        "rows": rows[-MAX_ROWS:],
        "invalid_lines": current["invalid_lines"] + sum(source["invalid_lines"] for source in history),
        "trimmed": current["trimmed"] or any(source["trimmed"] for source in history) or len(rows) > MAX_ROWS,
    }


class Dashboard:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.sync = {"enabled": bool(args.remote), "last_success": None, "error": None}
        self.history_remote_metrics = []
        if args.runs_config:
            value = json.loads(args.runs_config.read_text())
            if (
                not isinstance(value, dict)
                or value.get("schema_version") != 1
                or not isinstance(value.get("runs"), list)
                or not value["runs"]
            ):
                raise ValueError("runs config requires schema_version=1 and nonempty runs")
            specs = value["runs"]
            base = args.runs_config.resolve().parent
        else:
            specs = [
                {
                    "id": "default",
                    "metrics": str(args.metrics.resolve()),
                    "name": args.name,
                    "model": args.model,
                    "backend": args.backend,
                    "format": args.source_format,
                    "batch_size": args.batch_size,
                    "stage_steps": args.stage_steps,
                    "total_steps": args.total_steps,
                    "save_interval": args.save_interval,
                    "history_metrics": [str(path.resolve()) for path in args.history_metrics or []],
                }
            ]
            history_dir = args.metrics.parent / "history"
            for index, remote_path in enumerate(args.history_remote_metrics or []):
                local_path = history_dir / f"{index}.jsonl"
                specs[0]["history_metrics"].append(str(local_path.resolve()))
                self.history_remote_metrics.append((remote_path, local_path))
            if args.metadata:
                metadata = json.loads(args.metadata.read_text())
                if not isinstance(metadata, dict) or set(metadata) - {"parameters", "metric_labels", "loss_semantics"}:
                    raise ValueError("metadata accepts parameters, metric_labels and loss_semantics only")
                specs[0].update(metadata)
            base = Path.cwd()
        self.runs = {}
        for spec in specs:
            if not isinstance(spec, dict) or not isinstance(spec.get("metrics"), str):
                raise ValueError("each run requires a metrics path")
            run = dict(spec)
            for key in ("name", "model", "backend", "loss_semantics"):
                if key in run and not isinstance(run[key], str):
                    raise ValueError("run descriptions must be strings")
            name = run.get("id", "")
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name)
                or name in self.runs
            ):
                raise ValueError("run ids must be unique safe identifiers")
            run["metrics"] = (base / run["metrics"]).resolve()
            history_metrics = run.get("history_metrics", [])
            if not isinstance(history_metrics, list) or any(not isinstance(path, str) for path in history_metrics):
                raise ValueError("history_metrics must be a list of paths")
            run["history_metrics"] = [(base / path).resolve() for path in history_metrics]
            if run.get("format", "auto") not in ("auto", "jsonl", "csv", "trainer-state"):
                raise ValueError("unsupported metric format")
            for key in ("batch_size", "stage_steps", "total_steps", "save_interval"):
                if run.get(key) is not None and (type(run[key]) is not int or run[key] <= 0):
                    raise ValueError("run counts must be positive integers or omitted")
            if run.get("stage_steps") and run.get("total_steps") and run["stage_steps"] > run["total_steps"]:
                raise ValueError("stage exceeds total steps")
            for key in ("field_map", "metric_labels", "parameters"):
                values = run.get(key, {})
                if not isinstance(values, dict) or any(
                    not isinstance(k, str) or not isinstance(v, (str, int, float, bool)) for k, v in values.items()
                ):
                    raise ValueError("run maps must contain scalar values")
                if key != "parameters" and any(not isinstance(v, str) for v in values.values()):
                    raise ValueError("field names and labels must be strings")
                if any(isinstance(v, float) and not math.isfinite(v) for v in values.values()):
                    raise ValueError("metadata must be finite")
            self.runs[name] = run

    def list_runs(self):
        return [
            {"id": key, "name": run.get("name", key), "model": run.get("model", "未声明")}
            for key, run in self.runs.items()
        ]

    def snapshot(self, run_id=None):
        run = self.runs[run_id or next(iter(self.runs))]
        with self.lock:
            sync = dict(self.sync)
        current = read_metrics(
            run["metrics"],
            run.get("batch_size"),
            source_format=run.get("format", "auto"),
            field_map=run.get("field_map"),
        )
        history = [
            read_metrics(
                path,
                run.get("batch_size"),
                source_format=run.get("format", "auto"),
                field_map=run.get("field_map"),
            )
            for path in run.get("history_metrics", [])
        ]
        return {
            **combine_history(current, history),
            "server_time": time.time(),
            "sync": sync,
            "run": {
                **{
                    key: value
                    for key, value in run.items()
                    if key not in ("metrics", "field_map", "history_metrics")
                },
                "name": run.get("name", run["id"]),
                "source": str(run["metrics"]),
                "parameters": run.get("parameters", {}),
                "loss_semantics": run.get("loss_semantics", "生产者提供的数值; 未声明瞬时值或区间均值"),
                "metric_labels": run.get("metric_labels", {}),
            },
        }

    def mirror(self, stop):
        """Atomic local replacement. SSH failure retains the last successful snapshot."""
        while not stop.is_set():
            try:
                sources = [(self.args.remote_metrics, self.args.metrics)] + self.history_remote_metrics
                source_modified = None
                for remote_path, local_path in sources:
                    result = subprocess.run(
                        [
                            "ssh",
                            "-oBatchMode=yes",
                            "-oConnectTimeout=8",
                            self.args.remote,
                            "test -f "
                            + remote_path
                            + " || exit 44; stat -c %Y "
                            + remote_path
                            + "; head -c 33554433 "
                            + remote_path,
                        ],
                        capture_output=True,
                        timeout=20,
                        check=True,
                    )
                    timestamp, payload = result.stdout.split(b"\n", 1)
                    if len(payload) > MAX_BYTES:
                        raise ValueError("远端日志超过读取上限; 未替换本地缓存")
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = local_path.with_suffix(".pending")
                    temporary.write_bytes(payload)
                    temporary.replace(local_path)
                    if local_path == self.args.metrics:
                        source_modified = float(timestamp)
                with self.lock:
                    self.sync.update(last_success=time.time(), source_modified_at=source_modified, error=None)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                with self.lock:
                    self.sync["error"] = (
                        ("等待远端训练日志创建 (尚未绑定已启动的训练)" if self.args.metrics.exists() else None)
                        if isinstance(exc, subprocess.CalledProcessError) and exc.returncode == 44
                        else f"{type(exc).__name__}: 远端同步失败, 保留上次数据"
                    )
            stop.wait(self.args.interval)


def make_server(dashboard, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            route = parsed.path
            if route == "/":
                body, content_type, status = HTML.read_bytes(), "text/html; charset=utf-8", 200
            elif route == "/api/metrics":
                try:
                    data, status = dashboard.snapshot(parse_qs(parsed.query).get("run", [None])[0]), 200
                except KeyError:
                    self.send_error(404, "unknown configured run")
                    return
                except (OSError, ValueError):
                    data, status = {"error": "无法读取指标文件, 请检查权限、格式或文件大小"}, 503
                body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
                content_type = "application/json; charset=utf-8"
            elif route == "/api/runs":
                body = json.dumps(dashboard.list_runs(), ensure_ascii=False).encode()
                content_type, status = "application/json; charset=utf-8", 200
            else:
                self.send_error(404)
                return
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--history-metrics", type=Path, action="append")
    parser.add_argument(
        "--runs-config", type=Path, help="Configured runs; relative metric paths resolve beside this JSON"
    )
    parser.add_argument("--metadata", type=Path, help="Single-run parameters and metric labels JSON")
    parser.add_argument("--model", default="未声明")
    parser.add_argument("--backend", default="未声明")
    parser.add_argument("--source-format", choices=("auto", "jsonl", "csv", "trainer-state"), default="auto")
    parser.add_argument("--name", default="本地训练指标")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--stage-steps", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--remote", help="Optional SSH alias, e.g. yam-server")
    parser.add_argument("--remote-metrics", help="Absolute remote JSONL file; no shell metacharacters")
    parser.add_argument(
        "--history-remote-metrics",
        action="append",
        help="Parent-run JSONL files on the same SSH host; repeated options are prepended to the current run",
    )
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args(argv)
    if bool(args.metrics) == bool(args.runs_config):
        parser.error("choose exactly one of --metrics and --runs-config")
    if args.runs_config and (
        args.remote
        or args.remote_metrics
        or args.metadata
        or args.history_metrics
        or args.history_remote_metrics
    ):
        parser.error("multi-run sources are local; mirror each remote source separately")
    if any(
        value is not None and value <= 0
        for value in (args.batch_size, args.stage_steps, args.total_steps, args.save_interval, args.interval)
    ):
        parser.error("counts and interval must be positive")
    if args.stage_steps and args.total_steps and args.stage_steps > args.total_steps:
        parser.error("stage steps must not exceed total steps")
    if args.remote or args.remote_metrics or args.history_remote_metrics:
        if not args.remote or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", args.remote):
            parser.error("invalid SSH alias")
        if not args.remote_metrics or not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.remote_metrics):
            parser.error("remote-metrics must be an absolute path without shell metacharacters")
        if any(not re.fullmatch(r"/[A-Za-z0-9_./-]+", path) for path in args.history_remote_metrics or []):
            parser.error("history remote metrics must be absolute paths without shell metacharacters")
    if (args.history_metrics or args.history_remote_metrics) and not args.metrics:
        parser.error("history metrics require a single --metrics source")
    return args


def main():
    args = parse_args()
    dashboard = Dashboard(args)
    stop = threading.Event()
    if args.remote:
        threading.Thread(target=dashboard.mirror, args=(stop,), daemon=True).start()
    server = make_server(dashboard, args.port)
    print(f"Training dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
