"""Read-only, local training dashboard; standard library only, no model/token usage.

Serve a fixed JSONL path. Optionally mirror that one file over SSH every 10 seconds.
Never starts training, edits checkpoints, or exposes the repository as a file server.
"""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time

HTML = Path(__file__).with_name("training_dashboard.html")
MAX_BYTES = 32 * 1024 * 1024
MAX_ROWS = 20_000


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_metrics(path, batch_size):
    """Ignore partial writes; a backward step invalidates the abandoned future branch."""
    if not path.exists():
        return {"rows": [], "modified_at": None, "invalid_lines": 0, "trimmed": False}
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("指标文件超过 32 MiB; 请归档旧日志或提高经过审核的读取上限")
    rows = deque(maxlen=MAX_ROWS)
    invalid = 0
    count = 0
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        try:
            source = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            invalid += 1
            continue
        if not isinstance(source, dict):
            invalid += 1
            continue
        if any(isinstance(value, float) and not math.isfinite(value) for value in source.values()):
            invalid += 1
        step = source.get("step")
        if not number(step) or step < 0 or int(step) != step:
            continue
        row = {key: value for key, value in source.items() if number(value)}
        row["step"] = int(step)
        seconds = row.get("step_seconds", row.get("seconds"))
        if seconds is not None and seconds > 0:
            row["step_seconds"] = seconds
            row.setdefault("samples_per_second", batch_size / seconds)
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


class Dashboard:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.sync = {"enabled": bool(args.remote), "last_success": None, "error": None}

    def snapshot(self):
        with self.lock:
            sync = dict(self.sync)
        return {
            **read_metrics(self.args.metrics, self.args.batch_size),
            "server_time": time.time(),
            "sync": sync,
            "run": {
                "name": self.args.name,
                "source": str(self.args.metrics),
                "batch_size": self.args.batch_size,
                "stage_steps": self.args.stage_steps,
                "total_steps": self.args.total_steps,
                "save_interval": self.args.save_interval,
                "parameters": {
                    "模型": "Pi0.5 base · 全量微调",
                    "全局 batch": self.args.batch_size,
                    "FSDP": "4 卡",
                    "精度": "BF16 计算 / FP32 状态",
                    "EMA": "关闭",
                    "峰值学习率": "2.5e-5",
                    "Warmup": "1,000 steps",
                    "学习率周期": f"cosine / {self.args.total_steps:,} steps",
                    "优化器": "AdamW · β 0.9 / 0.95",
                    "梯度裁剪": "1.0",
                    "数据": "YAM Lego · 4,458 train episodes",
                    "Action horizon": "50",
                },
            },
        }

    def mirror(self, stop):
        """Atomic local replacement. SSH failure retains the last successful snapshot."""
        while not stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "ssh",
                        "-oBatchMode=yes",
                        "-oConnectTimeout=8",
                        self.args.remote,
                        "test -f "
                        + self.args.remote_metrics
                        + " || exit 44; stat -c %Y "
                        + self.args.remote_metrics
                        + "; head -c 33554433 "
                        + self.args.remote_metrics,
                    ],
                    capture_output=True,
                    timeout=20,
                    check=True,
                )
                timestamp, payload = result.stdout.split(b"\n", 1)
                source_modified = float(timestamp)
                if len(payload) > MAX_BYTES:
                    raise ValueError("远端日志超过读取上限; 未替换本地缓存")
                self.args.metrics.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.args.metrics.with_suffix(".pending")
                temporary.write_bytes(payload)
                temporary.replace(self.args.metrics)
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
            route = self.path.split("?", 1)[0]
            if route == "/":
                body, content_type, status = HTML.read_bytes(), "text/html; charset=utf-8", 200
            elif route == "/api/metrics":
                try:
                    data, status = dashboard.snapshot(), 200
                except (OSError, ValueError):
                    data, status = {"error": "无法读取指标文件, 请检查权限、格式或文件大小"}, 503
                body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
                content_type = "application/json; charset=utf-8"
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
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--name", default="pi05 · lego sorting / full finetune")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage-steps", type=int, default=40_000)
    parser.add_argument("--total-steps", type=int, default=162_097)
    parser.add_argument("--save-interval", type=int, default=20_000)
    parser.add_argument("--remote", help="Optional SSH alias, e.g. yam-server")
    parser.add_argument("--remote-metrics", help="Absolute remote JSONL file; no shell metacharacters")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.stage_steps, args.total_steps, args.save_interval, args.interval) <= 0:
        parser.error("counts and interval must be positive")
    if args.stage_steps > args.total_steps:
        parser.error("stage steps must not exceed total steps")
    if args.remote or args.remote_metrics:
        if not args.remote or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", args.remote):
            parser.error("invalid SSH alias")
        if not args.remote_metrics or not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.remote_metrics):
            parser.error("remote-metrics must be an absolute path without shell metacharacters")
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
