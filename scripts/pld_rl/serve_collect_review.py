"""Reusable localhost review server; only derived review files are writable."""

import argparse
import datetime
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.review_policy import human_reward
from yam_pld_rl.review_policy import merge_rewards


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--curator", default="curator-v2")
    p.add_argument("--vision", default="vision-v1")
    args = p.parse_args()
    root = args.dataset.resolve()
    candidates = json.loads((root / args.curator / "candidates.json").read_text())
    judged = {c["candidate_id"]: c for c in json.loads((root / args.vision / "candidates.json").read_text())}
    by_id = {c["candidate_id"]: c for c in candidates if c.get("outcome")}
    import hashlib  # noqa: PLC0415

    identity = hashlib.sha256((root / args.curator / "candidates.json").read_bytes()).hexdigest()
    folder = root / "reviews"
    folder.mkdir(exist_ok=True)
    state_file = folder / "user-review-state.json"
    previous = json.loads(state_file.read_text()) if state_file.exists() else {"identity": identity, "reviews": {}}
    if previous["identity"] != identity:
        raise ValueError("saved reviews belong to a different evidence release")
    state = previous["reviews"]
    lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def reply(self, value, status=200):
            data = (json.dumps(value, ensure_ascii=False) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/reviews":
                with lock:
                    self.reply({"identity": identity, "reviews": state})
                return
            if path == "/api/rewards":
                with lock:
                    self.reply({"identity": identity, **merge_rewards(candidates, judged, state)})
                return
            if path in ("/", "/review.html", "/review-collect13.html"):
                self.path = "/review.html"
            super().do_GET()

        def do_POST(self):
            host = self.headers.get("Host")
            origin = self.headers.get("Origin")
            if host not in (f"127.0.0.1:{args.port}", f"localhost:{args.port}") or (origin and origin not in (
                    f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}")):
                self.reply({"error": "same-origin localhost review only"}, 403)
                return
            if urlsplit(self.path).path != "/api/reviews":
                self.reply({"error": "unknown endpoint"}, 404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 262144:
                    raise ValueError("invalid review request size")
                body = json.loads(self.rfile.read(size))
                if body["identity"] != identity:
                    raise ValueError("review belongs to a different evidence release")
                changes = body["changes"]
                if not isinstance(changes, dict) or any(not isinstance(c, dict) for c in changes.values()):
                    raise ValueError("changes must map candidate IDs to review choices")
                updated = {}
                for cid, choice in changes.items():
                    if cid not in by_id or choice.get("status", "") not in (
                            "", "held", "empty", "not_lifted", "no_closure", "dropped", "exclude"):
                        raise ValueError("unknown candidate/outcome")
                    record = {k: choice.get(k) for k in ("status", "grasp_intent_confirmed", "held_then_lost_confirmed", "note")}
                    record.update(reviewer=choice.get("reviewer") or "user_review", candidate_sha256=by_id[cid]["candidate_sha256"])
                    record["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                    try:
                        if record["status"] not in ("", "exclude"):
                            human_reward(by_id[cid], record)
                        record.update(reward_valid=record["status"] != "", error=None)
                    except (ValueError, KeyError, IndexError) as exc:
                        record.update(reward_valid=False, error=str(exc))
                    updated[cid] = record
                with lock:
                    state.update(updated)
                    tmp = state_file.with_suffix(".tmp")
                    tmp.write_text(json.dumps({"identity": identity, "reviews": state}, ensure_ascii=False, indent=2) + "\n")
                    tmp.replace(state_file)
                    with (folder / "review-events.jsonl").open("a") as stream:
                        stream.write(json.dumps({"identity": identity, "changes": updated}, ensure_ascii=False) + "\n")
                self.reply({"identity": identity, "reviews": updated})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply({"error": str(exc)}, 400)

    print(json.dumps({"url": f"http://127.0.0.1:{args.port}/review.html", "dataset": str(root)}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
