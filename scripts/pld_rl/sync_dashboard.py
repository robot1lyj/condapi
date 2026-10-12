"""Mirror four configured runs read-only for the existing VLA dashboard."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

IDS = ["calql-success-s0", "calql-success-s1", "calql-mixed-s0", "td-success-s0"]
FILES = ["metrics.jsonl", "run.json", "status.json", "latest-evaluation.json"]


def sync(host, root, destination, ids=IDS):
    # Root is transmitted as JSON on stdin, never interpolated into shell code.
    source = """import base64,json,pathlib,sys
root=pathlib.Path(json.loads(sys.stdin.readline()))
rows=[]
for run in __IDS__:
 for name in __FILES__:
  path=root/run/name
  if path.is_file():
   data=path.read_bytes()
   if len(data)>32*1024*1024: raise ValueError('metric file too large')
   rows.append({'path':run+'/'+name,'data':base64.b64encode(data).decode(),'mtime':path.stat().st_mtime})
print(json.dumps(rows))
""".replace("__IDS__", repr(ids)).replace("__FILES__", repr(FILES))
    command = "python3 -c " + shlex.quote(source)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        input=json.dumps(root) + "\n",
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    rows = json.loads(result.stdout)
    permitted = {run + "/" + name for run in ids for name in FILES}
    identities = {}
    for row in rows:
        if row["path"] not in permitted:
            raise ValueError("unexpected remote file")
        payload = base64.b64decode(row["data"], validate=True)
        path = destination / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.utime(temporary, (row["mtime"], row["mtime"]))
        temporary.replace(path)
        identities[row["path"]] = hashlib.sha256(payload).hexdigest()
    return identities


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="yam-server")
    p.add_argument("--remote-root", required=True)
    p.add_argument("--local-root", type=Path, required=True)
    p.add_argument("--interval", type=int, default=10)
    p.add_argument("--once", action="store_true")
    p.add_argument("--ids", help="comma-separated immutable run IDs; defaults to the first four experiments")
    args = p.parse_args()
    args.local_root.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            status = {
                "last_success": time.time(),
                "files": sync(args.host, args.remote_root, args.local_root, args.ids.split(",") if args.ids else IDS),
                "error": None,
            }
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            status = {"checked_at": time.time(), "error": str(exc)}
        (args.local_root / "sync-state.json").write_text(json.dumps(status, indent=2) + "\n")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
