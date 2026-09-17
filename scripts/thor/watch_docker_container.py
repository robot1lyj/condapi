"""Wait while one Docker inference container is running, without `docker wait`.

The MAXN session can use this as its foreground command. Brief restarts are
tolerated; an explicit or lasting stop lets the session restore 120W/clocks.
"""

import argparse
import subprocess
import time


def running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--poll-s", type=float, default=2.0)
    parser.add_argument("--stop-grace-s", type=float, default=12.0)
    args = parser.parse_args()
    if args.poll_s <= 0 or args.stop_grace_s < 0:
        parser.error("Polling interval must be positive and stop grace nonnegative")
    if not running(args.container):
        parser.error("Inference container must already be running")
    print(f"Watching inference container {args.container}", flush=True)
    stopped_at = None
    while True:
        if running(args.container):
            stopped_at = None
        elif stopped_at is None:
            stopped_at = time.monotonic()
        elif time.monotonic() - stopped_at >= args.stop_grace_s:
            print(f"Inference container {args.container} stopped; restore performance state", flush=True)
            return
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
