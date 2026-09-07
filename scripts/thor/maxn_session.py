"""Run one foreground inference command in MAXN, then restore 120W and saved clocks.

Run on the Thor host with sudo, not inside a container. No startup service is
installed. SIGINT/SIGTERM stop the command before restoring clocks; SIGKILL or
power loss cannot execute Python cleanup, so recheck power mode after a crash.
"""

import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import tempfile


def run_checked(*args):
    result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    return result.stdout


@contextlib.contextmanager
def performance_session(backup, run=run_checked):
    before = run("nvpmodel", "-q")
    if "NV Power Mode: 120W\n1" not in before.strip():
        raise RuntimeError("Start from the normal 120W mode; refusing to snapshot an unknown performance state")
    run("jetson_clocks", "--store", str(backup))
    try:
        run("nvpmodel", "-m", "0")
        if "NV Power Mode: MAXN\n0" not in run("nvpmodel", "-q").strip():
            raise RuntimeError("MAXN did not become active")
        # Keep the saved automatic fan policy. Do not use --fan: this BSP's
        # wildcard also writes an invalid 255 to pwm1_enable. Log temperatures
        # during actual runs; max clocks do not override thermal protection.
        run("jetson_clocks")
        run("jetson_clocks", "--show")
        yield
    finally:
        try:
            run("jetson_clocks", "--restore", str(backup))
        finally:
            run("nvpmodel", "-m", "1")
            run("nvpmodel", "-q")


def run_foreground(command):
    child = subprocess.Popen(command, start_new_session=True)

    def stop(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        return child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Specify a foreground inference command after --")
    if os.geteuid() != 0:
        parser.error("Run with sudo on Thor; power controls require root")
    if "Thor" not in Path("/proc/device-tree/model").read_text():
        parser.error("This launcher is specific to the Thor developer kit")
    with Path("/run/lock/thor-inference-maxn.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_dir = Path(tempfile.mkdtemp(prefix="thor-maxn-"))
        print(f"Clock backup (retained for recovery): {state_dir}", flush=True)

        def interrupted(signum, _frame):
            raise SystemExit(128 + signum)

        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            with performance_session(state_dir / "clocks.conf"):
                code = run_foreground(command)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
