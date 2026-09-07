"""Record a foreground inference command and its own Thor telemetry process."""

import argparse
import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Specify a command")
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.prefix.with_suffix(".log").open("x") as log,
        args.prefix.with_suffix(".tegrastats").open("x") as telemetry,
    ):
        monitor = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=telemetry, stderr=subprocess.STDOUT)
        try:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    with args.prefix.with_suffix(".exit.json").open("x") as output:
        json.dump({"exit_code": code, "finished_at": datetime.datetime.now(datetime.UTC).isoformat()}, output)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
