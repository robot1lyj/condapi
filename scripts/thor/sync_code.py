"""Fast-forward Thor to the workstation commit already published on Gitea.

No automatic stashing, overwriting, image rebuild, or inference restart.
The first adoption of Thor's rsync copy is completed separately after key access.
"""

import argparse
from pathlib import Path
import re
import shlex
import subprocess


def checked(command, cwd=None):
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def sync(local_root, host, remote_root, run=checked):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]*", host):
        raise ValueError("Use a plain SSH host alias or address")

    def local(*args):
        return run(["git", *args], cwd=local_root)

    def remote(*args):
        return run(["ssh", "-o", "BatchMode=yes", host, shlex.join(["git", "-C", remote_root, *args])])

    if local("branch", "--show-current") != "main" or local("status", "--porcelain"):
        raise RuntimeError("Commit workstation changes on main before syncing")
    revision = local("rev-parse", "HEAD")
    advertised = local("ls-remote", "origin", "refs/heads/main").split()
    if not advertised or advertised[0] != revision:
        raise RuntimeError("Push this workstation commit to Gitea main first")
    if remote("remote", "get-url", "origin") != local("remote", "get-url", "origin"):
        raise RuntimeError("Thor origin must match the configured workstation Gitea repository")
    # An unborn HEAD fails here instead of overwriting the original rsync copy.
    remote("rev-parse", "--verify", "HEAD")
    if remote("branch", "--show-current") != "main" or remote("status", "--porcelain"):
        raise RuntimeError("Thor has local changes or is not on main; nothing overwritten")
    remote("fetch", "origin", "main")
    if remote("rev-parse", "origin/main") != revision:
        raise RuntimeError("Gitea changed during synchronization; retry after checking the workstation")
    remote("merge", "--ff-only", revision)
    if remote("rev-parse", "HEAD") != revision:
        raise RuntimeError("Thor revision verification failed")
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="thor-usb")
    parser.add_argument("--remote-root", default="/home/wuyan-lyj/condapi")
    args = parser.parse_args()
    try:
        revision = sync(Path(__file__).resolve().parents[2], args.host, args.remote_root)
    except subprocess.CalledProcessError as error:
        parser.exit(1, error.stderr + "\nInitial Gitea authorization/checkout may still be pending.\n")
    except (RuntimeError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    print(f"Thor synchronized: {revision}; container images and power mode unchanged")


if __name__ == "__main__":
    main()
