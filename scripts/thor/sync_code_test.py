import shlex

import pytest

from scripts.thor.sync_code import sync


def fake_git(*, dirty=False, published="new"):
    calls = []
    remote_head = "old"

    def run(command, cwd=None):
        nonlocal remote_head
        remote = command[0] == "ssh"
        args = shlex.split(command[-1])[3:] if remote else command[1:]
        calls.append((remote, args))
        if args == ["branch", "--show-current"]:
            return "main"
        if args == ["status", "--porcelain"]:
            return " M code.py" if remote and dirty else ""
        if args == ["ls-remote", "origin", "refs/heads/main"]:
            return f"{published}\trefs/heads/main"
        if args == ["remote", "get-url", "origin"]:
            return "ssh://git@gitea/project.git"
        if args[:2] == ["rev-parse", "--verify"]:
            return remote_head
        if args == ["rev-parse", "HEAD"]:
            return remote_head if remote else "new"
        if args == ["rev-parse", "origin/main"]:
            return "new"
        if args == ["merge", "--ff-only", "new"]:
            remote_head = "new"
        return ""

    return run, calls


def test_sync_fast_forwards_published_revision():
    run, calls = fake_git()
    assert sync("/local", "thor-usb", "/remote", run) == "new"
    assert (True, ["merge", "--ff-only", "new"]) in calls


def test_sync_preserves_thor_changes():
    run, calls = fake_git(dirty=True)
    with pytest.raises(RuntimeError, match="local changes"):
        sync("/local", "thor-usb", "/remote", run)
    assert not any(args[0] in ("fetch", "merge", "reset", "stash") for _, args in calls)


def test_sync_refuses_unpublished_workstation_revision():
    run, calls = fake_git(published="old")
    with pytest.raises(RuntimeError, match="Push"):
        sync("/local", "thor-usb", "/remote", run)
    assert not any(remote for remote, _args in calls)
