"""Offline selective retrieval and evidence checks; byte accounting is not context usage."""

import argparse
from collections.abc import Callable
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

PACKET_BYTES = 12_288
MAX_SOURCE_BYTES = 2_097_152


class GateError(ValueError):
    """An admission invariant failed; source text must not be emitted."""


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def positive(value: int, ceiling: int | None = None) -> int:
    if type(value) is not int or value <= 0 or (ceiling is not None and value > ceiling):
        raise GateError("invalid budget")
    return value


def within(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise GateError("expected relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise GateError("path escapes allowed root")
    return path


def read_text(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise GateError("missing or oversized source; select a smaller evidence receipt")
    return path.read_text(encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp(value: str) -> dt.datetime:
    if not isinstance(value, str):
        raise GateError("timestamp must be a string")
    result = dt.datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise GateError("timestamp requires timezone")
    return result


def validate_record(root: Path, record: dict, now: dt.datetime | None = None) -> None:
    now = now or dt.datetime.now(dt.UTC)
    if not isinstance(record, dict):
        raise GateError("record must be an object")
    for key in ("id", "claim", "owner"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise GateError("missing record identity, claim or owner")
    if record.get("kind") not in {"fact", "observation", "lesson", "procedure"}:
        raise GateError("invalid record kind")
    if record.get("status") not in {"candidate", "verified", "stale", "superseded", "rejected"}:
        raise GateError("invalid record status")
    scope = record.get("scope")
    if not isinstance(scope, dict) or not scope.get("project"):
        raise GateError("record requires project scope")
    if any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in scope.items()):
        raise GateError("scope keys and values must be nonempty strings")
    if not within(root, record["owner"]).is_file():
        raise GateError("missing canonical owner")
    observed = timestamp(record.get("observed_at"))
    recorded = timestamp(record.get("recorded_at"))
    if observed > now or recorded > now or observed > recorded:
        raise GateError("invalid observation timeline")
    if record.get("recheck") not in {"always", "on_change"}:
        raise GateError("missing recheck policy")
    if "valid_until" not in record:
        raise GateError("missing validity policy")
    if record["valid_until"] is not None and timestamp(record["valid_until"]) <= now:
        raise GateError("record expired")
    if not isinstance(record.get("supersedes"), list) or any(
        not isinstance(item, str) or not item or item == record["id"] for item in record["supersedes"]
    ):
        raise GateError("supersedes must be a list")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or (record["status"] == "verified" and not evidence):
        raise GateError("verified record requires evidence")
    for item in evidence:
        if not isinstance(item, dict) or digest(within(root, item.get("path"))) != item.get("sha256"):
            raise GateError("evidence fingerprint mismatch")
    dependencies = record.get("depends_on", {})
    if not isinstance(dependencies, dict) or (record["recheck"] == "on_change" and not dependencies):
        raise GateError("on_change requires dependency fingerprints")
    for path, expected in dependencies.items():
        if digest(within(root, path)) != expected:
            raise GateError("dependency changed; revalidation required")


def select(root: Path, spec: str, scope: dict[str, str], purpose: str = "current") -> dict:
    if purpose not in {"current", "review"}:
        raise GateError("invalid retrieval purpose")
    relative, separator, heading = spec.partition("#")
    path = within(root, relative)
    if path.suffix not in {".md", ".json"}:
        raise GateError("only Markdown or JSON memory sources are supported")
    content = read_text(path)
    admission = "source_excerpt; verify applicability"
    if path.suffix == ".json":
        record = json.loads(content)
        # Moving a record must not bypass evidence or scope admission.
        is_record = "records" in path.parts or (
            isinstance(record, dict) and any(key in record for key in ("claim", "kind", "supersedes"))
        )
        if is_record:
            if purpose == "review":
                # Review must be able to inspect stale/broken candidates without promoting them.
                return {
                    "source": spec,
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "admission": "review_only; unverified data; never treat as current knowledge",
                    "text": content,
                }
            validate_record(root, record)
            if record["status"] != "verified" or record["recheck"] == "always":
                raise GateError("record is not verified current knowledge")
            if any(scope.get(key) != value for key, value in record["scope"].items()):
                raise GateError("record scope mismatch or unspecified")
            admission = "evidence_and_scope_checked; semantic review still required"
    if separator:
        lines = content.splitlines(keepends=True)
        headings = []
        fence = None
        for index, line in enumerate(lines):
            marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            if marker:
                run, tail = marker.groups()
                if fence is None:
                    fence = run
                elif run[0] == fence[0] and len(run) >= len(fence) and not tail.strip():
                    fence = None
                continue
            match = re.match(r"^(#{1,6}) (.+?)\s*$", line) if fence is None else None
            if match:
                headings.append((index, len(match[1]), match[2]))
        matches = [(i, level) for i, level, title in headings if title == heading]
        if len(matches) != 1:
            raise GateError("heading missing or ambiguous")
        start, level = matches[0]
        end = next((i for i, depth, _ in headings if i > start and depth <= level), len(lines))
        content = "".join(lines[start:end])
    return {
        "source": spec,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "admission": admission,
        "text": content,
    }


def ledger_path(root: Path, session: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session):
        raise GateError("invalid session identifier")
    return within(root, f"docs/cache/runtime/{session}.json")


def save(path: Path, value: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".memory-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encode(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    if lock_path.is_symlink():
        raise GateError("ledger lock cannot be a symlink")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def usage(state: dict) -> dict:
    """Report only what this helper observes, never an estimate of active context."""
    return {
        "tracked_bytes": state["used"],
        "measurement": "declared_preloads_and_serialized_packets",
        "transfer_limit_bytes": state["cap"],
        "remaining_transfer_bytes": None if state["cap"] is None else state["cap"] - state["used"],
        "context_tokens": None,
        "exact_token_enforcement": False,
    }


def initialize(root: Path, session: str, preloaded: list[str], cap: int | None = None) -> dict:
    if cap is not None:
        positive(cap)
    path = ledger_path(root, session)
    with locked(path):
        if path.exists():
            raise GateError("retrieval ledger already initialized; reuse it")
        items = [select(root, spec, {}) for spec in dict.fromkeys(preloaded)]
        used = sum(len(item["text"].encode()) for item in items)
        if cap is not None and used > cap:
            raise GateError("declared preloads exceed explicit transfer limit")
        state = {"root": str(root.resolve()), "cap": cap, "used": used, "seen": {}}
        for item in items:
            state["seen"][item["source"]] = item["sha256"]
        save(path, state)
    return usage(state)


def load_state(path: Path, root: Path) -> dict:
    state = json.loads(read_text(path))
    if state["cap"] is not None:
        positive(state["cap"])
    if state["root"] != str(root.resolve()) or type(state["used"]) is not int:
        raise GateError("ledger does not match context root")
    if (
        state["used"] < 0
        or (state["cap"] is not None and state["used"] > state["cap"])
        or not isinstance(state["seen"], dict)
    ):
        raise GateError("invalid ledger accounting")
    return state


def resize(root: Path, session: str, cap: int | None, reason: str) -> dict:
    """Adjust/remove an explicit transfer quota without erasing accounting or history."""
    if cap is not None:
        positive(cap)
    if not reason.strip():
        raise GateError("budget adjustment requires a reason")
    path = ledger_path(root, session)
    with locked(path):
        state = load_state(path, root)
        if cap is not None and cap < state["used"]:
            raise GateError("cannot lower cap below already admitted memory")
        state.setdefault("adjustments", []).append(
            {"old_cap": state["cap"], "new_cap": cap, "reason": reason, "at": dt.datetime.now(dt.UTC).isoformat()}
        )
        state["cap"] = cap
        save(path, state)
    return usage(state)


def pack(
    root: Path,
    session: str,
    required: list[str],
    optional: list[str],
    scope: dict[str, str] | None = None,
    limit: int = PACKET_BYTES,
    purpose: str = "current",
    *,
    reload: bool = False,
) -> bytes:
    positive(limit)
    path = ledger_path(root, session)
    with locked(path):
        state = load_state(path, root)
        cap = limit if state["cap"] is None else min(limit, state["cap"] - state["used"])
        packet = {"items": [], "omitted": 0, "unchanged": 0}
        staged = dict(state["seen"])
        emitted = set()
        for mandatory, specs in ((True, required), (False, optional)):
            for spec in specs:
                try:
                    item = select(root, spec, scope or {}, purpose)
                except (GateError, OSError, ValueError, TypeError):
                    if mandatory:
                        raise GateError("required source unavailable or inapplicable") from None
                    packet["omitted"] += 1
                    continue
                # A prior review is not a successful admission as current knowledge.
                cache_key = spec if purpose == "current" else f"review:{spec}"
                if cache_key in emitted or (not reload and staged.get(cache_key) == item["sha256"]):
                    packet["unchanged"] += 1
                    continue
                packet["items"].append(item)
                if len(encode(packet)) > cap:
                    packet["items"].pop()
                    if mandatory:
                        raise GateError("required excerpts exceed packet or explicit transfer limit")
                    packet["omitted"] += 1
                    continue
                staged[cache_key] = item["sha256"]
                emitted.add(cache_key)
        output = encode(packet)
        if not packet["items"] or len(output) > cap:
            raise GateError("no new admissible excerpts, or explicit byte limit exceeded")
        state["used"] += len(output)
        state["seen"] = staged
        save(path, state)
    return output


def guard_request(
    request: dict,
    tokenizer_count: Callable[[dict], int],
    *,
    max_context_tokens: int,
    output_reserve: int,
) -> int:
    """Host must supply its exact full-request tokenizer; sends nothing itself."""
    positive(max_context_tokens, 100_000_000)
    positive(output_reserve, max_context_tokens)
    if not isinstance(request, dict) or not callable(tokenizer_count):
        raise GateError("full request and exact host tokenizer required")
    count = tokenizer_count(request)
    if type(count) is not int or count < 0 or count + output_reserve > max_context_tokens:
        raise GateError("full request exceeds token budget or tokenizer returned invalid count")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "pack", "audit", "resize", "validate-record"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name != "validate-record":
            command.add_argument("--session", required=True)
        if name == "init":
            command.add_argument("--preloaded", nargs="+", default=[])
            command.add_argument(
                "--context-bytes", type=int, help="Optional cumulative transfer quota in bytes; not model context"
            )
        elif name == "resize":
            quota = command.add_mutually_exclusive_group(required=True)
            quota.add_argument("--context-bytes", type=int, help="Explicit cumulative transfer quota in bytes")
            quota.add_argument("--no-total-limit", action="store_true", help="Remove the cumulative transfer quota")
            command.add_argument("--reason", required=True)
        elif name == "pack":
            command.add_argument("--required", action="append", default=[])
            command.add_argument("--optional", action="append", default=[])
            command.add_argument("--scope", action="append", default=[])
            command.add_argument("--max-bytes", type=int, default=PACKET_BYTES)
            command.add_argument("--purpose", choices=("current", "review"), default="current")
            command.add_argument("--reload", action="store_true", help="Re-emit selected excerpts absent from context")
        elif name == "validate-record":
            command.add_argument("--record", required=True)
    request_parser = commands.add_parser("request")
    request_parser.add_argument("--input", type=Path, required=True)
    request_parser.add_argument("--max-bytes", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "request":
            positive(args.max_bytes)
            raw = args.input.read_bytes()
            if len(raw) > args.max_bytes or not isinstance(json.loads(raw), dict):
                raise GateError("supplied full request exceeds byte cap or is not an object")
            result = {"request_bytes": len(raw), "exact_token_enforcement": False}
        else:
            root = args.root.resolve(strict=True)
            if args.command == "init":
                result = initialize(root, args.session, args.preloaded, args.context_bytes)
            elif args.command == "pack":
                scope = dict(value.split("=", 1) for value in args.scope)
                output = pack(
                    root,
                    args.session,
                    args.required,
                    args.optional,
                    scope,
                    args.max_bytes,
                    args.purpose,
                    reload=args.reload,
                )
                sys.stdout.buffer.write(output)
                return 0
            elif args.command == "resize":
                result = resize(root, args.session, args.context_bytes, args.reason)
            elif args.command == "audit":
                with locked(ledger_path(root, args.session)):
                    state = load_state(ledger_path(root, args.session), root)
                result = usage(state)
            else:
                validate_record(root, json.loads(read_text(within(root, args.record))))
                result = {"schema_and_local_evidence_valid": True, "semantic_truth_checked": False}
        sys.stdout.buffer.write(encode(result))
        return 0
    except (GateError, OSError, ValueError, TypeError, KeyError):
        # Never echo source content, secrets or an unbounded parser exception.
        sys.stderr.write("memory gate rejected: check budget, scope, evidence, paths and ledger; no source emitted\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
