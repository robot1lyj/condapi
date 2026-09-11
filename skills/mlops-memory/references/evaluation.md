# Behavioral acceptance

Run the standard-library tests with `python -m unittest discover -s skills/mlops-memory/tests -p '*_test.py'`.
Use temporary repositories and synthetic evidence; never access production IPCs, start GPU work or create actual training results for a memory test.

## Replay cases

- Long Chinese text, emoji and JSON escape characters: exact output bytes stay within budget; optional sections are omitted whole; source files remain unchanged.
- Mandatory evidence exceeds budget: no partial text or silent safety-clause truncation.
- Repeated calls/turns without an explicit quota: retrieval can pass the old 32,768-byte threshold; reported bytes are never presented as active-context tokens. Duplicate selections are not replayed by default.
- Explicit project quota and old ledgers: a failed admission preserves the ledger; changing/removing a quota preserves accounting and requires a recorded reason. No silent quota removal or new-ID bypass.
- A necessary complete section exceeds the default packet size: configurable packet limits allow it where authorized, while keeping conditions and source text intact.
- Needed evidence absent after compaction: selective `--reload` rechecks evidence/scope, emits and counts it again without duplicating within one packet.
- Concurrent packers: ledger serialization prevents overspending an explicit quota and partial accounting.
- Changed norm/config under the same path: old dependency fingerprint invalidates the lesson.
- Historical Slurm observation: never claim the job still runs without a new live observation.
- OpenArm/Piper vs YAM, server vs Thor: mismatched or missing scope excludes a record.
- Port opened, lower loss, BF16 passed: none alone proves a deployable FP8 engine or robot success.
- Evidence absent/corrupt, future timestamp, expired conclusion: no verified-current reuse.
- Full request budget includes history, tool schemas and reserved output; missing tokenizer/invalid counts fail closed.

## Task-level review

- Resume deployment with a large training history: retrieve the current deployment contract and relevant checkpoint evidence; leave unrelated experiments in their original owner.
- An experiment summary omits units or contradicts current configuration: expand the source and dependencies before deciding; smaller input alone is not success.
- Long logs: filter to the failure and required surrounding evidence outside the model; do not pass the full log in successive small packets.
- A complete task checkpoint preserves objective, user constraints, confirmed results, unresolved evidence, sources and next action. Producing it does not mean host compaction occurred.
- No full-request telemetry: proceed with selective retrieval and state the measurement boundary when relevant; do not invent context percentages or stop because the old default counter was exhausted.

Measure missing required evidence, stale/mismatched reuse, provenance completeness, unnecessary retrieval, task recovery correctness and regressions. Report actual context tokens only when the host measures them; otherwise label retrieval bytes separately. Automated tests measure mechanisms; semantic case review must still check causal claims and applicability. Fixed tests alone do not prove real-world improvement.
