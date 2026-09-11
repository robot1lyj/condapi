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
- Capability entrypoint/config changes: old verification stops being admissible; invocation/check text is never executed by memory tools.
- A verified unsuccessful attempt: discovery exposes its verdict as historical evidence, never as a recommended repair. Confounded experiments may remain inconclusive.
- Runtime mismatch/unknown observation: preserve expected value, measured value, unit/time and recheck method; a hash match never becomes live readiness.
- Problem search: current scope precedes lexical ranking; candidate/stale/broken records only appear in explicitly unverified review. Search omits evidence bodies, accounts for its output bytes and does not suppress a subsequent full record read.

## Task-level review

- Resume deployment with a large training history: retrieve the current deployment contract and relevant checkpoint evidence; leave unrelated experiments in their original owner.
- An experiment summary omits units or contradicts current configuration: expand the source and dependencies before deciding; smaller input alone is not success.
- Long logs: filter to the failure and required surrounding evidence outside the model; do not pass the full log in successive small packets.
- A complete task checkpoint preserves objective, user constraints, confirmed results, unresolved evidence, sources and next action. Producing it does not mean host compaction occurred.
- No full-request telemetry: proceed with selective retrieval and state the measurement boundary when relevant; do not invent context percentages or stop because the old default counter was exhausted.

Measure missing required evidence, stale/mismatched reuse, provenance completeness, unnecessary retrieval, task recovery correctness and regressions. Report actual context tokens only when the host measures them; otherwise label retrieval bytes separately. Automated tests measure mechanisms; semantic case review must still check causal claims and applicability. Fixed tests alone do not prove real-world improvement.

## Lightweight task replay record

For a meaningful change to a reusable procedure or problem route, preserve a small case at the existing project evaluation/report owner. Keep these groups, using null for measurements not obtained:

- **Case:** task prompt, applicable scope, frozen raw input/evidence references, and the initial user constraints. Remove credentials. Label synthetic cases explicitly.
- **Expected behavior:** required facts/conditions, admissible source references, useful next check and conclusions that the evidence cannot support. Keep this answer key out of the evaluated agent's initial context.
- **Comparison:** baseline memory version and candidate memory version, relevant model/tool/environment settings, and which historical fixes are visible to each run. Use the same inputs/settings where possible; record differences as confounders.
- **Observed behavior:** recovered next action, missed conditions, stale/mismatched reuse, repeated ineffective interventions without a changed retry condition, irrelevant excerpts loaded, elapsed time and measured retrieval bytes. Host-measured context tokens are a separate optional metric.
- **Decision:** retain, revise or revert the scoped update, with observed evidence and remaining uncertainty. A shorter run is not a pass when it loses a required condition or relies on a future result.

Replay diagnosis from frozen logs/configs in an isolated workspace; simulate tool responses if necessary and label them. Do not execute stored training/deployment commands as part of memory testing. A test showing retrieval finds the expected record verifies the lookup mechanism, not that an agent made the right engineering decision. Until a comparable task-level run exists, report real-world benefit as unmeasured.
