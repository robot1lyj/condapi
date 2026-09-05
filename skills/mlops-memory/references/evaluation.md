# Behavioral acceptance

Run the standard-library tests with `python -m unittest discover -s skills/mlops-memory/tests -p '*_test.py'`.
Use temporary repositories and synthetic evidence; never access production IPCs, start GPU work or create actual training results for a memory test.

## Replay cases

- Long Chinese text, emoji and JSON escape characters: exact output bytes stay within budget; optional sections are omitted whole; source files remain unchanged.
- Mandatory evidence exceeds budget: no partial text or silent safety-clause truncation.
- Repeated calls/turns: cumulative usage cannot exceed the fixed context cap; duplicate selections are not replayed.
- Concurrent packers: ledger serialization prevents overspending and partial accounting.
- Changed norm/config under the same path: old dependency fingerprint invalidates the lesson.
- Historical Slurm observation: never claim the job still runs without a new live observation.
- OpenArm/Piper vs YAM, server vs Thor: mismatched or missing scope excludes a record.
- Port opened, lower loss, BF16 passed: none alone proves a deployable FP8 engine or robot success.
- Evidence absent/corrupt, future timestamp, expired conclusion: no verified-current reuse.
- Full request budget includes history, tool schemas and reserved output; missing tokenizer/invalid counts fail closed.

Measure budget violations, stale/mismatched reuse, provenance completeness, missing required evidence and replay regressions. Automated tests measure mechanisms; semantic case review must still check causal claims and applicability. Fixed tests alone do not prove real-world improvement.
