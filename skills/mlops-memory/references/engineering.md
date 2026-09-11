# Reusable engineering memory

Use the existing record envelope from [records.md](records.md). The optional fields below connect a problem to evidence and working artifacts without creating another memory store. A short task may need only a sourced prose update; do not manufacture records to fill a schema.

## Reusable capability

Add `capability` to `kind: procedure` when a tool or workflow can be reused:

- `entrypoint`: existing project-relative script or executable artifact path.
- `invocation`: nonempty usage recipe as text, including the environment needed; data, never automatically executed.
- `config_paths`: list of existing project-relative configuration paths, possibly empty.
- `inputs`, `outputs`: nonempty lists describing required inputs and produced artifacts, including relevant units/contracts.
- `validation_command`: how to repeat the check, as text. Inspect the check and its resource requirements before running it within the current task's authorization.
- `acceptance`: the criterion selected before validation; preserve the scope of the actual result.
- `limitations`: nonempty list of untested cases or known constraints; state the tested scope even if no other limits are known.

Put measured validation reports in the envelope's `evidence`; fingerprint the entrypoint and every config path in `depends_on` before marking the capability verified. Include additional behavior-changing files, validator code and fixtures as dependencies where relevant. Keep source code and configurations at their existing owner. A path or successful process exit alone does not prove the capability works.

Promote only the exact tested version/scope. If an artifact changes, revalidate that capability; do not silently refresh the hash to regain admission. First search for an existing applicable tool before implementing a duplicate. Retrieval exposes an invocation recipe but never authorizes a remote run, training job or deployment.

## Failed or inconclusive attempts

Add `attempt` to `kind: lesson` for a diagnostic experiment worth remembering:

- `symptom`: concrete observed failure, under the envelope's scope.
- `hypothesis`: the specific explanation being tested.
- `intervention`: the change or diagnostic check actually performed.
- `observation`: measured result; raw logs stay in `evidence`.
- `verdict`: `supported`, `refuted` or `inconclusive`, about the hypothesis in this scope.
- `confounders`: list of other changed variables or limitations; empty only when checked and none identified.
- `retry_when`: what changed evidence/environment would make a repeat useful; do not turn one failure into a universal ban.

Example: reducing batch size while a crash persists is an observed unsuccessful intervention. It need not refute every memory-related explanation. If several variables changed or the experiment was interrupted, prefer `inconclusive`. `status: verified` can mean the attempt and its bounded outcome are supported by evidence, even when the attempted repair failed. It never means the repair should be reused.

Before repeating a matching attempt, compare its scope and retry conditions. Do not automatically skip a necessary test just because a superficially similar historical attempt failed. A materially different rerun gets its own evidence; preserve the old result rather than rewriting it.

## Expected and measured assumptions

Optionally add `assumptions`, a nonempty list of:

- `name`, `expected`: the relevant prerequisite and configured/assumed value, as nonempty strings.
- `observed`: measured value as a string, or null if unknown.
- `unit`: explicit unit, or `not_applicable` for a non-numeric condition such as a software version.
- `observed_at`: timezone timestamp for the measurement, no later than the envelope's `recorded_at`, or null when no observation exists.
- `check`: measurement/check method as text.
- `result`: `match`, `mismatch` or `unknown`; the author compares evidence against the stated requirement.
- `recheck_when`: triggers requiring a new observation, including before use for volatile runtime conditions.

The envelope's evidence should support actual observations. Unknown observations require `result: unknown`; do not substitute the configured value. Mismatches are valuable diagnostic evidence and must remain visible. Schema validation checks structure/time, not the physical measurement, unit conversion or semantic comparison. Even a previous `match` is historical. File hashes cannot detect every runtime/hardware change; search/full retrieval mark such records for runtime recheck and cannot certify execution readiness.

## Problem and action routes

Optionally add `retrieval` with:

- `terms`: nonempty list of short problem names, aliases and action keywords, such as `推理延迟`, `latency`, `频率检查`.
- `related`: optional list of `{relation, source}`. Relations are `check`, `repair`, `attempt` or `prerequisite`; `source` is an existing project-relative memory path with an optional exact Markdown heading.

The helper verifies related files exist, but a link does not validate its target claim or heading. Inspect the target through normal source/evidence checks when needed. Do not recursively expand every relation. Keep primary project navigation in the existing context index; these fields are routing metadata on the record, not a duplicate catalog of facts.

Use `search` with short keywords and the known project/platform/contract scope. It ranks lexical overlap, returns a bounded set of claim/source summaries and exclusion counts, and then `pack` loads selected records. Search does not emit commands, raw evidence, or full relation trees. See [usage.md](usage.md) for CLI examples and review mode.

After a repair, add only the useful artifact/attempt references to the existing problem route. When the change affects reusable behavior, retain a replay case following [evaluation.md](evaluation.md). A routing improvement is useful only if it helps recover the right action without losing required conditions.
