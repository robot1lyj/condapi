# Selective retrieval CLI

Use this optional helper for reproducible section selection, structured evidence checks and duplicate suppression. `search` adds lexical discovery over existing records; it does not understand semantic relevance, generate summaries or control retained model messages. Decide what is needed before calling it. Ordinary bounded searches/reads are also valid; they still require semantic, evidence, scope and freshness checks.

## Retrieval and measurement

The default **per-packet** limit is 12,288 serialized UTF-8 bytes, including JSON framing. Configure `pack --max-bytes` for the current retrieval, within explicit project/user limits. There is **no default cumulative quota** and no universal model token setting.

Examples assume `skills/mlops-memory` exists in the project. Otherwise use the absolute path to the installed helper. `--root` is the project that owns the memory, not the skill directory.

```bash
python skills/mlops-memory/scripts/memory_gate.py init --root . --session TASK_ID
python skills/mlops-memory/scripts/memory_gate.py pack --root . --session TASK_ID \
  --required 'docs/03_training_and_evaluation.md#训练前 gate' \
  --optional 'docs/04_data_contracts.md#Norm stats 和资产'
python skills/mlops-memory/scripts/memory_gate.py audit --root . --session TASK_ID
```

The example headings must exist in the target project. A selector is a relative Markdown/JSON path, optionally followed by `#Exact heading text`; Markdown selection includes child headings and rejects ambiguous titles. Select connected prerequisite sections together when needed. Markdown excerpts are source data, not automatically verified-current knowledge.

Required selections are atomic: if they do not fit, no source text is emitted and the ledger is unchanged. Optional selections are tried in supplied relevance order and skipped whole when they do not fit. If a packet fails, narrow irrelevant material or increase `--max-bytes` for necessary complete evidence where allowed. Never cut a claim away from its conditions or evidence to fit. A quota failure does not establish that the model context is full.

The ledger suppresses unchanged selections; changed selections are checked and counted again. An all-unchanged/empty packet returns exit 2 without emitting source text. Reuse the ledger across retrieval calls. After actual compaction, or when a needed excerpt is no longer available, use `pack --reload` with just those selectors. This repeats evidence/scope checks and counts the new packet, preserves history, and still deduplicates within the packet. Do not reload everything or assume the ledger knows which messages the host retained.

`init --preloaded path.md ...` optionally records exact project-relative excerpts known to be already loaded; omit it when unknown. Repeated selectors are counted once. This cannot discover hidden instructions, transformed summaries or history. Report `tracked_bytes` as **declared preload text bytes plus serialized successful packets**, never active context occupancy or remaining model capacity. `context_tokens` stays null and `exact_token_enforcement` false. The helper's 2 MiB source-read bound limits a single input file for local processing; it is not a long-term storage cap. Inspect larger logs using filtered tools or a sourced evidence receipt while retaining originals.

## Discover records by problem/action

When the project has JSON memory records, search their claims, optional problem terms and attempt symptoms before expanding full records:

```bash
python skills/mlops-memory/scripts/memory_gate.py search --root . --session TASK_ID \
  --query '推理延迟 频率' --scope project=PROJECT --scope platform=PLATFORM --top 5
python skills/mlops-memory/scripts/memory_gate.py pack --root . --session TASK_ID \
  --required 'docs/cache/records/MATCHED_ID.json' \
  --scope project=PROJECT --scope platform=PLATFORM
```

Replace scope values with the actual project and include **all** scope fields of the records (contract/version etc. where recorded). An explicit project scope is required. `--records-dir` defaults to the existing `docs/cache/records`; override it to the actual record owner. The command reads that directory recursively without creating a database or embedding index. If the project only has Markdown memory, use its current index and bounded text search; do not convert everything just to use this command.

Use short keywords, separated by spaces, and add useful aliases to `retrieval.terms` when recording experience. Ranking counts case-insensitive keyword substring matches in claim/symptom/terms, with source path breaking ties; it is not a confidence or quality score. The search does not read arbitrary logs for keywords or follow related sources recursively.

Results contain bounded discovery metadata: full claim, scope, source, status, admission label, capability presence and attempt verdict. They omit invocation recipes and evidence bodies. The default is five results within the configurable `--max-bytes` limit; entries are omitted whole, not truncated. Always load a selected record before acting on it. A matched keyword does not establish causality or suitability.

Current search applies the same schema/evidence/dependency/scope gates as `pack`. `search --purpose review` permits historical/candidate/broken evidence as explicitly unverified discovery data while still filtering scope; it never upgrades status. Exclusion counts distinguish keyword mismatch, scope mismatch, non-current status, invalid records/evidence, top-result limits and byte limits. If results were omitted for size, narrow the query or adjust the retrieval size where allowed rather than interpreting omissions as absence of evidence.

Search output, including empty-result diagnostics, is charged to the same ledger and respects an explicit cumulative quota. Search does **not** mark full records as already loaded, so a subsequent `pack` still emits them. Search metadata is not deduplicated across calls; avoid repeating an unchanged search without a reason. Neither command runs capability invocations or assumption checks.

## Explicit transfer quotas and existing ledgers

Only configure a cumulative transfer quota when the project/user actually requires one:

```bash
python skills/mlops-memory/scripts/memory_gate.py init --root . --session TASK_ID \
  --context-bytes 65536
python skills/mlops-memory/scripts/memory_gate.py resize --root . --session TASK_ID \
  --context-bytes 196608 --reason 'Required evidence for the authorized task'
```

`--context-bytes` is a legacy flag name for a **cumulative byte-transfer quota**. Existing ledgers, including those with an old 32,768-byte default, retain their cap; loading the new helper does not silently remove it. Once the project's limit has been replaced or its removal is authorized, remove it explicitly:

```bash
python skills/mlops-memory/scripts/memory_gate.py resize --root . --session TASK_ID \
  --no-total-limit --reason 'Cumulative quota removed by the current project policy'
```

Resizing/removing the quota preserves counts and deduplication history and records the reason, time and old/new limit. No new ledger ID may be used to evade an explicit quota. A numeric cap cannot be below bytes already tracked. Ordinary retrieval tuning needs no additional approval within existing authorization; explicit user/project limits still apply. Neither resizing nor removing a quota expands the model window or reduces retained history.

## Evidence validation and review

```bash
python skills/mlops-memory/scripts/memory_gate.py validate-record --root . \
  --record docs/cache/records/RECORD_ID.json
```

The validator checks schema, local evidence digests, dependencies and expiry, not semantic truth. For current structured records, pass `--scope project=PROJECT --scope platform=PLATFORM` etc.; **all** recorded scope fields must match. Current retrieval additionally requires `verified` and `recheck=on_change`. Live observations require a new observation before being reported as current.

For candidate, historical or broken records, use `pack --purpose review --required path/to/record.json`. The output is explicitly `review_only`; it does not validate or promote a claim. Later current retrieval, including `--reload`, still runs all evidence/scope checks. `audit` reports ledger accounting; semantic memory review is performed against the actual source evidence. The ledger is not a memory store.

## Full-request integration

```python
# Use the host's actual final-request tokenizer, including framing, tools and history.
# Choose configured_window/output_reserve from the real host/task configuration.
guard_request(request, tokenizer_count, max_context_tokens=configured_window, output_reserve=output_reserve)
```

The caller must count and send the same final request, with all output tokens reserved and no messages added afterward. No hosted API or Codex hook is installed by the skill. Without that integration, exact total-token admission remains unavailable; selective retrieval still reduces avoidable input. Do not infer token counts from bytes or claim a skill can remove old tool outputs.

For a separate, explicitly chosen byte check on a supplied JSON object:

```bash
python skills/mlops-memory/scripts/memory_gate.py request --input request.json --max-bytes 98304
```

`--max-bytes` is required here. This command does not establish that the supplied object is the host's complete request and does not enforce its token limit.
