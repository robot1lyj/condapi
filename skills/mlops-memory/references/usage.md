# Budgeted CLI

The defaults are transport budgets: packet 12,288 bytes, cumulative memory 32,768 bytes. They are not tokenizer estimates. A ledger spans all turns sharing a context. It counts serialized successful packets, including metadata; already loaded instructions must be charged at initialization. Diagnostics print no source text.

Initialize a fresh context once (include all actually preloaded memory; this is an example for this repository):

```bash
python skills/mlops-memory/scripts/memory_gate.py init --root . --session SESSION_ID \
  --preloaded AGENTS.md docs/cache/kernel.md docs/cache/context_index.md \
  skills/mlops-memory/SKILL.md
```

Read a whole short document or an exact Markdown heading (including its child headings); ambiguous headings fail. Each `--required` or `--optional` is `relative/path.md` or `relative/path.md#Heading text`:

```bash
python skills/mlops-memory/scripts/memory_gate.py pack --root . --session SESSION_ID \
  --required 'docs/03_training_and_evaluation.md#训练前 gate' \
  --optional 'docs/04_data_contracts.md#Norm stats 和资产'
python skills/mlops-memory/scripts/memory_gate.py audit --root . --session SESSION_ID
```

Mandatory selections are atomic: if they do not fit, no source text is returned and no charge is committed. Optional selections are admitted in supplied relevance order and skipped whole when they do not fit. Output includes omitted and unchanged counts. Changed content can be loaded again and is charged again; unchanged selections are not repeated. An empty packet is rejected, so repeated calls cannot leak unaccounted framing.

Do not use a series of new session IDs to evade the cumulative cap. Only the host can confirm that prior messages have actually left the context. Neither clearing a terminal nor starting a new user turn establishes that fact.

Validate a candidate/verified record without printing its contents:

```bash
python skills/mlops-memory/scripts/memory_gate.py validate-record --root . \
  --record docs/cache/records/RECORD_ID.json
```

The tool verifies schema, local evidence digests and expiry; remote evidence remains a candidate until audited locally. For retrieved records, request the same scope using `--scope project=condapi --scope platform=thor` etc.; every scope field recorded in an admitted record must match. Retrieval requires `verified`, an evidence digest match, an explicit expiry/recheck policy, and current matching scope. A ledger is not an evidence store.

For consolidation/auditing, use `pack --purpose review --required docs/cache/records/RECORD_ID.json`. This exposes candidates, historical or broken records explicitly labeled `review_only`; it does not validate or promote their claims. Subsequent current-knowledge retrieval still runs all scope/evidence checks. `audit` reports ledger usage; semantic memory audit is performed by the skill using source evidence.

Host integration:

```python
# tokenizer_count must count the host's actual final request including chat framing,
# tool schemas/results and retained history; reserve includes all output tokens.
guard_request(request, tokenizer_count, max_context_tokens=32768, output_reserve=4096)
```

The caller sends exactly that checked request without adding messages afterward. If exact token counting is unavailable, use a byte cap on the supplied request as an additional control, and state that exact total-token admission is not implemented:

```bash
python skills/mlops-memory/scripts/memory_gate.py request --input request.json --max-bytes 98304
```

No hosted API or Codex hook is installed automatically. Record the host integration status honestly.
