# Controlled evolution

## First principles and engineering control

Start with the decision and the minimum evidence needed to make it. Define inputs, outputs, constraints, uncertainties and measurable acceptance criteria. Include coupled effects: reducing one context component may hide a dependency needed by another decision.

Apply a feedback loop to memory quality, not to the robot's physical controller:

| Control concept | Memory implementation |
|---|---|
| Desired state | Correct decisions with bounded context and recoverable evidence |
| Observation | Tool reports, local metrics, user corrections, replay outcomes |
| Error | Unsupported conclusion, stale reuse, missed constraint, budget overflow |
| Correction | Small scoped record/route/procedure update |
| Actuation limit | Existing task authorization, immutable project boundaries |
| Stability check | Regression replay, limited update scope, reversible version |

This is our practical adaptation of engineering cybernetics. It does not assert a formal convergence/stability theorem. Qian's qualitative-to-quantitative systems approach motivates combining expert hypotheses with measured evidence; an LLM's confidence is not a measurement.

## Update lifecycle

1. Observe and preserve evidence; distinguish facts, hypotheses, user requirements and external references.
2. Compare expected/actual outcome. Record confounders; multiple changed variables do not identify a cause.
3. Propose a candidate with scope, owner, dependencies and counterexamples. Deduplicate by claim/scope/evidence, not just wording.
4. Validate evidence and applicability; replay old cases and a held-out case before changing a reusable procedure. A directly measured fact can be verified for its exact scope without requiring arbitrary repeated trials.
5. Apply a minimal patch to the owner, record why in the project changelog and refresh any sourced kernel projection. Failed candidates retain their reason; source artifacts are not rewritten.
6. Compare outcome with baseline. If regressions appear, supersede/revert the specific update while preserving unrelated work. Archive low-use historical information out of the hot path, never erase unique evidence to satisfy a budget.

When a loop produces reusable engineering work, retain the tested artifact entrypoint and usage conditions, not just a success summary. Preserve unsuccessful and inconclusive hypotheses with their scope and retry triggers. Use the optional fields in [engineering.md](engineering.md); connect them through the existing problem route. An unchanged file fingerprint is insufficient evidence that a volatile runtime assumption still holds.

For changes to reusable procedures or retrieval routes, evaluate task recovery against a frozen previous memory snapshot using the lightweight case format in [evaluation.md](evaluation.md). Measure repeated unsuccessful interventions, missing constraints and unnecessary retrieval alongside correctness. If no comparable rerun has occurred, label the proposed benefit unmeasured. Do not run training or hardware operations just to generate a memory evaluation.

A candidate must not change acceptance thresholds to pass itself. Hard project constraints can change only on an applicable user instruction. Ordinary evidence maintenance within an authorized task needs no extra approval. Remote runs and deployment still follow the scope of that task.

## Two useful reflection questions

- Did the change improve a measurable decision, or merely make the summary sound more certain? Does the evidence support the stated scope and cause?
- Did context reduction preserve conditions, contradictions and open questions? Can a fresh context recover the next action without importing obsolete defaults?

Do not prescribe endless reflection on every task. Use explicit user iteration requirements and observed failures; stop when acceptance criteria pass and no unresolved regression remains.

## Design references

- [ACE](https://arxiv.org/abs/2510.04618): structured incremental evolution of contextual playbooks.
- [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd): artifacts, executions and lineage.
- [钱学森，谈地理科学的内容及研究方法，1991](https://doi.org/10.11821/xb199103001): systems view and qualitative-to-quantitative integration; our application to software memory is an engineering interpretation.
- H. S. Tsien, *Engineering Cybernetics*, McGraw-Hill, 1954: historical control-theory foundation, not a source of claims about modern LLM memory performance.
- [General Robotics, Introducing Auto Engineering for Robotics, 2026-09-09](https://www.generalrobotics.company/post/introducing-auto-engineering-for-robotics): inspiration for retaining reusable artifacts, failed attempts and measured operational assumptions. The schemas here are our adaptation; this skill does not implement GRID's robotics execution platform.
