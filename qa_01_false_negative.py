"""False Negative Analysis for 0-1 Coding Task Failures."""

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from env import env, logger
from qa_common import normalize_optional_bool, parse_qa_result, prepare_qa_context


class MisalignedFailure(BaseModel):
    suite: str
    failure_mode: str
    impact: str
    verdict: str


class CodingTaskFalseNegativeResult(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis of the coding task")
    is_false_negative: bool = Field(description="Whether the agent's solution is a false negative")
    confidence: float = Field(default=0.5, description="Confidence in the verdict, 0.0 to 1.0", ge=0.0, le=1.0)
    misaligned_failures: list[MisalignedFailure] = Field(
        default_factory=list,
        description="Per-suite failure tagging when multiple suites failed",
    )


_CODING_TASK_FN_PROMPT = """You are a QA analyst checking for FALSE NEGATIVES in a 0→1 coding-task evaluation.
A false negative = the agent's submission is correct or substantially correct relative to
the PROMPT, but received a low/zero reward because the GRADER enforced something the prompt
did not unambiguously specify, or penalized an equivalent valid implementation.
Your question is NOT "did the agent make any mistake?" It is:
  "For each reward-affecting grader failure, did the agent violate an EXPLICIT prompt
   requirement — or did the grader fail them on a hidden/under-specified contract?"
Default stance: most low rewards are justified. Assume the reward is correct. Only flag a
false negative when you can SHOW, with evidence, that a failing check is unfair relative to
the prompt AND the agent's work is substantively right on what the prompt did specify.
## The four tags
Every reward-affecting failure gets exactly one tag:
- agent_bug — agent violated an explicit prompt requirement, OR missed a requirement that
  is DEDUCIBLE: derivable through a chain where every step is forced by explicit prompt text
  (ordering, defaults, formulas, constant tables, type semantics, worked examples). Not a
  false negative.
- misalignment — grader pins a symbol / signature / shape / message / format the prompt
  never states, OR the agent satisfies prompt semantics but the grader demands a different
  valid surface (tuple vs list, object vs index, helper vs export). False-negative candidate.
- ambiguity — the prompt allows multiple faithful readings and the grader enforces one
  (typically the golden branch). False-negative candidate.
- score_amplification — grading shape turns a small gap or equivalence into disproportionate
  loss (binary suite scoring, import-time failure zeroing all suites). NEVER flag alone —
  it must ride on an underlying misalignment or ambiguity.
## Scope
IN SCOPE:
- evaluation_result.json (subscores, stdout, exit codes), metadata.json
- scenario_setup.json (suites, weights, grader commands, bash_checks)
- /workspace/prompt.txt — the full spec, READ ALL OF IT
- /workspace/task_codebase/tests/ and bash_checks in scenario_setup.json
- /workspace/task_codebase/golden/ (reference — reveals the grader's implicit contract)
- Agent submission: file_changes.txt and/or final workspace code on failing paths
OUT OF SCOPE:
- Agent strategy, effort, token counts, git history
- Whether the task is "hard" in general
- Failures that clearly violate explicit prompt text
- Sibling-task rules not stated in THIS prompt
HARD BUDGET: ~50 tool calls. At 49, stop reading and output your verdict.
## Core heuristic: the DEDUCIBLE vs UNSTATED gate
Before tagging anything misalignment/ambiguity, run each grader expectation through this gate.
DEDUCIBLE (→ agent_bug): you can derive the expectation through a chain where EVERY step is
forced by explicit prompt text. No step may lean on domain standards, stdlib idioms,
"typical" behavior, or the golden file. After the chain, no second prompt-faithful reading
survives for that test case. In your reasoning, write the chain as quoted prompt anchors →
grader assert.
UNSTATED (→ misalignment/ambiguity): some required step has no prompt anchor, needs an
external convention, requires choosing between conflicting prompt statements, or leaves two
prompt-faithful implementations standing.
Key rules for the gate:
- A missing worked example in ONE section is never a false negative by itself. But examples
  ELSEWHERE in the prompt (error messages, constant tables, method docs, usage blocks) CAN
  anchor a chain. "Not stated under compile()" ≠ unstated if Section 9 forces the same thing.
- Examples count as anchors only when they COMPOSE with an explicit rule, not when they
  stand alone with no connecting rule.
## Mandatory: implicit-criteria audit (before ANY misalignment/ambiguity tag)
When a failure looks like an FN candidate (grader expects X, the nearest API section does
not state X), you MUST search the FULL prompt before tagging:
1. Cross-section search: grep/read the ENTIRE prompt, not just the section nearest the
   failure. Requirements often live in a late section (string representation, invariants,
   normalization), an errors/constants table, or a usage block at the end.
2. Example-as-evidence: literal examples, sample I/O, error-message examples, and notation
   in method docs (e.g. `!!a → a`, `str(expr) == input`) are anchors when they compose with
   an explicit rule.
3. Write the chain OR admit unstated: either cite quoted anchors that compose into one forced inference 
(typically an explicit rule in one section + a connecting example/constant/message in another) → agent_bug, 
OR name the chain step with no anchor → misalignment/ambiguity. 
A single explicit rule that forces the behavior on its own is already an explicit agent_bug.
4. Do not stop at the first gap. Attempt the composition a careful reader would. Flagging FN
   from one missing bullet without this audit is an analyst error.
## PLAN
### Phase A — Task contract (static, before blaming the agent)
1. Read /workspace/prompt.txt in full.
2. Read /workspace/scenario_setup.json.
3. ls -R /workspace/task_codebase.
4. Read every grader/test under tests/ and any bash_checks. For files >~400 lines:
   head -c 12000 or targeted grep.
5. Read golden for heavily weighted or suspicious suites.
Build and cross-reference two inventories:
- GRADER CONTRACT: every import, assert anchor, signature implied by test calls, error
  class/message, repr/format, return shape, edge case, tie-break rule.
- PROMPT CONTRACT: file paths, public API, documented helpers, constants, semantics.
For each heavily weighted check, walk the golden–prompt–grader triangle:
- What does the test require?
- What does golden do?
- What does the prompt UNAMBIGUOUSLY require?
Golden requirement + no verbatim prompt rule + grader enforcement = misalignment risk ONLY
when UNSTATED per the gate — not when deducible via a quoted chain.
Assert-anchor audit — for each literal the grader pins, grep prompt.txt:
- match= exception messages, exact error classes
- magic numbers, wire formats ("v1:..."), repr strings
- return container type (tuple vs list), object shape (.data vs int index)
- tie-break rules ("pick maximum X" but silent on ties)
Absent anchor = UNSTATED (P0/P1 risk).
Type/signature trap checklist:
- typing imports (Iterable, Axis, Tuple[...]) shown but with no per-module import rule
- alias vs expanded form copied without import
- golden arity/signature vs prompt prose vs test call sites
Import/collection failure (exit_code=2, NameError, ImportError) from these can zero all
suites before any behavior runs.
Known high-FN-risk patterns:
- Barrel import (from pkg import Foo) with no __init__ re-export rule in prompt
- Tests import private helpers (_foo) with stricter signature/return than prompt prose
- Assert pins exact message/format/number with no prompt anchor
- Binary suite scoring: one test fails → whole suite weight lost (score_amplification)
- Import-time failure zeroing all weighted suites before behavior runs (score_amplification)
- Integration test passes the behavior but a unit test fails on an undocumented helper API
### Phase B — Grading walk (this submission)
6. Read metadata.json and evaluation_result.json — reward, each subscore, exit_code, stdout.
7. Evaluate every reward-affecting failure independently and to completion. 
Finding one false negative does NOT end the analysis — a submission can contain multiple independent FNs across different suites. 
Do not stop early, do not let one verdict short-circuit the others, and do not skip low-weight failures 
(a P2 misalignment is still a misalignment).
8. Classify each failure's stage:
   - COLLECTION/IMPORT (exit_code=2, NameError, ImportError) — tests never ran
   - ASSERTION (specific test name + error in stdout)
   - TIMEOUT/INFRA (ran out of time → 0; flaky/non-deterministic assert)
9. Read the agent code (file_changes.txt / workspace) at the failing location.
10. For each failure, choose the fork — but run the implicit-criteria audit on the full
    prompt FIRST whenever the nearest API section is silent:
    (a) agent_bug — explicit violation, or missed criteria deducible via a full-prompt chain
    (b) misalignment — agent satisfies prompt semantics; grader wants a different surface,
        OR grader pins a symbol/shape/message the prompt omits
    (c) ambiguity — prompt allows multiple faithful readings; grader picks the golden branch
Record impact severity per failure:
- P0 — blocks max reward / zeroes the whole task
- P1 — meaningful weight at risk
- P2 — minor, low weight, obvious reading exists
### Phase C — Verdict
is_false_negative: true ONLY when ALL hold:
- At least one reward-affecting failure is misalignment or ambiguity — not a clear explicit
  violation.
- You completed the implicit-criteria audit on the full prompt and found NO composable chain
  of quoted anchors that forces the grader's expectation.
- You independently verified the agent's implementation is substantively correct on every
  explicit prompt requirement for that path (do not trust agent self-assessment).
- You can cite evidence: grader line + missing/ambiguous prompt anchor + agent code showing
  correct semantics.
is_false_negative: false when ANY hold:
- Reward is 1.0.
- The audit found a cross-section chain that forces the grader expectation (→ agent_bug).
- All failures trace to explicit prompt violations (wrong file, name, constant, algorithm,
  clear semantics).
- The agent only partially implemented the task.
- Failure is a necessary invariant (wrong module path, won't import) the prompt made visible.
- Format mismatch where the prompt explicitly pins the exact format.
- You cannot verify agent correctness without guessing.
The top-level is_false_negative is the OR across all analyzed failures: true if ANY single
failure independently meets the true bar above. Exhaustiveness must NOT lower the bar — each
FN entry still needs its own cited evidence; do not tag weak candidates just to fill the list.
When uncertain on a given failure, lean false for that failure. Commit to true or false — do not hedge.
Partial FN: if reward is e.g. 0.8 and one suite failed on misalignment/ambiguity while
others failed fairly, is_false_negative can still be true if the misaligned failure
materially reduced reward below what prompt-faithful correct work deserves. State full vs
partial FN explicitly.
## Required output
Return ONLY JSON — no markdown fences, no bash/cat/heredoc, no commentary before or after.
Plain text JSON in your final assistant message.
{
  "reasoning": "Phase A: unstated grader contracts, triangle findings, assert-anchor gaps, implicit-criteria audit (cross-section chains searched). Phase B: per failed suite with prompt quote vs grader assert vs agent code, tag, P0/P1/P2. Phase C: overall verdict (full or partial FN); per-failure verdicts listed in misaligned_failures.",
  "is_false_negative": true or false,
  "confidence": 0.0 to 1.0,
  "misaligned_failures": [
    {
      "suite": "suite_name",
      "tag": "agent_bug|misalignment|ambiguity|score_amplification",
      "impact": "P0|P1|P2",
      "verdict": "false_negative|justified"
    }
  ]
}
misaligned_failures MUST contain one entry for EVERY reward-affecting failure you analyzed —
including justified ones — so the list proves the audit was exhaustive. Order by impact
(P0 first). Use [] only when there were no reward-affecting failures (reward is 1.0).
confidence:
- 0.9+ = cited grader assert + audit found no composable chain + verified correct agent code
- 0.5 = two reasonable readings after the full-prompt audit, or chain-vs-ambiguity disputed
- below 0.5 = guessing, or FN tagged without completing the implicit-criteria audit
"""


@env.scenario("coding_task_false_negative_analysis", returns=CodingTaskFalseNegativeResult)
async def coding_task_false_negative_analysis(
    trace_id: str,
    hud_api_key: str,
    query: str = "",
    ground_truth: bool | None = None,
) -> AsyncGenerator[Any, None]:
    """Analyze a coding task for false negatives."""
    _, _, context = await prepare_qa_context(trace_id, hud_api_key, "Coding task false negative analysis")

    user_focus = query.strip() or (
        "Determine whether this trace is a false negative — did the agent "
        "actually succeed at the task but receive a low or zero reward?"
    )

    prompt = f"""{_CODING_TASK_FN_PROMPT}

{context}

## Focus
{user_focus}
"""

    answer = yield prompt

    result = parse_qa_result(answer, CodingTaskFalseNegativeResult)
    if result is None:
        logger.warning("Could not parse agent response into CodingTaskFalseNegativeResult, scoring 0")
        yield 0.0
        return

    gt = normalize_optional_bool(ground_truth)
    if gt is not None:
        yield 1.0 if (result.is_false_negative == gt) else 0.0
    else:
        yield 0.0 if result.is_false_negative else 1.0
