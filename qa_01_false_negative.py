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

A false negative occurs when the agent's submission is correct or substantially correct
relative to the PROMPT, but received a low or zero reward because the GRADER enforced
something the prompt did not unambiguously specify — or penalized an equivalent valid
implementation.

Your primary question is NOT "did the agent make any mistake?" It is:

  "For each reward-affecting grader failure, did the agent violate an explicit prompt
   requirement — or did the grader fail them on a hidden/under-specified contract?"

Most low rewards are justified. Start by assuming the reward is correct. Only set
is_false_negative: true when you can show, with evidence, that the failing check(s)
are unfair relative to the prompt AND the agent's work is substantively right on what
the prompt did specify.

## Scope

IN SCOPE:
- evaluation_result.json (subscores, stdout, exit codes)
- scenario_setup.json (suite names, weights, grader commands, bash_checks)
- /workspace/prompt.txt (full spec — READ ALL OF IT)
- /workspace/task_codebase/tests/ AND bash_checks embedded in scenario_setup.json
- /workspace/task_codebase/golden/ (reference — shows grader's implicit contract)
- Agent submission: file_changes.txt and/or final workspace code for failing paths

OUT OF SCOPE:
- Agent strategy, effort, token counts, git history
- Whether the task is "hard" in general (separate task-quality audit)
- Failures that clearly violate explicit prompt text
- Sibling-task rules not stated in THIS prompt (irrelevant to this eval)

HARD STEP BUDGET: ~50 tool calls. At 49, stop reading and output your verdict.

## PLAN — do in order

### Phase A — Task contract (static, before blaming the agent)

1. cat /workspace/prompt.txt
2. cat /workspace/scenario_setup.json
3. ls -R /workspace/task_codebase
4. Read every grader/test script under task_codebase/tests/ AND any bash_checks in
   scenario_setup.json. For files over ~400 lines: head -c 12000 "$f" or targeted grep.
5. Read golden for heavily weighted or suspicious suites

Build two inventories, then cross-reference:
- GRADER CONTRACT: every import, assert anchor, signature implied by test calls, error
  class/message, repr format, return shape, edge case, tie-break rule
- PROMPT CONTRACT: file paths, public API, documented helpers, constants, semantics

For each grader requirement, grep the prompt. Before marking anything UNSTATED, apply
the composability gate:

**DEDUCIBLE (NOT misalignment — agent reasoning failure):** You can derive the grader
expectation through a chain where EVERY step is forced by explicit prompt text
(ordering, defaults, formulas, constant tables, type semantics). No step may rely on
domain standards, stdlib idioms, "typical" behavior, or golden. After the chain, no
second prompt-faithful reading remains for this test case. The agent violated explicit
ordering/defaults or failed to apply composed rules. Tag justified /
agent_reasoning_failure — NOT spec_ambiguity or FN. In reasoning, write the chain as
quoted prompt anchors → grader assert.

**UNSTATED (misalignment risk):** Any required step lacks a prompt anchor, requires
external convention, requires choosing between conflicting prompt statements, or leaves
two prompt-faithful implementations (spec_ambiguity).

**Missing worked example in one section alone is NEVER FN by itself.** But examples
elsewhere in the prompt CAN anchor a composability chain (see implicit-criteria audit
below) — do not treat "not stated under compile()" as unstated if Section 9, error
messages, constant tables, or method docs elsewhere force the same behavior.

**Implicit-criteria audit (MANDATORY before any FN or UNSTATED tag):**
When a failure looks like a false-negative candidate — grader expects X, the failing
API section does not verbatim state X — you MUST double-check the FULL prompt for
implicit criteria before tagging misalignment or is_false_negative: true:

1. **Cross-section search:** Grep/read the entire prompt, not just the module or API
   nearest the failure. Requirements often live in a late section (e.g. string
   representation, normalization properties), an errors/constants table, or a worked
   usage block at the end while the parser/compiler section stays silent.
2. **Example-as-evidence (with chain):** Literal examples, sample I/O, error-message
   examples, and notation in method docs (e.g. `!!a → a`, `str(expr) == input`) count as
   prompt anchors when they compose with an explicit rule — not when they stand alone
   with no connecting rule.
3. **Write the chain or admit UNSTATED:** Either cite quoted anchors from ≥2 prompt
   locations forming one forced inference → tag **agent_reasoning_failure / justified**,
   OR show which chain step has no prompt anchor → may tag UNSTATED / FN.
4. **Do not stop at the first gap:** Agents often miss implicit details spread across
   sections; your job before calling FN is to attempt the same composition a careful
   reader would. Skipping this audit and flagging FN from one missing bullet is an
   analyst error.

**Golden–prompt–grader triangle** (explicit step for each heavily weighted check):
- What does the test require?
- What does golden do?
- What does prompt unambiguously require?

Golden requirement + absent verbatim rule + grader enforcement = misalignment risk ONLY
when UNSTATED per the gate above — not when fully deducible via quoted chain.

**Systematic assert-anchor audit:** For each test literal the grader pins, grep prompt.txt:
- match= exception messages, exact error classes
- magic numbers, wire formats ("v1:..."), repr strings
- return container type (tuple vs list), object shape (.data on Slot vs int index)
- tie-break rules when prompt says "pick maximum X" but not what to do on ties
Absent anchor = UNSTATED requirement (P0/P1 risk).

**Type/signature trap checklist:**
- typing imports (Iterable, Axis, Tuple[...]) shown in prompt but no per-module import rule
- SlotDivision alias vs expanded Iterable form copied elsewhere without import
- golden function arity/signature vs prompt prose — compare golden signatures to prompt
  signatures to test call sites
Import/collection failure (exit_code=2, NameError, ImportError) from these alone can
zero all suites before behavior is tested.

**Integration vs unit hotspot:** Integration tests pass the behavior but a unit test fails
only on an undocumented helper API → disproportionate penalty if suite scoring is binary
and prompt never required that API shape.

Known P0 patterns (high false-negative risk):
- Barrel import (from pkg import Foo) with no __init__ re-export rule in prompt
- Tests import private helpers (_foo) with stricter signature/return type than prompt prose
- Assert pins exact message/format/number with no prompt anchor
- Binary suite scoring: one test fails → whole suite weight lost
- Import-time failure zeroing all weighted suites before any behavior runs

### Phase B — Grading walk (this submission)

6. Read metadata.json and evaluation_result.json — reward, each subscore, exit_code, stdout
7. Enumerate EVERY failed suite/check that affected reward (name + weight + value)
8. For EACH failed suite, classify failure stage:
   - COLLECTION/IMPORT (exit_code=2, NameError, ImportError) — tests never ran
   - ASSERTION (specific test name + error in stdout)
   - TIMEOUT/INFRA (grader ran out of time, defaulted to 0; flaky non-deterministic assert)
9. Read file_changes.txt / agent code at the failing location
10. For each failure, answer the three-way fork:

    Before choosing (c) MISALIGNMENT or tagging FN: run the **implicit-criteria audit**
    (Phase A) on the full prompt — especially when the nearest API section is silent but
    examples, error messages, or late-section invariants elsewhere may force the behavior.

    (a) AGENT BUG — violates explicit prompt requirement, or missed implicit criteria
        deducible from a full-prompt composability chain → NOT a false negative
    (b) EQUIVALENCE — agent satisfies prompt semantics; grader wants different surface
        (tuple vs list, Slot vs index, nested helper vs module export) → likely FN
    (c) MISALIGNMENT — grader requires X; prompt never unambiguously states X;
        golden may show X but prompt does not → FN if agent work is otherwise correct

Tag each failure in reasoning with failure_mode when applicable:
- prompt_grader_misalignment — grader pins symbol/signature/shape/message prompt omits
- spec_ambiguity — prompt allows multiple faithful readings; grader picks golden branch
  (distinct from misalignment: prompt is ambiguous, grader is consistent with golden)
- scoring_amplification — grading shape turns small gap/equivalence into disproportionate
  loss (NEVER flag scoring shape alone — pair with underlying spec gap or equivalence trap)
- semantic_equivalence — correct behavior, different valid representation prompt allows

Include impact severity in reasoning: P0 (blocks max reward / whole-task zero),
P1 (meaningful weight at risk), P2 (minor, low weight, obvious reading exists).

### Phase C — Verdict

is_false_negative: true only when all of the following hold:

At least one reward-affecting failure is type (b) prompt–grader misalignment or (c) spec ambiguity / semantic equivalence — not a clear explicit violation.
You completed the implicit-criteria audit on the full prompt (rules, defaults, ordering, cross-section examples) and found no composable chain of quoted anchors that would force the grader’s expectation.
You independently verified the agent’s implementation is substantively correct on every explicit prompt requirement for that behavior path (do not trust agent self-assessment).
You can cite evidence: grader line + missing/ambiguous prompt anchor + agent code showing correct semantics.
is_false_negative: false when any of the following hold:

Reward is 1.0 (by definition).
Implicit-criteria audit found a cross-section composability chain that forces the grader expectation (agent overlooked implicit spec — agent_reasoning_failure, not FN).
All failures trace to explicit prompt violations (wrong file, name, constant, algorithm, semantics where prompt is clear).
Agent only partially implemented the task.
Failure is a necessary invariant (e.g. wrong module path, won’t import) and the prompt made that invariant visible or unambiguous.
Format mismatch when the prompt explicitly pins the exact format (fair failure).
You cannot verify agent correctness without guessing.
When uncertain, lean false. Do not hedge — commit to true or false.
Partial FN: If reward is e.g. 0.8 and one suite failed on misalignment while others failed fairly, is_false_negative can still be true 
if the misaligned failure materially reduced reward below what prompt-faithful correct work deserves. State full or partial FN explicitly.

## Required output

Return ONLY JSON — no markdown fences, no bash/cat/heredoc to print it, no commentary
before or after. Plain text JSON in your final assistant message.

{
  "reasoning": "Phase A: key unstated grader contracts, triangle findings, assert-anchor gaps, implicit-criteria audit (cross-section chains searched). 
  Phase B: per failed suite with prompt quote vs grader assert vs agent code, failure_mode tags, P0/P1/P2. Phase C: overall verdict (full or partial FN).",
  "is_false_negative": true or false,
  "confidence": 0.0 to 1.0,
  "misaligned_failures": [
    {
      "suite": "suite_name",
      "failure_mode": "prompt_grader_misalignment|spec_ambiguity|scoring_amplification|semantic_equivalence",
      "impact": "P0|P1|P2",
      "verdict": "false_negative|justified"
    }
  ]
}

misaligned_failures is optional — include when multiple suites failed and tagging aids clarity.
Omit or use [] when all failures are justified or only one failure matters.

confidence:
- 0.9+ = cited grader assert + implicit-criteria audit found no composable chain +
  verified correct agent code
- 0.5 = two reasonable readings after full-prompt audit, or chain vs ambiguity disputed
- below 0.5 = guessing, or FN tagged without completing implicit-criteria audit
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
