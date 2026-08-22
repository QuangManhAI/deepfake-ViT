# How to Set Up an AI Agent Workflow (v2)

## Step 1: Set default rules for the AI agent

Place these in `agents/rules/`. These are the **always-on** files — the only
ones that should be loaded into the agent's persistent system prompt.
Keep this set small and stable (e.g. naming conventions, folder structure,
codebase audit rules, smoke test checklist). Everything else is read
on-demand by the agent when relevant to the current task, not preloaded.

## Step 2: Define your objective in PURPOSE.md

## Step 3: Consolidate your objective in PURPOSE.md via AI agent

Have the agent do a clarifying interview.

- Ask the agent to ask you questions, one at a time, up to ~5, covering:
  - **Problem/motivation** — what's missing or broken without this project
  - **Success criteria** — what "done" looks like, ideally measurable
    (e.g. "val accuracy > X%", not just "good performance")
  - **Scope boundaries** — what's explicitly *out* of scope
  - **Constraints** — compute, time, dataset, dependencies on other
    projects/tools
  - **Audience/context** — coursework, portfolio, production, manager-facing
- After the questions, the agent drafts a updated PURPOSE.md itself — you review and edit before locking it in.
- If the agent's draft misses or misreads something, that's a signal to
  clarify further before moving to Step 3 (the roadmap), since a vague or
  wrong PURPOSE.md will propagate into every downstream phase.

## Step 4: Prompt the AI to generate a plan

Base it on `agents/templates/PROJECT_ROADMAP_TEMPLATE.md` and `PURPOSE.md`.
Optionally supply `agents/ML_PIPELINE_REFERENCE_v3.md` if you want the plan
to follow that pipeline more strictly.

## Step 5: Fill out agents/phases

One file per phase, following the roadmap, using
`agents/phases/PHASE_TEMPLATE.md`.

## Step 6: Update agents/rules/FOLDER_STRUCTURE.md

Do this after Step 4, **and revisit it after every phase** — folder
structure drifts as implementation proceeds, so this isn't a one-time step.

## Step 7: Create the README.md at project root

## Step 8: Read the generated roadmap and phase docs

Not a one-off — re-read the roadmap and the relevant phase doc at the
start of every session/phase, not just once here.

## Step 9: Implement phase by phase

- Save progress into `agents/progress/` as you go (checkpoint-based,
  human-in-the-loop — stop and review after each phase, don't chain
  autonomously into the next).
- **Run smoke tests** (per `agents/templates/SMOKE_TEST_CHECKLIST.md`) before
  considering a phase implementation complete.

## Step 10: Audit BEFORE marking a phase done

Prompt the AI to generate a codebase audit (using
`agents/templates/CODEBASE_AUDIT_TEMPLATE.md`) **before** the phase is
marked done — not after. The audit should gate completion:

- If the audit is clean, mark the phase done.
- If it flags issues, either fix them, or explicitly document the decision
  not to (rationale written into the phase doc or progress file), the same
  way a "won't fix" decision would be reported to a manager. Don't leave
  it as a silent gap.

Only after the audit passes (or issues are explicitly accepted) does the
phase count as done and the next phase begin.
