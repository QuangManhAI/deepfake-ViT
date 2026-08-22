# Agent AI Knowledge Base & Control Layer

Welcome to the **Agent AI Knowledge Base** for the `[PROJECT_NAME]` repository.
This folder serves as the "second brain", memory, and behavioral control layer
for AI agents working on this project.

---

## 🏗️ Architecture Overview

`agents/` governs a layered ML pipeline; each layer owns code, docs, and status:

```
Data (src/data) → Models (src/models) → Training scripts (src/training, script-only)
    → Evaluation (src/eval) → Analysis notebooks (notebooks/, test/demo/visualize only)
                ↕ artifacts (experiments/runs|results) ↕
Governance (this folder): rules → phases → progress → experiments → bugs
```

- **rules/** — binding conventions the agent must follow: naming, folder
  structure, MD format, notebook policy, **logging & checkpoints** (naming,
  format, resume procedure), **5W1H result reporting**, codebase audit.
- **phases/** — how each pipeline stage works (data prep → model → training → eval).
- **progress/** — live status of each phase (`*_STATUS.md`).
- **experiments/** — experiment plans and results with 5W1H.
- **bugs/** — resolved runtime issues.
- **templates/** — reusable skeletons (phase doc, progress status, audit, smoke test).
- **references/** — external technical guides.

Training runs **only from scripts** (`src/training/*.py`,
`src/experiments/*_train.py`); notebooks analyze artifacts. Logging/checkpoint
rules: [rules/LOGGING_CHECKPOINT_RULES.md](rules/LOGGING_CHECKPOINT_RULES.md).
Result reporting (5W1H): [rules/RESULTS_REPORTING.md](rules/RESULTS_REPORTING.md).

---

## 📁 Directory Architecture

```
agents/
├── README.md                      # Entry point & navigation guide (this file)
├── OVERVIEW.md                    # Core project overview, goals & phase roadmap
├── PURPOSE.md                     # Original project brief & requirements
├── HOW_TO_SETUP_AI_AGENT.md       # Step-by-step agent workflow setup
├── ML_PIPELINE_REFERENCE_v3.md    # End-to-end ML pipeline reference (steps 1-18+)
│
├── experiments/                   # Experiment reports & technical comparisons
│   ├── README.md                  # Index of experiments
│   └── EXPERIMENT_TEMPLATE.md     # Skeleton for new experiment docs
├── rules/                         # Guidelines, standards & philosophy for Agent AI
│   ├── AGENT_AI.md                # Agent AI philosophy & behavior layer rules
│   ├── CODEBASE_AUDIT.md          # Mandatory pre-task codebase audit checklist
│   ├── FOLDER_STRUCTURE.md        # Source of truth for repository directory layout
│   ├── MD_CONVENTION.md           # Documentation format standards
│   ├── NAMING_CONVENTION.md       # Naming rules for files, code & experiments
│   ├── NOTEBOOK_HEADER_CONVENTION.md # Standardized headers for Jupyter notebooks
│   ├── LOGGING_CHECKPOINT_RULES.md  # Logging, checkpoint format, resume procedure
│   └── RESULTS_REPORTING.md         # 5W1H rules for every reported result
│
├── phases/                        # Step-by-step pipeline & phase specifications
│   ├── PHASE_TEMPLATE.md          # Template for phase docs (DATA_PREP, MODEL, ...)
│   └── <PHASE>.md                 # One file per phase, filled from the template
│
├── templates/                     # Standard templates & checklists for agents
│   ├── PROJECT_ROADMAP_TEMPLATE.md  # Roadmap skeleton (milestones, phases, tasks)
│   ├── PHASE_DOC_TEMPLATE.md        # Template for writing new phase documentation
│   ├── PROGRESS_STATUS_TEMPLATE.md  # Template for tracking task status
│   ├── CODEBASE_AUDIT_TEMPLATE.md   # Reusable audit report skeleton
│   └── SMOKE_TEST_CHECKLIST.md      # Pre-run smoke test verification checklist
│
├── progress/                      # Live progress tracking per phase
│   └── <PHASE>_STATUS.md          # One per phase, from PROGRESS_TEMPLATE.md
│
├── bugs/                          # Documented bug reports
│   ├── README.md                  # Bug index & troubleshooting directory
│   └── BUG_TEMPLATE.md            # Skeleton for new bug reports
│
└── references/                    # External guides & technical reference notes
    ├── REFERENCE_TEMPLATE.md      # Skeleton for new reference docs
    ├── OPTUNA_DB_GUIDE.md         # Guide for Optuna hyperparameter tracking DB
    └── GIT_AND_RELEASE_BEST_PRACTICES.md  # Git commits, CI, releases & approval gate
```

---

## 🚀 Guidelines for AI Agents

1. **Before starting a multi-file task**: Run the procedure in
   [rules/CODEBASE_AUDIT.md](rules/CODEBASE_AUDIT.md) to check for drift
   between documentation and real code.
2. **Before creating or modifying code**: Verify directory locations in
   [rules/FOLDER_STRUCTURE.md](rules/FOLDER_STRUCTURE.md) and naming rules in
   [rules/NAMING_CONVENTION.md](rules/NAMING_CONVENTION.md).
3. **Executing a Phase**: Read the matching phase doc in `phases/<phase>.md`
   and maintain status in `progress/<phase>_STATUS.md`.
4. **Before long runs**: Perform the verification steps in
   [templates/SMOKE_TEST_CHECKLIST.md](templates/SMOKE_TEST_CHECKLIST.md).
5. **New project?** Follow [HOW_TO_SETUP_AI_AGENT.md](HOW_TO_SETUP_AI_AGENT.md)
   to wire this knowledge base into the project workflow.
