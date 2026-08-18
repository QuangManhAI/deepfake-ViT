# FOLDER_STRUCTURE.md

Source of truth for where things live. The agent should check this
before creating any new file, and update it (with human approval)
if the structure changes.

```
project_root/
├── agents/                # Agent AI Knowledge Base (behavior layer, rules, plans, progress)
│   ├── README.md          # Index & guide for Agent AI
│   ├── OVERVIEW.md        # Core project overview & roadmap
│   ├── PURPOSE.md         # Original brief & requirements
│   ├── HOW_TO_SETUP_AI_AGENT.md  # Agent workflow setup guide
│   ├── rules/             # Guidelines & standards for Agent AI
│   │   ├── AGENT_AI.md    # Agent AI philosophy & behavior rules
│   │   ├── CODEBASE_AUDIT.md # Codebase audit procedure
│   │   ├── FOLDER_STRUCTURE.md
│   │   ├── MD_CONVENTION.md
│   │   ├── NAMING_CONVENTION.md
│   │   └── NOTEBOOK_HEADER_CONVENTION.md
│   ├── phases/            # Phase & pipeline documentation
│   │   ├── PHASE_TEMPLATE.md
│   │   └── <PHASE>.md     # e.g. DATA_PREP.md, MODEL.md, TRAINING_INFO.md, EVAL.md
│   ├── templates/         # Document & checklist templates
│   │   ├── PROJECT_ROADMAP_TEMPLATE.md
│   │   ├── PHASE_DOC_TEMPLATE.md
│   │   ├── PROGRESS_STATUS_TEMPLATE.md
│   │   ├── CODEBASE_AUDIT_TEMPLATE.md
│   │   └── SMOKE_TEST_CHECKLIST.md
│   ├── references/        # External guides & reference docs
│   │   ├── REFERENCE_TEMPLATE.md
│   │   └── <GUIDE>.md     # e.g. OPTUNA_DB_GUIDE.md, GIT_AND_RELEASE_BEST_PRACTICES.md
│   ├── experiments/       # Experiment reports, plans & technical comparisons
│   │   ├── README.md
│   │   └── <EXP>.md       # from EXPERIMENT_TEMPLATE.md
│   ├── progress/          # One status file per task/phase
│   │   └── <PHASE>_STATUS.md  # from PROGRESS_TEMPLATE.md
│   └── bugs/              # Documented bug reports
│       ├── README.md
│       └── BUG_<NN>_<SHORT>.md  # from BUG_TEMPLATE.md
├── data/
│   ├── raw/               # never edited by the agent
│   ├── processed/
│   └── external/
├── src/
│   ├── data/              # loading, cleaning, transforms, dataloaders
│   ├── models/            # model definitions
│   ├── training/          # training loops (script entry points)
│   ├── eval/              # metrics, evaluation scripts
│   ├── experiments/       # python experiment execution scripts
│   └── utils/
├── notebooks/             # exploratory & deliverable notebooks (no training)
├── configs/
├── experiments/           # run outputs, checkpoints, plots, results
└── tests/                 # smoke tests + unit tests
```

## Rules

- Agent must not create top-level folders without flagging it first
- Anything in `data/raw/` is read-only — never written to by any script
- Exploratory/throwaway code stays in `notebooks/`, not `src/`
- If actual folder structure diverges from this file, that's a
  CODEBASE_AUDIT.md finding, not something to silently "fix"
- Before working on a phase, read the matching `agents/phases/<phase>.md` and
  `agents/progress/<phase>_STATUS.md` yourself before asking the human for context.
