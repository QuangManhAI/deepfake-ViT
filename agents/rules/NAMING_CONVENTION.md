# NAMING_CONVENTION.md
Rules the agent must follow when creating files, variables, or functions.
Edit these to match your actual preferences — this is a starting point.

## Files & folders
- Scripts: `snake_case.py` (e.g. `train_model.py`, not `TrainModel.py`)
- Notebooks: `NN_short_purpose.ipynb` (e.g. `01_eda.ipynb`, `02_baseline.ipynb`)
- Docs: `UPPER_SNAKE_CASE.md`, matches the phase it documents (`DATA_PREP.md`, `TRAINING_INFO.md`)
- Config files: `config.yaml` or `config_<experiment_name>.yaml`
- Checkpoints/artifacts: `<model>_<dataset>_<date_or_run_id>.pt`

## Code
- Functions: `snake_case`, verb-first (`load_data`, `compute_metrics`, not `data_loader_thing`)
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private/internal helpers: prefix with `_`

## Experiments / runs
- Run IDs: `YYYYMMDD_short-description` (e.g. `20260729_baseline-xgb`)
- Never overwrite a previous run's output folder — always new run ID

## What the agent must NOT do
- Do not invent new naming schemes mid-project
- Do not rename existing files without flagging it to the human first
- If a name in code conflicts with this file, flag the conflict — do not silently pick one
