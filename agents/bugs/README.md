# 🐛 Bug Reports & Troubleshooting Directory

This directory contains documented bug reports, runtime error analysis, and
resolution guides encountered during the project lifecycle.

## 🏗️ Architecture Overview

Bugs are reported against specific pipeline layers and tracked to resolution:

```
Component (src/data, src/training, notebooks) → BUG-<n> report
    → Root cause → Fix (committed) → Status (Resolved)
```

Each report follows [BUG_TEMPLATE.md](BUG_TEMPLATE.md) and links the affected
modules. Bug IDs are referenced from the codebase audit report and progress
docs.

## 📋 Bug Index

| Bug ID | Title & Summary | Component / Module | Severity | Status |
| :---: | :--- | :--- | :---: | :---: |
| `BUG-01` | <One-sentence title + link> | `src/...` | High | Resolved ✅ |

---

## 🛠️ General Troubleshooting Guidelines

1. **`BrokenPipeError` in PyTorch DataLoaders**:
   Set `num_workers = 0` when running inside Jupyter Notebooks on Linux to
   avoid multiprocessing `forkserver` IPC pipe closures.
2. **`ModuleNotFoundError` for Project Submodules**:
   Ensure `PROJECT_ROOT` is dynamically detected and added to `sys.path`:
   ```python
   _cwd = Path(os.getcwd()).resolve()
   PROJECT_ROOT = _cwd if (_cwd / "src").exists() else (_cwd.parent if (_cwd.parent / "src").exists() else _cwd)
   if str(PROJECT_ROOT) not in sys.path:
       sys.path.insert(0, str(PROJECT_ROOT))
   ```
