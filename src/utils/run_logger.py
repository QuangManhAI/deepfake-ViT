"""Zero-dependency run logger + full-state checkpoint helpers.

Implements the persistence contract in
``agents/rules/LOGGING_CHECKPOINT_RULES.md`` for this repo's training loops:

- A run directory ``<output_dir>/<run_name>/`` holding checkpoints (best +
  every-epoch ``_last``), per-epoch JSONL history, and a config JSON.
- A timestamped log written to ``logs/<run_name>.log`` with rotation at
  20 MB (gzip, keep 3 archives) plus a live console progress line.
- ``save_full_checkpoint`` / ``load_full_checkpoint`` helpers for the
  full-state dict (model + optimizer + scheduler + epoch + global_step +
  best metrics + history + config + RNG seeds), loadable with
  ``weights_only=True``.
"""
import datetime as _dt
import gzip
import json
import os
import random
import time

import numpy as np
import torch


class RunLogger:
    """Console + rotating-file logger with progress and epoch summaries."""

    def __init__(self, run_dir, run_name, level="INFO"):
        self.run_name = run_name
        self.log_dir = os.path.join(run_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"{run_name}.log")
        self.level = level
        self._log(f"RunLogger started: {run_name}")

    def _rotate(self):
        if os.path.getsize(self.log_path) <= 20 * 1024 * 1024:
            return
        archives = sorted(
            f for f in os.listdir(self.log_dir)
            if f.startswith(f"{self.run_name}.") and f.endswith(".log.gz")
        )
        while len(archives) >= 3:
            os.remove(os.path.join(self.log_dir, archives.pop(0)))
        seq = len(archives) + 1
        arc_path = os.path.join(self.log_dir, f"{self.run_name}.{seq}.log.gz")
        with open(self.log_path, "rb") as src, gzip.open(arc_path, "wb") as dst:
            dst.write(src.read())
        open(self.log_path, "w").close()

    def _log(self, message, lvl="INFO"):
        line = f"[{time.strftime('%H:%M:%S')}] [{lvl}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def progress(self, message):
        print(message, end="\r", flush=True)

    def epoch_summary(self, epoch, total_epochs, metrics, seconds):
        msg = (f"Epoch {epoch}/{total_epochs} | "
               + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
               + f" | {seconds:.0f}s")
        self._log(msg, lvl="TRAIN")

    def info(self, message):
        self._log(message, lvl="INFO")

    def warn(self, message):
        self._log(message, lvl="WARN")

    def error(self, message):
        self._log(message, lvl="ERROR")


def make_run_dir(output_dir, run_name):
    """Create ``<output_dir>/<run_name>/`` with checkpoints/, logs/, metrics/."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{ts}_{run_name}")
    for sub in ("checkpoints", "logs", "metrics"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def find_latest_run_dir(output_dir, run_name):
    """Return the most-recently-modified ``<ts>_<run_name>/`` dir, or None."""
    import glob as _glob

    matches = sorted(
        _glob.glob(os.path.join(output_dir, f"*_{run_name}")),
        key=lambda p: os.path.getmtime(p),
    )
    return matches[-1] if matches else None


def _rng_snapshot():
    """JSON-safe RNG states: python, numpy, torch CPU/CUDA (hex strings)."""
    snap = {
        "python": _jsonable(random.getstate()),
        "numpy": _jsonable(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().numpy().tobytes().hex(),
        "torch_cuda": [
            s.numpy().tobytes().hex() for s in torch.cuda.get_rng_state_all()
        ],
    }
    # numpy state contains an array → hex-encode it; torch/numpy fallback to seeds.
    snap["numpy"] = _jsonable(np.random.get_state())
    try:
        json.dumps(snap)
    except TypeError:
        snap = {"python_seed": None, "numpy_seed": None, "note": "full state not JSON-safe"}
    return snap


def _restore_rng(snap):
    """Best-effort restore of python/numpy/torch RNG states from a snapshot.

    No-ops on missing/malformed entries; torch CPU is restored when present.
    """
    if not snap:
        return
    py = snap.get("python")
    if isinstance(py, list) and len(py) == 3 and isinstance(py[1], list):
        try:
            random.setstate((int(py[0]), tuple(int(x) for x in py[1]), py[2]))
        except (ValueError, TypeError):
            pass
    np_state = snap.get("numpy")
    if isinstance(np_state, list) and np_state and isinstance(np_state[0], str):
        try:
            keys = np_state[1].get("hex")
            arr = np.frombuffer(bytes.fromhex(keys), dtype=np_state[1].get("dtype", "uint32"))
            np.random.set_state((np_state[0], arr, int(np_state[2]), bool(np_state[3]), np_state[4]))
        except (ValueError, TypeError, KeyError):
            pass
    tc = snap.get("torch_cpu")
    if isinstance(tc, str) and tc:
        try:
            torch.set_rng_state(torch.ByteTensor(bytearray.fromhex(tc)))
        except (ValueError, RuntimeError):
            pass

def _jsonable(state):
    """Convert a python/numpy RNG state tuple into a JSON-safe structure.

    ``random.getstate()`` returns ``(version, [ints...], gauss_next)`` and
    ``np.random.get_state()`` returns ``(bitgen_name, keys_array, pos, has_gauss,
    cached_gaussian)`` — the numpy keys array is hex-encoded here.
    """
    if isinstance(state, tuple):
        out = []
        for part in state:
            if isinstance(part, np.ndarray):
                out.append({"dtype": str(part.dtype), "hex": part.tobytes().hex()})
            elif isinstance(part, (list, tuple)):
                out.append([int(x) for x in part])
            elif isinstance(part, float):
                out.append(part)
            elif part is None:
                out.append(None)
            else:
                out.append(part)
        return out
    return state


def save_full_checkpoint(path, model, optimizer, scheduler, epoch, global_step,
                         best_metrics, history, config, seed):
    """Save the full-state checkpoint dict (model+optimizer+scheduler+RNG+history)."""
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": best_metrics,
        "history": history,
        "config": config,
        "seed": seed,
        "rng": _rng_snapshot(),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "early_stop_triggered": bool(best_metrics.get("early_stop_triggered", False)),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    return ckpt


def load_full_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """Load a full-state checkpoint with ``weights_only=True`` and restore
    model (required) + optimizer + scheduler + RNG states. Returns the dict."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    _restore_rng(ckpt.get("rng"))
    return ckpt


def append_history_jsonl(metrics_path, epoch_metrics):
    """Append one JSON line per epoch (append-only, crash-safe)."""
    with open(metrics_path, "a") as f:
        f.write(json.dumps(epoch_metrics) + "\n")


def write_config_json(config_path, config):
    """Persist the run config (hyperparams + 5W1H description + outputs list)."""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
