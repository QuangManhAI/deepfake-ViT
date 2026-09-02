"""Return the currently selected protocol's CSV paths from protocol_config.json."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_protocol_config(path=None):
    """Load protocol configuration.

    Args:
        path: optional explicit config file. Defaults to data/protocol/protocol_config.json.

    Returns:
        dict with DATA_PROTOCOL, protocol_dir, train_csv, val_csv, test_csv.
    """
    if path is None:
        path = PROJECT_ROOT / "data" / "protocol" / "protocol_config.json"
    with open(path) as f:
        return json.load(f)


def get_protocol_csvs():
    """Return (train_csv, val_csv, test_csv) for the active protocol."""
    cfg = get_protocol_config()
    return cfg["train_csv"], cfg["val_csv"], cfg["test_csv"]


if __name__ == "__main__":
    cfg = get_protocol_config()
    print(json.dumps(cfg, indent=2))
