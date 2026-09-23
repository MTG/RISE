"""Loading of ``configs.yaml``.

The YAML file holds one mapping per experiment whose keys mirror the command line
options of that experiment. Values loaded from it become the argparse defaults,
so a flag given on the command line always wins over the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_FILE


def load_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the full configuration, or an empty mapping if the file is absent."""
    path = path or CONFIG_FILE
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return {section: values or {} for section, values in config.items()}


def experiment_parameters(args: Any) -> dict[str, Any]:
    """The reportable arguments of a run.

    Strips the private entries the parser uses to dispatch, and anything not worth
    recording, so that the same mapping can be echoed to the console and logged to
    MLflow without either having to know about the other.
    """
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and key != "experiment" and not callable(value)
    }
