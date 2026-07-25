"""
config_loader.py
----------------
Loads and validates YAML scenario configurations.

WHY THIS EXISTS:
  Every tuneable parameter lives in configs/scenario_N.yaml.
  This module is the single gateway between those files and
  all Python code. Nothing in src/ imports PyYAML directly —
  everything goes through load_config().

KEY CONCEPTS:
  - Strict validation: missing keys raise ConfigError immediately,
    not silently downstream where the bug is hard to trace.
  - Dot-access: returns a SimpleNamespace so you can write
    cfg.agents.count instead of cfg['agents']['count'].
  - [V2-READY]: load_config() accepts an optional override_dict
    parameter. V2's interactive constraint editor will pass
    user-edited values here without touching the YAML files.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# ── Custom exception ──────────────────────────────────────────────────────────

class ConfigError(ValueError):
    """Raised when a scenario YAML is missing required keys or has bad values."""
    pass


# ── Required top-level keys — validation guard ────────────────────────────────

_REQUIRED_KEYS = {"scenario", "grid", "agents", "items", "training", "rewards"}


# ── Public API ────────────────────────────────────────────────────────────────

def load_config(
    scenario_id: int,
    configs_dir: str | Path = "configs",
    override_dict: dict[str, Any] | None = None,   # [V2-READY]
) -> SimpleNamespace:
    """
    Load and validate a scenario YAML file.

    Parameters
    ----------
    scenario_id : int
        Which scenario to load (1, 2, or 3).
    configs_dir : str | Path
        Directory containing scenario_N.yaml files.
        Defaults to 'configs/' relative to the working directory.
    override_dict : dict | None
        [V2-READY] Key-value pairs that overwrite YAML values after
        loading. Nested keys use dot notation, e.g.
        {"agents.count": 6, "rewards.delivery_reward": 15.0}.

    Returns
    -------
    SimpleNamespace
        Deeply nested namespace mirroring YAML structure.
        Access via cfg.agents.count, cfg.rewards.delivery_reward, etc.

    Raises
    ------
    ConfigError
        If the YAML file is missing, unreadable, or fails validation.
    """
    config_path = Path(configs_dir) / f"scenario_{scenario_id}.yaml"

    if not config_path.exists():
        raise ConfigError(
            f"Config file not found: {config_path}. "
            f"Expected scenario_id in {{1, 2, 3}}."
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    _validate(raw, config_path)

    if override_dict:
        raw = _apply_overrides(raw, override_dict)

    return _dict_to_namespace(raw)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate(raw: dict, path: Path) -> None:
    """Check that all required top-level sections are present."""
    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ConfigError(
            f"Config {path} is missing required sections: {missing}"
        )

    if raw["agents"]["count"] < 1:
        raise ConfigError("agents.count must be >= 1")

    if raw["grid"]["width"] < 4 or raw["grid"]["height"] < 4:
        raise ConfigError("Grid dimensions must be at least 4x4")


def _apply_overrides(raw: dict, overrides: dict[str, Any]) -> dict:
    """
    Apply dot-notation overrides to the raw config dict.
    e.g. "agents.count" → raw["agents"]["count"] = value
    [V2-READY]: Called by interactive constraint editor.
    """
    import copy
    raw = copy.deepcopy(raw)
    for key_path, value in overrides.items():
        keys = key_path.split(".")
        target = raw
        for k in keys[:-1]:
            if k not in target:
                raise ConfigError(f"Override key path '{key_path}' is invalid.")
            target = target[k]
        target[keys[-1]] = value
    return raw


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict to a SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns
