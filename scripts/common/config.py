from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    if not isinstance(cfg_raw, dict):
        raise ValueError("Config must be a JSON object at top-level")
    cfg: Dict[str, Any] = cfg_raw

    # Optional local override (ignored by git): config.local.json next to base config.
    local_path = config_path.with_name("config.local.json")
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            local_raw = json.load(f)
        if not isinstance(local_raw, dict):
            raise ValueError("Local config must be a JSON object at top-level")
        cfg = _deep_merge(cfg, local_raw)

    return cfg

def get_cfg_value(cfg: Dict[str, Any], key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def get_path(cfg: Dict[str, Any], key: str, *, default: Optional[str] = None, base_dir: Optional[Path] = None) -> Path:
    val = get_cfg_value(cfg, key)
    if val is None:
        val = default
    if val is None:
        raise KeyError(f"Missing required config key: {key}")
    p = Path(str(val))
    if base_dir is not None and not p.is_absolute():
        p = (base_dir / p).resolve()
    return p

def get_int(cfg: Dict[str, Any], key: str, *, default: Optional[int] = None) -> int:
    val = get_cfg_value(cfg, key)
    if val is None:
        val = default
    if val is None:
        raise KeyError(f"Missing required config key: {key}")
    return int(val)

def get_str(cfg: Dict[str, Any], key: str, *, default: Optional[str] = None) -> str:
    val = get_cfg_value(cfg, key)
    if val is None:
        val = default
    if val is None:
        raise KeyError(f"Missing required config key: {key}")
    return str(val)

def get_bool(cfg: Dict[str, Any], key: str, *, default: Optional[bool] = None) -> bool:
    val = get_cfg_value(cfg, key)
    if val is None:
        val = default
    if val is None:
        raise KeyError(f"Missing required config key: {key}")

    if isinstance(val, bool):
        return val

    if isinstance(val, (int, float)):
        return bool(val)

    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean string for config key '{key}': {val!r}")

    raise ValueError(f"Invalid boolean type for config key '{key}': {type(val).__name__}")
