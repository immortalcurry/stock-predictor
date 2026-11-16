"""Loads config.toml into a simple dict and adds a few handy derived fields."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUP_KEYS = ["memory", "semis", "ai_infra", "big_tech"]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.toml"
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    uni = cfg["universe"]
    # Ticker -> group name. The first group wins if a ticker is listed twice.
    groups = {}
    for g in GROUP_KEYS:
        for t in uni.get(g, []):
            groups.setdefault(t, g)
    for t in uni["holdings"]:
        groups.setdefault(t, "holding")

    cfg["targets"] = list(groups)          # every ticker we predict
    # Pool = targets + extra training-only tickers (more examples for the model)
    cfg["pool"] = list(groups) + [t for t in uni.get("extra_training", []) if t not in groups]
    cfg["groups"] = groups
    cfg["holdings"] = list(uni["holdings"])
    cfg["data"].setdefault("start_overrides", {})
    return cfg
