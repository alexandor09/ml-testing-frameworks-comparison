from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ds" not in df.columns:
        raise ValueError(f"Missing required column 'ds' in {path}")
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def time_split_80_20(df: pd.DataFrame) -> SplitResult:
    """Same split rule as in main.py: sort by ds, last 20% rows are test."""
    df = df.sort_values("ds").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    return SplitResult(train=df.iloc[:split_idx].copy(), test=df.iloc[split_idx:].copy())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_latest_run_dir(parent_dir: str) -> str:
    """Pick latest timestamped run folder like reports/run_pass/20260110_150247."""
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(f"Run directory not found: {parent_dir}")
    candidates: List[str] = []
    for name in os.listdir(parent_dir):
        p = os.path.join(parent_dir, name)
        if os.path.isdir(p) and RUN_ID_RE.match(name):
            candidates.append(name)
    if not candidates:
        raise FileNotFoundError(f"No run folders found in {parent_dir}")
    candidates.sort()
    return os.path.join(parent_dir, candidates[-1])


def parse_xy_ratio(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse strings like:
    - "3/3"
    - "2/2 a=10"
    Returns (x, y) or None.
    """
    if not s:
        return None
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(s))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


