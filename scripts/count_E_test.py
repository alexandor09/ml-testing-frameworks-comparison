from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from _utils import ensure_dir, load_csv, time_split_80_20


def iqr_outliers_count(values: np.ndarray, k: float = 1.5) -> int:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return int(np.sum((values < lo) | (values > hi)))


ACTIVE_TYPES = {
    "pass": ["missing_y", "missing_price", "dup_ds"],
    "dr": ["drift_features", "outliers_y", "outliers_price", "out_of_range_price"],
    "ideal": [],  # контроль FP: все E=0
}


def _iqr_outlier_rows(series, k: float = 1.5) -> list[int]:
    s = series.dropna()
    v = s.to_numpy(dtype=float)
    if v.size == 0:
        return []
    q1 = np.percentile(v, 25)
    q3 = np.percentile(v, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    mask = (v < lo) | (v > hi)
    idx = s.index.to_list()
    return [idx[i] for i in np.where(mask)[0].tolist() if i < len(idx)]


def scenario_from_path(path: str) -> str:
    base = os.path.basename(path).lower()
    for s in ("pass", "dr", "ideal", "small", "big"):
        if s in base:
            return s
    return os.path.splitext(base)[0]


def compute_E_test_for_csv(csv_path: str, scenario: str) -> Dict[str, Any]:
    df = load_csv(csv_path)
    split = time_split_80_20(df)
    test = split.test

    out: Dict[str, Any] = {
        "csv": csv_path,
        "scenario": scenario,
        "active_types": ACTIVE_TYPES.get(scenario, []),
        "n_rows": int(len(df)),
        "n_test": int(len(test)),
    }

    # Row sets for honest intersections (row_id = index after sort+reset, same as main.py)
    e_rows: Dict[str, Any] = {
        "missing_y_rows": [],
        "missing_price_rows": [],
        "dup_ds_rows": [],
        "outliers_y_rows": [],
        "outliers_price_rows": [],
        "out_of_range_price_rows": [],
        "drifted_feature_names": [],
    }

    if scenario == "pass":
        if "y" in test.columns:
            e_rows["missing_y_rows"] = test.index[test["y"].isna()].to_list()
        if "price" in test.columns:
            e_rows["missing_price_rows"] = test.index[test["price"].isna()].to_list()
        if "ds" in test.columns:
            e_rows["dup_ds_rows"] = test.index[test["ds"].duplicated(keep=False)].to_list()

        out["missing_y"] = int(len(e_rows["missing_y_rows"]))
        out["missing_price"] = int(len(e_rows["missing_price_rows"]))
        out["dup_ds"] = int(len(e_rows["dup_ds_rows"]))

        # Inactive for pass
        out["drift_features"] = 0
        out["outliers_y"] = 0
        out["outliers_price"] = 0
        out["out_of_range_price"] = 0

    elif scenario == "dr":
        e_rows["drifted_feature_names"] = ["price", "promotion", "y"]  # injected by generator

        if "y" in test.columns:
            e_rows["outliers_y_rows"] = _iqr_outlier_rows(test["y"], k=1.5)
        if "price" in test.columns:
            e_rows["outliers_price_rows"] = _iqr_outlier_rows(test["price"], k=1.5)
            p = test["price"]
            e_rows["out_of_range_price_rows"] = test.index[p.notna() & ((p < 80) | (p > 120))].to_list()

        out["drift_features"] = int(len(e_rows["drifted_feature_names"]))
        out["outliers_y"] = int(len(e_rows["outliers_y_rows"]))
        out["outliers_price"] = int(len(e_rows["outliers_price_rows"]))
        out["out_of_range_price"] = int(len(e_rows["out_of_range_price_rows"]))

        # Inactive for dr
        out["missing_y"] = 0
        out["missing_price"] = 0
        out["dup_ds"] = 0

    else:
        # ideal (or unknown): E=0 for all types (control false positives)
        out["missing_y"] = 0
        out["missing_price"] = 0
        out["dup_ds"] = 0
        out["drift_features"] = 0
        out["outliers_y"] = 0
        out["outliers_price"] = 0
        out["out_of_range_price"] = 0

    out["E_rows"] = e_rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Count E_test (injected issues) on test split (last 20% by time).")
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV paths, e.g. data/pass.csv data/dr.csv data/ideal.csv")
    parser.add_argument("--out", default="reports/mini_table_E_test.json", help="Output JSON path")
    args = parser.parse_args()

    results: Dict[str, Any] = {
        "split_rule": "sort by ds, test = last 20% (80/20)",
        "row_id_definition": "row_id is the row index after sorting by ds and resetting index (same as main.py).",
        "scenarios": {},
    }
    for p in args.inputs:
        scen = scenario_from_path(p)
        results["scenarios"][scen] = compute_E_test_for_csv(p, scenario=scen)

    ensure_dir(os.path.dirname(args.out) or ".")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


