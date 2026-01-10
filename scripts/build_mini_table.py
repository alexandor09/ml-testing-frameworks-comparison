from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Optional

from _utils import ensure_dir


ACTIVE_TYPES = {
    "pass": ["missing_y", "missing_price", "dup_ds"],
    "dr": ["drift_features", "outliers_y", "outliers_price", "out_of_range_price"],
    # ideal is control: E=0, but we still report EF_total and FP for the union of types
    "ideal": ["missing_y", "missing_price", "dup_ds", "drift_features", "outliers_y", "outliers_price", "out_of_range_price"],
}

UNITS = {
    "missing_y": "rows",
    "missing_price": "rows",
    "dup_ds": "rows",
    "drift_features": "features",
    "outliers_y": "rows",
    "outliers_price": "rows",
    "out_of_range_price": "rows",
}

# What to output in the final table per scenario.
# For `pass` we also include "false alarm" lines (E_test=0) for drift/outliers/range.
ROWS_BY_SCENARIO = {
    "pass": ["missing_y", "missing_price", "dup_ds", "drift_features", "outliers_y", "outliers_price", "out_of_range_price"],
    "dr": ["drift_features", "outliers_y", "outliers_price", "out_of_range_price"],
    "ideal": ["missing_y", "missing_price", "dup_ds", "drift_features", "outliers_y", "outliers_price", "out_of_range_price"],
}

FRAMEWORKS = ["gx", "evidently", "alibi", "nannyml"]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _as_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _ratio(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num / den)


def _fmt_num(v: Optional[int]) -> str:
    return "NA" if v is None else str(int(v))


def _fmt_metric(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final mini-table from E_test and EF_test.")
    parser.add_argument("--e-test", default="reports/mini_table_E_test.json", help="Path to E_test JSON")
    parser.add_argument("--ef-test", default="reports/mini_table_EF_test.json", help="Path to EF_test JSON")
    parser.add_argument("--out", default="reports/mini_table_final.csv", help="Output CSV path")
    args = parser.parse_args()

    e = _read_json(args.e_test)
    ef = _read_json(args.ef_test)

    ensure_dir(os.path.dirname(args.out) or ".")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)

        header = ["scenario", "problem_type", "unit", "n_test", "E_test"]
        for fw in FRAMEWORKS:
            header += [
                f"EF_total_{fw}",
                f"EF_true_{fw}",
                f"FP_{fw}",
                f"Recall_{fw}",
                f"Precision_{fw}",
                f"FalseAlarm_{fw}",
            ]
        w.writerow(header)

        for scenario, types in ROWS_BY_SCENARIO.items():
            if scenario not in e.get("scenarios", {}):
                continue
            if scenario not in ef.get("runs", {}):
                continue

            for problem_type in types:
                unit = UNITS.get(problem_type, "")
                # E sets
                e_rows = _get_nested(e, "scenarios", scenario, "E_rows") or {}
                n_test_val = _get_nested(e, "scenarios", scenario, "n_test")
                n_test = int(n_test_val) if n_test_val is not None else 0
                if problem_type == "drift_features":
                    e_set = set(e_rows.get("drifted_feature_names", []) or [])
                else:
                    e_set = set(e_rows.get(f"{problem_type}_rows", []) or [])
                e_int = len(e_set)

                row = [scenario, problem_type, unit, n_test, e_int]

                for fw in FRAMEWORKS:
                    fw_block = _get_nested(ef, "runs", scenario, "frameworks", fw) or {}
                    if problem_type == "drift_features":
                        raw = fw_block.get("drifted_feature_names", None)
                        ef_set = None if raw is None else set(raw or [])
                    else:
                        raw = fw_block.get(f"{problem_type}_rows", None)
                        ef_set = None if raw is None else set(raw or [])

                    if ef_set is None:
                        ef_total = None
                        ef_true = None
                        fp = None
                        recall = None
                        precision = None
                        false_alarm = None
                    else:
                        ef_total = len(ef_set)
                        ef_true = len(e_set.intersection(ef_set))
                        fp = ef_total - ef_true

                        # Recall is not applicable when E_test == 0 (false-alarm lines)
                        recall = None if e_int == 0 else _ratio(ef_true, e_int)
                        # Precision is not applicable when EF_total == 0
                        precision = None if (ef_total == 0 or e_int == 0) else _ratio(ef_true, ef_total)

                        # False alarm metric for E_test == 0:
                        # - for row-based: EF_total / n_test
                        # - for drift features: EF_total / 3 (max features)
                        if e_int == 0:
                            if unit == "rows":
                                false_alarm = None if n_test <= 0 else _ratio(ef_total, n_test)
                            elif unit == "features":
                                false_alarm = _ratio(ef_total, 3)
                            else:
                                false_alarm = None
                        else:
                            false_alarm = None

                    row += [
                        _fmt_num(ef_total),
                        _fmt_num(ef_true),
                        _fmt_num(fp),
                        _fmt_metric(recall),
                        _fmt_metric(precision),
                        _fmt_metric(false_alarm),
                    ]

                w.writerow(row)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


