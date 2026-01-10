from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Optional

from _utils import ensure_dir, find_latest_run_dir, parse_xy_ratio


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_gx_unexpected_counts(gx_validation_path: str) -> Dict[str, int]:
    """
    Map GX expectations to our EF units (counts of rows / extra-rows).
    Returns dict with keys: missing_y, missing_price, dup_ds, out_of_range_price.
    """
    data = _read_json(gx_validation_path)
    out = {"missing_y": 0, "missing_price": 0, "dup_ds": 0, "out_of_range_price": 0}
    results = data.get("results", []) if isinstance(data, dict) else []

    for r in results:
        cfg = (r or {}).get("expectation_config", {}) or {}
        rtype = cfg.get("type")
        kwargs = cfg.get("kwargs", {}) or {}
        col = kwargs.get("column")
        res = (r or {}).get("result", {}) or {}
        unexpected = int(res.get("unexpected_count", 0) or 0)

        if rtype == "expect_column_values_to_not_be_null" and col == "y":
            out["missing_y"] = unexpected
        elif rtype == "expect_column_values_to_not_be_null" and col == "price":
            out["missing_price"] = unexpected
        elif rtype in ("expect_column_values_to_be_unique", "expect_column_values_to_be_unique") and col == "ds":
            out["dup_ds"] = unexpected
        elif rtype == "expect_column_values_to_be_between" and col == "price":
            # out-of-range price counts (missing values are not in unexpected_count for between)
            out["out_of_range_price"] = unexpected

    return out


def _extract_alibi_drift_features(alibi_drift_path: str) -> int:
    data = _read_json(alibi_drift_path)
    per = data.get("is_drift_per_feature", {}) if isinstance(data, dict) else {}
    return int(sum(1 for v in per.values() if bool(v)))


def _extract_alibi_outliers_y(alibi_outliers_path: str) -> Optional[int]:
    data = _read_json(alibi_outliers_path)
    if isinstance(data, dict) and "outliers_detected_y" in data:
        try:
            return int(data["outliers_detected_y"])
        except Exception:
            return None
    return None


def _extract_alibi_outliers_price(alibi_outliers_path: str) -> Optional[int]:
    data = _read_json(alibi_outliers_path)
    if isinstance(data, dict) and "outliers_detected_price" in data:
        try:
            return int(data["outliers_detected_price"])
        except Exception:
            return None
    return None


def _scenario_run_dir(explicit: Optional[str], default_parent: str) -> str:
    if explicit:
        return explicit
    return find_latest_run_dir(default_parent)


def _load_comparison_summary(run_dir: str) -> Dict[str, Any]:
    return _read_json(os.path.join(run_dir, "comparison_summary.json"))


def _abs_artifact_path(run_dir: str, rel_path: str) -> str:
    return os.path.join(run_dir, rel_path.replace("/", os.sep).replace("\\", os.sep))


def _parse_int(s: Any) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def build_EF_for_run(run_dir: str, scenario: str) -> Dict[str, Any]:
    """
    Returns:
      {scenario, run_dir, frameworks: {fw: {metric: value_or_null}}}
    """
    summary = _load_comparison_summary(run_dir)
    frameworks: Dict[str, Any] = {}

    for fw_name, fw_res in summary.items():
        fw_out: Dict[str, Any] = {
            # scalar counts (backward compatible)
            "missing_y": None,
            "missing_price": None,
            "dup_ds": None,
            "drift_features": None,
            "outliers_y": None,
            "outliers_price": None,
            "out_of_range_price": None,
            # sets for intersections
            # IMPORTANT: use None for "not supported / not produced", and list for "measured".
            "missing_y_rows": None,
            "missing_price_rows": None,
            "dup_ds_rows": None,
            "outliers_y_rows": None,
            "outliers_price_rows": None,
            "out_of_range_price_rows": None,
            "drifted_feature_names": None,
        }

        artifacts = fw_res.get("artifacts", []) or []
        check_vals = fw_res.get("check_values", {}) or {}

        if fw_name == "gx":
            # Prefer gx_issue_rows.json (exact row_ids per type for EF_total/EF_true/FP).
            gx_rows_rel = None
            for a in artifacts:
                if str(a).endswith("gx_issue_rows.json"):
                    gx_rows_rel = str(a)
                    break
            if gx_rows_rel:
                rows = _read_json(_abs_artifact_path(run_dir, gx_rows_rel))
                fw_out["missing_y_rows"] = rows.get("missing_y_rows", []) or []
                fw_out["missing_price_rows"] = rows.get("missing_price_rows", []) or []
                fw_out["dup_ds_rows"] = rows.get("dup_ds_rows", []) or []
                fw_out["out_of_range_price_rows"] = rows.get("out_of_range_price_rows", []) or []
                fw_out["missing_y"] = len(fw_out["missing_y_rows"])
                fw_out["missing_price"] = len(fw_out["missing_price_rows"])
                fw_out["dup_ds"] = len(fw_out["dup_ds_rows"])
                fw_out["out_of_range_price"] = len(fw_out["out_of_range_price_rows"])
            else:
                # Fallback: parse gx_validation.json (counts only, no row_ids)
                gx_val_rel = None
                for a in artifacts:
                    if str(a).endswith("gx_validation.json"):
                        gx_val_rel = str(a)
                        break
                if gx_val_rel:
                    gx_counts = _extract_gx_unexpected_counts(_abs_artifact_path(run_dir, gx_val_rel))
                    fw_out.update(gx_counts)

            # Drift feature names from gx_drift.json if present
            gx_drift_rel = None
            for a in artifacts:
                if str(a).endswith("gx_drift.json"):
                    gx_drift_rel = str(a)
                    break
            if gx_drift_rel:
                d = _read_json(_abs_artifact_path(run_dir, gx_drift_rel))
                per = d.get("is_drift_per_feature", {}) if isinstance(d, dict) else {}
                fw_out["drifted_feature_names"] = sorted([k for k, v in per.items() if bool(v)])
                fw_out["drift_features"] = len(fw_out["drifted_feature_names"])

            # If drift file missing, fall back to ratio "X/Y"
            if fw_out["drift_features"] is None:
                ratio = parse_xy_ratio(check_vals.get("data_drift"))
                if ratio:
                    fw_out["drift_features"] = ratio[0]

        elif fw_name == "alibi":
            # drift features via alibi_drift.json
            drift_rel = None
            out_rel = None
            for a in artifacts:
                s = str(a)
                if s.endswith("alibi_drift.json"):
                    drift_rel = s
                if s.endswith("alibi_outliers.json"):
                    out_rel = s
            if drift_rel:
                fw_out["drift_features"] = _extract_alibi_drift_features(_abs_artifact_path(run_dir, drift_rel))
                try:
                    d = _read_json(_abs_artifact_path(run_dir, drift_rel))
                    per = d.get("is_drift_per_feature", {}) if isinstance(d, dict) else {}
                    fw_out["drifted_feature_names"] = sorted([k for k, v in per.items() if bool(v)])
                except Exception:
                    pass
            if out_rel:
                fw_out["outliers_y"] = _extract_alibi_outliers_y(_abs_artifact_path(run_dir, out_rel))
                fw_out["outliers_price"] = _extract_alibi_outliers_price(_abs_artifact_path(run_dir, out_rel))
                try:
                    d = _read_json(_abs_artifact_path(run_dir, out_rel))
                    fw_out["outliers_y_rows"] = d.get("outlier_indices", []) or []
                    fw_out["outliers_price_rows"] = d.get("outlier_price_indices", []) or []
                except Exception:
                    pass

            # missing_* are not provided by Alibi artifacts in our project -> keep None

        elif fw_name == "nannyml":
            ratio = parse_xy_ratio(check_vals.get("data_drift"))
            if ratio:
                fw_out["drift_features"] = ratio[0]
            fw_out["outliers_y"] = _parse_int(check_vals.get("outliers"))
            fw_out["outliers_price"] = _parse_int(check_vals.get("outliers_price"))

            # row sets
            out_rel = None
            drift_rel = None
            for a in artifacts:
                s = str(a)
                if s.endswith("nannyml_outliers.json"):
                    out_rel = s
                if s.endswith("nannyml_drift_features.json"):
                    drift_rel = s
            if out_rel:
                d = _read_json(_abs_artifact_path(run_dir, out_rel))
                fw_out["outliers_y_rows"] = d.get("outliers_y_rows", []) or []
                fw_out["outliers_price_rows"] = d.get("outliers_price_rows", []) or []
            if drift_rel:
                d = _read_json(_abs_artifact_path(run_dir, drift_rel))
                fw_out["drifted_feature_names"] = d.get("drifted_feature_names", []) or []
                fw_out["drift_features"] = len(fw_out["drifted_feature_names"])

        elif fw_name == "evidently":
            ratio = parse_xy_ratio(check_vals.get("data_drift"))
            if ratio:
                fw_out["drift_features"] = ratio[0]
            fw_out["outliers_y"] = _parse_int(check_vals.get("outliers"))
            fw_out["outliers_price"] = _parse_int(check_vals.get("outliers_price"))

            out_rel = None
            drift_rel = None
            for a in artifacts:
                s = str(a)
                if s.endswith("evidently_outliers.json"):
                    out_rel = s
                if s.endswith("evidently_drift.json"):
                    drift_rel = s
            if out_rel:
                d = _read_json(_abs_artifact_path(run_dir, out_rel))
                fw_out["outliers_y_rows"] = d.get("outliers_y_rows", []) or []
                fw_out["outliers_price_rows"] = d.get("outliers_price_rows", []) or []
            if drift_rel:
                d = _read_json(_abs_artifact_path(run_dir, drift_rel))
                per = d.get("is_drift_per_feature", {}) if isinstance(d, dict) else {}
                fw_out["drifted_feature_names"] = sorted([k for k, v in per.items() if bool(v)])
                fw_out["drift_features"] = len(fw_out["drifted_feature_names"])

        frameworks[fw_name] = fw_out

    return {"scenario": scenario, "run_dir": run_dir, "frameworks": frameworks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Count EF_test (detected issues) from framework artifacts on test split.")
    parser.add_argument("--run-pass", default=None, help="Path to reports/run_pass/<timestamp>. If omitted, latest is used.")
    parser.add_argument("--run-dr", default=None, help="Path to reports/run_dr/<timestamp>. If omitted, latest is used.")
    parser.add_argument("--run-ideal", default=None, help="Path to reports/run_ideal/<timestamp>. If omitted, latest is used (if exists).")
    parser.add_argument("--out-json", default="reports/mini_table_EF_test.json", help="Output JSON path")
    parser.add_argument("--out-csv", default="reports/mini_table_EF_test.csv", help="Output CSV path")
    args = parser.parse_args()

    run_pass_dir = _scenario_run_dir(args.run_pass, "reports/run_pass")
    run_dr_dir = _scenario_run_dir(args.run_dr, "reports/run_dr")
    run_ideal_dir = None
    try:
        run_ideal_dir = _scenario_run_dir(args.run_ideal, "reports/run_ideal")
    except Exception:
        run_ideal_dir = None

    ef = {
        "split_rule": "EF_test is taken from framework artifacts produced on test split (last 20% by time).",
        "runs": {
            "pass": build_EF_for_run(run_pass_dir, "pass"),
            "dr": build_EF_for_run(run_dr_dir, "dr"),
        },
    }
    if run_ideal_dir:
        ef["runs"]["ideal"] = build_EF_for_run(run_ideal_dir, "ideal")

    ensure_dir(os.path.dirname(args.out_json) or ".")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(ef, f, ensure_ascii=False, indent=2)

    # Flat CSV (one row per scenario+framework)
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "framework",
                "missing_y",
                "missing_price",
                "dup_ds",
                "drift_features",
                "outliers_y",
                "outliers_price",
                "out_of_range_price",
            ]
        )
        for scen in ("pass", "dr", "ideal"):
            if scen not in ef["runs"]:
                continue
            for fw, vals in ef["runs"][scen]["frameworks"].items():
                w.writerow(
                    [
                        scen,
                        fw,
                        vals.get("missing_y") if vals.get("missing_y") is not None else "NA",
                        vals.get("missing_price") if vals.get("missing_price") is not None else "NA",
                        vals.get("dup_ds") if vals.get("dup_ds") is not None else "NA",
                        vals.get("drift_features") if vals.get("drift_features") is not None else "NA",
                        vals.get("outliers_y") if vals.get("outliers_y") is not None else "NA",
                        vals.get("outliers_price") if vals.get("outliers_price") is not None else "NA",
                        vals.get("out_of_range_price") if vals.get("out_of_range_price") is not None else "NA",
                    ]
                )

    print(f"Saved: {args.out_json}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()


