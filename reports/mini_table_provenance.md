## Mini-table provenance (where EF_* values come from)

Conventions:
- **row_id** = row index after sorting by `ds` and resetting index (same as `main.py` split)
- **NA** means "this metric was not produced by this framework in current configuration"
- **—** means "not applicable" (e.g., Recall when \(E\_test = 0\), Precision when \(EF\_total = 0\))
- **FalseAlarm** is used only when \(E\_test = 0\): for row-based issues it is \(EF\_total / n\_test\), for drift_features it is \(EF\_total / 3\)

### Injected issues (E_test)
- Computed by `scripts/count_E_test.py` on **test split** (last 20% by time).
- Outliers/out_of_range use IQR (k=1.5) and range price ∉ [80;120].

### Detected issues (EF_total / EF_true)
We take EF sets from framework artifacts (test split), then compute intersections vs E sets:

- **GX**
  - missing/dup/range row_id sets: `gx/gx_issue_rows.json`
  - drift features: `gx/gx_drift.json` (`is_drift_per_feature`)
- **Evidently**
  - outliers row_id sets: `evidently/evidently_outliers.json`
  - drift features: `evidently/evidently_drift.json` (`is_drift_per_feature`)
- **Alibi Detect**
  - outliers row_id sets: `alibi/alibi_outliers.json` (`outlier_indices`, `outlier_price_indices`)
  - drift features: `alibi/alibi_drift.json` (`is_drift_per_feature`)
- **NannyML**
  - outliers row_id sets: `nannyml/nannyml_outliers.json`
  - drift features: `nannyml/nannyml_drift_features.json` (`drifted_feature_names`)


