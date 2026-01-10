import pandas as pd
import nannyml as nml
import os
import numpy as np
import json
from .base import BaseAdapter
from .monitoring import measure_time_and_peak_rss_mb
from typing import Dict, Any

class NannyMLAdapter(BaseAdapter):
    def run_checks(self, train_df: pd.DataFrame, test_df: pd.DataFrame, predictions: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        issues_detected = 0
        artifacts = []
        checks_performed = {
            'data_quality': True,
            'data_drift': True,
            'outliers': True,
            'model_performance': True
        }
        check_values = {
            'data_quality': None,
            'data_drift': None,
            'outliers': None,
            'model_performance': None
        }

        def _run() -> None:
            nonlocal issues_detected, artifacts, check_values
            train_data = train_df.copy()
            test_data = test_df.copy()

            if 'yhat' not in train_data.columns and 'y' in train_data.columns:
                train_data['yhat'] = train_data['y']  # Заглушка

            chunk_size = max(1, len(test_data) // 5)

            # 1. Performance (регрессия)
            calc = nml.PerformanceCalculator(
                y_pred='yhat',
                y_true='y',
                timestamp_column_name='ds',
                problem_type='regression',
                metrics=['mae', 'rmse'],
                chunk_size=chunk_size,
            )

            calc.fit(train_data)
            perf_results = calc.calculate(test_data)

            fig = perf_results.plot()
            fig_path = os.path.join(output_dir, "nannyml_performance.html")
            fig.write_html(fig_path)
            artifacts.append(fig_path)

            # Извлекаем значения метрик производительности
            perf_df = perf_results.filter(period='analysis').to_df()
            mae_value = None
            rmse_value = None
            
            for metric in ['mae', 'rmse']:
                alerts = perf_results.filter(period='analysis', metrics=[metric]).to_df()
                alert_cols = [c for c in alerts.columns if 'alert' in c and metric in c]
                if alert_cols:
                    issues_detected += int(alerts[alert_cols[0]].sum())
                
                # Извлекаем средние значения метрик - пробуем разные варианты колонок
                metric_cols = []
                for c in perf_df.columns:
                    if isinstance(c, str):
                        c_lower = c.lower()
                        if metric in c_lower and ('value' in c_lower or 'metric' in c_lower or metric.upper() in c):
                            metric_cols.append(c)
                    elif isinstance(c, tuple):
                        # Для MultiIndex колонок
                        c_str = str(c).lower()
                        if metric in c_str and ('value' in c_str or 'metric' in c_str):
                            metric_cols.append(c)
                
                if metric_cols:
                    try:
                        metric_values = perf_df[metric_cols[0]].dropna()
                        if len(metric_values) > 0:
                            avg_value = float(metric_values.mean())
                            if metric == 'mae':
                                mae_value = avg_value
                            elif metric == 'rmse':
                                rmse_value = avg_value
                    except Exception:
                        pass
            
            if checks_performed['model_performance']:
                if mae_value is not None or rmse_value is not None:
                    check_values['model_performance'] = {
                        'mae': mae_value,
                        'rmse': rmse_value
                    }
                else:
                    # Fallback: вычисляем метрики напрямую из predictions
                    if 'yhat' in test_data.columns and 'y' in test_data.columns:
                        try:
                            from sklearn.metrics import mean_absolute_error, mean_squared_error
                            # Удаляем строки, где хотя бы одно значение NaN, чтобы сохранить соответствие
                            df_clean = test_data[['y', 'yhat']].dropna()
                            if len(df_clean) > 0:
                                y_true = df_clean['y'].values
                                y_pred = df_clean['yhat'].values
                                mae_val = float(mean_absolute_error(y_true, y_pred))
                                rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                                check_values['model_performance'] = {
                                    'mae': mae_val,
                                    'rmse': rmse_val
                                }
                        except Exception:
                            pass

            # 2. Data drift (если есть фичи)
            # Для мини-таблицы drift_features считаем среди {price, promotion, y} (0..3).
            drift_features = [c for c in ['price', 'promotion', 'y'] if c in train_data.columns and c in test_data.columns]
            if drift_features:
                drift_calc = nml.UnivariateDriftCalculator(
                    column_names=drift_features,
                    timestamp_column_name='ds',
                    chunk_size=chunk_size,
                )

                drift_calc.fit(train_data)
                drift_results = drift_calc.calculate(test_data)

                drift_fig = drift_results.plot()
                drift_fig_path = os.path.join(output_dir, "nannyml_drift.html")
                drift_fig.write_html(drift_fig_path)
                artifacts.append(drift_fig_path)

                drift_alerts = drift_results.filter(period='analysis').to_df()
                # В NannyML алерты выставляются по чанкам (по времени), поэтому суммарное число алертов
                # может быть больше числа фич. Для человеко-читаемого DD считаем:
                # - drifted_features: сколько фич хотя бы раз дали alert
                # - total_alerts: сколько alert'ов всего (по всем чанкам и фичам)
                total_alerts = 0
                drifted_features = 0

                alert_cols = [c for c in drift_alerts.columns if 'alert' in str(c).lower()]
                drifted_feature_names = set()
                for c in alert_cols:
                    col_alerts = int(drift_alerts[c].sum())
                    total_alerts += col_alerts
                    issues_detected += col_alerts
                    if col_alerts > 0:
                        drifted_features += 1
                        c_lower = str(c).lower()
                        if "price" in c_lower:
                            drifted_feature_names.add("price")
                        elif "promotion" in c_lower:
                            drifted_feature_names.add("promotion")
                        elif "('y" in c_lower or " y" in c_lower or c_lower.endswith("y_alert"):
                            drifted_feature_names.add("y")
                
                # Сохраняем значение data_drift
                if checks_performed['data_drift']:
                    total_features = len(drift_features)
                    # Формат для таблицы (max_len=12 в main.py): "X/Y a=Z"
                    # где X/Y — число задрифтевших фич, a=Z — число алертов по чанкам.
                    check_values['data_drift'] = f"{drifted_features}/{total_features} a={total_alerts}"

                # Артефакт: какие именно признаки задрифтило (для EF_true/FP по признакам)
                try:
                    drift_path = os.path.join(output_dir, "nannyml_drift_features.json")
                    with open(drift_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"drifted_feature_names": sorted(drifted_feature_names)},
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    artifacts.append(drift_path)
                except Exception:
                    pass
            
            # Data Quality: проверяем пропуски и некорректные значения
            if checks_performed['data_quality']:
                quality_issues = 0
                # Проверяем пропуски
                for col in ['ds', 'y']:
                    if col in test_data.columns:
                        missing_count = int(test_data[col].isna().sum())
                        quality_issues += missing_count
                # Проверяем некорректные значения для y
                if 'y' in test_data.columns:
                    invalid_y = int((test_data['y'] < 0).sum())
                    quality_issues += invalid_y
                check_values['data_quality'] = int(quality_issues)
            
            # Outliers: используем IQR метод для обнаружения выбросов
            if checks_performed['outliers'] and 'y' in test_data.columns:
                try:
                    y_values = test_data['y'].dropna().values
                    if len(y_values) > 0:
                        Q1 = np.percentile(y_values, 25)
                        Q3 = np.percentile(y_values, 75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers_count = int(np.sum((y_values < lower_bound) | (y_values > upper_bound)))
                        check_values['outliers'] = int(outliers_count)
                except Exception:
                    pass

            # Outliers for price (same IQR method), stored in a separate key for mini-table scripts
            if checks_performed['outliers'] and 'price' in test_data.columns:
                try:
                    p_values = test_data['price'].dropna().values
                    if len(p_values) > 0:
                        Q1 = np.percentile(p_values, 25)
                        Q3 = np.percentile(p_values, 75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers_count_p = int(np.sum((p_values < lower_bound) | (p_values > upper_bound)))
                        check_values['outliers_price'] = int(outliers_count_p)
                except Exception:
                    pass

            # Для precision/FP: сохраняем row_id (индексы строк test) выбросов по y/price
            try:
                out_json = {"outliers_y_rows": [], "outliers_price_rows": []}
                if 'y' in test_data.columns:
                    y_series = test_data['y'].dropna()
                    yv = y_series.values
                    if len(yv) > 0:
                        Q1 = np.percentile(yv, 25)
                        Q3 = np.percentile(yv, 75)
                        IQR = Q3 - Q1
                        lo = Q1 - 1.5 * IQR
                        hi = Q3 + 1.5 * IQR
                        mask = (yv < lo) | (yv > hi)
                        idx = y_series.index.tolist()
                        out_json["outliers_y_rows"] = [idx[i] for i in np.where(mask)[0].tolist() if i < len(idx)]
                if 'price' in test_data.columns:
                    p_series = test_data['price'].dropna()
                    pv = p_series.values
                    if len(pv) > 0:
                        Q1 = np.percentile(pv, 25)
                        Q3 = np.percentile(pv, 75)
                        IQR = Q3 - Q1
                        lo = Q1 - 1.5 * IQR
                        hi = Q3 + 1.5 * IQR
                        mask = (pv < lo) | (pv > hi)
                        idx = p_series.index.tolist()
                        out_json["outliers_price_rows"] = [idx[i] for i in np.where(mask)[0].tolist() if i < len(idx)]
                out_path = os.path.join(output_dir, "nannyml_outliers.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out_json, f, ensure_ascii=False, indent=2)
                artifacts.append(out_path)
            except Exception:
                pass

        _, execution_time, memory_peak = measure_time_and_peak_rss_mb(_run)
        
        return {
            "execution_time_sec": execution_time,
            "memory_peak_mb": memory_peak,
            "issues_detected": int(issues_detected),
            "coverage_score": self.calculate_coverage_score(checks_performed),
            "checks_performed": checks_performed,
            "check_values": check_values,
            "artifacts": artifacts
        }
