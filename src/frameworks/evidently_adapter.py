import pandas as pd
import os
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from .base import BaseAdapter
from .monitoring import measure_time_and_peak_rss_mb
from typing import Dict, Any

class EvidentlyAdapter(BaseAdapter):
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
            # main.py добавляет 'yhat' в train_df и test_df, но перестрахуемся
            train_ref = train_df.copy()
            current = test_df.copy()
            if 'yhat' not in train_ref.columns and 'y' in train_ref.columns:
                train_ref['yhat'] = train_ref['y']

            # ColumnMapping менялся между версиями Evidently, поэтому делаем максимально совместимо
            column_mapping_obj = None
            try:
                from evidently import ColumnMapping  # type: ignore

                column_mapping_obj = ColumnMapping(
                    target='y',
                    prediction='yhat',
                    datetime='ds',
                    numerical_features=[c for c in ['price', 'promotion'] if c in train_ref.columns],
                )
            except Exception:
                # Старые версии могут принимать dict или не принимать mapping вообще
                column_mapping_obj = {
                    'target': 'y',
                    'prediction': 'yhat',
                    'datetime': 'ds',
                    'numerical_features': [c for c in ['price', 'promotion'] if c in train_ref.columns],
                }

            report = Report(metrics=[
                DataDriftPreset(),
                DataSummaryPreset(),
            ])

            try:
                # В новых версиях run() возвращает объект с результатами
                result = report.run(reference_data=train_ref, current_data=current, column_mapping=column_mapping_obj)
            except TypeError:
                result = report.run(reference_data=train_ref, current_data=current)
            
            # Если run() вернул None (старые версии), используем сам report
            if result is None:
                result = report

            report_path = os.path.join(output_dir, "evidently_report.html")
            
            # В разных версиях API методы могут отличаться
            try:
                if hasattr(result, 'save_html'):
                    result.save_html(report_path)
                elif hasattr(result, 'save'):
                    result.save(report_path)
                else:
                    # Fallback: сохраним как json если html не доступен
                    json_path = report_path.replace('.html', '.json')
                    if hasattr(result, 'save_json'):
                        result.save_json(json_path)
                    elif hasattr(result, 'json'):
                        with open(json_path, 'w') as f:
                            f.write(result.json())
            except Exception as e:
                print(f"Error saving Evidently report: {e}")
            artifacts.append(report_path)

            # Нормализованное правило issues_detected: считаем 1 проблему, если дрейфит значимая доля колонок
            report_data = {}
            for attr in ("as_dict", "dict", "to_dict"):
                if hasattr(result, attr):
                    try:
                        report_data = getattr(result, attr)()
                        break
                    except Exception:
                        pass

            drift_share = None
            drift_count = None
            total_columns = None
            
            # Извлекаем значения из отчета - пробуем разные варианты структуры
            metrics_list = report_data.get('metrics', []) if isinstance(report_data, dict) else []
            if not metrics_list and isinstance(report_data, dict):
                # Пробуем найти метрики в других местах структуры
                for key in ['metric_results', 'metric_result', 'results']:
                    if key in report_data:
                        metrics_list = report_data[key]
                        if isinstance(metrics_list, dict):
                            metrics_list = [metrics_list]
                        break
            
            for metric in metrics_list:
                if not isinstance(metric, dict):
                    continue
                    
                name = metric.get('metric_name') or metric.get('name') or metric.get('metric')
                value = metric.get('value') or metric.get('result') or {}
                
                # Ищем метрики дрифта
                if name:
                    name_lower = str(name).lower()
                    if 'drift' in name_lower or name in ('DriftedColumnsCount', 'DataDriftPreset', 'ColumnDriftMetric'):
                        if isinstance(value, dict):
                            drift_share = value.get('share') or value.get('drift_share')
                            drift_count = value.get('number_of_drifted_columns') or value.get('drifted_columns') or value.get('drifted_count')
                            total_columns = value.get('number_of_columns') or value.get('total_columns') or value.get('total_count')
                            if drift_share is not None or (drift_count is not None and total_columns is not None):
                                break
                        elif isinstance(value, (int, float)):
                            # Если значение - просто число
                            drift_count = int(value) if isinstance(value, (int, float)) else None
                            total_columns = len([c for c in ['price', 'promotion', 'y'] if c in train_ref.columns and c in current.columns])
                            if drift_count is not None:
                                break
                
                # Пытаемся найти метрики производительности
                if name and 'performance' in str(name).lower() and isinstance(value, dict):
                    if 'mae' in value or 'rmse' in value:
                        check_values['model_performance'] = {
                            'mae': value.get('mae'),
                            'rmse': value.get('rmse')
                        }

            if drift_share is not None and drift_share > 0.5:
                issues_detected += 1
            
            # Сохраняем значения проверок
            if checks_performed['data_drift']:
                # Приводим все к формату "X/Y" для единообразия
                if drift_count is not None and total_columns is not None:
                    check_values['data_drift'] = f"{int(drift_count)}/{int(total_columns)}"
                elif drift_share is not None:
                    # Конвертируем процент в формат "X/Y"
                    drift_features = [c for c in ['price', 'promotion', 'y'] if c in train_ref.columns and c in current.columns]
                    if drift_features:
                        total_cols = len(drift_features)
                        drifted_cols = int(round(drift_share * total_cols))
                        check_values['data_drift'] = f"{drifted_cols}/{total_cols}"
                    else:
                        # Если не можем определить признаки, используем процент как есть
                        check_values['data_drift'] = f"{drift_share:.1%}"
                else:
                    # Fallback: используем прямое сравнение распределений
                    try:
                        drift_features = [c for c in ['price', 'promotion', 'y'] if c in train_ref.columns and c in current.columns]
                        if drift_features:
                            from scipy import stats
                            drift_detected = 0
                            for feature in drift_features:
                                try:
                                    train_vals = train_ref[feature].dropna().values
                                    test_vals = current[feature].dropna().values
                                    if len(train_vals) > 0 and len(test_vals) > 0:
                                        _, p_val = stats.ks_2samp(train_vals, test_vals)
                                        if p_val < 0.05:
                                            drift_detected += 1
                                except Exception:
                                    pass
                            check_values['data_drift'] = f"{drift_detected}/{len(drift_features)}"
                    except Exception:
                        pass
            
            if checks_performed['data_quality']:
                # Для data_quality используем issues_detected как индикатор
                check_values['data_quality'] = int(issues_detected)
            
            # Outliers: используем IQR метод для обнаружения выбросов
            if checks_performed['outliers'] and 'y' in current.columns:
                try:
                    y_values = current['y'].dropna().values
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
            
            # Model Performance: если не нашли в отчете, вычисляем напрямую
            if checks_performed['model_performance'] and check_values['model_performance'] is None:
                if 'yhat' in current.columns and 'y' in current.columns:
                    try:
                        from sklearn.metrics import mean_absolute_error, mean_squared_error
                        # Удаляем строки, где хотя бы одно значение NaN, чтобы сохранить соответствие
                        df_clean = current[['y', 'yhat']].dropna()
                        if len(df_clean) > 0:
                            y_true = df_clean['y'].values
                            y_pred = df_clean['yhat'].values
                            mae = float(mean_absolute_error(y_true, y_pred))
                            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                            check_values['model_performance'] = {
                                'mae': mae,
                                'rmse': rmse
                            }
                    except Exception:
                        pass

        _, execution_time, memory_peak = measure_time_and_peak_rss_mb(_run)
        
        return {
            "execution_time_sec": execution_time,
            "memory_peak_mb": memory_peak,
            "issues_detected": issues_detected,
            "coverage_score": self.calculate_coverage_score(checks_performed),
            "checks_performed": checks_performed,
            "check_values": check_values,
            "artifacts": artifacts
        }
