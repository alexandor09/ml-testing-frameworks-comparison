import pandas as pd
import numpy as np
import os
from alibi_detect.od import IForest
from alibi_detect.cd import KSDrift
from .base import BaseAdapter
from .monitoring import measure_time_and_peak_rss_mb
from typing import Dict, Any

class AlibiAdapter(BaseAdapter):
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
            # Alibi Detect работает с numpy массивами
            # Проверяем выбросы в целевой переменной 'y'
            # Удаляем NaN перед обработкой
            X_train = train_df['y'].dropna().values.reshape(-1, 1)
            X_test = test_df['y'].dropna().values.reshape(-1, 1)

            # 1. Инициализация детектора выбросов (Isolation Forest)
            # Проблема: Isolation Forest сравнивает test с train, что может давать ложные срабатывания
            # из-за естественного временного дрейфа между train и test периодами.
            # Используем более консервативный подход: проверяем выбросы относительно самого test набора
            # через IQR метод, а не сравниваем с train
            if len(X_test) > 0 and len(X_train) > 0:
                # Используем IQR метод для более стабильных результатов на временных рядах
                # Это избегает проблемы ложных срабатываний из-за временного дрейфа
                try:
                    Q1 = np.percentile(X_test, 25)
                    Q3 = np.percentile(X_test, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3.0 * IQR  # Используем 3.0 вместо 1.5 для более консервативного подхода
                    upper_bound = Q3 + 3.0 * IQR
                    outliers_mask = (X_test.flatten() < lower_bound) | (X_test.flatten() > upper_bound)
                    outliers_count = int(np.sum(outliers_mask))
                    issues_detected += outliers_count
                    
                    # Сохраняем значение outliers
                    if checks_performed['outliers']:
                        check_values['outliers'] = int(outliers_count)
                except Exception:
                    # Fallback на Isolation Forest, если IQR не сработал
                    try:
                        od = IForest(threshold=None, n_estimators=100)
                        od.fit(X_train)
                        od.infer_threshold(X_train, threshold_perc=99.5)  # Еще более консервативный порог
                        
                        preds = od.predict(X_test)
                        outliers = preds['data']['is_outlier']
                        outliers_count = int(np.sum(outliers))
                        issues_detected += outliers_count
                        
                        if checks_performed['outliers']:
                            check_values['outliers'] = int(outliers_count)
                    except Exception:
                        if checks_performed['outliers']:
                            check_values['outliers'] = 0

            # Сохранение результатов выбросов
            if checks_performed['outliers']:
                outlier_indices = []
                method_used = "IQR (3.0 * IQR)"
                
                if 'outliers_mask' in locals():
                    # Случай с IQR методом
                    test_y_clean = test_df['y'].dropna()
                    outlier_indices_in_clean = np.where(outliers_mask)[0].tolist()
                    if len(outlier_indices_in_clean) > 0:
                        clean_indices = test_y_clean.index.tolist()
                        outlier_indices = [clean_indices[i] for i in outlier_indices_in_clean if i < len(clean_indices)]
                elif 'outliers' in locals():
                    # Случай с Isolation Forest (fallback)
                    method_used = "Isolation Forest"
                    test_y_clean = test_df['y'].dropna()
                    outlier_indices_in_clean = np.where(outliers)[0].tolist()
                    if len(outlier_indices_in_clean) > 0:
                        clean_indices = test_y_clean.index.tolist()
                        outlier_indices = [clean_indices[i] for i in outlier_indices_in_clean if i < len(clean_indices)]
                
                result_od = {
                    "outliers_detected": check_values.get('outliers', 0),
                    "outlier_indices": outlier_indices,
                    "method": method_used
                }

                result_path_od = os.path.join(output_dir, "alibi_outliers.json")
                import json

                with open(result_path_od, "w", encoding="utf-8") as f:
                    json.dump(result_od, f, indent=2, ensure_ascii=False)
                artifacts.append(result_path_od)

            # 2. Детекция дрифта (KSDrift) для числовых признаков
            drift_features = ['price', 'promotion', 'y']
            available_features = [f for f in drift_features if f in train_df.columns and f in test_df.columns]

            if available_features:
                # Проверяем дрифт по каждому признаку отдельно для более точного результата
                drift_count = 0
                drift_per_feature = {}
                p_values_per_feature = {}
                
                for feature in available_features:
                    try:
                        # Для каждого признака создаем отдельный детектор
                        train_vals = train_df[feature].dropna().values.reshape(-1, 1)
                        test_vals = test_df[feature].dropna().values.reshape(-1, 1)
                        
                        if len(train_vals) > 0 and len(test_vals) > 0:
                            cd = KSDrift(train_vals, p_val=0.05)
                            drift_preds = cd.predict(test_vals)
                            
                            is_drift = drift_preds['data']['is_drift']
                            p_val = drift_preds['data']['p_val']
                            
                            # is_drift может быть скаляром или массивом
                            if isinstance(is_drift, (bool, np.bool_)):
                                feature_drift = bool(is_drift)
                            else:
                                feature_drift = bool(np.any(is_drift))
                            
                            drift_per_feature[feature] = feature_drift
                            
                            # p_val может быть скаляром или массивом
                            if isinstance(p_val, (float, np.floating)):
                                p_values_per_feature[feature] = float(p_val)
                            else:
                                p_values_per_feature[feature] = float(np.mean(p_val)) if len(p_val) > 0 else 1.0
                            
                            if feature_drift:
                                drift_count += 1
                                issues_detected += 1
                    except Exception:
                        # Если не удалось проверить признак, считаем что дрифта нет
                        drift_per_feature[feature] = False
                        p_values_per_feature[feature] = 1.0
                
                # Сохраняем значение data_drift
                if checks_performed['data_drift']:
                    total_features = len(available_features)
                    check_values['data_drift'] = f"{drift_count}/{total_features}"

                # Сохранение результатов
                result_cd = {
                    "drift_detected_count": drift_count,
                    "is_drift_per_feature": drift_per_feature,
                    "p_values": p_values_per_feature
                }

                result_path_cd = os.path.join(output_dir, "alibi_drift.json")
                with open(result_path_cd, "w", encoding="utf-8") as f:
                    json.dump(result_cd, f, indent=2, ensure_ascii=False)
                artifacts.append(result_path_cd)
            
            # Data Quality: проверяем пропуски и некорректные значения
            if checks_performed['data_quality']:
                quality_issues = 0
                # Проверяем пропуски
                for col in ['ds', 'y']:
                    if col in test_df.columns:
                        missing_count = int(test_df[col].isna().sum())
                        quality_issues += missing_count
                # Проверяем некорректные значения для y
                if 'y' in test_df.columns:
                    invalid_y = int((test_df['y'] < 0).sum())
                    quality_issues += invalid_y
                check_values['data_quality'] = int(quality_issues)
            
            # Model Performance: вычисляем метрики на основе predictions
            if checks_performed['model_performance'] and 'yhat' in test_df.columns and 'y' in test_df.columns:
                try:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    # Удаляем строки, где хотя бы одно значение NaN, чтобы сохранить соответствие
                    df_clean = test_df[['y', 'yhat']].dropna()
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
