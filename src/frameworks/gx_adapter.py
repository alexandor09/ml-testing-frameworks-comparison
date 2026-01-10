import pandas as pd
import great_expectations as gx
import os
import numpy as np
from .base import BaseAdapter
from .monitoring import measure_time_and_peak_rss_mb
from typing import Dict, Any

class GXAdapter(BaseAdapter):
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
            # Настройка контекста GX
            context = gx.get_context(mode="ephemeral")

            # Создание источников данных
            ds_name = "my_datasource"
            if hasattr(context, "data_sources"):
                datasource = context.data_sources.add_pandas(ds_name)
            else:
                datasource = context.sources.add_pandas(ds_name)
            
            train_batch = None
            test_batch = None
            train_batch_request = None
            test_batch_request = None

            # Добавление ассетов
            try:
                if hasattr(datasource, "read_dataframe"):
                    # GX 1.0+ Fluent API: read_dataframe возвращает Batch
                    train_batch = datasource.read_dataframe(dataframe=train_df, asset_name="train_data")
                    test_batch = datasource.read_dataframe(dataframe=test_df, asset_name="test_data")
                else:
                    raise AttributeError("No read_dataframe")
            except Exception:
                # Fallback
                try:
                    # Пытаемся получить существующие ассеты или создать новые
                    try:
                        train_asset = datasource.get_asset("train_data")
                    except (ValueError, LookupError):
                        train_asset = datasource.add_dataframe_asset(name="train_data")
                    
                    try:
                        test_asset = datasource.get_asset("test_data")
                    except (ValueError, LookupError):
                        test_asset = datasource.add_dataframe_asset(name="test_data")

                    # Пробуем разные способы создания batch_request
                    try:
                        train_batch_request = train_asset.build_batch_request(dataframe=train_df)
                        test_batch_request = test_asset.build_batch_request(dataframe=test_df)
                    except TypeError:
                        train_batch_request = train_asset.build_batch_request(batch_data=train_df)
                        test_batch_request = test_asset.build_batch_request(batch_data=test_df)
                except Exception:
                    pass

            # Создание Expectation Suite
            suite_name = "my_suite"
            try:
                # В некоторых версиях GX метод возвращает suite - сохраняем ссылку,
                # иначе ниже может получиться NameError на `suite`.
                suite = context.add_or_update_expectation_suite(expectation_suite_name=suite_name)
            except AttributeError:
                # GX 1.0+
                if hasattr(context, "suites"):
                    try:
                        context.suites.delete(suite_name)
                    except Exception:
                        pass
                    # Создаем suite через context.suites
                    # Важно: в GX 1.0 ExpectationSuite может требовать импорта
                    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
                else:
                    # Fallback for older versions that might use add_expectation_suite
                    try:
                        context.add_expectation_suite(expectation_suite_name=suite_name)
                    except Exception as e:
                        print(f"DEBUG: Could not create expectation suite: {e}")

            # Ожидания качества данных
            # В GX 1.0 ожидания добавляются в suite, а не через validator
            if hasattr(suite, "add_expectation"):
                try:
                    import great_expectations.expectations as gxe
                    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="ds"))
                    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="y"))
                    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="y", min_value=0))
                    # Генераторы дают price примерно в диапазоне 80..120 (а в drift-сценарии могут быть выше),
                    # поэтому 100 здесь давал ложные срабатывания даже на "ideal".
                    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="price", min_value=0, max_value=130))
                    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="promotion", value_set=[0, 1]))
                except (ImportError, AttributeError):
                    # Fallback if module structure is different
                    pass

            # Получение валидатора
            if train_batch is not None:
                validator = context.get_validator(batch_list=[train_batch], expectation_suite_name=suite_name)
            elif train_batch_request is not None:
                validator = context.get_validator(batch_request=train_batch_request, expectation_suite_name=suite_name)
            else:
                raise RuntimeError("Could not create validator: no batch or batch_request")

            # Если старый API (validator.expect_...), добавляем ожидания через validator
            if not hasattr(suite, "add_expectation"):
                # Old way
                validator.expect_column_values_to_be_not_null(column="ds")
                validator.expect_column_values_to_be_not_null(column="y")
                validator.expect_column_values_to_be_between(column="y", min_value=0)  # Спрос неотрицательный

                # Ожидания для регрессоров
                validator.expect_column_values_to_be_between(column="price", min_value=0, max_value=130)
                validator.expect_column_values_to_be_in_set(column="promotion", value_set=[0, 1])

                validator.save_expectation_suite(discard_failed_expectations=False)

            # Валидация test_df
            if test_batch is not None:
                # Если есть batch, используем валидацию напрямую (Checkpoint сложнее настроить с batch_list)
                checkpoint_result = context.get_validator(batch_list=[test_batch], expectation_suite_name=suite_name).validate()
            elif test_batch_request is not None:
                checkpoint = context.add_or_update_checkpoint(
                    name="my_checkpoint",
                    validations=[
                        {
                            "batch_request": test_batch_request,
                            "expectation_suite_name": suite_name,
                        },
                    ],
                )
                checkpoint_result = checkpoint.run()
            else:
                checkpoint_result = None

            # Обработка результатов
            if checkpoint_result:
                # checkpoint.run() возвращает CheckpointResult, validate() возвращает ExpectationValidationResult
                # Нужно привести к общему виду или обрабатывать по-разному
                
                # Если это CheckpointResult
                if hasattr(checkpoint_result, "list_validation_results"):
                    results = checkpoint_result.list_validation_results()
                else:
                    # Это ExpectationValidationResult
                    results = [checkpoint_result]

                for validation_result in results:
                    stats = validation_result.statistics
                    unsuccessful = stats.get("unsuccessful_expectations", 0)
                    issues_detected += unsuccessful
                    
                    # Сохраняем значения проверок
                    if checks_performed['data_quality']:
                        check_values['data_quality'] = int(unsuccessful)
                    if checks_performed['outliers']:
                        # Для outliers считаем как часть data_quality проверок
                        check_values['outliers'] = int(unsuccessful)

                    # Сохранение результатов в JSON
                    res_path = os.path.join(output_dir, "gx_validation.json")
                    with open(res_path, "w", encoding="utf-8") as f:
                        import json
                        # to_json_dict может быть методом или атрибутом
                        if hasattr(validation_result, "to_json_dict"):
                            res_dict = validation_result.to_json_dict()
                        else:
                            # Fallback serialization
                            res_dict = str(validation_result)
                            
                        json.dump(res_dict, f, indent=2, ensure_ascii=False)
                    artifacts.append(res_path)
            
            # Data Drift: сравниваем распределения признаков между train и test
            if checks_performed['data_drift']:
                drift_features = [c for c in ['price', 'promotion', 'y'] if c in train_df.columns and c in test_df.columns]
                if drift_features:
                    drift_count = 0
                    total_features = len(drift_features)
                    
                    for feature in drift_features:
                        # Используем KS тест для проверки дрифта
                        from scipy import stats
                        try:
                            train_values = train_df[feature].dropna().values
                            test_values = test_df[feature].dropna().values
                            if len(train_values) > 0 and len(test_values) > 0:
                                _, p_value = stats.ks_2samp(train_values, test_values)
                                if p_value < 0.05:  # Статистически значимый дрифт
                                    drift_count += 1
                        except Exception:
                            pass
                    
                    check_values['data_drift'] = f"{drift_count}/{total_features}"
            
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
                except Exception as e:
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
