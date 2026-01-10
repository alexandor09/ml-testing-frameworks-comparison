import argparse
import os
import pandas as pd
import json
import logging
import shutil
import warnings
from datetime import datetime
import random
import numpy as np
from src.model import ForecastModel, calculate_metrics
from src.reporting import save_json_report, generate_dashboard, determine_best_framework, save_split_demonstration

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Уменьшаем шум от сторонних библиотек (это не ошибки, но захламляет консоль).
logging.getLogger("great_expectations").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

def load_data(input_path: str, fmt: str) -> pd.DataFrame:
    if fmt == 'csv':
        df = pd.read_csv(input_path)
    elif fmt == 'json':
        df = pd.read_json(input_path)
    else:
        raise ValueError("Неподдерживаемый формат")
    
    # Убедимся, что 'ds' - это datetime
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
    else:
        raise ValueError("В данных отсутствует обязательная колонка 'ds'.")

    if 'y' not in df.columns:
        raise ValueError("В данных отсутствует обязательная колонка 'y'.")
        
    return df


def _format_check_value(checks, check_vals, check_type: str, max_len: int) -> str:
    if not checks.get(check_type, False):
        return "-"
    val = check_vals.get(check_type)
    if val is None:
        return "N/A"
    if isinstance(val, dict):
        if "mae" in val and val["mae"] is not None:
            return f"MAE:{val['mae']:.2f}"[:max_len]
        if "rmse" in val and val["rmse"] is not None:
            return f"RMSE:{val['rmse']:.2f}"[:max_len]
        return str(val)[:max_len]
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val.is_integer():
            formatted = f"{int(val)}"
        elif isinstance(val, float):
            formatted = f"{val:.2f}"
        else:
            formatted = f"{int(val)}"
        return formatted[:max_len]
    return str(val)[:max_len]


def _time_split_and_demo(df: pd.DataFrame, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same as main.py: sort by ds, test is last 20%. Also prints and saves split artifacts."""
    df = df.sort_values("ds").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Demo print
    print("\n" + "=" * 100)
    print("Демонстрация временного разбиения данных (Train/Test Split)")
    print("=" * 100)
    print(f"\nОбщий размер данных: {len(df)} строк")
    print(f"Train: {len(train_df)} строк ({len(train_df)/len(df)*100:.1f}%) - с {train_df['ds'].min()} по {train_df['ds'].max()}")
    print(f"Test:  {len(test_df)} строк ({len(test_df)/len(df)*100:.1f}%) - с {test_df['ds'].min()} по {test_df['ds'].max()}")

    n_show = 5
    print(f"\n--- Последние {n_show} строк Train (перед разделением) ---")
    train_tail = train_df[["ds", "y", "price", "promotion"]].tail(n_show)
    print(train_tail.to_string(index=True))

    print(f"\n--- Первые {n_show} строк Test (после разделения) ---")
    test_head = test_df[["ds", "y", "price", "promotion"]].head(n_show)
    print(test_head.to_string(index=True))

    # Continuity check
    last_train_date = train_df["ds"].max()
    first_test_date = test_df["ds"].min()
    date_gap = (first_test_date - last_train_date).days

    print("\n--- Проверка непрерывности ---")
    print(f"Последняя дата Train: {last_train_date}")
    print(f"Первая дата Test:     {first_test_date}")
    print(f"Разрыв между Train и Test: {date_gap} дней")
    if date_gap == 1:
        print("OK: даты идут последовательно без пропусков")
    elif date_gap > 1:
        print(f"WARNING: есть пропуск в {date_gap - 1} дней между train и test")
    else:
        print("OK: даты идут последовательно")

    print("=" * 100)
    print()

    split_info = {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_percent": float(len(train_df) / len(df) * 100),
        "test_percent": float(len(test_df) / len(df) * 100),
        "train_date_range": {"start": str(train_df["ds"].min()), "end": str(train_df["ds"].max())},
        "test_date_range": {"start": str(test_df["ds"].min()), "end": str(test_df["ds"].max())},
        "date_gap_days": int(date_gap),
        "train_tail": train_tail.to_dict("records"),
        "test_head": test_head.to_dict("records"),
    }

    save_json_report(split_info, os.path.join(output_dir, "train_test_split.json"))
    save_split_demonstration(train_df, test_df, split_info, os.path.join(output_dir, "train_test_split.html"))
    print(f"Демонстрация train/test split сохранена: {os.path.join(output_dir, 'train_test_split.html')}")
    return train_df, test_df


def _normalize_artifacts(artifacts, run_dir: str):
    """Преобразует пути артефактов в относительные (относительно папки прогона)."""
    if not artifacts:
        return []
    out = []
    for a in artifacts:
        a_str = str(a)
        try:
            out.append(os.path.relpath(a_str, start=run_dir))
        except Exception:
            out.append(a_str)
    return out


def _print_comparison_table(comparison_summary: dict, metrics: dict) -> None:
    """Print comparison table to console (auto-compact for narrow terminals)."""
    term_w = shutil.get_terminal_size(fallback=(120, 20)).columns
    if term_w < 170:
        _print_comparison_table_compact(comparison_summary, term_w)
    else:
        _print_comparison_table_full(comparison_summary, metrics, term_w)


def _print_comparison_table_compact(comparison_summary: dict, term_w: int) -> None:
    print("\n--- Сводка сравнения ---")
    cols = [
        ("Фреймворк", 12),
        ("Время", 7),
        ("Память", 8),
        ("Пробл", 5),
        ("DQ", 6),
        ("DD", 10),
        ("Out", 6),
        ("MP", 10),
        ("Art", 3),
    ]

    def _row(values):
        return " | ".join(f"{str(v):<{w}}"[:w] for (_, w), v in zip(cols, values))

    print(_row([c[0] for c in cols]))
    print("-" * min(term_w, len(_row(["-" * c[1] for c in cols]))))

    for name, res in comparison_summary.items():
        checks = res.get("checks_performed", {})
        check_vals = res.get("check_values", {})
        dq = _format_check_value(checks, check_vals, "data_quality", max_len=6)
        dd = _format_check_value(checks, check_vals, "data_drift", max_len=10)
        out = _format_check_value(checks, check_vals, "outliers", max_len=6)
        mp = _format_check_value(checks, check_vals, "model_performance", max_len=10)
        print(
            _row(
                [
                    name,
                    f"{res['execution_time_sec']:.3f}",
                    f"{res['memory_peak_mb']:.0f}",
                    int(res["issues_detected"]),
                    dq,
                    dd,
                    out,
                    mp,
                    len(res.get("artifacts", [])),
                ]
            )
        )


def _print_comparison_table_full(comparison_summary: dict, metrics: dict, term_w: int) -> None:
    print("\n--- Сводка сравнения ---")
    cols = [
        ("Фреймворк", 15),
        ("Время (с)", 10),
        ("Память (МБ)", 10),
        ("Проблемы", 8),
        ("Покрытие", 8),
        ("DQ", 12),
        ("DD", 12),
        ("Out", 8),
        ("MP", 12),
        ("MAE", 8),
        ("RMSE", 8),
        ("MAPE", 8),
        ("Артефакты", 8),
    ]

    def _row(values):
        return " | ".join(f"{str(v):<{w}}"[:w] for (_, w), v in zip(cols, values))

    print(_row([c[0] for c in cols]))
    print("-" * min(term_w, len(_row(["-" * c[1] for c in cols]))))

    for name, res in comparison_summary.items():
        checks = res.get("checks_performed", {})
        check_vals = res.get("check_values", {})
        dq = _format_check_value(checks, check_vals, "data_quality", max_len=12)
        dd = _format_check_value(checks, check_vals, "data_drift", max_len=12)
        out = _format_check_value(checks, check_vals, "outliers", max_len=8)
        mp = _format_check_value(checks, check_vals, "model_performance", max_len=12)
        print(
            _row(
                [
                    name,
                    f"{res['execution_time_sec']:.4f}",
                    f"{res['memory_peak_mb']:.2f}",
                    int(res["issues_detected"]),
                    f"{res['coverage_score']:.2f}",
                    dq,
                    dd,
                    out,
                    mp,
                    f"{metrics['MAE']:.2f}",
                    f"{metrics['RMSE']:.2f}",
                    f"{metrics['MAPE']:.4f}",
                    len(res.get("artifacts", [])),
                ]
            )
        )


def _build_adapters(framework: str | None):
    """
    Создает словарь адаптеров.

    Важно: импортируем адаптеры лениво, чтобы можно было запускать один фреймворк
    без установки зависимостей остальных (удобно для Google Colab демо).
    """
    adapters = {}

    def _import_one(name: str):
        if name == "gx":
            from src.frameworks.gx_adapter import GXAdapter  # noqa: WPS433 (runtime import)
            return GXAdapter()
        if name == "evidently":
            from src.frameworks.evidently_adapter import EvidentlyAdapter  # noqa: WPS433
            return EvidentlyAdapter()
        if name == "alibi":
            from src.frameworks.alibi_adapter import AlibiAdapter  # noqa: WPS433
            return AlibiAdapter()
        if name == "nannyml":
            from src.frameworks.nannyml_adapter import NannyMLAdapter  # noqa: WPS433
            return NannyMLAdapter()
        raise ValueError(f"Unknown framework: {name}")

    all_names = ["gx", "evidently", "alibi", "nannyml"]
    names = [framework] if framework else all_names

    for name in names:
        try:
            adapters[name] = _import_one(name)
        except Exception as e:
            if framework:
                # Если пользователь явно выбрал фреймворк — падаем сразу и объясняем.
                raise RuntimeError(
                    f"Не удалось импортировать адаптер '{name}'. "
                    f"Проверьте зависимости (pip install -r requirements.txt). "
                    f"Оригинальная ошибка: {e}"
                ) from e
            # Если это полный прогон — пропускаем недоступный фреймворк, но продолжаем.
            logging.warning(f"Пропускаю фреймворк '{name}': не удалось импортировать ({e}).")

    return adapters

def main():
    # Убираем известный шумный FutureWarning (не влияет на результат)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"google\.api_core\._python_version_support",
    )

    parser = argparse.ArgumentParser(description="CLI для сравнения ML-фреймворков")
    parser.add_argument("--input", required=True, help="Путь к входному датасету")
    parser.add_argument("--format", choices=['csv', 'json'], default='csv', help="Формат входных данных")
    parser.add_argument("--output", required=True, help="Директория для отчетов")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")
    parser.add_argument("--framework", choices=['gx', 'evidently', 'alibi', 'nannyml'], help="Запуск конкретного фреймворка")
    
    args = parser.parse_args()

    # Фиксируем seed (для воспроизводимости стохастических частей)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Создание директории вывода
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Запуск {run_id}")
    
    # Загрузка данных
    df = load_data(args.input, args.format)

    # Временное разбиение: сортируем по времени и берем последние 20% как test (как в main.py)
    train_df, test_df = _time_split_and_demo(df, output_dir)

    # Обучение модели
    logging.info("Обучение модели Prophet...")
    
    # Prophet не умеет работать с NaN в регрессорах, поэтому заполняем их для обучения
    train_df_filled = train_df.ffill().bfill()
    
    model = ForecastModel()
    model.train(train_df_filled)
    
    # Прогнозирование
    logging.info("Генерация прогнозов...")
    
    # Также заполняем пропуски для прогноза
    train_preds = model.predict(train_df_filled)
    test_df_filled = test_df.ffill().bfill()
    test_preds = model.predict(test_df_filled)
    
    # Добавляем 'yhat' в датафреймы для адаптеров
    train_df['yhat'] = train_preds['yhat'].values
    test_df['yhat'] = test_preds['yhat'].values
    
    predictions = test_preds
    
    # Расчет метрик модели
    metrics = calculate_metrics(test_df['y'], predictions['yhat'])
    save_json_report(metrics, os.path.join(output_dir, "model_metrics.json"))
    logging.info(f"Метрики модели: {metrics}")
    
    # Фреймворки для запуска (ленивый импорт адаптеров)
    selected_adapters = _build_adapters(args.framework)

    comparison_summary = _run_adapters(
        selected_adapters=selected_adapters,
        train_df=train_df,
        test_df=test_df,
        predictions=predictions,
        output_dir=output_dir,
        single_framework=args.framework,
        metrics=metrics,
    )

    # Сохранение сводки сравнения
    save_json_report(comparison_summary, os.path.join(output_dir, "comparison_summary.json"))
    
    # Генерация дашборда и итоговой сводки (если запущены все)
    if not args.framework:
        logging.info("Генерация дашборда и итоговой сводки...")
        generate_dashboard(comparison_summary, os.path.join(output_dir, "dashboard.html"), test_df=test_df, predictions=predictions)
        
        best_fw = determine_best_framework(comparison_summary)
        final_summary = {
            "best_framework": best_fw,
            "details": comparison_summary.get(best_fw, {})
        }
        save_json_report(final_summary, os.path.join(output_dir, "final_summary.json"))
        
        _print_comparison_table(comparison_summary, metrics)
        
        # print(f"\nЛучший фреймворк: {best_fw}")

def _run_adapters(
    *,
    selected_adapters: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: str,
    single_framework: str | None,
    metrics: dict,
) -> dict:
    """Run all selected adapters and return comparison_summary dict."""
    comparison_summary: dict = {}

    from tqdm import tqdm

    def _print_single_framework_result(name: str, result: dict) -> None:
        checks = result.get("checks_performed", {})
        print(f"\n--- Результаты {name} ---")
        print(f"Время выполнения: {result['execution_time_sec']:.4f}с")
        print(f"Пик памяти: {result['memory_peak_mb']:.2f}МБ")
        print(f"Обнаружено проблем: {result['issues_detected']}")
        print(f"Оценка покрытия: {result['coverage_score']:.2f}")
        print("Выполненные проверки:")
        print(f"  - Data Quality: {'Да' if checks.get('data_quality', False) else 'Нет'}")
        print(f"  - Data Drift: {'Да' if checks.get('data_drift', False) else 'Нет'}")
        print(f"  - Outliers: {'Да' if checks.get('outliers', False) else 'Нет'}")
        print(f"  - Model Performance: {'Да' if checks.get('model_performance', False) else 'Нет'}")
        print(f"Артефакты: {result.get('artifacts', [])}")
        print(f"Метрики модели: {metrics}")

    def _error_result(e: Exception) -> dict:
        return {
            "execution_time_sec": 0,
            "memory_peak_mb": 0,
            "issues_detected": -1,
            "coverage_score": 0,
            "checks_performed": {
                "data_quality": False,
                "data_drift": False,
                "outliers": False,
                "model_performance": False,
            },
            "check_values": {
                "data_quality": None,
                "data_drift": None,
                "outliers": None,
                "model_performance": None,
            },
            "artifacts": [],
            "error": str(e),
        }

    iterator = tqdm(selected_adapters.items(), desc="Выполнение проверок", unit="framework")
    for name, adapter in iterator:
        iterator.set_description(f"Запуск {name}")
        logging.info(f"Запуск {name}...")
        fw_output_dir = os.path.join(output_dir, name)
        os.makedirs(fw_output_dir, exist_ok=True)

        try:
            result = adapter.run_checks(train_df, test_df, predictions, fw_output_dir)
            result["artifacts"] = _normalize_artifacts(result.get("artifacts", []), output_dir)
            comparison_summary[name] = result

            if single_framework:
                _print_single_framework_result(name, result)

        except Exception as e:
            logging.error(f"Ошибка при запуске {name}: {e}")
            comparison_summary[name] = _error_result(e)

    return comparison_summary


if __name__ == "__main__":
    main()
