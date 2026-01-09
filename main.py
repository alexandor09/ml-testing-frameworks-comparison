import argparse
import os
import pandas as pd
import json
import logging
from datetime import datetime
import random
import numpy as np
from src.model import ForecastModel, calculate_metrics
from src.frameworks.gx_adapter import GXAdapter
from src.frameworks.evidently_adapter import EvidentlyAdapter
from src.frameworks.alibi_adapter import AlibiAdapter
from src.frameworks.nannyml_adapter import NannyMLAdapter
from src.reporting import save_json_report, generate_dashboard, determine_best_framework, save_split_demonstration

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def main():
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

    # Временное разбиение: сортируем по времени и берем последние N точек как test
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Разделение данных (последние 20% - тест)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Демонстрация train/test split по времени
    print("\n" + "=" * 100)
    print("Демонстрация временного разбиения данных (Train/Test Split)")
    print("=" * 100)
    print(f"\nОбщий размер данных: {len(df)} строк")
    print(f"Train: {len(train_df)} строк ({len(train_df)/len(df)*100:.1f}%) - с {train_df['ds'].min()} по {train_df['ds'].max()}")
    print(f"Test:  {len(test_df)} строк ({len(test_df)/len(df)*100:.1f}%) - с {test_df['ds'].min()} по {test_df['ds'].max()}")
    
    # Показываем последние строки train и первые строки test
    n_show = 5  # Количество строк для показа
    print(f"\n--- Последние {n_show} строк Train (перед разделением) ---")
    train_tail = train_df[['ds', 'y', 'price', 'promotion']].tail(n_show)
    print(train_tail.to_string(index=True))
    
    print(f"\n--- Первые {n_show} строк Test (после разделения) ---")
    test_head = test_df[['ds', 'y', 'price', 'promotion']].head(n_show)
    print(test_head.to_string(index=True))
    
    # Проверка непрерывности дат
    last_train_date = train_df['ds'].max()
    first_test_date = test_df['ds'].min()
    date_gap = (first_test_date - last_train_date).days
    
    print("\n--- Проверка непрерывности ---")
    print(f"Последняя дата Train: {last_train_date}")
    print(f"Первая дата Test:     {first_test_date}")
    print(f"Разрыв между Train и Test: {date_gap} дней")
    if date_gap == 1:
        print("✓ Даты идут последовательно без пропусков")
    elif date_gap > 1:
        print(f"⚠ Внимание: есть пропуск в {date_gap - 1} дней между train и test")
    else:
        print("✓ Даты идут последовательно")
    
    print("=" * 100)
    print()
    
    # Сохраняем информацию о split в файлы
    split_info = {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_percent": float(len(train_df)/len(df)*100),
        "test_percent": float(len(test_df)/len(df)*100),
        "train_date_range": {
            "start": str(train_df['ds'].min()),
            "end": str(train_df['ds'].max())
        },
        "test_date_range": {
            "start": str(test_df['ds'].min()),
            "end": str(test_df['ds'].max())
        },
        "date_gap_days": int(date_gap),
        "train_tail": train_tail.to_dict('records'),
        "test_head": test_head.to_dict('records')
    }
    
    # Сохраняем в JSON
    save_json_report(split_info, os.path.join(output_dir, "train_test_split.json"))
    
    # Сохраняем в HTML файл
    save_split_demonstration(train_df, test_df, split_info, os.path.join(output_dir, "train_test_split.html"))
    
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
    
    # Фреймворки для запуска
    adapters = {
        'gx': GXAdapter(),
        'evidently': EvidentlyAdapter(),
        'alibi': AlibiAdapter(),
        'nannyml': NannyMLAdapter()
    }
    
    if args.framework:
        selected_adapters = {args.framework: adapters[args.framework]}
    else:
        selected_adapters = adapters
        
    comparison_summary = {}
    
    # Запуск фреймворков
    from tqdm import tqdm
    
    # Используем tqdm для отображения прогресса по фреймворкам
    iterator = tqdm(selected_adapters.items(), desc="Выполнение проверок", unit="framework")
    
    for name, adapter in iterator:
        # Обновляем описание прогресс-бара
        iterator.set_description(f"Запуск {name}")
        logging.info(f"Запуск {name}...")
        fw_output_dir = os.path.join(output_dir, name)
        os.makedirs(fw_output_dir, exist_ok=True)
        
        try:
            result = adapter.run_checks(train_df, test_df, predictions, fw_output_dir)
            result["artifacts"] = _normalize_artifacts(result.get("artifacts", []), output_dir)
            comparison_summary[name] = result
            
            # Вывод в консоль для одиночного запуска
            if args.framework:
                checks = result.get('checks_performed', {})
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
                
        except Exception as e:
            logging.error(f"Ошибка при запуске {name}: {e}")
            comparison_summary[name] = {
                "execution_time_sec": 0,
                "memory_peak_mb": 0,
                "issues_detected": -1,
                "coverage_score": 0,
                "checks_performed": {
                    'data_quality': False,
                    'data_drift': False,
                    'outliers': False,
                    'model_performance': False
                },
                "check_values": {
                    'data_quality': None,
                    'data_drift': None,
                    'outliers': None,
                    'model_performance': None
                },
                "artifacts": [],
                "error": str(e)
            }

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
        
        # Таблица в консоли
        print("\n--- Сводка сравнения ---")
        print(f"{'Фреймворк':<15} | {'Время (с)':<10} | {'Память (МБ)':<10} | {'Проблемы':<8} | {'Покрытие':<8} | {'DQ':<12} | {'DD':<12} | {'Out':<8} | {'MP':<12} | {'MAE':<8} | {'RMSE':<8} | {'MAPE':<8} | {'Артефакты':<8}")
        print("-" * 150)
        for name, res in comparison_summary.items():
            checks = res.get('checks_performed', {})
            check_vals = res.get('check_values', {})
            
            # Форматируем значения проверок
            def format_check_value(check_type, max_len=12):
                if not checks.get(check_type, False):
                    return '-'
                val = check_vals.get(check_type)
                if val is None:
                    return 'N/A'
                if isinstance(val, dict):
                    # Для model_performance может быть словарь с метриками
                    if 'mae' in val and val['mae'] is not None:
                        return f"MAE:{val['mae']:.2f}"[:max_len]
                    elif 'rmse' in val and val['rmse'] is not None:
                        return f"RMSE:{val['rmse']:.2f}"[:max_len]
                    return str(val)[:max_len]
                if isinstance(val, (int, float)):
                    # Всегда показываем как целое число для счетчиков
                    if isinstance(val, float) and val.is_integer():
                        formatted = f"{int(val)}"
                    elif isinstance(val, float):
                        # Если это действительно дробное число (например, процент), показываем с 2 знаками
                        formatted = f"{val:.2f}"
                    else:
                        formatted = f"{int(val)}"
                    return formatted[:max_len]
                formatted_str = str(val)[:max_len]
                return formatted_str
            
            dq = format_check_value('data_quality', max_len=12)
            dd = format_check_value('data_drift', max_len=12)
            out = format_check_value('outliers', max_len=8)
            mp = format_check_value('model_performance', max_len=12)
            
            print(f"{name:<15} | {res['execution_time_sec']:<10.4f} | {res['memory_peak_mb']:<10.2f} | {res['issues_detected']:<8} | {res['coverage_score']:<8.2f} | {dq:<12} | {dd:<12} | {out:<8} | {mp:<12} | {metrics['MAE']:<8.2f} | {metrics['RMSE']:<8.2f} | {metrics['MAPE']:<8.4f} | {len(res.get('artifacts', [])):<8}")
        
        # print(f"\nЛучший фреймворк: {best_fw}")

if __name__ == "__main__":
    main()
