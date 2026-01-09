Этот проект сравнивает 4 фреймворка для тестирования ML (Great Expectations, Evidently, Alibi Detect, NannyML) на задаче прогнозирования спроса с использованием Prophet.

## Требования
- Windows 10 (x64) или macOS/Linux
- Python 3.10+

## Установка

1. **Создание виртуального окружения**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

2. **Установка зависимостей**
   ```bash
   pip install -r requirements.txt
   ```

## Использование

### 1. Генерация данных
Проект включает генератор данных.
```bash
python src/data_gen.py
```
Это создаст `data/small.csv` и `data/big.csv`.

python generate_scenarios.py

Это создаст  идеальные данные (ideal)...
✓ Сохранено data\ideal.csv (20000 строк)
✓ Сохранено data\ideal.json (20000 строк)

данных с пропусками и дубликатами (pass)...
✓ Сохранено data\pass.csv (20600 строк)
✓ Сохранено data\pass.json (20600 строк)

 данных с дрейфом и выбросами (dr)...
✓ Сохранено data\dr.csv (20000 строк)
✓ Сохранено data\dr.json (20000 строк)

**Особенности генератора данных (Реалистичная симуляция рынка):**
Генератор (`src/data_gen.py`) создает высокореалистичные временные ряды, используя стохастическую симуляцию рыночной среды:
*   **Переключение режимов (Regime Switching):** Рынок переключается между состояниями "Стабильность", "Бум" и "Рецессия" с помощью цепей Маркова. Это меняет тренды и волатильность.
*   **Умное ценообразование:** Спрос зависит от *относительной цены* (наша цена vs цена скрытого конкурента) с учетом эластичности.
*   **Маркетинговый Adstock:** Эффект от промо-акций затухает плавно (накопительный эффект), а не исчезает мгновенно.
*   **Сложная сезонность:** Комбинация недельной и годовой сезонности + специальные паттерны для праздников (Новый год, Black Friday).
*   **Дрейф и Аномалии:** В тестовой выборке симулируется агрессивное поведение конкурента (дрейф концепции) и добавляются случайные выбросы.

### 2. Запуск сравнения
Показать справку CLI:
```bash
python main.py --help
```

Запуск **всех 4 фреймворков** на `data/small.csv`:
```bash
python main.py --input data/small.csv --format csv --output reports/run_small
```

Запуск **всех 4 фреймворков** на `data/big.csv`:
```bash
python main.py --input data/big.csv --format csv --output reports/run_big
```

Запуск **всех 4 фреймворков** на новых сценариях данных:

**Идеальные данные (ideal.csv):**
```bash
python main.py --input data/ideal.csv --format csv --output reports/run_ideal
```

**Данные с пропусками и дубликатами (pass.csv):**
```bash
python main.py --input data/pass.csv --format csv --output reports/run_pass
```

**Данные с дрейфом и выбросами (dr.csv):**
```bash
python main.py --input data/dr.csv --format csv --output reports/run_dr
```

Аналогично для JSON формата:
```bash
python main.py --input data/ideal.json --format json --output reports/run_ideal_json
python main.py --input data/pass.json --format json --output reports/run_pass_json
python main.py --input data/dr.json --format json --output reports/run_dr_json
```

Запуск **всех 4 фреймворков** на `data/small.json`:
```bash
python main.py --input data/small.json --format json --output reports/run_small_json
```

Запуск **всех 4 фреймворков** на `data/big.json`:
```bash
python main.py --input data/big.json --format json --output reports/run_big_json
```

Запуск конкретного фреймворка (например, Evidently):
```bash
python main.py --input data/small.csv --format csv --output reports/run_evidently --framework evidently
```

Запуск конкретного фреймворка (например, Great Expectations):
```bash
python main.py --input data/small.csv --format csv --output reports/run_gx --framework gx
```

Запуск конкретного фреймворка (например, Alibi Detect):
```bash
python main.py --input data/small.csv --format csv --output reports/run_alibi --framework alibi
```

Запуск конкретного фреймворка (например, NannyML):
```bash
python main.py --input data/small.csv --format csv --output reports/run_nannyml --framework nannyml
```

### 3. Просмотр результатов
Результаты сохраняются в директории `output` (например, `reports/run_small/<timestamp>/`).
- `dashboard.html`: Графики сравнения.
- `comparison_summary.json`: Метрики для всех фреймворков.
- `final_summary.json`: Выбор лучшего фреймворка.
- `model_metrics.json`: Производительность модели Prophet.
- `<framework_name>/`: Специфические артефакты для каждого фреймворка.

## Структура проекта
- `src/`: Исходный код
  - `frameworks/`: Адаптеры для каждого инструмента
  - `data_gen.py`: Генератор данных
  - `model.py`: Обертка для Prophet
  - `reporting.py`: Логика дашборда
- `main.py`: Точка входа CLI
