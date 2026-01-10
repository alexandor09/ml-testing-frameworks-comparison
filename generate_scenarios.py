import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

def generate_base_data(n_rows: int, start_date: str = '2023-01-01', seed: int = 42) -> pd.DataFrame:
    """
    Генерирует базовые данные без артефактов.
    Используется как основа для всех сценариев.
    """
    np.random.seed(seed)
    dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(n_rows)]
    
    # Переключение режимов рынка (Markov Chain)
    states = np.zeros(n_rows, dtype=int)
    transition_matrix = [
        [0.98, 0.01, 0.01],  # Из Stable
        [0.05, 0.90, 0.05],  # Из Boom
        [0.02, 0.01, 0.97]   # Из Recession
    ]
    
    current_state = 0
    for i in range(1, n_rows):
        current_state = np.random.choice([0, 1, 2], p=transition_matrix[current_state])
        states[i] = current_state
        
    regime_params = {
        0: (0.05, 1.0),   # Stable
        1: (0.15, 1.5),   # Boom
        2: (-0.05, 2.0)   # Recession
    }
    
    # Генерация тренда
    trend = np.zeros(n_rows)
    trend[0] = 100
    volatility_mult = np.ones(n_rows)
    
    for i in range(1, n_rows):
        slope, vol = regime_params[states[i]]
        trend[i] = trend[i-1] + slope + np.random.normal(0, 0.1)
        volatility_mult[i] = vol

    # Сезонность
    t = np.arange(n_rows)
    seasonality = 15 * np.sin(2 * np.pi * t / 365) + 10 * np.cos(2 * np.pi * t / 7)
    
    # Праздники
    holidays = np.zeros(n_rows)
    for i, d in enumerate(dates):
        if d.month == 12 and d.day > 25:
            holidays[i] = 30
        if d.month == 1 and d.day < 10:
            holidays[i] = -10
        if d.month == 11 and d.day > 25 and d.day < 30:
            holidays[i] = 40

    # Цена
    price = np.zeros(n_rows)
    current_price = 100.0
    for i in range(n_rows):
        if i % 7 == 0:
            current_price += np.random.uniform(-5, 5)
            current_price = max(80, min(120, current_price))
        price[i] = current_price
        
    # Цена конкурента
    comp_price = np.zeros(n_rows)
    comp_price[0] = 100.0
    for i in range(1, n_rows):
        target = price[i-1]
        change = (target - comp_price[i-1]) * 0.1 + np.random.normal(0, 2)
        comp_price[i] = comp_price[i-1] + change
    
    relative_price = price / comp_price
    elasticity_effect = np.power(relative_price, -1.5) * 100 - 100

    # Промо-акции
    raw_promo = np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    adstock_promo = np.zeros(n_rows)
    decay = 0.7
    for i in range(1, n_rows):
        if raw_promo[i] == 1:
            adstock_promo[i] = 1.0 + adstock_promo[i-1] * decay
        else:
            adstock_promo[i] = adstock_promo[i-1] * decay
            
    marketing_effect = adstock_promo * 30

    # Сборка
    base_demand = trend + seasonality + holidays + elasticity_effect + marketing_effect
    
    ar_noise = np.zeros(n_rows)
    for i in range(1, n_rows):
        shock = np.random.normal(0, 5 * volatility_mult[i])
        ar_noise[i] = 0.7 * ar_noise[i-1] + shock
        
    y = base_demand + ar_noise
    y = np.maximum(y, 0)

    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'price': price,
        'promotion': raw_promo
    })
    
    return df


def generate_ideal_data(n_rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует "идеальные" данные без проблем для всех фреймворков.

    Важно:
    - В `main.py` используется строгое разбиение по времени (последние 20% = test).
    - На больших выборках KS/дрейф-тесты очень чувствительны, поэтому даже нормальный тренд/сезонность
      могут выглядеть как "дрейф".
    - Чтобы "ideal" стабильно проходил проверки, делаем данные стационарными и повторяем одинаковые блоки:
      тогда train и test имеют одинаковое эмпирическое распределение.
    """
    rng = np.random.default_rng(seed)

    # Делаем блок кратным будущему split'у (80/20) в main.py.
    # При n_rows=20000 split_idx=16000, block_size=1000 => и train, и test состоят из целых блоков.
    block_size = 1000
    if n_rows % block_size != 0:
        # Чтобы не усложнять математику, просто подгоним n_rows вверх до кратности block_size
        n_rows = int(np.ceil(n_rows / block_size) * block_size)

    n_blocks = n_rows // block_size

    # Блок: умеренные, "безопасные" диапазоны.
    # y держим в [90, 110], чтобы IQR-детекторы выбросов (особенно в Alibi) не находили outliers.
    t = np.arange(block_size)
    promotion_block = rng.choice([0, 1], size=block_size, p=[0.95, 0.05]).astype(int)
    price_block = 95 + 2.0 * np.sin(2 * np.pi * t / 30) + rng.uniform(-0.5, 0.5, size=block_size)
    price_block = np.clip(price_block, 90, 100)

    y_block = (
        100
        + 3.0 * np.sin(2 * np.pi * t / 7)
        - 2.0 * promotion_block
        + rng.uniform(-1.0, 1.0, size=block_size)
    )
    y_block = np.clip(y_block, 90, 110)

    # Тиражируем блоки
    y = np.tile(y_block, n_blocks)
    price = np.tile(price_block, n_blocks)
    promotion = np.tile(promotion_block, n_blocks)

    dates = [datetime.strptime('2023-01-01', '%Y-%m-%d') + timedelta(days=i) for i in range(n_rows)]
    return pd.DataFrame({'ds': dates, 'y': y, 'price': price, 'promotion': promotion})


def generate_data_with_missing_duplicates(n_rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует данные с пропусками и дубликатами.
    """
    df = generate_base_data(n_rows, seed=seed)
    
    # Добавляем пропуски (5% в price, 2% в y)
    n_missing_price = int(n_rows * 0.05)
    n_missing_y = int(n_rows * 0.02)
    
    missing_price_indices = np.random.choice(range(n_rows), size=n_missing_price, replace=False)
    missing_y_indices = np.random.choice(range(n_rows), size=n_missing_y, replace=False)
    
    df.loc[missing_price_indices, 'price'] = np.nan
    df.loc[missing_y_indices, 'y'] = np.nan
    
    # Добавляем дубликаты (3% данных)
    n_duplicates = int(n_rows * 0.03)
    duplicate_indices = np.random.choice(range(n_rows), size=n_duplicates, replace=False)
    
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Сортируем по дате
    df = df.sort_values('ds').reset_index(drop=True)
    
    return df


def generate_data_with_drift_outliers(n_rows: int = 20000, seed: int = 43) -> pd.DataFrame:
    """
    Генерирует данные с сильным дрейфом и выбросами.
    """
    # Берем "чистую" базу (как ideal), а дрейф вводим ТОЛЬКО в тестовом хвосте (последние 20%),
    # потому что `main.py` сравнивает train(0..80%) и test(80..100%).
    df = generate_ideal_data(n_rows=n_rows, seed=seed)
    
    # Дрейф начинается в тестовом периоде
    drift_start = int(len(df) * 0.8)
    rng = np.random.default_rng(seed + 1000)
    
    # 1) Дрейф в цене: сильный сдвиг вверх (выйдет за ожидания GX и даст заметный drift)
    n_tail = len(df) - drift_start
    df.loc[drift_start:, 'price'] = df.loc[drift_start:, 'price'] + 60 + rng.normal(0, 3, size=n_tail)
    df.loc[drift_start:, 'price'] = df.loc[drift_start:, 'price'].clip(90, 180)
    
    # 2) Дрейф в промо: промо становится намного чаще
    promo_drift = rng.choice([0, 1], size=n_tail, p=[0.60, 0.40]).astype(int)
    df.loc[drift_start:, 'promotion'] = promo_drift
    
    # 3) Дрейф в y: структурный сдвиг + смена влияния промо (коэф-т "переворачивается"),
    # чтобы модель, обученная на train, деградировала на test (NannyML performance alerts).
    y_tail = df.loc[drift_start:, 'y'].to_numpy()
    promo_tail = df.loc[drift_start:, 'promotion'].to_numpy()
    df.loc[drift_start:, 'y'] = (
        y_tail
        - 25.0
        + 15.0 * promo_tail
        + rng.normal(0, 8, size=n_tail)
    )
    df.loc[drift_start:, 'y'] = np.maximum(df.loc[drift_start:, 'y'], 0)
    
    # 4) Выбросы: только в тестовом хвосте (чтобы не "обучить" модель на выбросах)
    n_outliers = int(n_tail * 0.07)
    outlier_indices = rng.choice(np.arange(drift_start, len(df)), size=n_outliers, replace=False)
    
    outlier_values = rng.choice([-60, -40, 80, 120], size=n_outliers)
    df.loc[outlier_indices, 'y'] = np.maximum(df.loc[outlier_indices, 'y'] + outlier_values, 0)
    
    # 5) Дополнительные выбросы в цене (тоже в хвосте)
    price_outlier_indices = rng.choice(np.arange(drift_start, len(df)), size=int(n_tail * 0.05), replace=False)
    df.loc[price_outlier_indices, 'price'] = rng.choice([150, 180, 200], size=len(price_outlier_indices))
    df.loc[price_outlier_indices, 'price'] = df.loc[price_outlier_indices, 'price'].clip(90, 200)
    
    return df


def save_data(df: pd.DataFrame, output_dir: str, name: str):
    """Сохраняет DataFrame в CSV и JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV
    csv_path = os.path.join(output_dir, f'{name}.csv')
    df.to_csv(csv_path, index=False)
    # В Windows-консолях с CP1251 символы вроде "✓" могут ломать вывод (UnicodeEncodeError).
    print(f"Сохранено {csv_path} ({len(df)} строк)")
    
    # JSON
    json_path = os.path.join(output_dir, f'{name}.json')
    df_json = df.copy()
    df_json['ds'] = df_json['ds'].dt.strftime('%Y-%m-%d')
    df_json.to_json(json_path, orient='records', indent=2)
    print(f"Сохранено {json_path} ({len(df_json)} строк)")


if __name__ == "__main__":
    print("=" * 60)
    print("Генерация сценариев данных")
    print("=" * 60)
    
    output_dir = 'data'
    n_rows = 20000
    
    # 1. Идеальные данные
    print("\n1. Генерация идеальных данных (ideal)...")
    df_ideal = generate_ideal_data(n_rows=n_rows, seed=42)
    save_data(df_ideal, output_dir, 'ideal')
    
    # 2. Данные с пропусками и дубликатами
    print("\n2. Генерация данных с пропусками и дубликатами (pass)...")
    df_pass = generate_data_with_missing_duplicates(n_rows=n_rows, seed=42)
    save_data(df_pass, output_dir, 'pass')
    
    # 3. Данные с дрейфом и выбросами
    print("\n3. Генерация данных с дрейфом и выбросами (dr)...")
    df_dr = generate_data_with_drift_outliers(n_rows=n_rows, seed=43)
    save_data(df_dr, output_dir, 'dr')
    
    print("\n" + "=" * 60)
    print("Генерация завершена!")
    print("=" * 60)
    print(f"\nСгенерированные файлы:")
    print(f"  - {output_dir}/ideal.csv и {output_dir}/ideal.json")
    print(f"  - {output_dir}/pass.csv и {output_dir}/pass.json")
    print(f"  - {output_dir}/dr.csv и {output_dir}/dr.json")

