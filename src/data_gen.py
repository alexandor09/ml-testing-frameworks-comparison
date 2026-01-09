import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

def generate_data(n_rows: int, start_date: str = '2023-01-01', seed: int = 42) -> pd.DataFrame:
    """
    Генерирует высокореалистичные временные ряды для прогнозирования спроса.
    Симуляция включает:
    1. Переключение режимов рынка (Stable, Boom, Recession) через цепи Маркова.
    2. Эластичность спроса от относительной цены (vs конкурент).
    3. Маркетинговый эффект Adstock (затухание промо).
    4. Сложную сезонность и праздники.
    5. Авторегрессию (инерцию спроса).
    """
    np.random.seed(seed)
    dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i) for i in range(n_rows)]
    
    # --- 1. Переключение режимов рынка (Markov Chain) ---
    # 0: Stable, 1: Boom, 2: Recession
    states = np.zeros(n_rows, dtype=int)
    # Матрица переходов: [Stable, Boom, Recession] -> [Stable, Boom, Recession]
    transition_matrix = [
        [0.98, 0.01, 0.01], # Из Stable
        [0.05, 0.90, 0.05], # Из Boom (быстрее заканчивается)
        [0.02, 0.01, 0.97]  # Из Recession (затяжная)
    ]
    
    current_state = 0 # Start with Stable
    for i in range(1, n_rows):
        current_state = np.random.choice([0, 1, 2], p=transition_matrix[current_state])
        states[i] = current_state
        
    # Параметры для режимов: [Base Trend Slope, Volatility Multiplier]
    regime_params = {
        0: (0.05, 1.0),  # Stable: слабый рост, норм волатильность
        1: (0.15, 1.5),  # Boom: сильный рост, высокая активность
        2: (-0.05, 2.0)  # Recession: падение, паника (высокая волатильность)
    }
    
    # Генерация тренда на основе режимов
    trend = np.zeros(n_rows)
    trend[0] = 100 # Start level
    volatility_mult = np.ones(n_rows)
    
    for i in range(1, n_rows):
        slope, vol = regime_params[states[i]]
        trend[i] = trend[i-1] + slope + np.random.normal(0, 0.1) # Stochastic trend
        volatility_mult[i] = vol

    # --- 2. Сезонность и Праздники ---
    t = np.arange(n_rows)
    # Годовая (сильная) + Недельная (сильная)
    seasonality = 15 * np.sin(2 * np.pi * t / 365) + 10 * np.cos(2 * np.pi * t / 7)
    
    # Праздники (Новый год - пик, Январь - спад)
    holidays = np.zeros(n_rows)
    for i, d in enumerate(dates):
        if d.month == 12 and d.day > 25: # Предновогодний бум
            holidays[i] = 30
        if d.month == 1 and d.day < 10: # Январское затишье
            holidays[i] = -10
        if d.month == 11 and d.day > 25 and d.day < 30: # Black Friday (примерно)
            holidays[i] = 40

    # --- 3. Цена, Конкуренты и Эластичность ---
    # Наша цена (случайные изменения раз в неделю)
    price = np.zeros(n_rows)
    current_price = 100.0
    for i in range(n_rows):
        if i % 7 == 0:
            current_price += np.random.uniform(-5, 5)
            current_price = max(80, min(120, current_price))
        price[i] = current_price
        
    # Цена конкурента (коррелирует с нашей, но с лагом и шумом)
    comp_price = np.zeros(n_rows)
    comp_price[0] = 100.0
    for i in range(1, n_rows):
        # Конкурент реагирует на нашу цену с задержкой
        target = price[i-1]
        change = (target - comp_price[i-1]) * 0.1 + np.random.normal(0, 2)
        comp_price[i] = comp_price[i-1] + change
    
    # Относительная цена
    relative_price = price / comp_price
    # Эластичность: если мы дороже на 10%, спрос падает на 15% (k=-1.5)
    elasticity_effect = np.power(relative_price, -1.5) * 100 - 100 # Центрируем около 0

    # --- 4. Маркетинг и Adstock ---
    # Промо-акции (редкие, но меткие)
    raw_promo = np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    # Adstock эффект (затухание)
    adstock_promo = np.zeros(n_rows)
    decay = 0.7
    for i in range(1, n_rows):
        if raw_promo[i] == 1:
            adstock_promo[i] = 1.0 + adstock_promo[i-1] * decay # Накопление
        else:
            adstock_promo[i] = adstock_promo[i-1] * decay
            
    marketing_effect = adstock_promo * 30 # Сила промо

    # --- 5. Сборка и AR (Авторегрессия) ---
    y = np.zeros(n_rows)
    base_demand = trend + seasonality + holidays + elasticity_effect + marketing_effect
    
    # AR(1) процесс для шума/инерции
    ar_noise = np.zeros(n_rows)
    for i in range(1, n_rows):
        # Шум зависит от волатильности режима
        shock = np.random.normal(0, 5 * volatility_mult[i])
        ar_noise[i] = 0.7 * ar_noise[i-1] + shock
        
    y = base_demand + ar_noise
    y = np.maximum(y, 0) # Неотрицательный спрос

    # --- 6. Артефакты данных (для тестов) ---
    # Дрейф данных в тесте (последние 20%): конкурент демпингует
    test_start = int(n_rows * 0.8)
    # Конкурент резко снижает цены -> наша относительная цена растет -> спрос падает
    # Но мы это "не видим" в колонке price, видим только падение продаж
    # Чтобы это было честно, мы должны это отразить в данных. 
    # Но мы отдаем только 'price' (нашу). 
    # Дрейф будет в том, что связь y ~ price изменится (скрытый фактор).
    
    # Аномалии (выбросы)
    n_outliers = int(n_rows * 0.005)
    outlier_indices = np.random.choice(range(n_rows), size=n_outliers, replace=False)
    y[outlier_indices] += np.random.choice([-50, 50], size=n_outliers)
    y = np.maximum(y, 0)

    # Пропуски в цене
    n_missing = int(n_rows * 0.005)
    missing_indices = np.random.choice(range(n_rows), size=n_missing, replace=False)
    price_with_nan = price.copy()
    price_with_nan[missing_indices] = np.nan

    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'price': price_with_nan,
        'promotion': raw_promo # Отдаем сырой флаг промо, модель должна сама выучить лаги (или нет)
    })
    
    return df

def save_data(df: pd.DataFrame, output_dir: str, name: str):
    """Сохраняет DataFrame в CSV и JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV
    csv_path = os.path.join(output_dir, f'{name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Сохранено {csv_path}")
    
    # JSON
    json_path = os.path.join(output_dir, f'{name}.json')
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d') # Даты в строку для JSON
    df.to_json(json_path, orient='records', indent=2)
    print(f"Сохранено {json_path}")

if __name__ == "__main__":
    # Малый датасет
    print("Генерация малого датасета...")
    df_small = generate_data(n_rows=1000)
    save_data(df_small, 'data', 'small')
    
    # Большой датасет
    print("Генерация большого датасета...")
    df_big = generate_data(n_rows=20000)
    save_data(df_big, 'data', 'big')
