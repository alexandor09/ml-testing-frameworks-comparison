import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

class ForecastModel:
    def __init__(self):
        self.model = None

    def train(self, df: pd.DataFrame):
        """
        Обучает модель Prophet.
        Ожидает df с колонками: ds, y, price, promotion
        """
        # На больших train-наборах (10k+ точек) Prophet/CmdStan может упасть по памяти
        # на слабых машинах (std::bad_alloc). Для нашего учебного проекта достаточно
        # обучаться на ограниченном окне последних наблюдений.
        max_train_rows = 5000
        train_df = df.tail(max_train_rows).copy() if len(df) > max_train_rows else df

        # Prophet по умолчанию генерирует доверительные интервалы через 1000 симуляций
        # (uncertainty_samples=1000). На датасетах 20k+ это может приводить к выделению
        # больших матриц и падению с `_ArrayMemoryError` даже на "ideal".
        # Уменьшаем число сэмплов: интервалы остаются, но память/время резко падают.
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            uncertainty_samples=50,
            # Чуть упрощаем модель, чтобы быстрее/стабильнее обучаться на Windows
            n_changepoints=15,
        )
        self.model.add_regressor('price')
        self.model.add_regressor('promotion')
        self.model.fit(train_df)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует прогнозы.
        Возвращает DataFrame с колонками: ds, yhat, yhat_lower, yhat_upper, predicted_revenue
        """
        if self.model is None:
            raise ValueError("Модель не обучена.")
        
        forecast = self.model.predict(df)
        
        # Расчет прогнозируемой выручки (predicted_revenue = yhat * price)
        # 'price' должен быть во входном df
        
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        
        if 'price' in df.columns:
             result['predicted_revenue'] = result['yhat'] * df['price']
        else:
             result['predicted_revenue'] = 0.0
             
        return result

def calculate_metrics(y_true, y_pred):
    """
    Считает MAE, RMSE, MAPE.
    Удаляет NaN значения перед вычислением метрик.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Удаляем NaN значения (пропуски в данных)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        # Если все значения NaN, возвращаем нулевые метрики
        return {
            "MAE": 0.0,
            "RMSE": 0.0,
            "MAPE": 0.0
        }

    mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
    rmse = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))

    eps = 1e-8
    mape = float(np.mean(np.abs((y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + eps))))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
