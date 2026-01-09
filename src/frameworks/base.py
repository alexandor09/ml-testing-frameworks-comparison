from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List

class BaseAdapter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run_checks(self, train_df: pd.DataFrame, test_df: pd.DataFrame, predictions: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        Запускает проверки фреймворка.
        
        Аргументы:
            train_df: Обучающая выборка (эталон)
            test_df: Тестовая выборка (текущая)
            predictions: Прогнозы модели
            output_dir: Папка для сохранения артефактов
            
        Возвращает:
            Словарь с ключами:
                - execution_time_sec: время выполнения (сек)
                - memory_peak_mb: пиковая память (МБ)
                - issues_detected: количество найденных проблем
                - coverage_score: оценка покрытия (0.0 - 1.0)
                - artifacts: список путей к файлам
        """
        pass
    
    def calculate_coverage_score(self, checks_performed: Dict[str, bool]) -> float:
        """
        Считает покрытие на основе выполненных проверок.
        Ключи: 'data_quality', 'data_drift', 'outliers', 'model_performance'
        """
        score = sum(checks_performed.values()) / 4.0
        return score
