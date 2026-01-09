import os
import threading
import time
from typing import Callable, Tuple, TypeVar

import psutil

T = TypeVar("T")


def measure_time_and_peak_rss_mb(func: Callable[[], T], sample_interval_sec: float = 0.02) -> Tuple[T, float, float]:
    """
    Запускает `func` и измеряет:
    - execution_time_sec: время выполнения
    - memory_peak_mb: пиковая RSS память текущего процесса python во время выполнения `func`

    Используется подход с сэмплированием (psutil). Это соответствует требованию ЛР4:
    "мониторинг через psutil, брать максимум".
    """
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    stop = threading.Event()

    def _sampler() -> None:
        nonlocal peak_rss
        while not stop.is_set():
            try:
                rss = process.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                # Если psutil временно недоступен, продолжаем сэмплирование
                pass
            time.sleep(sample_interval_sec)

    t0 = time.time()
    thread = threading.Thread(target=_sampler, name="rss-peak-sampler", daemon=True)
    thread.start()
    try:
        result = func()
    finally:
        stop.set()
        thread.join(timeout=1.0)
    t1 = time.time()

    execution_time_sec = t1 - t0
    memory_peak_mb = peak_rss / 1024 / 1024
    return result, execution_time_sec, memory_peak_mb



