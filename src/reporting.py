import json
import os
from typing import Dict, Any
from html import escape
from datetime import datetime
import pandas as pd
import numpy as np

def _convert_to_native_types(obj):
    """Конвертирует numpy типы в стандартные Python типы для JSON сериализации."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (pd.Timestamp, datetime)):
        # Преобразуем pandas.Timestamp и datetime в строку в ISO-формате
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

def save_json_report(data: Dict[str, Any], output_path: str):
    """Сохраняет словарь в JSON."""
    # Конвертируем numpy типы перед сериализацией
    converted_data = _convert_to_native_types(data)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

def generate_dashboard(comparison_summary: Dict[str, Any], output_path: str, test_df: pd.DataFrame = None, predictions: pd.DataFrame = None):
    """
    Генерирует HTML дашборд с графиками сравнения и прогноза.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = list(comparison_summary.keys())
    times = [float(comparison_summary[n].get('execution_time_sec', 0.0) or 0.0) for n in names]
    mems = [float(comparison_summary[n].get('memory_peak_mb', 0.0) or 0.0) for n in names]
    issues = [int(comparison_summary[n].get('issues_detected', 0) or 0) for n in names]
    coverage = [float(comparison_summary[n].get('coverage_score', 0.0) or 0.0) for n in names]

    # Графики метрик фреймворков
    figs = [
        ("Время выполнения (сек)", go.Figure(data=[go.Bar(x=names, y=times)])),
        ("Пиковая память RSS (МБ)", go.Figure(data=[go.Bar(x=names, y=mems)])),
        ("Количество проблем", go.Figure(data=[go.Bar(x=names, y=issues)])),
        ("Покрытие тестами", go.Figure(data=[go.Bar(x=names, y=coverage)])),
    ]
    for title, fig in figs:
        fig.update_layout(title=title, xaxis_title="Фреймворк", yaxis_title=title, template="plotly_white")

    chart_html_parts = []
    for i, (_, fig) in enumerate(figs):
        chart_html_parts.append(fig.to_html(include_plotlyjs=False, full_html=False))

    # График прогноза (если переданы данные)
    forecast_html = ""
    forecast_fig = None
    if test_df is not None and predictions is not None:
        try:
            # Убедимся, что даты совпадают и данные синхронизированы
            # Синхронизируем по датам для корректного сравнения
            test_df_clean = test_df[['ds', 'y']].copy()
            predictions_clean = predictions[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].copy()
            
            # Объединяем по датам для синхронизации
            merged = pd.merge(
                test_df_clean, 
                predictions_clean, 
                on='ds', 
                how='inner',
                suffixes=('', '_pred')
            )
            
            # Удаляем NaN для корректного отображения
            mask = ~(merged['y'].isna() | merged['yhat'].isna())
            merged_clean = merged[mask].copy()
            
            fig_forecast = go.Figure()
            
            if len(merged_clean) > 0:
                # Доверительный интервал (рисуем первым, чтобы был под линиями)
                fig_forecast.add_trace(go.Scatter(
                    x=merged_clean['ds'], 
                    y=merged_clean['yhat_upper'],
                    mode='lines', 
                    line=dict(width=0), 
                    showlegend=False, 
                    hoverinfo='skip',
                    name='Верхняя граница'
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=merged_clean['ds'], 
                    y=merged_clean['yhat_lower'],
                    mode='lines', 
                    line=dict(width=0), 
                    fill='tonexty', 
                    fillcolor='rgba(0, 100, 255, 0.15)',
                    name='Доверительный интервал (80%)',
                    hoverinfo='skip'
                ))
                
                # Факт (y) - реальные значения
                fig_forecast.add_trace(go.Scatter(
                    x=merged_clean['ds'], 
                    y=merged_clean['y'],
                    mode='lines+markers', 
                    name='Факт (y)',
                    line=dict(color='#2c3e50', width=2),
                    marker=dict(size=3, color='#2c3e50', opacity=0.6),
                    hovertemplate='<b>Факт</b><br>Дата: %{x}<br>Значение: %{y:.2f}<extra></extra>'
                ))
                
                # Прогноз (yhat) - предсказанные значения
                fig_forecast.add_trace(go.Scatter(
                    x=merged_clean['ds'], 
                    y=merged_clean['yhat'],
                    mode='lines+markers', 
                    name='Прогноз (yhat)',
                    line=dict(color='#3498db', width=2.5, dash='dash'),
                    marker=dict(size=3, color='#3498db', opacity=0.6),
                    hovertemplate='<b>Прогноз</b><br>Дата: %{x}<br>Значение: %{y:.2f}<extra></extra>'
                ))
                
                # Настройка оси X с ползунком и начальным зумом
                xaxis_config = dict(
                    rangeslider=dict(visible=True),
                    type="date",
                    title="Дата (тестовый интервал)"
                )
                
                # Если точек много, показываем только последние 500 по умолчанию
                if len(merged_clean) > 500:
                    last_date = merged_clean['ds'].max()
                    start_view = merged_clean['ds'].iloc[-500]
                    xaxis_config['range'] = [start_view, last_date]
            else:
                raise ValueError("Нет данных для отображения после синхронизации")

            fig_forecast.update_layout(
                title="Сравнение факта (y) и прогноза (yhat) на тестовом интервале",
                xaxis=xaxis_config,
                yaxis_title="Значение",
                template="plotly_white",
                height=600,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            forecast_html = fig_forecast.to_html(include_plotlyjs=False, full_html=False)
            forecast_fig = fig_forecast  # Сохраняем для отдельного сохранения
        except Exception as e:
            print(f"Ошибка генерации графика прогноза: {e}")
            import traceback
            traceback.print_exc()
            forecast_html = f"<p>Ошибка генерации графика: {e}</p>"

    rows_html = []
    for name in names:
        res = comparison_summary[name] or {}
        arts = res.get('artifacts', []) or []
        links = []
        for a in arts:
            a_str = str(a)
            links.append(f'<a href="{escape(a_str)}">{escape(os.path.basename(a_str))}</a>')
        rows_html.append(
            "<tr>"
            f"<td>{escape(str(name))}</td>"
            f"<td>{float(res.get('execution_time_sec', 0.0) or 0.0):.4f}</td>"
            f"<td>{float(res.get('memory_peak_mb', 0.0) or 0.0):.2f}</td>"
            f"<td>{int(res.get('issues_detected', 0) or 0)}</td>"
            f"<td>{float(res.get('coverage_score', 0.0) or 0.0):.2f}</td>"
            f"<td>{', '.join(links)}</td>"
            "</tr>"
        )

    html_content = f"""
<html>
<head>
  <meta charset="utf-8"/>
  <title>Сравнение ML-фреймворков</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .full-width {{ grid-column: 1 / -1; }}
    .card {{ border: 1px solid #eee; padding: 12px; border-radius: 10px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background-color: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>Сравнение ML-фреймворков</h1>
  
  <div class="grid">
    <div class="card full-width">{forecast_html}</div>
    <div class="card">{chart_html_parts[0]}</div>
    <div class="card">{chart_html_parts[1]}</div>
    <div class="card">{chart_html_parts[2]}</div>
    <div class="card">{chart_html_parts[3]}</div>
  </div>

  <h2>Сводная таблица</h2>
  <table>
    <tr>
      <th>Фреймворк</th>
      <th>Время (с)</th>
      <th>Память (МБ)</th>
      <th>Проблемы</th>
      <th>Покрытие</th>
      <th>Артефакты</th>
    </tr>
    {''.join(rows_html)}
  </table>
</body>
</html>
""".strip()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Сохраняем отдельный файл с графиком y vs yhat, если он был создан
    if forecast_fig is not None:
        forecast_path = output_path.replace('dashboard.html', 'forecast_plot.html')
        try:
            forecast_fig.write_html(forecast_path, include_plotlyjs='cdn')
            print(f"График прогноза сохранен: {forecast_path}")
        except Exception as e:
            print(f"Ошибка сохранения графика прогноза: {e}")

def save_split_demonstration(train_df: pd.DataFrame, test_df: pd.DataFrame, split_info: Dict[str, Any], output_path: str):
    """
    Сохраняет HTML файл с демонстрацией train/test split по времени.
    """
    from html import escape
    
    n_show = 5
    train_tail = train_df[['ds', 'y', 'price', 'promotion']].tail(n_show)
    test_head = test_df[['ds', 'y', 'price', 'promotion']].head(n_show)
    
    # Форматируем таблицы в HTML
    def df_to_html_table(df, title):
        html = f"<h3>{title}</h3>\n"
        html += "<table style='border-collapse: collapse; width: 100%; margin-bottom: 20px;'>\n"
        html += "<thead><tr style='background-color: #f2f2f2;'>"
        html += "<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Индекс</th>"
        for col in df.columns:
            html += f"<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{escape(str(col))}</th>"
        html += "</tr></thead>\n<tbody>\n"
        
        for idx, row in df.iterrows():
            html += "<tr>"
            html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{idx}</td>"
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    val_str = "NaN"
                elif isinstance(val, (int, float)):
                    val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                else:
                    val_str = str(val)
                html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{escape(val_str)}</td>"
            html += "</tr>\n"
        
        html += "</tbody></table>\n"
        return html
    
    train_table = df_to_html_table(train_tail, f"Последние {n_show} строк Train (перед разделением)")
    test_table = df_to_html_table(test_head, f"Первые {n_show} строк Test (после разделения)")
    
    html_content = f"""
<html>
<head>
  <meta charset="utf-8"/>
  <title>Демонстрация Train/Test Split</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #34495e; margin-top: 30px; }}
    h3 {{ color: #555; }}
    .info-box {{
      background-color: #ecf0f1;
      border-left: 4px solid #3498db;
      padding: 15px;
      margin: 20px 0;
    }}
    .success {{ color: #27ae60; font-weight: bold; }}
    .warning {{ color: #e67e22; font-weight: bold; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Демонстрация временного разбиения данных (Train/Test Split)</h1>
  
  <div class="info-box">
    <h2>Общая информация о разбиении</h2>
    <p><strong>Общий размер данных:</strong> {split_info['total_rows']} строк</p>
    <p><strong>Train:</strong> {split_info['train_rows']} строк ({split_info['train_percent']:.1f}%)</p>
    <p style="margin-left: 20px;">Период: с {split_info['train_date_range']['start']} по {split_info['train_date_range']['end']}</p>
    <p><strong>Test:</strong> {split_info['test_rows']} строк ({split_info['test_percent']:.1f}%)</p>
    <p style="margin-left: 20px;">Период: с {split_info['test_date_range']['start']} по {split_info['test_date_range']['end']}</p>
  </div>
  
  <h2>Проверка непрерывности дат</h2>
  <div class="info-box">
    <p><strong>Последняя дата Train:</strong> {split_info['train_date_range']['end']}</p>
    <p><strong>Первая дата Test:</strong> {split_info['test_date_range']['start']}</p>
    <p><strong>Разрыв между Train и Test:</strong> {split_info['date_gap_days']} дней</p>
    <p class="{'success' if split_info['date_gap_days'] == 1 else 'warning'}">
      {'✓ Даты идут последовательно без пропусков' if split_info['date_gap_days'] == 1 else f'⚠ Внимание: есть пропуск в {split_info["date_gap_days"] - 1} дней между train и test'}
    </p>
  </div>
  
  <h2>Демонстрация последовательности данных</h2>
  <p>Ниже показаны последние строки обучающей выборки и первые строки тестовой выборки. 
     Видно, что данные отсортированы по времени и не перемешаны.</p>
  
  {train_table}
  
  {test_table}
  
  <div class="info-box">
    <h3>Вывод</h3>
    <p>Разбиение выполнено <strong>по времени</strong> (temporal split), а не случайным образом:</p>
    <ul>
      <li>Данные отсортированы по дате (ds) перед разделением</li>
      <li>Train содержит более ранние даты (первые 80% данных)</li>
      <li>Test содержит более поздние даты (последние 20% данных)</li>
      <li>Даты идут последовательно, что видно из таблиц выше</li>
    </ul>
  </div>
</body>
</html>
""".strip()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Демонстрация train/test split сохранена: {output_path}")

def determine_best_framework(comparison_summary: Dict[str, Any]) -> str:
    """
    Определяет лучший фреймворк:
    1) макс. coverage_score
    2) мин. execution_time_sec
    3) мин. memory_peak_mb
    """
    candidates = []
    for name, res in (comparison_summary or {}).items():
        if not isinstance(res, dict):
            continue
        cov = float(res.get('coverage_score', 0.0) or 0.0)
        t = float(res.get('execution_time_sec', float('inf')) or float('inf'))
        m = float(res.get('memory_peak_mb', float('inf')) or float('inf'))
        candidates.append((name, cov, t, m))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (-x[1], x[2], x[3], str(x[0])))
    return candidates[0][0]
