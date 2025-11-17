"""
Streamlit веб-интерфейс для Parking AI (ПОЛНАЯ ВЕРСИЯ)
С автоматической детекцией парковочных мест
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from pathlib import Path

# ============ ПРАВИЛЬНЫЕ ИМПОРТЫ ============
# Добавляем пути для импортов
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Импорты модулей (используйте ОДИН из вариантов ниже)

# ВАРИАНТ 1: Прямые импорты (если src в PYTHONPATH)
try:
    from pipeline import ParkingPipeline
    from analyzer import DensityAnalyzer
    from recommender import Recommender
except ImportError:
    # ВАРИАНТ 2: Импорты через src (если структура пакета)
    from src.pipeline import ParkingPipeline
    from src.analyzer import DensityAnalyzer
    from src.recommender import Recommender

# ============ КОНФИГУРАЦИЯ СТРАНИЦЫ ============
st.set_page_config(
    page_title="Parking AI | Анализ парковок",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ УЛУЧШЕННЫЕ СТИЛИ CSS ============
st.markdown("""
<style>
    /* Основные стили */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1f77b4 0%, #667eea 50%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-in;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    .zone-card {
        border: 3px solid #ddd;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .zone-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .critical { 
        border-color: #ff4444; 
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    }
    .warning { 
        border-color: #ffaa00; 
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    }
    .normal { 
        border-color: #00cc66; 
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }
    .empty { 
        border-color: #aaaaaa; 
        background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
    }
    
    .progress-bar {
        width: 100%;
        height: 30px;
        background: #e0e0e0;
        border-radius: 15px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00cc66 0%, #ffaa00 70%, #ff4444 100%);
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    
    .badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 5px;
    }
    
    .badge-critical { background: #ff4444; color: white; }
    .badge-warning { background: #ffaa00; color: white; }
    .badge-normal { background: #00cc66; color: white; }
    .badge-empty { background: #aaaaaa; color: white; }
    
    .stButton>button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Стили для парковочных мест */
    .parking-space-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .space-free {
        border-left: 5px solid #00cc66;
    }
    
    .space-occupied {
        border-left: 5px solid #ff4444;
    }
</style>
""", unsafe_allow_html=True)

# ============ ИНИЦИАЛИЗАЦИЯ СЕССИИ ============
def load_pipeline(conf_threshold):
    """Загрузка пайплайна с заданным порогом уверенности"""
    try:
        return ParkingPipeline(conf_threshold=conf_threshold, use_auto_detection=True)
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке пайплайна: {str(e)}")
        return None

# ============ ФУНКЦИИ ВИЗУАЛИЗАЦИИ ============

def create_occupancy_chart(density_data):
    """Создание графика загруженности зон"""
    zones = list(density_data.keys())
    occupancy = [data['occupancy'] for data in density_data.values()]
    vehicles = [data['vehicles'] for data in density_data.values()]
    
    colors = []
    for occ in occupancy:
        if occ >= 100:
            colors.append('#ff4444')
        elif occ >= 85:
            colors.append('#ffaa00')
        elif occ >= 70:
            colors.append('#ffa500')
        elif occ > 0:
            colors.append('#00cc66')
        else:
            colors.append('#aaaaaa')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=zones,
        y=occupancy,
        text=[f"{o}%" for o in occupancy],
        textposition='outside',
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Загруженность: %{y}%<br>ТС: %{customdata}<extra></extra>',
        customdata=vehicles,
        name='Загруженность'
    ))
    
    fig.update_layout(
        title={
            'text': '📊 Загруженность зон парковки',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1f77b4'}
        },
        xaxis_title='Зоны',
        yaxis_title='Загруженность (%)',
        yaxis=dict(range=[0, max(occupancy) + 20] if occupancy else [0, 100]),
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_vehicle_types_chart(summary):
    """Круговая диаграмма типов ТС"""
    if not summary['by_type']:
        return None
    
    types = list(summary['by_type'].keys())
    counts = list(summary['by_type'].values())
    
    type_icons = {
        'car': '🚗 Легковые',
        'truck': '🚛 Грузовики',
        'bus': '🚌 Автобусы',
        'motorcycle': '🏍️ Мотоциклы'
    }
    
    labels = [type_icons.get(t, t) for t in types]
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Количество: %{value}<br>Доля: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': '🚙 Распределение типов транспорта',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1f77b4'}
        },
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def create_capacity_gauge(summary):
    """Круговой индикатор общей загруженности"""
    occupancy = summary['total_occupancy']
    
    if occupancy >= 100:
        color = '#ff4444'
    elif occupancy >= 85:
        color = '#ffaa00'
    elif occupancy >= 70:
        color = '#ffa500'
    else:
        color = '#00cc66'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=occupancy,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Общая загруженность", 'font': {'size': 24}},
        delta={'reference': 70, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 70], 'color': '#c8e6c9'},
                {'range': [70, 85], 'color': '#ffecb3'},
                {'range': [85, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_parking_spaces_chart(space_occupancy):
    """График статистики парковочных мест"""
    if not space_occupancy:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Свободно', 'Занято'],
            y=[space_occupancy['free'], space_occupancy['occupied']],
            marker=dict(color=['#00cc66', '#ff4444']),
            text=[space_occupancy['free'], space_occupancy['occupied']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🅿️ Статистика парковочных мест',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1f77b4'}
        },
        yaxis_title='Количество мест',
        height=350,
        template='plotly_white'
    )
    
    return fig

def create_zone_comparison_table(density_data):
    """Таблица сравнения зон"""
    data = []
    
    level_icons = {
        'critical': '🔴',
        'warning': '🟡',
        'busy': '🟠',
        'normal': '🟢',
        'empty': '⚪'
    }
    
    for zone, info in sorted(density_data.items()):
        data.append({
            'Зона': f"{level_icons.get(info['level'], '❓')} {zone}",
            'ТС': info['vehicles'],
            'Занято': f"{info['space_used']:.1f}",
            'Вместимость': info['capacity'],
            'Загруженность': f"{info['occupancy']}%",
            'Доступно': f"{info['available']:.1f}",
            'Статус': info['level'].upper()
        })
    
    df = pd.DataFrame(data)
    
    return df

# ============ ГЛАВНАЯ ФУНКЦИЯ ============
def main():
    # Анимированный заголовок
    st.markdown('<h1 class="main-header">🚗 Parking AI — Интеллектуальный анализ парковок</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by YOLOv8 & CUDA | Real-time Detection & Analysis | Auto Parking Space Detection</p>',
                unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://img.icons8.com/fluency/96/parking.png", width=96)
        
        st.markdown("### ⚙️ ПАНЕЛЬ УПРАВЛЕНИЯ")
        st.markdown("---")
        
        # Режим работы
        mode = st.radio(
            "🎯 Выберите режим:",
            ["📸 Анализ изображения", "🔧 Калибровка системы", "🎥 Обработка видео", "📊 Демо", "ℹ️ О системе"],
            label_visibility="visible"
        )
        
        st.markdown("---")
        
        # Параметры детектора
        with st.expander("🔧 Настройки детектора", expanded=True):
            conf_threshold = st.slider(
                "Порог уверенности",
                min_value=0.1,
                max_value=0.9,
                value=0.25,
                step=0.05,
                help="Минимальная уверенность для детекции"
            )
            
            draw_zones = st.checkbox("Показать зоны", value=True)
            draw_detections = st.checkbox("Показать детекции", value=True)
            draw_spaces = st.checkbox("Показать парковочные места", value=True)
            show_charts = st.checkbox("Показать графики", value=True)
        
        st.markdown("---")
        
        # Статистика системы
        with st.expander("📊 Статистика системы"):
            pipeline_info = {
                'Детектор': 'YOLOv8m',
                'Устройство': 'CUDA (GPU)',
                'Зон парковки': '4',
                'Классов ТС': '4',
                'Авто-детекция мест': '✅'
            }
            
            for key, value in pipeline_info.items():
                st.metric(key, value)
        
        st.markdown("---")
        
        # Информация
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; color: white;'>
            <h4 style='margin: 0;'>💡 Возможности</h4>
            <ul style='margin: 0.5rem 0;'>
                <li>Детекция транспорта</li>
                <li>Анализ зон</li>
                <li>Автоматическое определение мест</li>
                <li>Рекомендации</li>
                <li>Обработка видео</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ============ РЕЖИМ: КАЛИБРОВКА СИСТЕМЫ ============
    if mode == "🔧 Калибровка системы":
        st.markdown("## 🔧 Калибровка системы автоматического определения мест")
        
        st.info("""
        **📌 Калибровка требуется один раз** для автоматического обнаружения парковочных мест.
        
        Загрузите изображение **пустой парковки** или парковки с **четкой разметкой**.
        
        Система автоматически обнаружит:
        - Вертикальные границы парковочных мест
        - Горизонтальные границы рядов
        - Координаты и размеры каждого места
        """)
        
        calibration_file = st.file_uploader(
            "📸 Загрузите изображение для калибровки",
            type=["jpg", "jpeg", "png"],
            help="Рекомендуется изображение с пустой парковкой для лучших результатов"
        )
        
        if calibration_file is not None:
            # Превью
            col1, col2 = st.columns([1, 1])
            
            with col1:
                calibration_image = Image.open(calibration_file)
                st.image(calibration_image, caption="Исходное изображение", use_container_width=True)
                st.info(f"📐 Размер: {calibration_image.size[0]}x{calibration_image.size[1]} px")
            
            if st.button("🚀 ЗАПУСТИТЬ КАЛИБРОВКУ", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Загрузка изображения
                    calibration_np = np.array(calibration_image)
                    
                    if len(calibration_np.shape) == 3:
                        calibration_np = cv2.cvtColor(calibration_np, cv2.COLOR_RGB2BGR)
                    
                    status_text.text("🔄 Инициализация пайплайна...")
                    progress_bar.progress(20)
                    
                    pipeline = load_pipeline(conf_threshold)
                    
                    if pipeline is None:
                        st.error("❌ Не удалось загрузить пайплайн")
                        return
                    
                    status_text.text("🔍 Обнаружение парковочных мест...")
                    progress_bar.progress(50)
                    
                    # Калибровка
                    success = pipeline.calibrate_parking_spaces(calibration_np)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Калибровка завершена!")
                    
                    if success:
                        with col2:
                            st.success(f"✅ Калибровка завершена успешно!")
                            
                            num_spaces = len(pipeline.auto_detector.parking_spaces)
                            st.metric("Обнаружено парковочных мест", num_spaces)
                            
                            # Показать визуализацию
                            viz_path = Path('outputs/auto_detection/step3_parking_spaces.jpg')
                            if viz_path.exists():
                                viz_image = Image.open(viz_path)
                                st.image(viz_image, caption="Обнаруженные места", use_container_width=True)
                        
                        st.markdown("---")
                        st.success("""
                        **✅ Калибровка завершена!**
                        
                        📁 Результаты сохранены в:
                        - `config/auto_detected_spaces.pkl`
                        - `outputs/auto_detection/`
                        
                        💡 Теперь можете использовать режим **"Анализ изображения"** для обработки парковки.
                        """)
                        
                    else:
                        st.error("""
                        ❌ Калибровка не удалась
                        
                        **Возможные причины:**
                        - Плохое качество изображения
                        - Отсутствие четкой разметки
                        - Недостаточное освещение
                        
                        Попробуйте другое изображение.
                        """)
                
                except Exception as e:
                    st.error(f"❌ Ошибка при калибровке: {str(e)}")
                    import traceback
                    with st.expander("Показать детали ошибки"):
                        st.code(traceback.format_exc())
                
                finally:
                    progress_bar.empty()
                    status_text.empty()
        
        else:
            st.warning("👆 Загрузите изображение для начала калибровки")
    
    # ============ РЕЖИМ: АНАЛИЗ ИЗОБРАЖЕНИЯ ============
    elif mode == "📸 Анализ изображения":
        st.markdown("## 📸 Загрузите изображение парковки")
        
        uploaded_file = st.file_uploader(
            "Перетащите изображение сюда или нажмите для выбора",
            type=["jpg", "jpeg", "png"],
            help="Поддерживаемые форматы: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Загрузка изображения
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Конвертация RGB -> BGR
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Превью
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📥 Исходное изображение")
                st.image(image, use_container_width=True)
                st.info(f"📐 Размер: {image.size[0]}x{image.size[1]} px")
            
            # Кнопка анализа
            analyze_btn = st.button(
                "🚀 ЗАПУСТИТЬ АНАЛИЗ",
                type="primary",
                use_container_width=True,
                help="Начать детекцию и анализ парковки"
            )
            
            if analyze_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 Инициализация пайплайна...")
                    progress_bar.progress(20)
                    pipeline = load_pipeline(conf_threshold)
                    
                    if pipeline is None:
                        st.error("❌ Не удалось загрузить пайплайн")
                        return
                    
                    # Загрузка ранее обнаруженных мест
                    if pipeline.auto_detector:
                        pipeline.auto_detector.load_spaces()
                    
                    status_text.text("🔍 Детекция транспортных средств...")
                    progress_bar.progress(50)
                    
                    # Обработка
                    result = pipeline.process(
                        image_np,
                        draw_zones=draw_zones,
                        draw_detections=draw_detections,
                        draw_spaces=draw_spaces
                    )
                    
                    status_text.text("📊 Анализ данных...")
                    progress_bar.progress(80)
                    
                    result_rgb = cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB)
                    
                    status_text.text("✅ Готово!")
                    progress_bar.progress(100)
                    
                    # Результат
                    with col2:
                        st.markdown("### 📤 Результат анализа")
                        st.image(result_rgb, use_container_width=True)
                        st.success(f"✅ Обнаружено {len(result['detections'])} транспортных средств")
                    
                    # Очистка прогресса
                    import time
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.markdown("---")
                    
                    # ============ ПАНЕЛЬ МЕТРИК ============
                    st.markdown("## 📊 ОБЩАЯ СТАТИСТИКА")
                    
                    summary = result['summary']
                    
                    # Проверяем наличие данных о парковочных местах
                    has_parking_spaces = 'parking_spaces' in summary and summary['parking_spaces']
                    
                    if has_parking_spaces:
                        # С парковочными местами - 7 метрик
                        metric_cols = st.columns(7)
                        
                        space_info = summary['parking_spaces']
                        
                        metrics_data = [
                            ("🚗", "Всего ТС", summary['total_vehicles'], ""),
                            ("📈", "Загруженность зон", f"{summary['total_occupancy']}%", ""),
                            ("🅿️", "Всего парковочных мест", space_info['total'], ""),
                            ("✅", "Свободных мест", space_info['free'], ""),
                            ("🚗", "Занято мест", space_info['occupied'], ""),
                            ("📊", "Загруженность парковки", f"{space_info['occupancy_rate']}%", ""),
                            ("🔴", "Критичных зон", summary['critical_zones'], "")
                        ]
                    else:
                        # Без парковочных мест - 5 метрик
                        metric_cols = st.columns(5)
                        
                        metrics_data = [
                            ("🚗", "Всего ТС", summary['total_vehicles'], ""),
                            ("📈", "Загруженность", f"{summary['total_occupancy']}%", ""),
                            ("🅿️", "Зон", summary['zones_count'], ""),
                            ("🔴", "Критичных", summary['critical_zones'], ""),
                            ("🟡", "Предупреждений", summary['warning_zones'], "")
                        ]
                    
                    # Отображение метрик
                    for col, (icon, label, value, delta) in zip(metric_cols, metrics_data):
                        with col:
                            st.metric(
                                label=f"{icon} {label}",
                                value=value,
                                delta=delta if delta else None
                            )
                    
                    st.markdown("---")
                    
                    # ============ БЛОК ПАРКОВОЧНЫХ МЕСТ ============
                    if has_parking_spaces:
                        st.markdown("## 🅿️ ИНФОРМАЦИЯ О ПАРКОВОЧНЫХ МЕСТАХ")
                        
                        space_info = summary['parking_spaces']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                                <h2 style='margin: 0;'>🅿️</h2>
                                <h3 style='margin: 10px 0;'>{space_info['total']}</h3>
                                <p style='margin: 0;'>Всего мест</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #00cc66 0%, #00aa55 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                                <h2 style='margin: 0;'>✅</h2>
                                <h3 style='margin: 10px 0;'>{space_info['free']}</h3>
                                <p style='margin: 0;'>Свободно</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                                <h2 style='margin: 0;'>🚗</h2>
                                <h3 style='margin: 10px 0;'>{space_info['occupied']}</h3>
                                <p style='margin: 0;'>Занято</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            # Цвет зависит от загруженности
                            if space_info['occupancy_rate'] >= 90:
                                bg_color = "linear-gradient(135deg, #ff4444 0%, #cc0000 100%)"
                            elif space_info['occupancy_rate'] >= 70:
                                bg_color = "linear-gradient(135deg, #ffaa00 0%, #ff8800 100%)"
                            else:
                                bg_color = "linear-gradient(135deg, #00cc66 0%, #00aa55 100%)"
                            
                            st.markdown(f"""
                            <div style='background: {bg_color}; 
                                        padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                                <h2 style='margin: 0;'>📊</h2>
                                <h3 style='margin: 10px 0;'>{space_info['occupancy_rate']}%</h3>
                                <p style='margin: 0;'>Загруженность</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Прогресс-бар
                        st.markdown("### Заполненность парковки")
                        progress_percentage = space_info['occupancy_rate'] / 100
                        st.progress(progress_percentage)
                        
                        st.markdown(f"""
                        <p style='text-align: center; color: #666;'>
                            {space_info['occupied']} из {space_info['total']} мест занято 
                            ({space_info['free']} свободно)
                        </p>
                        """, unsafe_allow_html=True)
                        
                        # График парковочных мест
                        if show_charts:
                            fig_spaces = create_parking_spaces_chart(space_info)
                            if fig_spaces:
                                st.plotly_chart(fig_spaces, use_container_width=True)
                        
                        st.markdown("---")
                    
                    # ============ ГРАФИКИ И ВИЗУАЛИЗАЦИЯ ============
                    if show_charts:
                        st.markdown("## 📈 ВИЗУАЛИЗАЦИЯ ДАННЫХ")
                        
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            # График загруженности
                            fig_occupancy = create_occupancy_chart(result['density'])
                            st.plotly_chart(fig_occupancy, use_container_width=True)
                        
                        with chart_col2:
                            # Круговая диаграмма типов
                            fig_types = create_vehicle_types_chart(summary)
                            if fig_types:
                                st.plotly_chart(fig_types, use_container_width=True)
                            else:
                                st.info("Типы ТС не обнаружены")
                        
                        # Индикатор загруженности
                        st.markdown("### 🎯 Индикатор общей загруженности")
                        fig_gauge = create_capacity_gauge(summary)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        st.markdown("---")
                    
                    # ============ ДЕТАЛЬНАЯ СТАТИСТИКА ПО ЗОНАМ ============
                    st.markdown("## 🗺️ ДЕТАЛЬНАЯ СТАТИСТИКА ПО ЗОНАМ")
                    
                    # Таблица сравнения
                    with st.expander("📋 Таблица сравнения зон", expanded=True):
                        df_zones = create_zone_comparison_table(result['density'])
                        st.dataframe(
                            df_zones,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # Карточки зон
                    density = result['density']
                    zone_cols = st.columns(2)
                    
                    for idx, (zone_name, data) in enumerate(sorted(density.items())):
                        col = zone_cols[idx % 2]
                        
                        with col:
                            level_class = {
                                'critical': 'critical',
                                'warning': 'warning',
                                'normal': 'normal',
                                'empty': 'empty',
                                'busy': 'normal'
                            }.get(data['level'], 'normal')
                            
                            level_icon = {
                                'critical': '🔴',
                                'warning': '🟡',
                                'busy': '🟠',
                                'normal': '🟢',
                                'empty': '⚪'
                            }.get(data['level'], '❓')
                            
                            progress_width = min(data['occupancy'], 100)
                            
                            st.markdown(f"""
                            <div class="zone-card {level_class}">
                                <h3>{level_icon} {zone_name}</h3>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: {progress_width}%">
                                        {data['occupancy']}%
                                    </div>
                                </div>
                                <p><strong>🚗 Транспорта:</strong> {data['vehicles']} шт</p>
                                <p><strong>📊 Занято места:</strong> {data['space_used']:.1f}/{data['capacity']}</p>
                                <p><strong>✅ Доступно:</strong> {data['available']:.1f} мест</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if data['by_type']:
                                type_badges = ""
                                for vtype, count in data['by_type'].items():
                                    type_badges += f'<span class="badge badge-normal">{vtype}: {count}</span>'
                                st.markdown(f"**Типы ТС:** {type_badges}", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # ============ РЕКОМЕНДАЦИИ ============
                    st.markdown("## 💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ")
                    
                    recommendations = result['recommendations']
                    
                    priority_filter = st.multiselect(
                        "🔍 Фильтр по приоритету:",
                        options=[1, 2, 3, 4, 5],
                        default=[1, 2, 3],
                        format_func=lambda x: {
                            1: "🔴 Критичные",
                            2: "🟡 Важные",
                            3: "🟠 Информационные",
                            4: "🟢 Нормальные",
                            5: "✅ Резервные"
                        }[x]
                    )
                    
                    filtered_recs = [r for r in recommendations if r['priority'] in priority_filter]
                    
                    if filtered_recs:
                        for rec in filtered_recs:
                            with st.expander(
                                f"{rec['icon']} {rec['title']}", 
                                expanded=(rec['priority'] <= 2)
                            ):
                                st.markdown(f"**{rec['message']}**")
                                st.markdown(f"_{rec['details']}_")
                                
                                if rec['level'] == 'critical':
                                    st.error(rec['action'])
                                elif rec['level'] == 'warning':
                                    st.warning(rec['action'])
                                else:
                                    st.info(rec['action'])
                                
                                st.caption(f"🎯 Приоритет: {rec['priority']} | 💥 Влияние: {rec['impact']}")
                    else:
                        st.success("✅ Нет рекомендаций для выбранных приоритетов")
                    
                    st.markdown("---")
                    
                    # ============ ЭКСПОРТ ============
                    st.markdown("## 💾 ЭКСПОРТ РЕЗУЛЬТАТОВ")
                    
                    export_col1, export_col2, export_col3 = st.columns(3)
                    
                    output_dir = Path('outputs')
                    output_dir.mkdir(exist_ok=True)
                    
                    with export_col1:
                        output_path = output_dir / 'result_annotated.jpg'
                        cv2.imwrite(str(output_path), result['annotated'])
                        
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="📥 Скачать изображение",
                                data=f,
                                file_name=f"parking_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                    
                    with export_col2:
                        recommender = Recommender()
                        report = recommender.format_report(recommendations)
                        
                        st.download_button(
                            label="📄 Скачать отчёт (TXT)",
                            data=report,
                            file_name=f"parking_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with export_col3:
                        df_zones = create_zone_comparison_table(result['density'])
                        csv = df_zones.to_csv(index=False, encoding='utf-8-sig')
                        
                        st.download_button(
                            label="📊 Скачать таблицу (CSV)",
                            data=csv,
                            file_name=f"zones_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                
                finally:
                    if 'progress_bar' in locals():
                        progress_bar.empty()
                    if 'status_text' in locals():
                        status_text.empty()
        
        else:
            st.info("👆 **Загрузите изображение парковки для начала анализа**")
            
            st.markdown("### 🖼️ Примеры изображений")
            st.markdown("""
            Вы можете использовать:
            - Фото парковки с высоты (drone view)
            - Кадры с камер видеонаблюдения
            - Спутниковые снимки парковок
            """)
    
    # ============ ОСТАЛЬНЫЕ РЕЖИМЫ ============
    elif mode == "🎥 Обработка видео":
        st.markdown("## 🎥 Обработка видео парковки")
        st.info("Функция в разработке...")
    
    elif mode == "📊 Демо":
        st.markdown("## 📊 Демонстрационный режим")
        st.info("Функция в разработке...")
    
    elif mode == "ℹ️ О системе":
        st.markdown("## ℹ️ О системе Parking AI")
        st.markdown("""
        ### Возможности системы
        
        - **Детекция транспорта**: YOLOv8m с точностью >95%
        - **Анализ зон**: 4 зоны парковки
        - **Автоматическое определение мест**: Калибровка по изображению
        - **Статистика**: Подсчет ТС и парковочных мест
        - **Рекомендации**: Интеллектуальная система советов
        
        ### Технологии
        
        - Python 3.8+
        - YOLOv8m (Ultralytics)
        - OpenCV
        - Streamlit
        - Scikit-learn (DBSCAN)
        """)

# ============ ЗАПУСК ============
if __name__ == "__main__":
    main()
