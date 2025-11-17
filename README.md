# parking-ai
# Parking AI - Интеллектуальная система анализа парковок

## Описание

Система для автоматического анализа парковочных мест с использованием YOLOv8, автоматической детекции парковочных мест и веб-интерфейса на Streamlit.

## Возможности

- Детекция транспортных средств с YOLOv8m
- Автоматическое обнаружение парковочных мест
- Анализ загруженности зон
- Интеллектуальные рекомендации
- Веб-интерфейс на Streamlit
- Поддержка видео

## Технологии

- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- Scikit-learn
- NumPy, Pandas, Plotly

## Установка

### 1. Клонировать репозиторий
```bash
git clone https://github.com/gvbelavin/parking-ai.git
cd parking-ai

2. Создать виртуальное окружение
Windows:

bash
python -m venv venv  
venv\Scripts\activate  
Linux/Mac:

bash
python3 -m venv venv  
source venv/bin/activate 

3. Установить зависимости
bash
pip install -r requirements.txt  
Основные зависимости:

ultralytics - YOLOv8
opencv-python - Обработка изображений
streamlit - Веб-интерфейс
scikit-learn - Машинное обучение
numpy, pandas, plotly - Анализ данных

4. Скачать веса YOLOv8
Веса загрузятся автоматически при первом запуске или вручную:

bash
# Автоматическая загрузка  
yolo detect predict model=yolov8m.pt  

# Или скачать вручную  
mkdir -p weights  
cd weights  
# Скачайте yolov8m.pt с https://github.com/ultralytics/assets/releases

-- ЗАПУСК --
Веб-интерфейс (Streamlit)
bash
streamlit run app.py  
Откроется браузер по адресу: http://localhost:8501

Анализ парковки
Выберите режим "Анализ изображения"
Загрузите изображение парковки
Нажмите "ЗАПУСТИТЬ АНАЛИЗ"
Просмотрите результаты:

<img width="1489" height="728" alt="image" src="https://github.com/user-attachments/assets/be1e94f0-f3c0-4c24-bea7-34192165bf4d" />

Детекция транспорта
Загруженность зон
Свободные/занятые места
Рекомендации

-- Авторы --
Даниил Шелестов
Александр Чернякин
Глеб Белавин
