"""
Автоматический детектор парковочных мест на основе разметки
Без ручной разметки - полностью автоматическое обнаружение
"""

import cv2
import numpy as np
from pathlib import Path
import pickle
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import json

class AutomaticParkingDetector:
    """
    Автоматическое обнаружение парковочных мест на основе:
    - Детекции линий разметки
    - Геометрического анализа
    - Кластеризации пространства
    """
    
    def __init__(self, config_path='config/auto_parking_config.json'):
        """
        Инициализация автоматического детектора
        
        Args:
            config_path: путь к конфигурации
        """
        self.config_path = Path(config_path)
        self.parking_spaces = []
        
        # Параметры детекции линий
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 80
        self.hough_min_line_length = 100
        self.hough_max_line_gap = 10
        
        # Параметры парковочных мест
        self.typical_space_width = 250  # пикселей (примерно)
        self.typical_space_height = 500  # пикселей (примерно)
        self.min_space_width = 150
        self.max_space_width = 400
        
        # Параметры кластеризации
        self.dbscan_eps = 30
        self.dbscan_min_samples = 2
        
        self.load_config()
    
    def load_config(self):
        """Загрузка конфигурации из файла"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.canny_low = config.get('canny_low', self.canny_low)
                    self.canny_high = config.get('canny_high', self.canny_high)
                    self.hough_threshold = config.get('hough_threshold', self.hough_threshold)
                    self.typical_space_width = config.get('typical_space_width', self.typical_space_width)
                    self.typical_space_height = config.get('typical_space_height', self.typical_space_height)
                print(f"✅ Конфигурация загружена из {self.config_path}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки конфигурации: {e}")
    
    def save_config(self):
        """Сохранение текущей конфигурации"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            'canny_low': self.canny_low,
            'canny_high': self.canny_high,
            'hough_threshold': self.hough_threshold,
            'hough_min_line_length': self.hough_min_line_length,
            'hough_max_line_gap': self.hough_max_line_gap,
            'typical_space_width': self.typical_space_width,
            'typical_space_height': self.typical_space_height,
            'dbscan_eps': self.dbscan_eps,
            'dbscan_min_samples': self.dbscan_min_samples
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✅ Конфигурация сохранена в {self.config_path}")
    
    def detect_parking_lines(self, image):
        """
        Детекция линий парковочной разметки
        
        Args:
            image: изображение парковки
        
        Returns:
            list: список обнаруженных линий [(x1, y1, x2, y2), ...]
        """
        # Конвертация в grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применение Gaussian blur для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection (Canny)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Морфологические операции для улучшения линий
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Детекция линий (Hough Transform)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        if lines is None:
            return []
        
        # Преобразование формата
        detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append((x1, y1, x2, y2))
        
        return detected_lines
    
    def filter_vertical_lines(self, lines, angle_threshold=15):
        """
        Фильтрация вертикальных линий (границы парковочных мест)
        
        Args:
            lines: список линий
            angle_threshold: максимальное отклонение от вертикали (градусы)
        
        Returns:
            list: отфильтрованные вертикальные линии
        """
        vertical_lines = []
        
        for x1, y1, x2, y2 in lines:
            # Вычисление угла линии
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Проверка на вертикальность (угол близок к 90°)
            if 90 - angle_threshold <= angle <= 90 + angle_threshold:
                vertical_lines.append((x1, y1, x2, y2))
        
        return vertical_lines
    
    def filter_horizontal_lines(self, lines, angle_threshold=15):
        """
        Фильтрация горизонтальных линий (границы рядов)
        
        Args:
            lines: список линий
            angle_threshold: максимальное отклонение от горизонтали (градусы)
        
        Returns:
            list: отфильтрованные горизонтальные линии
        """
        horizontal_lines = []
        
        for x1, y1, x2, y2 in lines:
            # Вычисление угла линии
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Проверка на горизонтальность (угол близок к 0° или 180°)
            if angle <= angle_threshold or angle >= 180 - angle_threshold:
                horizontal_lines.append((x1, y1, x2, y2))
        
        return horizontal_lines
    
    def cluster_lines(self, lines, orientation='vertical'):
        """
        Кластеризация линий для определения отдельных границ мест
        
        Args:
            lines: список линий
            orientation: 'vertical' или 'horizontal'
        
        Returns:
            list: кластеризованные линии (средние значения)
        """
        if not lines:
            return []
        
        # Выбор координаты для кластеризации
        if orientation == 'vertical':
            # Используем X координату (среднюю)
            coords = np.array([[(x1 + x2) / 2] for x1, y1, x2, y2 in lines])
        else:
            # Используем Y координату (среднюю)
            coords = np.array([[(y1 + y2) / 2] for x1, y1, x2, y2 in lines])
        
        # DBSCAN кластеризация
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = clustering.fit_predict(coords)
        
        # Группировка по кластерам
        clustered_lines = []
        for label in set(labels):
            if label == -1:  # Шум
                continue
            
            cluster_indices = np.where(labels == label)[0]
            cluster_lines = [lines[i] for i in cluster_indices]
            
            # Вычисление средней линии кластера
            if orientation == 'vertical':
                avg_x = np.mean([[(x1 + x2) / 2] for x1, y1, x2, y2 in cluster_lines])
                min_y = min([min(y1, y2) for x1, y1, x2, y2 in cluster_lines])
                max_y = max([max(y1, y2) for x1, y1, x2, y2 in cluster_lines])
                clustered_lines.append((int(avg_x), int(min_y), int(avg_x), int(max_y)))
            else:
                avg_y = np.mean([[(y1 + y2) / 2] for x1, y1, x2, y2 in cluster_lines])
                min_x = min([min(x1, x2) for x1, y1, x2, y2 in cluster_lines])
                max_x = max([max(x1, x2) for x1, y1, x2, y2 in cluster_lines])
                clustered_lines.append((int(min_x), int(avg_y), int(max_x), int(avg_y)))
        
        return clustered_lines
    
    def generate_parking_spaces(self, vertical_lines, horizontal_lines, image_shape):
        """
        Генерация парковочных мест на основе вертикальных и горизонтальных линий
        
        Args:
            vertical_lines: вертикальные границы
            horizontal_lines: горизонтальные границы
            image_shape: размер изображения (height, width)
        
        Returns:
            list: список парковочных мест [(x, y, width, height), ...]
        """
        parking_spaces = []
        
        # Сортировка линий
        vertical_lines = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
        horizontal_lines = sorted(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
        
        height, width = image_shape[:2]
        
        # Если нет горизонтальных линий, используем границы изображения
        if not horizontal_lines:
            horizontal_lines = [(0, 0, width, 0), (0, height, width, height)]
        
        # Генерация мест на пересечениях
        for i in range(len(vertical_lines) - 1):
            x1_line = vertical_lines[i]
            x2_line = vertical_lines[i + 1]
            
            x1 = int((x1_line[0] + x1_line[2]) / 2)
            x2 = int((x2_line[0] + x2_line[2]) / 2)
            
            space_width = x2 - x1
            
            # Проверка на разумную ширину места
            if not (self.min_space_width <= space_width <= self.max_space_width):
                continue
            
            for j in range(len(horizontal_lines) - 1):
                y1_line = horizontal_lines[j]
                y2_line = horizontal_lines[j + 1]
                
                y1 = int((y1_line[1] + y1_line[3]) / 2)
                y2 = int((y2_line[1] + y2_line[3]) / 2)
                
                space_height = y2 - y1
                
                # Проверка на разумную высоту
                if space_height < 100:
                    continue
                
                parking_spaces.append({
                    'x': x1,
                    'y': y1,
                    'width': space_width,
                    'height': space_height,
                    'center': (x1 + space_width // 2, y1 + space_height // 2)
                })
        
        return parking_spaces
    
    def auto_detect_spaces(self, image, visualize=False):
        """
        Автоматическое обнаружение всех парковочных мест
        
        Args:
            image: изображение парковки
            visualize: показать промежуточные результаты
        
        Returns:
            list: список обнаруженных парковочных мест
        """
        print("\n" + "="*60)
        print("  АВТОМАТИЧЕСКОЕ ОБНАРУЖЕНИЕ ПАРКОВОЧНЫХ МЕСТ")
        print("="*60 + "\n")
        
        # Шаг 1: Детекция всех линий
        print("1. Детекция линий разметки...")
        all_lines = self.detect_parking_lines(image)
        print(f"   Обнаружено линий: {len(all_lines)}")
        
        if not all_lines:
            print("❌ Линии не обнаружены")
            return []
        
        # Шаг 2: Фильтрация вертикальных линий
        print("\n2. Фильтрация вертикальных линий...")
        vertical_lines = self.filter_vertical_lines(all_lines)
        print(f"   Вертикальных линий: {len(vertical_lines)}")
        
        # Шаг 3: Кластеризация вертикальных линий
        print("\n3. Кластеризация вертикальных линий...")
        clustered_vertical = self.cluster_lines(vertical_lines, 'vertical')
        print(f"   Уникальных границ: {len(clustered_vertical)}")
        
        # Шаг 4: Фильтрация горизонтальных линий
        print("\n4. Фильтрация горизонтальных линий...")
        horizontal_lines = self.filter_horizontal_lines(all_lines)
        print(f"   Горизонтальных линий: {len(horizontal_lines)}")
        
        # Шаг 5: Кластеризация горизонтальных линий
        print("\n5. Кластеризация горизонтальных линий...")
        clustered_horizontal = self.cluster_lines(horizontal_lines, 'horizontal')
        print(f"   Уникальных рядов: {len(clustered_horizontal)}")
        
        # Шаг 6: Генерация парковочных мест
        print("\n6. Генерация парковочных мест...")
        self.parking_spaces = self.generate_parking_spaces(
            clustered_vertical,
            clustered_horizontal,
            image.shape
        )
        print(f"   Обнаружено мест: {len(self.parking_spaces)}")
        
        # Визуализация
        if visualize:
            self._visualize_detection(
                image,
                all_lines,
                clustered_vertical,
                clustered_horizontal,
                self.parking_spaces
            )
        
        print("\n" + "="*60)
        print("  ✅ ОБНАРУЖЕНИЕ ЗАВЕРШЕНО")
        print("="*60 + "\n")
        
        return self.parking_spaces
    
    def _visualize_detection(self, image, all_lines, vertical_lines, 
                           horizontal_lines, parking_spaces):
        """Визуализация процесса обнаружения"""
        
        # Все линии
        img_all_lines = image.copy()
        for x1, y1, x2, y2 in all_lines:
            cv2.line(img_all_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Кластеризованные линии
        img_clustered = image.copy()
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(img_clustered, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(img_clustered, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Парковочные места
        img_spaces = image.copy()
        for i, space in enumerate(parking_spaces):
            x, y, w, h = space['x'], space['y'], space['width'], space['height']
            cv2.rectangle(img_spaces, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img_spaces, str(i+1), (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Сохранение
        output_dir = Path('outputs/auto_detection')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_dir / 'step1_all_lines.jpg'), img_all_lines)
        cv2.imwrite(str(output_dir / 'step2_clustered_lines.jpg'), img_clustered)
        cv2.imwrite(str(output_dir / 'step3_parking_spaces.jpg'), img_spaces)
        
        print(f"\n📁 Визуализация сохранена в {output_dir}")
    
    def check_occupancy(self, image, vehicle_detections):
        """
        Проверка занятости автоматически обнаруженных мест
        
        Args:
            image: изображение
            vehicle_detections: детекции от YOLOv8
        
        Returns:
            dict: информация о занятости мест
        """
        if not self.parking_spaces:
            print("⚠️ Сначала выполните auto_detect_spaces()")
            return None
        
        occupied_count = 0
        free_count = 0
        spaces_status = []
        
        for i, space in enumerate(self.parking_spaces):
            x, y, w, h = space['x'], space['y'], space['width'], space['height']
            
            # Проверка пересечения с детекциями транспорта
            is_occupied = self._check_vehicle_overlap(x, y, w, h, vehicle_detections)
            
            if is_occupied:
                occupied_count += 1
                status = 'occupied'
            else:
                free_count += 1
                status = 'free'
            
            spaces_status.append({
                'id': i,
                'position': (x, y),
                'width': w,
                'height': h,
                'center': space['center'],
                'status': status,
                'has_vehicle': is_occupied
            })
        
        total_spaces = len(self.parking_spaces)
        occupancy_rate = (occupied_count / total_spaces * 100) if total_spaces > 0 else 0
        
        return {
            'total_spaces': total_spaces,
            'occupied': occupied_count,
            'free': free_count,
            'occupancy_rate': round(occupancy_rate, 1),
            'spaces': spaces_status
        }
    
    def _check_vehicle_overlap(self, space_x, space_y, space_w, space_h, vehicle_detections):
        """Проверка пересечения места с детекциями транспорта"""
        space_rect = (space_x, space_y, space_x + space_w, space_y + space_h)
        
        for vehicle in vehicle_detections:
            vehicle_rect = (vehicle['x1'], vehicle['y1'], vehicle['x2'], vehicle['y2'])
            
            if self._rectangles_overlap(space_rect, vehicle_rect):
                return True
        
        return False
    
    def _rectangles_overlap(self, rect1, rect2):
        """Проверка пересечения прямоугольников"""
        x1_min, y1_min, x1_max, y1_max = rect1
        x2_min, y2_min, x2_max, y2_max = rect2
        
        return not (x1_max < x2_min or x2_max < x1_min or 
                   y1_max < y2_min or y2_max < y1_min)
    
    def draw_spaces(self, image, spaces_status):
        """Отрисовка парковочных мест"""
        result = image.copy()
        
        for space in spaces_status:
            x, y = space['position']
            w, h = space['width'], space['height']
            status = space['status']
            
            # Цвет в зависимости от статуса
            if status == 'occupied':
                color = (0, 0, 255)  # Красный
            else:
                color = (0, 255, 0)  # Зеленый
            
            # Рисуем прямоугольник
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Номер места
            cv2.putText(result, str(space['id'] + 1), (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def save_spaces(self, filename='config/auto_detected_spaces.pkl'):
        """Сохранение обнаруженных мест"""
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.parking_spaces, f)
        
        print(f"✅ Сохранено {len(self.parking_spaces)} мест в {filepath}")
    
    def load_spaces(self, filename='config/auto_detected_spaces.pkl'):
        """Загрузка сохраненных мест"""
        filepath = Path(filename)
        
        if not filepath.exists():
            print(f"❌ Файл {filepath} не найден")
            return False
        
        with open(filepath, 'rb') as f:
            self.parking_spaces = pickle.load(f)
        
        print(f"✅ Загружено {len(self.parking_spaces)} мест из {filepath}")
        return True


# ============ ТЕСТ ============
if __name__ == "__main__":
    # Пример использования
    detector = AutomaticParkingDetector()
    
    # Путь к тестовому изображению
    test_image_path = 'test_images/parking_lot.jpg'
    
    if Path(test_image_path).exists():
        # Загрузка изображения
        image = cv2.imread(test_image_path)
        
        # Автоматическое обнаружение мест
        spaces = detector.auto_detect_spaces(image, visualize=True)
        
        # Сохранение результатов
        if spaces:
            detector.save_spaces()
            detector.save_config()
        
        print(f"\n📊 ИТОГО: обнаружено {len(spaces)} парковочных мест")
    else:
        print(f"❌ Изображение не найдено: {test_image_path}")
