"""
Полный пайплайн обработки парковки (Windows)
ВЕРСИЯ С АВТОМАТИЧЕСКИМ ДЕТЕКТОРОМ ПАРКОВОЧНЫХ МЕСТ
"""

import cv2
import numpy as np
from detector import VehicleDetector
from zones import ParkingZones
from analyzer import DensityAnalyzer
from recommender import Recommender
from automatic_parking_detector import AutomaticParkingDetector

class ParkingPipeline:
    """Полный пайплайн: Детекция -> Зоны -> Анализ -> Рекомендации -> Парковочные места"""
    
    def __init__(self, conf_threshold=0.25, use_auto_detection=True):
        """
        Инициализация пайплайна
        
        Args:
            conf_threshold: порог уверенности для детектора (0.25 для YOLOv8m)
            use_auto_detection: использовать автоматический детектор парковочных мест
        """
        print(f"\n{'='*60}")
        print(f"  ИНИЦИАЛИЗАЦИЯ ПАЙПЛАЙНА")
        print(f"{'='*60}\n")
        
        # 1. Детектор транспорта (YOLOv8m с оптимальным порогом)
        print("1. Загрузка детектора...")
        self.detector = VehicleDetector(
            model='yolov8m.pt',
            conf_threshold=conf_threshold
        )
        
        # 2. Зоны парковки
        print("2. Инициализация зон парковки...")
        self.zones = ParkingZones()
        
        # 3. Анализатор плотности
        print("3. Инициализация анализатора...")
        self.analyzer = DensityAnalyzer()
        
        # 4. Система рекомендаций
        print("4. Инициализация рекомендаций...")
        self.recommender = Recommender()
        
        # 5. Автоматический детектор парковочных мест (НОВОЕ)
        if use_auto_detection:
            print("5. Инициализация автоматического детектора мест...")
            self.auto_detector = AutomaticParkingDetector()
        else:
            self.auto_detector = None
        
        print(f"\n{'='*60}")
        print(f"  ✅ ПАЙПЛАЙН ГОТОВ")
        print(f"{'='*60}\n")
    
    def calibrate_parking_spaces(self, calibration_image):
        """
        Калибровка системы - автоматическое обнаружение мест
        
        Args:
            calibration_image: эталонное изображение пустой парковки
        
        Returns:
            bool: True если калибровка успешна
        """
        if not self.auto_detector:
            print("❌ Автоматический детектор не инициализирован")
            return False
        
        print("\n🔧 КАЛИБРОВКА СИСТЕМЫ...")
        
        spaces = self.auto_detector.auto_detect_spaces(calibration_image, visualize=True)
        
        if spaces:
            self.auto_detector.save_spaces()
            print(f"✅ Калибровка завершена: {len(spaces)} мест")
            return True
        else:
            print("❌ Места не обнаружены")
            return False
    
    def process(self, image, draw_zones=True, draw_detections=True, draw_spaces=True):
        """
        Полная обработка изображения парковки с автоматической детекцией мест
        
        Args:
            image: входное изображение BGR
            draw_zones: рисовать зоны на изображении
            draw_detections: рисовать детекции
            draw_spaces: рисовать парковочные места
        
        Returns:
            dict: результаты обработки
        """
        print(f"\n{'='*60}")
        print(f"  ОБРАБОТКА ИЗОБРАЖЕНИЯ")
        print(f"{'='*60}\n")
        
        result_img = image.copy()
        
        # Шаг 1: Детекция транспорта
        print("Шаг 1/5: Детекция транспортных средств...")
        detections = self.detector.detect(image)
        print(f"   ✅ Обнаружено: {len(detections)} ТС")
        
        # Шаг 2: Распределение по зонам
        print("\nШаг 2/5: Распределение по зонам...")
        zones_assignment = self.zones.assign_vehicles(detections)
        
        for zone_name, vehicles in zones_assignment.items():
            print(f"   📍 {zone_name}: {len(vehicles)} ТС")
        
        # Шаг 3: Анализ плотности
        print("\nШаг 3/5: Анализ плотности...")
        zones_info = self.zones.get_all_zones()
        density_data = self.analyzer.analyze(zones_assignment, zones_info)
        
        for zone_name, data in density_data.items():
            status_icon = {
                'critical': '🔴',
                'warning': '🟡',
                'busy': '🟠',
                'normal': '🟢',
                'empty': '⚪'
            }.get(data['level'], '❓')
            
            print(f"   {status_icon} {zone_name}: {data['occupancy']}% загружено")
        
        # Шаг 4: Рекомендации
        print("\nШаг 4/5: Генерация рекомендаций...")
        recommendations = self.recommender.generate(density_data)
        print(f"   ✅ Сгенерировано {len(recommendations)} рекомендаций")
        
        # Шаг 5: Автоматическая детекция парковочных мест (НОВОЕ)
        space_occupancy = None
        if self.auto_detector and self.auto_detector.parking_spaces:
            print("\nШаг 5/5: Проверка занятости парковочных мест...")
            space_occupancy = self.auto_detector.check_occupancy(image, detections)
            
            if space_occupancy:
                print(f"   🅿️ Всего мест: {space_occupancy['total_spaces']}")
                print(f"   ✅ Свободно: {space_occupancy['free']}")
                print(f"   🚗 Занято: {space_occupancy['occupied']}")
                print(f"   📊 Загруженность: {space_occupancy['occupancy_rate']}%")
        
        # Визуализация
        if draw_zones:
            result_img = self.zones.draw(result_img, zones_assignment)
        
        if draw_detections:
            result_img = self._draw_detections(result_img, detections)
        
        if draw_spaces and space_occupancy:
            result_img = self.auto_detector.draw_spaces(
                result_img, space_occupancy['spaces']
            )
            result_img = self.auto_detector.draw_info_panel(
                result_img, space_occupancy
            )
        
        # Сводка
        summary = self._create_summary(detections, density_data, recommendations, space_occupancy)
        
        print(f"\n{'='*60}")
        print(f"  ✅ ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"{'='*60}\n")
        
        return {
            'annotated': result_img,
            'detections': detections,
            'zones': zones_assignment,
            'density': density_data,
            'recommendations': recommendations,
            'space_occupancy': space_occupancy,
            'summary': summary
        }
    
    def _draw_detections(self, image, detections):
        """
        Отрисовка детекций на изображении
        
        Args:
            image: исходное изображение
            detections: список детекций
        
        Returns:
            изображение с отрисованными детекциями
        """
        annotated = image.copy()
        
        # Цвета по типам ТС
        colors = {
            'car': (0, 255, 0),        # Зелёный
            'truck': (255, 165, 0),    # Оранжевый
            'bus': (0, 0, 255),        # Красный
            'motorcycle': (255, 0, 255) # Пурпурный
        }
        
        for det in detections:
            color = colors.get(det['class'], (255, 255, 255))
            
            # Рисуем бокс
            cv2.rectangle(annotated,
                         (det['x1'], det['y1']),
                         (det['x2'], det['y2']),
                         color, 2)
            
            # Метка с классом и уверенностью
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Размер текста
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Фон для текста
            cv2.rectangle(annotated,
                         (det['x1'], det['y1'] - h - 10),
                         (det['x1'] + w, det['y1']),
                         color, -1)
            
            # Текст
            cv2.putText(annotated, label,
                       (det['x1'], det['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
        
        return annotated
    
    def _create_summary(self, detections, density_data, recommendations, space_occupancy=None):
        """
        Создание сводной статистики с информацией о парковочных местах
        
        Args:
            detections: список детекций
            density_data: результаты анализа плотности
            recommendations: рекомендации
            space_occupancy: информация о парковочных местах
        
        Returns:
            dict со сводной статистикой
        """
        # Используем метод анализатора для получения статистики
        summary = self.analyzer.get_summary(density_data)
        
        # Добавляем количество рекомендаций
        summary['recommendations_count'] = len(recommendations)
        
        # Добавляем информацию о парковочных местах (НОВОЕ)
        if space_occupancy:
            summary['parking_spaces'] = {
                'total': space_occupancy['total_spaces'],
                'occupied': space_occupancy['occupied'],
                'free': space_occupancy['free'],
                'occupancy_rate': space_occupancy['occupancy_rate']
            }
        
        return summary
    
    def process_video(self, video_path, output_path, max_frames=None):
        """
        Обработка видео
        
        Args:
            video_path: путь к входному видео
            output_path: путь к выходному видео
            max_frames: максимальное количество кадров (None = все)
        """
        print(f"\n{'='*60}")
        print(f"  ОБРАБОТКА ВИДЕО")
        print(f"{'='*60}\n")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        # Параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Видео: {width}x{height} @ {fps} FPS")
        print(f"📊 Всего кадров: {total_frames}")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            print(f"⚙️  Обрабатываем: {total_frames} кадров\n")
        
        # Создание writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Не удалось создать выходной файл: {output_path}")
        
        frame_num = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret or (max_frames and frame_num >= max_frames):
                    break
                
                # Обработка кадра
                result = self.process(frame, draw_zones=True, draw_detections=True, draw_spaces=True)
                
                # Запись
                out.write(result['annotated'])
                
                frame_num += 1
                
                # Прогресс каждые 30 кадров
                if frame_num % 30 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"   Обработано: {frame_num}/{total_frames} ({progress:.1f}%)")
        
        finally:
            cap.release()
            out.release()
        
        print(f"\n✅ Видео сохранено: {output_path}")
        print(f"   Обработано кадров: {frame_num}")
        print(f"{'='*60}\n")
    
    def get_info(self):
        """
        Информация о пайплайне
        
        Returns:
            dict с информацией о компонентах
        """
        info = {
            'detector': self.detector.get_info(),
            'zones': len(self.zones.zones),
            'zone_names': list(self.zones.zones.keys())
        }
        
        # Добавляем информацию о парковочных местах
        if self.auto_detector and self.auto_detector.parking_spaces:
            info['parking_spaces'] = len(self.auto_detector.parking_spaces)
        
        return info


# ============ ТЕСТ ПАЙПЛАЙНА ============
if __name__ == "__main__":
    import os
    
    print("\n" + "="*60)
    print("  ТЕСТ ПОЛНОГО ПАЙПЛАЙНА С АВТОМАТИЧЕСКОЙ ДЕТЕКЦИЕЙ")
    print("="*60 + "\n")
    
    # Инициализация с автоматическим детектором
    pipeline = ParkingPipeline(conf_threshold=0.25, use_auto_detection=True)
    
    # Информация о пайплайне
    info = pipeline.get_info()
    print("\n📊 ИНФОРМАЦИЯ О ПАЙПЛАЙНЕ:")
    print(f"   Детектор: {info['detector']['device']}")
    print(f"   Зон: {info['zones']}")
    print(f"   Названия зон: {', '.join(info['zone_names'])}")
    if 'parking_spaces' in info:
        print(f"   Парковочных мест: {info['parking_spaces']}")
    print()
    
    # Тестовое изображение
    print("📸 Создание тестового изображения...")
    test_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
    
    # Добавляем "машины"
    for i in range(8):
        x = 200 + (i % 4) * 400
        y = 300 + (i // 4) * 400
        cv2.rectangle(test_img, (x, y), (x+200, y+150), (0, 0, 255), -1)
    
    # Калибровка (первый раз)
    print("\n🔧 КАЛИБРОВКА ПАРКОВОЧНЫХ МЕСТ...")
    calibration_success = pipeline.calibrate_parking_spaces(test_img)
    
    if calibration_success:
        print("✅ Калибровка выполнена успешно\n")
        
        # Обработка
        print("🚀 Запуск пайплайна с детекцией мест...\n")
        result = pipeline.process(test_img)
        
        # Результаты
        print("\n" + "="*60)
        print("  📊 РЕЗУЛЬТАТЫ")
        print("="*60)
        summary = result['summary']
        
        print(f"\n🚗 Всего ТС: {summary['total_vehicles']}")
        print(f"📈 Общая загруженность зон: {summary['total_occupancy']}%")
        print(f"🅿️  Зон: {summary['zones_count']}")
        print(f"🔴 Критичных зон: {summary['critical_zones']}")
        print(f"🟡 Предупреждений: {summary['warning_zones']}")
        print(f"💡 Рекомендаций: {summary['recommendations_count']}")
        
        # Информация о парковочных местах
        if 'parking_spaces' in summary:
            ps = summary['parking_spaces']
            print(f"\n🅿️  ПАРКОВОЧНЫЕ МЕСТА:")
            print(f"   Всего мест: {ps['total']}")
            print(f"   Занято: {ps['occupied']}")
            print(f"   Свободно: {ps['free']}")
            print(f"   Загруженность: {ps['occupancy_rate']}%")
        
        if summary['by_type']:
            print(f"\n🎨 По типам:")
            for vtype, count in summary['by_type'].items():
                print(f"   {vtype}: {count}")
        
        # Сохранение
        os.makedirs('outputs', exist_ok=True)
        output_file = 'outputs\\pipeline_test_with_spaces.jpg'
        cv2.imwrite(output_file, result['annotated'])
        print(f"\n💾 Результат сохранён: {output_file}")
        
        # Рекомендации (топ-3)
        if result['recommendations']:
            print(f"\n💡 ТОП-3 РЕКОМЕНДАЦИИ:")
            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"\n{i}. {rec['icon']} {rec['title']}")
                print(f"   {rec['message']}")
    else:
        print("❌ Калибровка не выполнена")
    
    print("\n" + "="*60)
    print("  ✅ ТЕСТ ЗАВЕРШЁН")
    print("="*60 + "\n")
