"""
Менеджер зон парковки (Windows) - ИСПРАВЛЕНО
"""

import yaml
import cv2
import numpy as np
import os

class ParkingZones:  # ← ИСПРАВЛЕНО: переименован из ZoneManager
    """Управление зонами парковки"""
    
    def __init__(self, config_path=None):
        """
        Инициализация менеджера зон
        
        Args:
            config_path: путь к YAML конфигу зон (опционально)
        """
        if config_path and os.path.exists(config_path):
            print(f"Загрузка зон из {config_path}...")
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.zones = data['zones']
            print(f"Загружено {len(self.zones)} зон")
        else:
            print("Использование зон по умолчанию...")
            # Зоны по умолчанию для изображения 1280x960
            self.zones = {
                'Zone_A': {
                    'coords': [0, 0, 640, 480],
                    'capacity': 20,
                    'type': 'standard',
                    'color': [0, 255, 0]
                },
                'Zone_B': {
                    'coords': [640, 0, 1280, 480],
                    'capacity': 15,
                    'type': 'standard',
                    'color': [255, 0, 0]
                },
                'Zone_C': {
                    'coords': [0, 480, 640, 960],
                    'capacity': 18,
                    'type': 'large',
                    'color': [0, 0, 255]
                },
                'Zone_D': {
                    'coords': [640, 480, 1280, 960],
                    'capacity': 12,
                    'type': 'standard',
                    'color': [0, 255, 255]
                }
            }
            print(f"Создано {len(self.zones)} зон по умолчанию")
    
    def assign_zone(self, detection):
        """
        Определить зону для детекции
        
        Args:
            detection: словарь с ключом 'center' (x, y)
        
        Returns:
            имя зоны или 'Unknown'
        """
        cx, cy = detection['center']
        
        for zone_name, zone_info in self.zones.items():
            x1, y1, x2, y2 = zone_info['coords']
            
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return zone_name
        
        return 'Unknown'
    
    def assign_vehicles(self, detections):  # ← НОВЫЙ МЕТОД
        """
        Распределить все детекции по зонам
        
        Args:
            detections: список детекций
        
        Returns:
            dict: {zone_name: [detections]}
        """
        zones_assignment = {zone: [] for zone in self.zones.keys()}
        zones_assignment['Unknown'] = []
        
        for det in detections:
            zone = self.assign_zone(det)
            zones_assignment[zone].append(det)
        
        # Удаляем пустую зону Unknown если она пустая
        if not zones_assignment['Unknown']:
            del zones_assignment['Unknown']
        
        return zones_assignment
    
    def get_zone_info(self, zone_name):
        """Получить информацию о зоне"""
        return self.zones.get(zone_name, None)
    
    def draw(self, image, zones_assignment=None):  # ← ИСПРАВЛЕНО: переименован из draw_zones
        """
        Нарисовать зоны на изображении
        
        Args:
            image: numpy array (BGR)
            zones_assignment: dict с распределением ТС по зонам (опционально)
        
        Returns:
            изображение с нарисованными зонами
        """
        overlay = image.copy()
        
        for zone_name, zone_info in self.zones.items():
            x1, y1, x2, y2 = zone_info['coords']
            color = tuple(zone_info['color'])
            
            # Прямоугольник зоны
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            
            # Полупрозрачная заливка
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Текст с информацией
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Если есть информация о распределении, показываем количество
            if zones_assignment and zone_name in zones_assignment:
                count = len(zones_assignment[zone_name])
                text = f"{zone_name}: {count} ТС ({zone_info['capacity']} мест)"
            else:
                text = f"{zone_name} ({zone_info['capacity']} мест)"
            
            text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
            
            # Фон для текста
            text_x = x1 + 10
            text_y = y1 + 40
            cv2.rectangle(overlay, 
                         (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         (0, 0, 0), -1)
            
            # Текст
            cv2.putText(overlay, text, (text_x, text_y),
                       font, 0.8, color, 2)
        
        # Смешивание (прозрачность 30%)
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        return result
    
    def get_all_zones(self):
        """Получить все зоны"""
        return self.zones.copy()


# ============ ТЕСТ ============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ТЕСТ PARKING ZONES")
    print("="*60 + "\n")
    
    zones = ParkingZones()
    
    # Тестовые детекции
    test_detections = [
        {'center': (300, 200), 'class': 'car'},
        {'center': (900, 200), 'class': 'truck'},
        {'center': (300, 700), 'class': 'bus'},
    ]
    
    # Распределение
    assignment = zones.assign_vehicles(test_detections)
    
    print("Распределение по зонам:")
    for zone, vehicles in assignment.items():
        print(f"  {zone}: {len(vehicles)} ТС")
    
    # Визуализация
    test_img = np.ones((960, 1280, 3), dtype=np.uint8) * 50
    result = zones.draw(test_img, assignment)
    
    os.makedirs('outputs', exist_ok=True)
    cv2.imwrite('outputs\\zones_test.jpg', result)
    print("\n✅ Тест завершен")
