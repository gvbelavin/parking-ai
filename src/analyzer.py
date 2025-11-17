"""
Анализатор плотности парковки - ИСПРАВЛЕНО
"""

from collections import defaultdict

class DensityAnalyzer:
    """Анализатор загруженности зон парковки"""
    
    def __init__(self):  # ← ИСПРАВЛЕНО: убран параметр zone_manager
        """Инициализация анализатора"""
        pass
    
    def analyze(self, zones_assignment, zones_info):  # ← ИСПРАВЛЕНО: добавлен параметр zones_info
        """
        Анализ детекций по зонам
        
        Args:
            zones_assignment: dict {zone_name: [detections]}
            zones_info: dict с информацией о зонах (из ParkingZones.get_all_zones())
        
        Returns:
            dict со статистикой по каждой зоне
        """
        results = {}
        
        for zone_name in zones_info.keys():
            # Получаем детекции для зоны
            vehicles = zones_assignment.get(zone_name, [])
            
            # Подсчет статистики
            count = len(vehicles)
            space_used = sum(v['size'] for v in vehicles)
            
            by_type = defaultdict(int)
            for v in vehicles:
                by_type[v['class']] += 1
            
            # Вместимость зоны
            capacity = zones_info[zone_name]['capacity']
            
            # Процент загруженности
            occupancy = (space_used / capacity) * 100 if capacity > 0 else 0
            
            results[zone_name] = {
                'vehicles': count,
                'space_used': round(space_used, 1),
                'capacity': capacity,
                'occupancy': round(min(occupancy, 100), 1),
                'by_type': dict(by_type),
                'level': self._get_level(occupancy),
                'available': max(0, capacity - space_used)
            }
        
        return results
    
    def _get_level(self, occupancy):
        """Определить уровень загруженности"""
        if occupancy >= 100:
            return 'critical'
        elif occupancy >= 85:
            return 'warning'
        elif occupancy >= 70:
            return 'busy'
        elif occupancy > 0:
            return 'normal'
        else:
            return 'empty'
    
    def get_summary(self, density_data):
        """Получить общую статистику"""
        total_vehicles = sum(d['vehicles'] for d in density_data.values())
        total_capacity = sum(d['capacity'] for d in density_data.values())
        total_used = sum(d['space_used'] for d in density_data.values())
        
        all_types = defaultdict(int)
        for d in density_data.values():
            for vtype, count in d['by_type'].items():
                all_types[vtype] += count
        
        return {
            'total_vehicles': total_vehicles,
            'total_capacity': total_capacity,
            'total_occupancy': round((total_used / total_capacity) * 100, 1) if total_capacity > 0 else 0,
            'zones_count': len(density_data),
            'by_type': dict(all_types),
            'critical_zones': sum(1 for d in density_data.values() if d['level'] == 'critical'),
            'warning_zones': sum(1 for d in density_data.values() if d['level'] == 'warning'),
            'empty_zones': sum(1 for d in density_data.values() if d['level'] == 'empty')
        }


# ============ ТЕСТ ============
if __name__ == "__main__":
    print("\nТЕСТ АНАЛИЗАТОРА\n")
    
    analyzer = DensityAnalyzer()
    
    # Тестовые данные
    zones_info = {
        'Zone_A': {'capacity': 20},
        'Zone_B': {'capacity': 15}
    }
    
    zones_assignment = {
        'Zone_A': [
            {'class': 'car', 'size': 1.0},
            {'class': 'truck', 'size': 2.5},
        ],
        'Zone_B': []
    }
    
    density = analyzer.analyze(zones_assignment, zones_info)
    
    print("Результаты:")
    for zone, data in density.items():
        print(f"  {zone}: {data['vehicles']} ТС, {data['occupancy']}%")
    
    print("\n✅ Тест завершен")
