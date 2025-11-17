"""
Генератор рекомендаций по оптимизации (Windows)
Время: 30 минут
"""

class Recommender:
    """Генератор рекомендаций по оптимизации парковки"""
    
    def __init__(self):
        """Инициализация генератора"""
        pass
    
    def generate(self, density_data):
        """
        Генерация рекомендаций на основе данных о плотности
        
        Args:
            density_data: результат DensityAnalyzer.analyze()
        
        Returns:
            список рекомендаций (отсортированный по приоритету)
        """
        recommendations = []
        
        # Анализ каждой зоны
        for zone, data in density_data.items():
            level = data['level']
            occupancy = data['occupancy']
            vehicles = data['vehicles']
            by_type = data['by_type']
            available = data['available']
            
            # КРИТИЧЕСКАЯ ПЕРЕГРУЗКА (>=100%)
            if level == 'critical':
                large_count = by_type.get('truck', 0) + by_type.get('bus', 0)
                
                recommendations.append({
                    'zone': zone,
                    'priority': 1,
                    'level': 'critical',
                    'icon': '🔴',
                    'title': f'ПЕРЕГРУЗКА: {zone}',
                    'message': f"Зона {zone} критически перегружена ({occupancy}%)! "
                              f"Обнаружено {vehicles} ТС при вместимости {data['capacity']} мест.",
                    'details': f"Крупногабаритных ТС: {large_count}",
                    'action': f"🚨 СРОЧНО: Перенаправить {large_count} крупногабаритных ТС в свободные зоны",
                    'impact': 'high'
                })
            
            # ПРЕДУПРЕЖДЕНИЕ (85-99%)
            elif level == 'warning':
                recommendations.append({
                    'zone': zone,
                    'priority': 2,
                    'level': 'warning',
                    'icon': '🟡',
                    'title': f'Предупреждение: {zone}',
                    'message': f"Зона {zone} почти заполнена ({occupancy}%).",
                    'details': f"Осталось ~{available:.1f} мест из {data['capacity']}",
                    'action': f"⚠️ Подготовить альтернативные зоны для перенаправления",
                    'impact': 'medium'
                })
            
            # ВЫСОКАЯ ЗАГРУЗКА (70-84%)
            elif level == 'busy':
                recommendations.append({
                    'zone': zone,
                    'priority': 3,
                    'level': 'info',
                    'icon': '🟠',
                    'title': f'Активная зона: {zone}',
                    'message': f"Зона {zone} активно используется ({occupancy}%).",
                    'details': f"{vehicles} ТС, доступно {available:.1f} мест",
                    'action': f"ℹ️ Мониторинг загруженности",
                    'impact': 'low'
                })
            
            # НОРМАЛЬНАЯ ЗАГРУЗКА (1-69%)
            elif level == 'normal':
                recommendations.append({
                    'zone': zone,
                    'priority': 4,
                    'level': 'success',
                    'icon': '🟢',
                    'title': f'Норма: {zone}',
                    'message': f"Зона {zone} в нормальном состоянии ({occupancy}%).",
                    'details': f"{vehicles} ТС, свободно {available:.1f} мест",
                    'action': f"✅ Без действий",
                    'impact': 'none'
                })
            
            # ПУСТАЯ ЗОНА (0%)
            elif level == 'empty':
                recommendations.append({
                    'zone': zone,
                    'priority': 5,
                    'level': 'success',
                    'icon': '✅',
                    'title': f'Свободна: {zone}',
                    'message': f"Зона {zone} полностью свободна.",
                    'details': f"Доступно {data['capacity']} мест",
                    'action': f"💡 Доступна для перенаправления из перегруженных зон",
                    'impact': 'positive'
                })
        
        # Межзональные рекомендации
        recommendations.extend(self._generate_cross_zone_recommendations(density_data))
        
        # Сортировка по приоритету
        return sorted(recommendations, key=lambda x: x['priority'])
    
    def _generate_cross_zone_recommendations(self, density_data):
        """Генерация рекомендаций по перераспределению между зонами"""
        cross_recs = []
        
        # Поиск перегруженных и свободных зон
        overloaded = [z for z, d in density_data.items() if d['level'] in ['critical', 'warning']]
        available = [z for z, d in density_data.items() if d['level'] in ['normal', 'empty'] and d['available'] > 2]
        
        if overloaded and available:
            overloaded_str = ', '.join(overloaded)
            available_str = ', '.join(available)
            
            cross_recs.append({
                'zone': 'Все зоны',
                'priority': 1,
                'level': 'info',
                'icon': '🔄',
                'title': 'Перераспределение нагрузки',
                'message': f"Обнаружен дисбаланс загруженности.",
                'details': f"Перегружены: {overloaded_str}. Свободны: {available_str}",
                'action': f"🔄 Перенаправить новые ТС из {overloaded_str} в {available_str}",
                'impact': 'high'
            })
        
        # Рекомендация по крупногабаритным ТС
        large_vehicles = {}
        for zone, data in density_data.items():
            large_count = data['by_type'].get('truck', 0) + data['by_type'].get('bus', 0)
            if large_count > 0:
                large_vehicles[zone] = large_count
        
        if len(large_vehicles) > 0:
            total_large = sum(large_vehicles.values())
            zones_str = ', '.join([f"{z}({c})" for z, c in large_vehicles.items()])
            
            cross_recs.append({
                'zone': 'Все зоны',
                'priority': 2,
                'level': 'info',
                'icon': '🚛',
                'title': 'Крупногабаритный транспорт',
                'message': f"Всего крупногабаритных ТС: {total_large}",
                'details': f"Распределение: {zones_str}",
                'action': f"📊 Рассмотреть выделение специальной зоны для грузовиков/автобусов",
                'impact': 'medium'
            })
        
        return cross_recs
    
    def format_report(self, recommendations):
        """
        Форматирование рекомендаций в текстовый отчёт
        
        Args:
            recommendations: список рекомендаций
        
        Returns:
            строка с форматированным отчётом
        """
        report = []
        report.append("="*60)
        report.append("  ОТЧЁТ ПО ОПТИМИЗАЦИИ ПАРКОВКИ")
        report.append("="*60)
        
        # Группировка по приоритетам
        by_priority = {}
        for rec in recommendations:
            priority = rec['priority']
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append(rec)
        
        priority_names = {
            1: "🔴 КРИТИЧНЫЕ",
            2: "🟡 ВАЖНЫЕ",
            3: "🟠 ИНФОРМАЦИОННЫЕ",
            4: "🟢 НОРМАЛЬНЫЕ",
            5: "✅ РЕЗЕРВНЫЕ"
        }
        
        for priority in sorted(by_priority.keys()):
            report.append(f"\n{priority_names.get(priority, f'Приоритет {priority}')}")
            report.append("-"*60)
            
            for rec in by_priority[priority]:
                report.append(f"\n{rec['icon']} {rec['title']}")
                report.append(f"   {rec['message']}")
                report.append(f"   {rec['details']}")
                report.append(f"   Действие: {rec['action']}")
        
        report.append("\n" + "="*60)
        return "\n".join(report)


# ============ ТЕСТ ГЕНЕРАТОРА РЕКОМЕНДАЦИЙ ============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ТЕСТ ГЕНЕРАТОРА РЕКОМЕНДАЦИЙ (Windows)")
    print("="*60 + "\n")
    
    recommender = Recommender()
    
    # Тестовые данные (разные сценарии)
    print("Сценарий 1: Критическая перегрузка")
    print("-"*60)
    
    test_data_1 = {
        'Zone_A': {
            'occupancy': 120.0, 'vehicles': 15, 'capacity': 20,
            'by_type': {'truck': 8, 'car': 7}, 'level': 'critical',
            'space_used': 24.0, 'available': -4.0
        },
        'Zone_B': {
            'occupancy': 88.0, 'vehicles': 10, 'capacity': 15,
            'by_type': {'car': 10}, 'level': 'warning',
            'space_used': 13.2, 'available': 1.8
        },
        'Zone_C': {
            'occupancy': 30.0, 'vehicles': 5, 'capacity': 18,
            'by_type': {'car': 5}, 'level': 'normal',
            'space_used': 5.4, 'available': 12.6
        },
        'Zone_D': {
            'occupancy': 0.0, 'vehicles': 0, 'capacity': 12,
            'by_type': {}, 'level': 'empty',
            'space_used': 0.0, 'available': 12.0
        }
    }
    
    recs_1 = recommender.generate(test_data_1)
    
    print(f"\nГенерировано рекомендаций: {len(recs_1)}")
    print("\nПервые 3 рекомендации:")
    for i, rec in enumerate(recs_1[:3], 1):
        print(f"\n{i}. {rec['icon']} {rec['title']}")
        print(f"   {rec['message']}")
        print(f"   Действие: {rec['action']}")
    
    # Форматированный отчёт
    print("\n" + "="*60)
    print("  ПОЛНЫЙ ОТЧЁТ")
    print("="*60)
    
    report = recommender.format_report(recs_1)
    print(report)
    
    # Сохранение отчёта
    import os
    os.makedirs('outputs', exist_ok=True)
    
    with open('outputs\\recommendations_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n✅ Отчёт сохранён: outputs\\recommendations_report.txt")
    
    print("\n" + "="*60)
    print("  ГЕНЕРАТОР РЕКОМЕНДАЦИЙ РАБОТАЕТ!")
    print("="*60)
    print("\nВремя: 2:00 / 6:00")
    print("Следующий шаг: src\\pipeline.py (ИНТЕГРАЦИЯ)")
