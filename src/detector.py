"""
Детектор транспортных средств - ИСПРАВЛЕНО
"""

import torch
from ultralytics import YOLO
import cv2

class VehicleDetector:
    """Детектор транспортных средств на базе YOLOv8"""
    
    # ← НОВОЕ: Размеры ТС для расчета занимаемого места
    VEHICLE_SIZES = {
        'car': 1.0,
        'motorcycle': 0.5,
        'truck': 2.5,
        'bus': 3.0
    }
    
    def __init__(self, model='yolov8m.pt', conf_threshold=0.25):
        """
        Инициализация детектора
        
        Args:
            model: путь к модели YOLO
            conf_threshold: порог уверенности
        """
        print(f"\n{'='*60}")
        print(f"  ИНИЦИАЛИЗАЦИЯ ДЕТЕКТОРА")
        print(f"{'='*60}\n")
        
        self.model = YOLO(model)
        self.conf_threshold = conf_threshold
        self.iou_threshold = 0.45
        self.imgsz = 1280
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Устройство: {self.device}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        print(f"Классы: {list(self.vehicle_classes.values())}")
        print(f"{'='*60}\n")
    
    def detect(self, image):
        """
        Детекция транспортных средств
        
        Args:
            image: изображение BGR (numpy array)
        
        Returns:
            list: список детекций с полями:
                - class: тип ТС
                - confidence: уверенность
                - x1, y1, x2, y2: координаты бокса
                - center: центр (x, y)
                - size: занимаемое место (в парковочных местах)
        """
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            agnostic_nms=True,
            max_det=300
        )
        
        detections = []
        boxes = results[0].boxes
        
        for box in boxes:
            cls = int(box.cls[0])
            
            if cls in self.vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                vehicle_type = self.vehicle_classes[cls]
                
                detections.append({
                    'class': vehicle_type,
                    'confidence': conf,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    'size': self.VEHICLE_SIZES.get(vehicle_type, 1.0)  # ← ИСПРАВЛЕНО: добавлено поле size
                })
        
        return detections
    
    def get_info(self):
        """Информация о детекторе"""
        return {
            'model': str(self.model.model),
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'classes': list(self.vehicle_classes.values())
        }


# ============ ТЕСТ ============
if __name__ == "__main__":
    import numpy as np
    
    print("\nТЕСТ ДЕТЕКТОРА\n")
    
    detector = VehicleDetector(conf_threshold=0.25)
    
    test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    detections = detector.detect(test_img)
    
    print(f"Найдено: {len(detections)} объектов")
    
    if detections:
        for det in detections:
            print(f"  {det['class']} (size={det['size']}, conf={det['confidence']:.3f})")
    
    print("\n✅ Тест завершен")
