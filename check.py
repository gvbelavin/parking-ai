import torch
import cv2
import numpy as np

print("=" * 60)
print("  ПРОВЕРКА ОКРУЖЕНИЯ (Windows)")
print("=" * 60)
print("\nУстановленные пакеты:")
print("  PyTorch:", torch.__version__)
print("  CUDA доступна:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
    print("  CUDA версия:", torch.version.cuda)
else:
    print("  GPU: НЕТ (будет использован CPU)")

print("  OpenCV:", cv2.__version__)
print("  NumPy:", np.__version__)

print("\n" + "=" * 60)
print("  ✅ ВСЁ УСТАНОВЛЕНО КОРРЕКТНО!")
print("=" * 60)
