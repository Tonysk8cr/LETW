import os
import cv2
import random

class Utilities:

    @staticmethod
    def get_video_paths(directory, extensions=('.mp4', '.avi', '.mov')):
        """Devuelve una lista de rutas de videos en el directorio dado."""
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(extensions)
        ]

    @staticmethod
    def flip_horizontal(frame):
        """Devuelve el frame volteado horizontalmente."""
        return cv2.flip(frame, 1)
    
    @staticmethod
    def random_augmentation(frame):
        """Aplica una transformación aleatoria entre varias opciones"""
        choice = random.choice(['flip', 'brightness', 'blur', 'rotate', 'none'])

        if choice == 'flip':
            return cv2.flip(frame, 1)
        
        elif choice == 'brightness':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            value = random.randint(-10, 10)
            hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif choice == 'blur':
            return cv2.GaussianBlur(frame, (5, 5), 0)
        
        elif choice == 'rotate':
            (h, w) = frame.shape[:2]
            center = (w // 2, h // 2)
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(frame, M, (w, h))
        
        else:
            return frame