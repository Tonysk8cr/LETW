# Developed by Anthony Villalobos 08/01/2025
# Updated by Anthony Villalobos 15/08/2025

import os
import cv2
import random
import logging

class Utilities:
    """Class used for secondary but important functions related to video processing.
    This class contains static methods to handle video paths, transformations, and augmentations."""

    @staticmethod
    def get_video_paths(directory, extensions=('.mp4', '.avi', '.mov')):
        """Devuelve una lista de rutas de videos en el directorio dado."""
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(extensions)
        ]
    
    @staticmethod
    def get_video_by_action(parent_directory, extensions=('.mp4', '.avi', '.mov')):
        """Devuelve un diccionario con claves como el nombre de la acción y valores como las rutas de video correspondientes."""
        video_dict = {} 
        for class_folder in os.listdir(parent_directory):
            class_path = os.path.join(parent_directory, class_folder)
            if os.path.isdir(class_path):
                videos = [
                    os.path.join(class_path, f)
                    for f in os.listdir(class_path)
                    if f.lower().endswith(extensions)
                ]
                if videos:
                    video_dict[class_folder.upper()] = videos
        return video_dict

    @staticmethod
    def flip_horizontal(frame):
        """Devuelve el frame volteado horizontalmente."""
        return cv2.flip(frame, 1)
    
    @staticmethod
    def random_augmentation(frame):
        """Aplica una transformación aleatoria entre varias opciones"""
        choice = random.choice(['brightness', 'none'])
        
        if choice == 'brightness':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            value = random.randint(-10, 10)
            hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        elif choice == 'none':
            return frame
        
        else:
            return frame
        
    @staticmethod
    def setup_logging(log_file='app.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)