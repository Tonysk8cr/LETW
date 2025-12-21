# Developed by Anthony Villalobos 08/01/2025
# Updated by Anthony Villalobos 15/08/2025

import logging
import os
import random

import cv2


class Utilities:
    """
    Class used for secondary but important functions related to video processing.
    This class contains static methods to handle video paths, transformations, and augmentations.
    Here we also set up logging for the application.
    """

    @staticmethod
    def get_video_paths(directory, extensions=(".mp4", ".avi", ".mov")):
        """Devuelve una lista de rutas de videos en el directorio dado."""
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]

    @staticmethod
    def get_video_by_action(parent_directory, extensions=(".mp4", ".avi", ".mov")):
        """Devuelve un diccionario con claves como el nombre de la acción y valores como las rutas de video correspondientes."""
        video_dict = {}
        for class_folder in os.listdir(parent_directory):
            class_path = os.path.join(parent_directory, class_folder)
            if os.path.isdir(class_path):
                videos = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(extensions)]
                if videos:
                    video_dict[class_folder.upper()] = videos
        return video_dict

    @staticmethod
    def training_paths():
        # Test videos path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(base_dir, ".", "Test_Videos")
        video_paths = os.path.normpath(video_path)

        # this was created just in case, but since we specify the path using a similar logic on the class this is used
        # Mp data path
        mp_data_path = os.path.join(base_dir, ".", "MP_Data")
        mp_data_paths = os.path.normpath(mp_data_path)
        return video_paths, mp_data_paths

    def model_route():
        # obtain model route
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "..", "action_recognition_model.h5")
        model_path = os.path.normpath(model_path)
        return model_path

    @staticmethod
    def flip_horizontal(frame):
        """Devuelve el frame volteado horizontalmente."""
        return cv2.flip(frame, 1)

    @staticmethod
    def random_augmentation(frame):
        """Aplica una transformación aleatoria entre varias opciones"""
        choice = random.choice(["brightness", "none"])

        if choice == "brightness":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            value = random.randint(-10, 10)
            hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if choice == "none":
            return frame

        return frame

    @staticmethod
    def setup_logging(log_file="app.log"):
        logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        return logging.getLogger(__name__)
