# Developed by Anthony Villalobos 08/01/2025
# Updated by Anthony Villalobos 15/08/2025

import logging
import random
from pathlib import Path

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
        dir_path = Path(directory)
        return [str(f) for f in dir_path.iterdir() if f.suffix.lower() in extensions]

    @staticmethod
    def get_video_by_action(parent_directory, extensions=(".mp4", ".avi", ".mov")):
        """Devuelve un diccionario con claves como el nombre de la acción y valores como las rutas de video correspondientes."""
        video_dict = {}
        parent_path = Path(parent_directory)
        for class_folder in parent_path.iterdir():
            if class_folder.is_dir():
                videos = [str(f) for f in class_folder.iterdir() if f.suffix.lower() in extensions]
                if videos:
                    video_dict[class_folder.name.upper()] = videos
        return video_dict

    @staticmethod
    def training_paths():
        # Test videos path
        base_dir = Path(__file__).resolve().parent
        video_path = base_dir / "Test_Videos"
        # mp data path
        mp_data_path = base_dir / "MP_Data"
        return str(video_path), str(mp_data_path)

    def model_route():
        # obtain model route
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir.parent.parent / "action_recognition_model.keras"
        return str(model_path.resolve())

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
