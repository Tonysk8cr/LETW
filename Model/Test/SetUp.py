# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 23/09/2025

from pathlib import Path

import numpy as np
from Utilities import Utilities


class SetUp:
    def __init__(self, repetitions, signs):
        self.signs = signs
        self.logger = Utilities.setup_logging()
        self.repetitions = repetitions

    def create_directories(self):
        # Ruta para los numpy arrays
        data_path = Path("Model") / "Test" / "MP_Data"
        actions = np.array(self.signs)
        number_sequences = self.repetitions

        print("Creando folders para los numpy arrays")
        self.logger.info("Creando folders para los numpy arrays")

        for action in actions:
            for sequence in range(number_sequences):
                folder_path = data_path / action / str(sequence)
                folder_path.mkdir(parents=True, exist_ok=True)  # Crea todos los dirs intermedios si no existen

        print(f"Directorios creados en {data_path} para las acciones: {actions}")
        self.logger.info(f"Directorios creados en {data_path} para las acciones: {actions}")

        # Ruta para los videos
        video_base_path = Path("Model") / "Test" / "Test_Videos"
        print("Creando directorio para los videos")
        self.logger.info("Creando directorio para los videos")

        video_base_path.mkdir(parents=True, exist_ok=True)  # Asegura que la carpeta base exista
        for action in actions:
            action_video_path = video_base_path / action
            action_video_path.mkdir(parents=True, exist_ok=True)  # Crea carpetas de acciones

        return str(data_path), actions, str(video_base_path)
