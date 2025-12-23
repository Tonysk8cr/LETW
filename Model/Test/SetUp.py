# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 23/09/2025

import os

import numpy as np
from Utilities import Utilities


class SetUp:
    def __init__(self, repetitions, signs):
        self.signs = signs
        self.logger = Utilities.setup_logging()
        self.repetitions = repetitions

    def create_directories(self):
        # Ruta para los numpy arrays
        data_path = os.path.join("Model", "Test", "MP_Data")
        actions = np.array(self.signs)
        number_sequences = self.repetitions

        print("Creando folders para los numpy arrays")
        self.logger.info("Creando folders para los numpy arrays")

        for action in actions:
            for sequence in range(number_sequences):
                folder_path = os.path.join(data_path, action, str(sequence))
                os.makedirs(folder_path, exist_ok=True)  # Crea todos los dirs intermedios si no existen

        print(f"Directorios creados en {data_path} para las acciones: {actions}")
        self.logger.info(f"Directorios creados en {data_path} para las acciones: {actions}")

        # Ruta para los videos
        video_base_path = os.path.join("Model", "Test", "Test_Videos")
        print("Creando directorio para los videos")
        self.logger.info("Creando directorio para los videos")

        os.makedirs(video_base_path, exist_ok=True)  # Asegura que la carpeta base exista
        for action in actions:
            action_video_path = os.path.join(video_base_path, action)
            os.makedirs(action_video_path, exist_ok=True)  # Crea carpetas de acciones

        return data_path, actions, video_base_path
