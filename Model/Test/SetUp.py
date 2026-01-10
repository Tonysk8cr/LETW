# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 23/09/2025

from pathlib import Path

import numpy as np
from Utilities import Utilities


class SetUp:
    def __init__(self, workspace_path: Path, repetitions: int, signs: list[str]):
        self.workspace_path = workspace_path
        self.signs = signs
        self.logger = Utilities.setup_logging()
        self.repetitions = repetitions

    def create_directories(self):
        # Ruta para los numpy arrays
        data_path = self.workspace_path / "MP_Data"
        actions = np.array(self.signs)
        number_sequences = self.repetitions

        print(f"Creando folders para los numpy arrays en {data_path}")
        self.logger.info(f"Creando folders para los numpy arrays en {data_path}")

        for action in actions:
            for sequence in range(number_sequences):
                folder_path = data_path / action / str(sequence)
                folder_path.mkdir(parents=True, exist_ok=True)  # Crea todos los dirs intermedios si no existen

        print(f"Directorios creados en {data_path} para las acciones: {actions}")
        self.logger.info(f"Directorios creados en {data_path} para las acciones: {actions}")

        # Ruta para los videos
        video_base_path = self.workspace_path / "Test_Videos"
        print(f"Creando directorio para los videos en {video_base_path}")
        self.logger.info(f"Creando directorio para los videos en {video_base_path}")

        video_base_path.mkdir(parents=True, exist_ok=True)  # Asegura que la carpeta base exista
        for action in actions:
            action_video_path = video_base_path / action
            action_video_path.mkdir(parents=True, exist_ok=True)  # Crea carpetas de acciones

        return str(data_path), actions, str(video_base_path)
