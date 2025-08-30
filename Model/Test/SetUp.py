# Developed by Anthony Villalobos 24/07/2025
# Updated by Anthony Villalobos 15/08/2025

import os
import numpy as np
from DataExtraction import DataExtractor
from Utilities import Utilities

class SetUp:
    def __init__(self, repetitions, signs):
        self.signs = signs
        self.logger = Utilities.setup_logging()
        self.repetitions = repetitions

    def create_directories(self):
        Data_Path = os.path.join("Model", "Test", "MP_Data")
        actions = np.array(self.signs)
        number_sequences = self.repetitions

        print("Creando folder para los numpy arrays")
        self.logger.info("Creando folder para los numpy arrays")
        for action in actions:
            for sequence in range(number_sequences):
                try:
                    os.makedirs(os.path.join(Data_Path, action, str(sequence)))
                except OSError as e:
                    pass
        print(f"Directorios creados en {Data_Path} para las acciones: {actions}")
        self.logger.info(f"Directorios creados en {Data_Path} para las acciones: {actions}")

        print("Creando directorio para los videos")
        self.logger.info("Creando directorio para los videos")
        video_path = os.mkdir(os.path.join("Model/Test/Test_Videos"))
        for action in actions:
            try:
                os.makedirs(os.path.join("Model/Test/Test_Videos", action))
            except OSError as e:
                pass


        return Data_Path, actions, video_path
