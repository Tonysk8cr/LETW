import os
import numpy as np
from DataExtraction import DataExtractor

class SetUp:
    def __init__(self):
        self.signs = DataExtractor().signs

    def create_directories(self):
        Data_Path = os.path.join("MP_Data")
        actions = np.array(self.signs)
        number_sequences = 50
        Sequence_length = 30

        print("Creando folder para los numpy arrays")
        for action in actions:
            for sequence in range(number_sequences):
                try:
                    os.makedirs(os.path.join(Data_Path, action, str(sequence)))
                except OSError as e:
                    pass
        print(f"Directorios creados en {Data_Path} para las acciones: {actions}")

        print("Creando directorio para los videos")
        for action in actions:
            try:
                os.makedirs(os.path.join("Test_Videos", action))
            except OSError as e:
                pass

        video_path = os.mkdir(os.path.join("Test_Videos"))
        return Data_Path, actions, number_sequences, Sequence_length, video_path
