import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from DataExtraction import DataExtractor



class DataLabelling:

    def __init__(self, repetitions=30):
        self.signs = DataExtractor().signs
        self.label_map = {label:num for num, label in enumerate(self.signs)}
        self.repetitions = repetitions
        self.mp_data = os.path.join(r"C:\Users\tonyi\LETW\Test\MP_Data")
        # Inicializar las coordenadas
        self.x_coordinate = None
        self.y_coordinate = None


    def label_data(self):
  

        sequences, labels = [], []

        print("Acciones detectadas:", self.signs)
        print("Repeticiones configuradas:", self.repetitions)
        for sign in self.signs:
            for sequence in range(self.repetitions):
                window = []
                for frame_num in range(self.repetitions):
                    res = np.load(os.path.join(self.mp_data, sign, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(self.label_map[sign])
        # Guardar como atributos de la clase
        self.x_coordinate = np.array(sequences)
        self.y_coordinate = to_categorical(labels).astype(int)
        print("Shape total de datos:", self.x_coordinate.shape)
        return self.x_coordinate, self.y_coordinate

    def split_data(self):
        # Verificar si los datos están cargados
        if self.x_coordinate is None:
            self.label_data()
        
        x_train, x_test, y_train, y_test = train_test_split(self.x_coordinate, self.y_coordinate, test_size=0.5)
        print(f"Datos de entrenamiento: {x_train.shape}, Etiquetas: {y_train.shape}")
        print(f"Datos de prueba: {x_test.shape}, Etiquetas: {y_test.shape}")
        return x_train, x_test, y_train, y_test