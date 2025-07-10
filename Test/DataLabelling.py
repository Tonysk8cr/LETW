# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 10/07/2025

import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from DataExtraction import DataExtractor



class DataLabelling:
    """Claas in charge of labelling the data extracted from the videos.
    It uses the DataExtractor to get the signs and the repetitions to label the data.
    Parameters:
    repetitions: Number of repetitions for each sign.
    Attributes:
    signs: List of signs to be used for labelling.
    label_map: Dictionary mapping each sign to a unique integer.
    repetitions: Number of repetitions for each sign.
    """

    def __init__(self, repetitions=30):
        self.signs = DataExtractor().signs
        self.label_map = {label: num for num, label in enumerate(self.signs)}
        self.repetitions = repetitions
        self.mp_data = os.path.join(r"C:\Users\tonyi\LETW\Test\MP_Data")
        self.x_coordinate = None
        self.y_coordinate = None


    def label_data(self):
        """Y is related to the labels of the signs, X is related to the coordinates of the keypoints.
        It uses one hot encoding to convert the labels into a format suitable for training.
        Most of the print statements are used for debugging purposes.
        Returns: X, Y"""
        sequences, labels = [], []
        for sign in self.signs:
            for seq in range(self.repetitions):
                window = []
                for frame_num in range(self.repetitions):
                    path = os.path.join(self.mp_data, sign, str(seq), f"{frame_num}.npy")
                    window.append(np.load(path))
                sequences.append(window)
                labels.append(self.label_map[sign])
        self.x_coordinate = np.array(sequences)            # (N, 30, 1662)
        self.y_coordinate = to_categorical(labels).astype(int)
        return self.x_coordinate, self.y_coordinate

    def split_data(self, test_size=0.15, val_size=0.15, random_state=42):
        """Method in charge of splitting the data into 3 sets: train, validation and test.
        test saves 15% of the data, val saves 15% of the data and train saves the rest.
        """
                   
        #Loads the data if it is not already loaded
        if self.x_coordinate is None:
            self.label_data()

        #Here we split the data 15% to evaluate the model once trained, the rest 85% that we separate later on
        x_tmp, x_test, y_tmp, y_test = train_test_split(self.x_coordinate, self.y_coordinate, test_size=test_size, random_state=random_state, stratify=self.y_coordinate)
        
        #From val_size we return the remaining 85% of the data this is used to indicate the value of x_tmp
        rel_val = val_size / (1 - test_size)

        #Here we split x_tmp in around 17% for x_val and y_val and the rest for x_train and y_train to save remaining 70% of the data
        #This is done to have a balanced dataset for training and validation
        x_train, x_val, y_train, y_val = train_test_split(x_tmp, y_tmp, test_size=rel_val, random_state=random_state, stratify=y_tmp)

        print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
        return x_train, x_val, x_test, y_train, y_val, y_test
    