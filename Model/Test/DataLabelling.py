# Developed by Anthony Villalobos 08/01/2025
# Updated by Anthony Villalobos 23/09/2025

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tf_keras.utils import to_categorical
from Utilities import Utilities


class DataLabelling:
    """Class in charge of labelling the data extracted from the videos.
    Parameters:
    repetitions: Number of repetitions for each sign.
    Attributes:
    signs: List of signs to be used for labelling.
    label_map: Dictionary mapping each sign to a unique integer.
    repetitions: Number of repetitions for each sign.
    """

    def __init__(self, repetitions, signs, frames, mp_path):
        self.signs = signs
        self.label_map = {label: num for num, label in enumerate(self.signs)}
        self.repetitions = repetitions
        self.frames = frames
        self.mp_data = mp_path
        self.x_coordinate = None
        self.y_coordinate = None
        self.logger = Utilities.setup_logging()

    def label_data(self):
        """Y corresponds to the labels of the signs; X corresponds to the keypoint coordinates.
        It uses one-hot encoding to convert the labels into a format suitable for training.
        Most of the print statements are used for debugging purposes.
        Returns: X, Y"""
        sequences, labels = [], []
        sequence_count = self.repetitions  # 100 as specified on the main class
        sequence_length = self.frames

        print(
            f"[LabelData] Iniciando etiquetado → acciones: {len(self.signs)}, "
            f"secuencias por acción: {sequence_count}, frames/seq: {sequence_length}"
        )
        self.logger.info(
            f"[LabelData] Start: {len(self.signs)} signs, {sequence_count} seq/sign, {sequence_length} frames/seq"
        )

        for sign in self.signs:
            print(f"\n[LabelData] Acción: {sign}")
            self.logger.info(f"[LabelData] Acción: {sign}")
            for seq in range(sequence_count):  # 30 sequences
                window = []
                for frame_num in range(sequence_length):
                    path = os.path.join(self.mp_data, sign, str(seq), f"{frame_num}.npy")
                    window.append(np.load(path))
                sequences.append(window)
                labels.append(self.label_map[sign])
        self.x_coordinate = np.array(sequences)  # (N, 30, 1662)
        self.y_coordinate = to_categorical(labels).astype(int)
        return self.x_coordinate, self.y_coordinate

    def split_data(self, test_size=0.15, val_size=0.15, random_state=42):
        """Method in charge of splitting the data into three sets: train, validation and test.
        The test set uses test_size (default 15%), the validation set uses val_size (default 15%),
        and the remaining data is used for training.
        """

        # Loads the data if it is not already loaded
        if self.x_coordinate is None:
            self.label_data()

        # First split: separate out the test set (test_size)
        x_tmp, x_test, y_tmp, y_test = train_test_split(
            self.x_coordinate,
            self.y_coordinate,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_coordinate,
        )

        # Compute relative validation size with respect to the remaining data after removing the test set
        rel_val = val_size / (1 - test_size)

        # Split x_tmp into validation and training sets using the relative validation size.
        # This yields approximately val_size of the original data for validation and the rest for training.
        x_train, x_val, y_train, y_val = train_test_split(
            x_tmp, y_tmp, test_size=rel_val, random_state=random_state, stratify=y_tmp
        )

        print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
        self.logger.info(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
        return x_train, x_val, x_test, y_train, y_val, y_test
