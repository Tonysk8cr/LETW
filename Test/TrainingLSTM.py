# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 10/07/2025

import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from DataLabelling import DataLabelling
from DataExtraction import DataExtractor

class TrainingLSTM:

    def __init__(self):
        self.model = Sequential()
        self.labeller = DataLabelling()
        self.signs = DataExtractor().signs

    def build_model(self):
        """
        Build the LSTM model.
        Parameters:
            input_shape: Shape of the input data (timesteps, features).
            num_classes: Number of output classes.
        """

    def __init__(self):
        self.model = Sequential()
        self.labeller = DataLabelling()
        self.signs = DataExtractor().signs

    def build_model(self):
        # Callbacks
        log_dir = os.path.join("Logs")
        tb = TensorBoard(log_dir=log_dir)
        es = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        rlrop = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Split data
        x_train, x_val, x_test, y_train, y_val, y_test = self.labeller.split_data()

        # Modelo
        # Capa LSTM 1
        self.model.add(LSTM(
            64,
            return_sequences=True,
            activation='tanh',
            input_shape=(30, 1662),
            recurrent_dropout=0.2
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        # Capa LSTM 2
        self.model.add(LSTM(
            64,
            return_sequences=True,
            activation='tanh',
            recurrent_dropout=0.2
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        # Capa LSTM 3
        self.model.add(LSTM(
            32,
            return_sequences=False,
            activation='tanh',
            recurrent_dropout=0.2
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        # Densas finales
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(len(self.signs), activation='softmax'))

        # Compilación
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=9e-4),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )

        # Entrenamiento
        self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2000,
            callbacks=[tb, es, rlrop]
        )

        # Evaluación final en test
        loss, acc = self.model.evaluate(x_test, y_test)
        print(f"\nEvaluación final en test set:\n  Pérdida: {loss:.4f} | Precisión: {acc:.4f}")

        
