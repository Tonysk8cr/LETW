# Developed by Anthony Villalobos 08/01/2025
# Updated by Anthony Villalobos 23/09/2025

import os
import numpy as np
import tensorflow as tf
from DataLabelling import DataLabelling
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import LSTM, BatchNormalization, Dense, Dropout
from keras.models import Sequential
from Utilities import Utilities


class TrainingLSTM:
    def __init__(self, signs, repetitions, frames, mp_path):
        self.model = Sequential()
        self.labeller = DataLabelling(signs=signs, repetitions=repetitions, frames=frames, mp_path=mp_path)
        self.signs = signs
        self.logger = Utilities.setup_logging()

        # Data attributes
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    # main menu options
    # requires self and tb as parameters
    def option1(self, tb):
        es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

        rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

        # Split data
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.labeller.split_data()

        # Model
        # LSTM Layer 1

        """64 units, return_sequences=True to return the full sequence for the next LSTM layer, 
        tanh activation function, recurrent_dropout to avoid overfitting
        BatchNormalization: Normalization to stabilize the learning process
        DropOut: turns some neurons off to avoid overfitting, 30% of the neurons will be turned off
        """
        self.model.add(
            LSTM(64, return_sequences=True, activation="tanh", input_shape=(30, 1662), recurrent_dropout=0.2)
        )
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        # LSTM Layer 2
        """Takes the data from the previous layer and returns the full sequence again. 
        this helps to refine the features extracted by the first layer."""
        self.model.add(LSTM(64, return_sequences=True, activation="tanh", recurrent_dropout=0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        # LSTM Layer 3
        """Takes the data from the previous layer and returns the last output.
        This is the last LSTM layer before the dense layers."""
        self.model.add(LSTM(32, return_sequences=False, activation="tanh", recurrent_dropout=0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        # Fianl Dense Layers
        """After the final LSTM Layer we return a vector of 32 units
        With this we want to move to the dense layers to classify the data.
        Relu is used here instead of tanh because it is more efficient for classification tasks.
        The final layer has the number of 16 to forze the model to learn the features of the data."""
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(len(self.signs), activation="softmax"))

        # Compile the model
        """We use Adam optimizer with a learning rate of 9e-4, categorical_crossentropy loss and categorical_accuracy metric."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=9e-4),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        # Train the model
        self.model.fit(
            self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), epochs=2000, callbacks=[tb, es, rlrop]
        )

        # Last evaluation in test
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        print(f"\nEvaluación final en test set:\n  Pérdida: {loss:.4f} | Precisión: {acc:.4f}")
        self.logger.info(f"\nEvaluación final en test set:\n  Pérdida: {loss:.4f} | Precisión: {acc:.4f}")
        return (self.x_test, self.y_test, self.x_val, self.y_val, self.x_train, self.y_train)

    # requires self, x_test and y_test as parameters
    def option2(self):
        print("\nProbando las predicciones en el conjunto de test:")
        self.logger.info("Probando las predicciones en el conjunto de test:")
        print("Predicciones para el primer video de test: ")
        self.logger.info("Predicciones para el primer video de test: ")
        res = self.model.predict(self.x_test)

        for i in range(len(self.x_test)):
            print("Resultado: ", self.signs[np.argmax(res[i])])
            self.logger.info(f"Resultado: {self.signs[np.argmax(res[i])]}")
            print("Valor en y: ", self.signs[np.argmax(self.y_test[i])])
            self.logger.info(f"Valor en y: {self.signs[np.argmax(self.y_test[i])]}")
            print("\n")

        return "Prueba de predicciones completada"

    def option3(self):
        self.model.save("action_recognition_model.keras")
        # self.model.load_weights('action_recognition_model.keras')

        return "Modelo guardado como 'action_recognition_model.keras'"

    def option4(self):
        print("Prueba de evaluación usando CM y Accuracy. \n")
        self.logger.info("Prueba de evaluación usando CM y Accuracy.")

        yhat = self.model.predict(self.x_test)
        ytrue = np.argmax(self.y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        clasiffication = multilabel_confusion_matrix(ytrue, yhat)
        print(clasiffication)
        model_precission = accuracy_score(ytrue, yhat)
        print(model_precission)

        return "Evaluación completada"

    def option5(self):
        print("Saliendo del entrenamiento del modelo LSTM. ¡Hasta luego!")
        exit_training = False
        return exit_training

    def build_model(self):
        # Callbacks
        log_dir = os.path.join("Logs")
        tb = TensorBoard(log_dir=log_dir)

        """
        es = early stopping
        rlrop = ReduceLROnPlateau
        Early stopping used to check on validation loss if it is not improving after 20 epochs,
        it will stop training and restore the best weights
        If val_loss is not improving after 5 epochs it will reduce the learning rate by a factor of 0.5
        This is important to avoid overfitting and to help the model converge
        """

        exit_training = True
        while exit_training:
            print(
                "Hola, estamos en el entrenamiento del modelo LSTM \n"
                "Opciones: \n"
                "1. Entrenar el modelo LSTM \n"
                "2. Hacer un prueba de comparacion \n"
                "3. Guardar el modelo \n"
                "4. Evaluar el modelo \n"
                "5. Salir \n"
            )

            user_choice = input("Seleccione una opción: ")
            self.logger.info(f"User selected option {user_choice} in TrainingLSTM")

            options = {
                "1": lambda: self.option1(tb),
                "2": self.option2,
                "3": self.option3,
                "4": self.option4,
                "5": self.option5,
            }

            menu_option = options.get(user_choice)
            if menu_option:
                result = menu_option()
                if result is False:
                    exit_training = False

            else:
                print("Opción no válida. Por favor, intente de nuevo. \n")
                return
