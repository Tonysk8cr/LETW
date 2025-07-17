# Developed by Anthony Villalobos 08/01/2025
# Updated by Anthony Villalobos 13/07/2025

import os
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from DataLabelling import DataLabelling
from DataExtraction import DataExtractor

class TrainingLSTM:

    def __init__(self):
        self.model = Sequential()
        self.labeller = DataLabelling()
        self.signs = DataExtractor().signs

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

            print("Hola, estamos en el entrenamiento del modelo LSTM \n" \
            "Opciones: \n" \
            "1. Entrenar el modelo LSTM \n" \
            "2. Hacer un prueba de comparacion \n"
            "3. Guardar el modelo \n" 
            "4. Evaluar el modelo \n"
            "5. Salir \n")
        
            user_choice = input("Seleccione una opción: ")

            if user_choice == '1':

                es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

                rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,min_lr=1e-6)

                # Split data
                x_train, x_val, x_test, y_train, y_val, y_test = self.labeller.split_data()

                # Model
                # LSTM Layer 1
                """64 units, return_sequences=True to return the full sequence for the next LSTM layer, 
                tanh activation function, recurrent_dropout to avoid overfitting
                BatchNormalization: Normalization to stabilize the learning process
                DropOut: turns some neurons off to avoid overfitting, 30% of the neurons will be turned off
                """
                self.model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 1662), recurrent_dropout=0.2))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(0.3)) 

                # LSTM Layer 2
                """Takes the data from the previous layer and returns the full sequence again. 
                this helps to refine the features extracted by the first layer."""
                self.model.add(LSTM(64, return_sequences=True, activation='tanh', recurrent_dropout=0.2))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(0.3))

                # LSTM Layer 3
                """Takes the data from the previous layer and returns the last output.
                This is the last LSTM layer before the dense layers."""
                self.model.add(LSTM(32, return_sequences=False, activation='tanh', recurrent_dropout=0.2))
                self.model.add(BatchNormalization())
                self.model.add(Dropout(0.3))

                # Fianl Dense Layers
                """After the final LSTM Layer we return a vector of 32 units
                With this we want to move to the dense layers to classify the data.
                Relu is used here instead of tanh because it is more efficient for classification tasks.
                The final layer has the number of 16 to forze the model to learn the features of the data."""
                self.model.add(Dense(32, activation='relu'))
                self.model.add(Dropout(0.3))
                self.model.add(Dense(16, activation='relu'))
                self.model.add(Dropout(0.3))
                self.model.add(Dense(len(self.signs), activation='softmax'))

                # Compile the model
                """We use Adam optimizer with a learning rate of 9e-4, categorical_crossentropy loss and categorical_accuracy metric."""
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=9e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

                # Train the model
                self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2000, callbacks=[tb, es, rlrop])

                # Last evaluation in test
                loss, acc = self.model.evaluate(x_test, y_test)
                print(f"\nEvaluación final en test set:\n  Pérdida: {loss:.4f} | Precisión: {acc:.4f}")


            elif user_choice == '2':

                print("\nProbando las predicciones en el conjunto de test:")
                print("Predicciones para el primer video de test: ")
                res = self.model.predict(x_test)

                for i in range(len(x_test)):

                    print("Resultado: ", self.signs[np.argmax(res[i])])
                    print("Valor en y: ", self.signs[np.argmax(y_test[i])])
                    print("\n")

                continue

            elif user_choice == '3':

                self.model.save('action_recognition_model.h5')
                #self.model.load_weights('action_recognition_model.h5')

                continue
            
            elif user_choice == '4':
                print("Prueba de evaluación usando CM y Accuracy. \n")

                yhat = self.model.predict(x_test)
                ytrue = np.argmax(y_test, axis=1).tolist()
                yhat = np.argmax(yhat, axis=1).tolist()
                clasiffication = multilabel_confusion_matrix(ytrue, yhat)
                print(clasiffication)
                model_precission = accuracy_score(ytrue, yhat)
                print(model_precission)

                continue
            
            elif user_choice == '5':
                print("Saliendo del entrenamiento del modelo LSTM. ¡Hasta luego!")
                exit_training = False
                return
            else:
                print("Opción no válida. Por favor, intente de nuevo. \n")
                return

