import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping
from DataLabelling import DataLabelling
from DataExtraction import DataExtractor

class TrainingLSTM:

    def __init__(self):
        self.model = Sequential()
        self.data_labeller = DataLabelling()
        self.signs = DataExtractor().signs

    def build_model(self):
        """
        Build the LSTM model.
        Parameters:
            input_shape: Shape of the input data (timesteps, features).
            num_classes: Number of output classes.
        """

        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        x_train, x_test, y_train, y_test =self.data_labeller.split_data()
        


        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(len(self.signs), activation='softmax'))


        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
        self.model.fit(x_train, y_train,
                       validation_data=(x_test, y_test),
                        epochs=2000, 
                        callbacks=[tb_callback, early_stop])
        
        loss, acc = self.model.evaluate(x_test, y_test)
        print(f"\n Evaluación final en test set:\n   Pérdida: {loss:.4f} | Precisión: {acc:.4f}")

        #test=self.model.summary()
        #print(test)

        
