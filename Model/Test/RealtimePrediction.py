# Developed by Anthony Villalobos 13/07/2025

import cv2
import mediapipe as mp
import numpy as np
from LandmarkDrawer import LandmarkDrawer
from KeypointExtractor import KeypointExtractor
from DataExtraction import DataExtractor
from ImageProcessor import ImageProcessor
from keras.models import load_model

class RealtimeDetection: 

    def __init__(self):
        self.extractor = KeypointExtractor()
        self.signs = DataExtractor().signs
        self.model = load_model(r"C:\Users\tonyi\LETW\action_recognition_model.h5")
        self.convert = ImageProcessor().mediapipe_detection
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawer = LandmarkDrawer(self.mp_drawing, self.mp_holistic)
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.treshold = 0.9
        self.colors = [
            (245,117,16),
            (117,245,16),
            (16,117,245),
            (245,16,117),
            (117,16,245),
            (16,245,117)
        ]

    def real_time_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se puede acceder a la camara")
            return None
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = self.convert(frame, holistic)
                print("resultados", type(results))

                self.drawer.draw(image, results)

                keypoints, success = self.extractor.extract(results)
                if not success or keypoints.shape != (1662,):
                    continue

                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                if len(self.sequence) == 30:
                    preds = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    predicted_class = np.argmax(preds)
                    confidence = preds[predicted_class]

                    print(f"Predicted: {self.signs[predicted_class]} with confidence {confidence:.3f}")

                    self.predictions.append(predicted_class)

                    if len(self.predictions) >= 10:
                        # Moda (valor más frecuente) de las últimas 10 predicciones
                        most_common = np.argmax(np.bincount(self.predictions[-10:]))

                        if most_common == predicted_class and confidence > self.treshold:
                            if len(self.sentence) == 0 or self.signs[predicted_class] != self.sentence[-1]:
                                print(f"Añadiendo signo: {self.signs[predicted_class]}")
                                self.sentence.append(self.signs[predicted_class])

                    if len(self.sentence) > 5:
                        self.sentence = self.sentence[-5:]

                    # Visualizar probabilidades
                    image = self.visualize(image, preds)

                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(self.sentence), (3, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Prueba', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            return self.sentence  # O lo que desees retornar

    def visualize(self, input_frame, results):
        output_frame = input_frame.copy()
        for num, prob in enumerate(results):
            if isinstance(prob, (np.ndarray, list)):
                if np.size(prob) == 1:
                    prob = prob.item()
                else:
                    prob = float(prob[0])
            color = self.colors[num % len(self.colors)]
            cv2.rectangle(output_frame, (0, 60 + num*40), (int(prob * 100), 90 + num*40), color, -1)
            cv2.putText(output_frame, self.signs[num], (0, 85 + num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return output_frame
