# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 02/06/2025

import cv2
import mediapipe as mp
import os
import numpy as np
from LandmarkDrawer import LandmarkDrawer
from KeypointExtractor import KeypointExtractor
from Utilities import Utilities

class DataExtractor:
    """
    Takes the data from one video and extracts the landmarks using MediaPipe Holistic.
    This is used to get the data upon which the model will be trained.
    """
    def __init__(self, repetitions=30):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawer = LandmarkDrawer(self.mp_drawing, self.mp_holistic)
        self.extractor = KeypointExtractor()
        self.signs = ["ANASCOR", "A-PARTIR-DE", "CAERSE-2", "CALZONCILLOS", "CIUDAD-QUESADA", "NOTA"]
        self.mp_data = os.path.join(r"C:\Users\tonyi\LETW\Test\MP_Data")
        self.repetitions = repetitions

    def mediapipe_detection(self, frame, model):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), results

    def process_video(self, video_path, transform=None):
        # Extraer el nombre de la acción del path del video
        video_filename = os.path.basename(video_path)
        action = None
        
        # Buscar qué acción corresponde al video
        for sign in self.signs:
            if sign in video_filename.upper():
                action = sign
                break
        
        if not action:
            print(f"No se pudo determinar la acción para el video: {video_filename}")
            return None, None
        
        print(f"\nProcesando video: {video_filename} como acción: {action}")
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for sequence in range(self.repetitions):
                
                # Alternar transformación por secuencia
                current_transform = None
                transform_name = "con transformación" if current_transform else "original"
                print(f"  Secuencia {sequence + 1}/{self.repetitions} ({transform_name})")
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"No se pudo abrir el video: {video_path}")
                    continue
                
                # Obtener información del video
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < self.repetitions:
                    print(f"    Advertencia: El video tiene solo {total_frames} frames, menos que los {self.repetitions} necesarios")
                
                # Calcular qué frames tomar (distribuidos uniformemente)
                frame_indices = np.linspace(0, total_frames - 1, self.repetitions, dtype=int)
                
                for i, frame_idx in enumerate(frame_indices):
                    # Ir al frame específico
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        print(f"    No se pudo leer el frame {frame_idx}")
                        continue
                    
                    # Aplicar transformación si existe
                    if current_transform:
                        frame = current_transform(frame)

                    # Procesar frame con MediaPipe
                    image, results = self.mediapipe_detection(frame, holistic)
                    self.drawer.draw(image, results)
                    cv2.imshow('Video Detection', image)
                    cv2.waitKey(1)
                    
                    # Extraer keypoints
                    keypoints, success = self.extractor.extract(results)
                    if success:
                        # Crear directorio si no existe
                        sequence_dir = os.path.join(self.mp_data, action, str(sequence))
                        os.makedirs(sequence_dir, exist_ok=True)
                        
                        # Guardar keypoints
                        npy_path = os.path.join(sequence_dir, f"{i}.npy")
                        np.save(npy_path, keypoints)
                        print(f"    Frame {i + 1}/{self.repetitions} guardado: {npy_path}")
                    else:
                        print(f"    Error extrayendo keypoints del frame {i + 1}")
                
                cap.release()
                print(f"  Completada secuencia {sequence + 1}")

        cv2.destroyAllWindows()
        return None, None