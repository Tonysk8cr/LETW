# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 02/06/2025

import cv2
import mediapipe as mp
import os
import itertools
import time
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
        self.drawer = LandmarkDrawer(self.mp_drawing, self.mp_holistic) # Instance of LandmarkDrawer to draw landmarks on the video frames
        self.extractor = KeypointExtractor() # Instance of KeypointExtractor to extract keypoints
        self.signs = ["ANASCOR", "A-PARTIR-DE", "CAERSE-2", "CALZONCILLOS", "CIUDAD-QUESADA", "NOTA"]
        self.mp_data = os.path.join(r"C:\Users\tonyi\LETW\Test\MP_Data") # Dir used to store the extracted data
        self.repetitions = repetitions # Number of repetitions for each video

    def mediapipe_detection(self, frame, model):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), results

    def process_video(self, video_path):
        """This method is the one in charge of processing the video and extracting the keypoints from the specified video path.
        Variables:  
        video_path: The path to the video file or directory containing videos.
        video_files: A list of video files to process.
        action: The action label derived from the video filename or directory name.
        squence: The current sequence number for the action being processed.
        frame, image: The current frame being processed and the image with drawn landmarks.
        results: The results from MediaPipe Holistic processing.
        keypoints: The extracted keypoints from the current frame.
        npy_path: The path where the keypoints will be saved as a .npy file.
        """
        if os.path.isdir(video_path):
            #Detects the video in the directory if it is a subfolder it considers it as an action and the videos inside it as the videos of that action
            video_files = Utilities.get_video_paths(video_path)
            if not video_files:
                print(f"No se encontraron videos en el directorio: {video_path}")
                return
            action = os.path.basename(video_path).upper()
            video_files = (video_files * ((self.repetitions // len(video_files)) + 1))[:self.repetitions] # Here we ensure that we have enough videos to process the required repetitions
        else:
            video_files = [video_path]
            video_filename = os.path.basename(video_path)
            action = None
            for sign in self.signs:
                if sign in video_filename.upper():
                    action = sign
                    break
            if not action:
                print(f"No se pudo determinar la acción para el video: {video_filename}")
                return

        print(f"\nProcesando acción: {action} con {len(video_files)} videos disponibles")

        #Main mediapipe holistic model

        with self.mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
            sequence = 0

            for video_idx, current_video in enumerate(video_files):
                if sequence >= self.repetitions:
                    break  # If we have processed enough repetitions, we stop processing more videos

                print(f"  Secuencia {sequence + 1}/{self.repetitions} → Usando video: {os.path.basename(current_video)}")

                cap = cv2.VideoCapture(current_video)
                if not cap.isOpened():
                    print(f"No se pudo abrir el video: {current_video}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < self.repetitions:
                    print(f"    Advertencia: El video tiene solo {total_frames} frames, menos que los {self.repetitions} necesarios")

                frame_indices = np.linspace(0, total_frames - 1, self.repetitions, dtype=int) # here we select the frames to process from the video

                for i, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) # Here we position the video to the frame we want to process, applying the model and landmarks
                    ret, frame = cap.read()

                    if not ret:
                        print(f"    No se pudo leer el frame {frame_idx}")
                        continue

                    image, results = self.mediapipe_detection(frame, holistic)
                    self.drawer.draw(image, results)
                    cv2.imshow('Video Detection', image)
                    cv2.waitKey(1)

                    keypoints, success = self.extractor.extract(results)
                    if success:
                        sequence_dir = os.path.join(self.mp_data, action, str(sequence))
                        os.makedirs(sequence_dir, exist_ok=True)
                        
                        #here we save the keypoints in a .npy file
                        npy_path = os.path.join(sequence_dir, f"{i}.npy")
                        np.save(npy_path, keypoints)
                        print(f"    Frame {i + 1}/{self.repetitions} guardado: {npy_path}")
                    else:
                        print(f"    Error extrayendo keypoints del frame {i + 1}")

                cap.release()
                print(f"  Completada secuencia {sequence + 1}")
                time.sleep(1)
                sequence += 1



        cv2.destroyAllWindows()
        return None, None
    

    