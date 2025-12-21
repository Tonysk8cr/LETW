# Developed by Anthony Villalobos 08/01/2025
# Updated by Anthony Villalobos 23/09/2025

import cv2
import mediapipe as mp
import os
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

    def __init__(self, repetitions, frames_per_sequence, signs, mp_path):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawer = LandmarkDrawer(
            self.mp_drawing, self.mp_holistic
        )  # Instance of LandmarkDrawer to draw landmarks on the video frames
        self.extractor = KeypointExtractor()  # Instance of KeypointExtractor to extract keypoints
        self.signs = signs
        self.mp_data = mp_path
        self.repetitions = repetitions  # Number of repetitions for each video
        self.frames_per_sequence = frames_per_sequence
        self.logger = Utilities.setup_logging()

    def mediapipe_detection(self, frame, model):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), results

    def process_video(self, video_path, confidence, transform=None):
        """This method is the one in charge of processing the video and extracting the keypoints from the specified video path.
        Variables:
        video_path: The path to the video file or directory containing videos.
        video_files: A list of video files to process.
        action: The action label derived from the video filename or directory name.
        sequence: The current sequence number for the action being processed.
        frame, image: The current frame being processed and the image with drawn landmarks.
        results: The results from MediaPipe Holistic processing.
        keypoints: The extracted keypoints from the current frame.
        npy_path: The path where the keypoints will be saved as a .npy file.
        """
        if os.path.isdir(video_path):
            # Detects whether video_path is a directory; if it is a subfolder it is considered an action
            # and the videos inside it are treated as that action's videos.
            video_files = Utilities.get_video_paths(video_path)
            if not video_files:
                print(f"No se encontraron videos en el directorio: {video_path}")
                self.logger.error(f"No se encontraron videos en el directorio: {video_path}")
                return
            action = os.path.basename(video_path).upper()
            video_files = (video_files * ((self.repetitions // len(video_files)) + 1))[
                : self.repetitions
            ]  # Here we ensure that we have enough videos to process the required repetitions
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
                self.logger.error(f"No se pudo determinar la acción para el video: {video_filename}")
                return

        print(f"\nProcesando acción: {action} con {len(video_files)} videos disponibles")

        # Main MediaPipe Holistic model

        with self.mp_holistic.Holistic(
            min_detection_confidence=confidence, min_tracking_confidence=confidence
        ) as holistic:
            sequence = 0

            for video_idx in range(self.repetitions):
                current_video = video_files[video_idx]
                print(f"Procesando video {video_idx + 1}/{len(video_files)}: {os.path.basename(current_video)}")
                self.logger.info(
                    f"Procesando video {video_idx + 1}/{len(video_files)}: {os.path.basename(current_video)}"
                )
                if sequence >= self.repetitions:
                    break  # If we have processed enough repetitions, we stop processing more videos

                print(
                    f"  Secuencia {sequence + 1}/{self.repetitions} → Usando video: {os.path.basename(current_video)}"
                )
                self.logger.info(
                    f"  Secuencia {sequence + 1}/{self.repetitions} → Usando video: {os.path.basename(current_video)}"
                )

                cap = cv2.VideoCapture(current_video)
                if not cap.isOpened():
                    print(f"No se pudo abrir el video: {current_video}")
                    self.logger.error(f"No se pudo abrir el video: {current_video}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < self.frames_per_sequence:
                    print(
                        f"    Advertencia: El video tiene solo {total_frames} frames, menos que los {self.repetitions} necesarios"
                    )
                    self.logger.warning(
                        f"    Advertencia: El video tiene solo {total_frames} frames, menos que los {self.repetitions} necesarios"
                    )

                frame_indices = np.linspace(
                    0, total_frames - 1, self.frames_per_sequence, dtype=int
                )  # Here we select the frames to process from the video

                for i, frame_idx in enumerate(frame_indices):
                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES, frame_idx
                    )  # Here we position the video to the frame we want to process, applying the model and landmarks
                    ret, frame = cap.read()

                    if not ret:
                        print(f"    No se pudo leer el frame {frame_idx}")
                        self.logger.error(f"    No se pudo leer el frame {frame_idx}")
                        continue

                    if transform:
                        frame = transform(frame)

                    image, results = self.mediapipe_detection(frame, holistic)
                    self.drawer.draw(image, results)
                    # Remove the comment to show the video with the landmarks; used during development and not required now
                    cv2.imshow("Video Detection", image)
                    cv2.waitKey(1)

                    keypoints, success = self.extractor.extract(results)
                    if success:
                        sequence_dir = os.path.join(self.mp_data, action, str(sequence))
                        os.makedirs(sequence_dir, exist_ok=True)

                        # Here we save the keypoints in a .npy file
                        npy_path = os.path.join(sequence_dir, f"{i}.npy")
                        np.save(npy_path, keypoints)
                        print(f"    Frame {i + 1}/{self.frames_per_sequence} guardado: {npy_path}")
                        self.logger.info(f"    Frame {i + 1}/{self.frames_per_sequence} guardado: {npy_path}")
                    else:
                        print(f"    Error extrayendo keypoints del frame {i + 1}")
                        self.logger.error(f"    Error extrayendo keypoints del frame {i + 1}")

                cap.release()
                print(f"  Completada secuencia {sequence + 1}")
                self.logger.info(f"  Completada secuencia {sequence + 1}")
                time.sleep(1)
                sequence += 1

        cv2.destroyAllWindows()
        return None, None
