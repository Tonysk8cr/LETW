# Developed by Anthony Villalobos 08/01/2025
# Updated by Anthony Villalobos 23/09/2025

import time
from pathlib import Path

import cv2
import numpy as np
from MediapipeHolistic import MediapipeHolistic
from Utilities import Utilities


class DataExtractor:
    """
    Takes the data from one video and extracts the landmarks using MediaPipe Holistic.
    This is used to get the data upon which the model will be trained.
    """

    KEYPOINT_SIZE = 1662  # pose 33*4 + face 468*3 + hands 2*21*3, same as MediapipeHolistic's empty keypoints
    # shortcut: 30% picked without measuring real data, tune once we know typical miss rates
    MAX_MISSING_RATIO = 0.3  # a sequence with more empty frames than this is dropped

    @staticmethod
    def _clear_sequence(sequence_dir):
        """Delete the .npy files of a sequence but keep its folder."""
        sequence_dir.mkdir(parents=True, exist_ok=True)
        for npy in sequence_dir.glob("*.npy"):
            npy.unlink()

    def __init__(self, repetitions, frames_per_sequence, signs, mp_path):
        self.signs = signs
        self.mp_data = Path(mp_path)
        self.repetitions = repetitions  # Number of repetitions for each video
        self.frames_per_sequence = frames_per_sequence
        self.logger = Utilities.setup_logging()

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
        video_path = Path(video_path)
        if video_path.is_dir():
            # Detects whether video_path is a directory; if it is a subfolder it is considered an action
            # and the videos inside it are treated as that action's videos.
            video_files = Utilities.get_video_paths(str(video_path))
            if not video_files:
                print(f"No se encontraron videos en el directorio: {video_path}")
                self.logger.error(f"No se encontraron videos en el directorio: {video_path}")
                return None
            action = video_path.name.upper()
            video_files = (video_files * ((self.repetitions // len(video_files)) + 1))[
                : self.repetitions
            ]  # Here we ensure that we have enough videos to process the required repetitions
        else:
            video_files = [str(video_path)] * self.repetitions
            video_filename = video_path.name
            action = None
            for sign in self.signs:
                if sign in video_filename.upper():
                    action = sign
                    break
            if not action:
                print(f"No se pudo determinar la acción para el video: {video_filename}")
                self.logger.error(f"No se pudo determinar la acción para el video: {video_filename}")
                return None

        print(f"\nProcesando acción: {action} con {len(video_files)} videos disponibles")

        # Main MediaPipe Holistic model

        with MediapipeHolistic(min_detection_confidence=confidence, min_tracking_confidence=confidence) as holistic:
            sequence = 0
            kept = dropped = 0

            for video_idx in range(self.repetitions):
                current_video = Path(video_files[video_idx])
                print(f"Procesando video {video_idx + 1}/{len(video_files)}: {current_video.name}")
                self.logger.info(f"Procesando video {video_idx + 1}/{len(video_files)}: {current_video.name}")

                print(f"  Secuencia {sequence + 1}/{self.repetitions} → Usando video: {current_video.name}")
                self.logger.info(f"  Secuencia {sequence + 1}/{self.repetitions} → Usando video: {current_video.name}")

                cap = cv2.VideoCapture(str(current_video))
                if not cap.isOpened():
                    print(f"No se pudo abrir el video: {current_video}")
                    self.logger.error(f"No se pudo abrir el video: {current_video}")
                    # drop this sequence so labelling can't load data left by an earlier run
                    self._clear_sequence(self.mp_data / action / str(sequence))
                    dropped += 1
                    sequence += 1
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < self.frames_per_sequence:
                    print(
                        f"    Advertencia: El video tiene solo {total_frames} frames, menos que los {self.frames_per_sequence} necesarios"
                    )
                    self.logger.warning(
                        f"    Advertencia: El video tiene solo {total_frames} frames, menos que los {self.frames_per_sequence} necesarios"
                    )

                frame_indices = np.linspace(
                    0, total_frames - 1, self.frames_per_sequence, dtype=int
                )  # Here we select the frames to process from the video

                sequence_dir = self.mp_data / action / str(sequence)
                self._clear_sequence(sequence_dir)  # no stale frames from earlier runs
                missing = 0  # frames in this sequence saved as zeros

                for i, frame_idx in enumerate(frame_indices):
                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES, frame_idx
                    )  # Here we position the video to the frame we want to process, applying the model and landmarks
                    ret, frame = cap.read()

                    if not ret:
                        print(f"    No se pudo leer el frame {frame_idx}")
                        self.logger.error(f"    No se pudo leer el frame {frame_idx}")
                        np.save(sequence_dir / f"{i}.npy", np.zeros(self.KEYPOINT_SIZE))  # Save empty keypoints if frame read fails
                        missing += 1
                        continue

                    if transform:
                        frame = transform(frame)

                    image, keypoints, success = holistic.process_frame(frame, draw_results=True)
                    # Remove the comment to show the video with the landmarks; used during development and not required now
                    cv2.imshow("Video Detection", image)
                    cv2.waitKey(1)

                    if success:
                        npy_path = sequence_dir / f"{i}.npy"
                        np.save(npy_path, keypoints)
                        print(f"    Frame {i + 1}/{self.frames_per_sequence} guardado: {npy_path}")
                        self.logger.info(f"    Frame {i + 1}/{self.frames_per_sequence} guardado: {npy_path}")
                    else:
                        missing += 1
                        np.save(sequence_dir / f"{i}.npy", keypoints)  # keypoints are already zeros when nothing was detected
                        print(f"   Frame {i + 1} sin detección de landmarks, guardando keypoints vacíos")
                        self.logger.error(f"   Frame {i + 1} sin detección de landmarks, guardando keypoints vacíos")

                cap.release()
                if missing > self.MAX_MISSING_RATIO * self.frames_per_sequence:
                    self._clear_sequence(sequence_dir)
                    dropped += 1
                    msg = f"  Secuencia {sequence + 1} descartada: {missing}/{self.frames_per_sequence} frames sin detección"
                    print(msg)
                    self.logger.warning(msg)
                else:
                    kept += 1
                    print(f"  Completada secuencia {sequence + 1}")
                    self.logger.info(f"  Completada secuencia {sequence + 1}")
                time.sleep(1)
                sequence += 1

        summary = f"Resumen {action}: {kept} secuencias guardadas, {dropped} descartadas"
        print(f"\n{summary}")
        self.logger.info(summary)
        cv2.destroyAllWindows()
        return None, None
