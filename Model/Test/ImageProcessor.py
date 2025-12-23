# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a video file instead of the camera
# Updated by Anthony Villalobos 23/09/2025

import cv2
import mediapipe as mp
from LandmarkDrawer import LandmarkDrawer
from Utilities import Utilities


class ImageProcessor:
    """
    Converts the image from BGR to RGB, which is the format used by MediaPipe.
    Uses MediaPipe's Holistic model to process the video frames and draw landmarks.
    Returns the last frame and the results with landmarks.
    """

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawer = LandmarkDrawer(self.mp_drawing, self.mp_holistic)
        self.logger = Utilities.setup_logging()

    def mediapipe_detection(self, frame, model):  # Processes the frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), results

    def process_video(
        self, video_path, confidence, transform=None
    ):  # Loads the video, processes it and draws the landmarks
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"No se pudo abrir el vídeo: {video_path}")
            self.logger.error(f"No se pudo abrir el vídeo: {video_path}")
            return None, None

        last_frame, last_result = None, None

        with self.mp_holistic.Holistic(
            min_detection_confidence=confidence, min_tracking_confidence=confidence
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if transform:
                    frame = transform(frame)

                image, results = self.mediapipe_detection(frame, holistic)

                if (
                    results.pose_landmarks
                    or results.face_landmarks
                    or results.left_hand_landmarks
                    or results.right_hand_landmarks
                ):
                    last_frame = frame
                    last_result = results

                self.drawer.draw(image, results)
                # Remove the comment to show the video with the landmarks; used during development, not needed now
                cv2.imshow("Video Detection", image)

                cap.read()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        return last_frame, last_result
