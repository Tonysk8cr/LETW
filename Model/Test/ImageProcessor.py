# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a video file instead of the camera
# Updated by Anthony Villalobos 23/09/2025

import cv2
from MediapipeHolistic import MediapipeHolistic
from Utilities import Utilities


class ImageProcessor:
    """
    Converts the image from BGR to RGB, which is the format used by MediaPipe.
    Uses MediaPipe's Holistic model to process the video frames and draw landmarks.
    Returns the last frame and the results with landmarks.
    """

    def __init__(self):
        self.logger = Utilities.setup_logging()

    def process_video(
        self, video_path, confidence, transform=None
    ):  # Loads the video, processes it and draws the landmarks
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"No se pudo abrir el vídeo: {video_path}")
            self.logger.error(f"No se pudo abrir el vídeo: {video_path}")
            return None, None

        last_keypoints, last_success = None, False

        with MediapipeHolistic(min_detection_confidence=confidence, min_tracking_confidence=confidence) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if transform:
                    frame = transform(frame)

                image, keypoints, success = holistic.process_frame(frame, draw_results=True)

                if success:
                    last_keypoints = keypoints
                    last_success = success

                # Remove the comment to show the video with the landmarks; used during development, not needed now
                cv2.imshow("Video Detection", image)

                cap.read()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        return last_keypoints, last_success
