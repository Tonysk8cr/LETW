# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 10/07/2025

import cv2
import mediapipe as mp
from LandmarkDrawer import LandmarkDrawer

class ImageProcessor:
    """
    Converts the image from bgr to rgb which is the format used by mediapipe.
    Uses mediapipes holistic model to process the video frames and draw landmarks.
    returns the last frame and result with landmarks."""
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawer = LandmarkDrawer(self.mp_drawing, self.mp_holistic)

    def mediapipe_detection(self, frame, model): #processes the frame with mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), results

    def process_video(self, video_path, transform=None): #loads the video, process it and draws the landmarks
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"No se pudo abrir el video: {video_path}")
            return None, None

        last_frame, last_result = None, None

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if transform:
                    frame = transform(frame)

                image, results = self.mediapipe_detection(frame, holistic)

                if results.pose_landmarks or results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                    last_frame = frame
                    last_result = results

                self.drawer.draw(image, results)
                #Eliminate the comment to show the video with the landmarks, used while coding not needed as we already know that it works
                cv2.imshow('Video Detection', image)
                
                cap.read(); cap.read(); cap.read()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        return last_frame, last_result