# Developed by Anthony Villalobos 08/01/2025
# Updated by Gemini 25/01/2026

import cv2
import mediapipe as mp
import numpy as np


class MediapipeHolistic:
    """
    A class to encapsulate the MediaPipe Holistic model and its usage.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the MediaPipe Holistic model.
        :param min_detection_confidence: Minimum detection confidence.
        :param min_tracking_confidence: Minimum tracking confidence.
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray, draw_landmarks: bool = False) -> tuple[np.ndarray, any]:
        """
        Processes a single frame with the MediaPipe Holistic model.
        :param frame: The frame to process.
        :param draw_results: Whether to draw the landmarks on the processed frame.
        :return: A tuple containing the processed image and the landmark results.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        processed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if draw_landmarks:
            self._draw_landmarks(processed_frame, results)

        return processed_frame, results

    def _draw_landmarks(self, frame: np.ndarray, results: any) -> None:
        """
        Draws the landmarks on a frame. This is a private helper method.
        :param frame: The frame to draw on.
        :param results: The landmark results from the MediaPipe Holistic model.
        """
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=(103, 207, 245), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
            )

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
            )

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.holistic.close()
