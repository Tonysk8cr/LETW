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

    def process_frame(self, frame: np.ndarray, draw_results: bool = False) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Processes a single frame with the MediaPipe Holistic model.
        :param frame: The frame to process.
        :param draw_results: Whether to draw the landmarks on the processed frame.
        :return: A tuple containing the processed image, the extracted keypoints, and a success flag.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        processed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if draw_results:
            self._draw_landmarks(processed_frame, results)

        keypoints, success = self._extract_keypoints(results)
        return processed_frame, keypoints, success

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

    def _normalize_pose_landmarks(self, pose_landmarks):
        """Normalize using shoulders as a reference."""
        if len(pose_landmarks) < 132:
            return pose_landmarks

        try:
            # Used to extract the coordinates of the shoulders (landmark 11 and 12)
            left_shoulder = pose_landmarks[11 * 4 : (11 * 4) + 3]
            right_shoulder = pose_landmarks[12 * 4 : (12 * 4) + 3]

            # Center and scale
            center = (left_shoulder + right_shoulder) / 2
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

            normalized_pose = []
            for i in range(0, 132, 4):
                if i + 3 < len(pose_landmarks):
                    point = pose_landmarks[i : i + 3]
                    normalized_point = (point - center) / (shoulder_distance + 1e-6)
                    normalized_pose.extend(normalized_point)
                    normalized_pose.append(pose_landmarks[i + 3])
            return np.array(normalized_pose)
        except Exception as e:
            print(f"Error in normalize_pose_landmarks: {e}")
            return pose_landmarks  # Return original if normalization fails

    def _normalize_hand_landmarks(self, hand_landmarks):
        """Normalize hand landmarks relative to the wrist using finger length as scale."""
        if len(hand_landmarks) < 63:
            return hand_landmarks

        try:
            wrist = hand_landmarks[0:3]
            middle_finger_tip = hand_landmarks[12 * 3 : (12 * 3) + 3]
            hand_scale = np.linalg.norm(middle_finger_tip - wrist)

            normalized_hand = []
            for i in range(0, 63, 3):
                point = hand_landmarks[i : i + 3]
                normalized_point = (point - wrist) / (hand_scale + 1e-6)
                normalized_hand.extend(normalized_point)
            return np.array(normalized_hand)
        except Exception as e:
            print(f"Error in normalize_hand_landmarks: {e}")
            return hand_landmarks  # Return original if normalization fails

    def _extract_keypoints(self, results):
        try:
            if not results or (
                not results.pose_landmarks
                and not results.face_landmarks
                and not results.left_hand_landmarks
                and not results.right_hand_landmarks
            ):
                return np.zeros(33 * 4 + 468 * 3 + 21 * 3 + 21 * 3), False

            pose = (
                np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
                if results.pose_landmarks
                else np.zeros(33 * 4)
            )
            face = (
                np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
                if results.face_landmarks
                else np.zeros(468 * 3)
            )
            left_hand = (
                np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
                if results.left_hand_landmarks
                else np.zeros(21 * 3)
            )
            right_hand = (
                np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
                if results.right_hand_landmarks
                else np.zeros(21 * 3)
            )

            # Apply normalized keypoints
            if results.pose_landmarks:
                pose = self._normalize_pose_landmarks(pose)
            if results.left_hand_landmarks:
                left_hand = self._normalize_hand_landmarks(left_hand)
            if results.right_hand_landmarks:
                right_hand = self._normalize_hand_landmarks(right_hand)

            return np.concatenate([pose, face, left_hand, right_hand]), True
        except Exception as e:
            print(f"Error extrayendo keypoints: {e}")
            return np.zeros(33 * 4 + 468 * 3 + 21 * 3 + 21 * 3), False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.holistic.close()
