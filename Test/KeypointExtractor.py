# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 10/07/2025

import numpy as np

class KeypointExtractor:
    """
    Meant to extract keypoints from MediaPipe results.
    Parameters:
        results: The results from the MediaPipe Holistic model which contains the landmarks.
    """
    @staticmethod
    def extract(results):
        try:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
            left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            #10/7/2025  we added this to print statements to check the number of landmarks extracted, done to debug
            #print(len(results.left_hand_landmarks.landmark))
            #print(len(results.right_hand_landmarks.landmark))
            return np.concatenate([pose, face, left_hand, right_hand]), True
        except Exception as e:
            print(f"Error extrayendo keypoints: {e}")
            return None, False