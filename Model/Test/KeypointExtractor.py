# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 10/07/2025
# Fixed broadcasting error - 17/07/2025

import numpy as np

class KeypointExtractor:
    """
    Meant to extract keypoints from MediaPipe results.
    Parameters:
        results: The results from the MediaPipe Holistic model which contains the landmarks.
    """

    @staticmethod
    def normalize_pose_landmarks(pose_landmarks):
        """Normalize using shoulders as a reference"""
        if len(pose_landmarks) < 132:
            return pose_landmarks
        
        try:
            # Used to extract the coordenates of the shoulders(landmark 11 and 12)
            left_shoulder = pose_landmarks[11*4: (11*4)+3]  # FIXED: removed extra colon
            right_shoulder = pose_landmarks[12*4: (12*4)+3]  # FIXED: removed extra colon

            # Center and escale
            center = (left_shoulder + right_shoulder) / 2
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

            normalized_pose = []
            for i in range(0, 132, 4):
                if i + 3 < len(pose_landmarks):
                    point = pose_landmarks[i:i+3]
                    normalized_point = (point - center) / (shoulder_distance + 1e-6)
                    normalized_pose.extend(normalized_point)
                    normalized_pose.append(pose_landmarks[i+3])
            return np.array(normalized_pose)
        except Exception as e:
            print(f"Error in normalize_pose_landmarks: {e}")
            return pose_landmarks  # Return original if normalization fails
    
    @staticmethod
    def normalize_hand_landmarks(hand_landmarks):
        if len(hand_landmarks) < 63:
            return hand_landmarks
        
        try:
            wrist = hand_landmarks[0:3]
            middle_finger_tip = hand_landmarks[12*3:(12*3)+3]
            hand_scale = np.linalg.norm(middle_finger_tip - wrist)

            normalized_hand = []
            for i in range(0, 63, 3):
                point = hand_landmarks[i:i+3]
                normalized_point = (point - wrist) / (hand_scale + 1e-6)
                normalized_hand.extend(normalized_point)
            return np.array(normalized_hand)
        except Exception as e:
            print(f"Error in normalize_hand_landmarks: {e}")
            return hand_landmarks  # Return original if normalization fails

    @staticmethod
    def extract(results):
        try:
            # FIXED: Added validation to prevent None results
            if not results:
                return np.zeros(33*4 + 468*3 + 21*3 + 21*3), False
            
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
            left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            
            # Apply normalized keypoints
            if results.pose_landmarks:
                pose = KeypointExtractor.normalize_pose_landmarks(pose)
            if results.left_hand_landmarks:
                left_hand = KeypointExtractor.normalize_hand_landmarks(left_hand)
            if results.right_hand_landmarks:
                right_hand = KeypointExtractor.normalize_hand_landmarks(right_hand)
            
            return np.concatenate([pose, face, left_hand, right_hand]), True
        except Exception as e:
            print(f"Error extrayendo keypoints: {e}")
            return np.zeros(33*4 + 468*3 + 21*3 + 21*3), False  # FIXED: Return proper sized array instead of None