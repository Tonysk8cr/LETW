# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 02/06/2025


class LandmarkDrawer:
    """
    This class is used to draw landmarks on the video frames.
    Parameters:
        frame, this is the frame taken from the video which will be processed.
        results, this is the results from the MediaPipe Holistic model which contains the landmarks.
    """
    def __init__(self, mp_drawing, mp_holistic):
        self.mp_drawing = mp_drawing
        self.mp_holistic = mp_holistic

    def draw(self, frame, results): 
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=(103,207,245), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1))

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS)

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS)

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS)