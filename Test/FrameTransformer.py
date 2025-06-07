# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 02/06/2025

import cv2

class FrameTransformer:
    """
    This is pretty small, it transforms the video, used to build some data augmentation.
    Parameters:
        frame: The frame to be transformed.
    """
    @staticmethod
    def flip_horizontal(frame): #turns the frame horizontally
        return cv2.flip(frame, 1)