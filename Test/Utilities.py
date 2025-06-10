import os
import cv2

class Utilities:

    @staticmethod
    def get_video_paths(directory, extensions=('.mp4', '.avi', '.mov')):
        """Devuelve una lista de rutas de videos en el directorio dado."""
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(extensions)
        ]

    @staticmethod
    def flip_horizontal(frame):
        """Devuelve el frame volteado horizontalmente."""
        return cv2.flip(frame, 1)