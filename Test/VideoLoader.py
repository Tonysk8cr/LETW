# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 02/06/2025

import os

class VideoLoader:
    """
    HI :)
    This class is used to load the video files from a specified directory.
    Parameters:
        directory: The directory where the video files are located.
        valid_extensions: A tuple of valid video file extensions.
    """
    def __init__(self, directory, valid_extensions=('.mp4', '.avi', '.mov', '.mkv')):
        self.directory = directory
        self.valid_extensions = valid_extensions

    def get_video_paths(self):
        #This will return a list of videos from the past specified
        return [
            os.path.join(self.directory, file)
            for file in os.listdir(self.directory)
            if file.lower().endswith(self.valid_extensions)
        ]