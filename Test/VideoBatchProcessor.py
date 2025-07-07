# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 02/06/2025

import time
import os
from KeypointExtractor import KeypointExtractor
from Utilities import Utilities
from ImageProcessor import ImageProcessor
from DataExtraction import DataExtractor

class VideoBatchProcessor:
    """
    Controls the batch processing of videos.
    Parameters:
        directory: The directory containing video files.
        repetitions: Number of times to process each video with transformations.
    """
    def __init__(self, directory, repetitions=30):
        self.directory = directory  # Here we store the directory where the videos are located
        self.extractor = KeypointExtractor() # Instance of KeypointExtractor to extract keypoints
        self.processor = ImageProcessor() # Instance of ImageProcessor to process the video frames
        self.data_extractor = DataExtractor() # Instance of DataExtractor to handle video processing
        self.repetitions = repetitions
        self.counter = 0

    def run(self):
        # Uses the static method to get video paths
        """This will process the videos in teh directory, but only if there is one video directly on the folder example
        /Videos
        -------/Videos/Action1.mp4
        -------/Videos/Action2.mp4
        If we add subfolders, it will not work, we need to use the get_video_by_action method"""
        video_paths = Utilities.get_video_paths(self.directory)
        self.counter = 0
        start_time = time.perf_counter()

        for video_path in video_paths:
            for i in range(self.repetitions):
                # Usar el método estático directamente
                transform = Utilities.flip_horizontal if i % 2 else None
                print(f"Procesando: {video_path} (repetición {i+1}/{self.repetitions})")

                frame, results = self.processor.process_video(video_path, transform)
                self.counter += 1

                if results:
                    keypoints, success = self.extractor.extract(results)
                    if success:
                        print(f"Keypoints extraídos correctamente, cantidad: {len(keypoints)}")
                    else:
                        print("Error extrayendo keypoints.")
                else:
                    print("No se detectaron landmarks.")
 
        duration = time.perf_counter() - start_time
        print(f"\nProcesados: {self.counter} videos\nDuración total: {duration:.2f}")

    def extract_single_path(self):
        """This extracts the keypoints"""
        # Usar el método estático directamente
        video_paths = Utilities.get_video_paths(self.directory)
        start_time = time.perf_counter()
        
        for video_path in video_paths:
            print(f"\n=== Procesando video: {video_path} ===")
            # Pasar la transformación flip_horizontal para que se alterne por secuencia
            self.data_extractor.process_video(video_path, transform=Utilities.flip_horizontal)

 
        duration = time.perf_counter() - start_time
        print(f"\nExtracción completada\nDuración total: {duration:.2f} segundos")


    def train(self):
        """This will extract the keypoints from the videos in the directory"""
        all_videos = Utilities.get_video_by_action(self.directory)
        self.counter = 0
        start_time = time.perf_counter()

        for action_name, video_paths in all_videos.items():
            print(f"\n=== Procesando acción: {action_name} ===")
            for video_path in video_paths:
                for i in range(self.repetitions):
                    print(f"Procesando: {video_path} (repetición {i+1}/{self.repetitions})")

                    frame, results = self.processor.process_video(video_path)
                    self.counter += 1

                    if results:
                        keypoints, success = self.extractor.extract(results)
                        if success:
                            print(f"Keypoints extraídos correctamente, cantidad: {len(keypoints)}")
                        else:
                            print("Error extrayendo keypoints.")
                    else:
                        print("No se detectaron landmarks.")


        duration = time.perf_counter() - start_time
        print(f"\nProcesados: {self.counter} videos\nDuración total: {duration:.2f}")

    def extract_parent_path(self):
        """Processes all videos in the parent directory, assuming they are organized by action."""
        action_video_dict = Utilities.get_video_by_action(self.directory)
        start_time = time.perf_counter()

        for action, video_paths in action_video_dict.items():
            print(f"\n=== Procesando acción: {action} ===")
            action_folder_path = os.path.dirname(video_paths[0])  # Todas están en la misma carpeta
            self.data_extractor.process_video(action_folder_path)

        duration = time.perf_counter() - start_time
        print(f"\nExtracción completada\nDuración total: {duration:.2f} segundos")
