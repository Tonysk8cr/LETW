# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 02/06/2025

import time
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
        self.directory = directory  # Guardamos el directorio directamente
        self.extractor = KeypointExtractor()
        self.processor = ImageProcessor()
        self.data_extractor = DataExtractor()
        self.repetitions = repetitions
        self.counter = 0

    def run(self):
        # Usar el método estático directamente
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

    def extract(self):
        """Extraer datos de videos y guardarlos automáticamente"""
        # Usar el método estático directamente
        video_paths = Utilities.get_video_paths(self.directory)
        start_time = time.perf_counter()
        
        for video_path in video_paths:
            print(f"\n=== Procesando video: {video_path} ===")
            # Pasar la transformación flip_horizontal para que se alterne por secuencia
            self.data_extractor.process_video(video_path, transform=Utilities.flip_horizontal)
 
        duration = time.perf_counter() - start_time
        print(f"\nExtracción completada\nDuración total: {duration:.2f} segundos")