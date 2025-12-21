# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 23/09/2025

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
        signs: The list of signs that the model can recognize.
        frames: Number of frames per sequence.
    """

    def __init__(self, directory, repetitions, signs, frames, confidence, mp_path):
        self.directory = directory  # Here we store the directory where the videos are located
        self.extractor = KeypointExtractor()  # Instance of KeypointExtractor to extract keypoints
        self.processor = ImageProcessor()  # Instance of ImageProcessor to process the video frames
        self.data_extractor = DataExtractor(
            repetitions=repetitions, signs=signs, frames_per_sequence=frames, mp_path=mp_path
        )  # Instance of DataExtractor to handle video processing
        self.repetitions = repetitions
        self.signs = signs
        self.frames = frames
        self.counter = 0
        self.logger = Utilities.setup_logging()
        self.confidence = confidence
        self.mp_path = mp_path

    def run(self):
        """This will process the videos in teh directory, but only if there is one video directly on the folder example
        /Videos
        -------/Videos/Action1.mp4
        -------/Videos/Action2.mp4
        If we add subfolders, it will not work, we need to use the get_video_by_action method
        Used when the user chooses option 3 and then 1 from the main menu
        """
        video_paths = Utilities.get_video_paths(self.directory)
        self.counter = 0
        start_time = time.perf_counter()

        for video_path in video_paths:
            for i in range(self.repetitions):
                transform = Utilities.flip_horizontal if i % 2 == 0 else None  # Alternate transformations
                # Use the method directly
                print(f"Procesando: {video_path} (repetición {i + 1}/{self.repetitions})")
                self.logger.info(f"Procesando: {video_path} (repetición {i + 1}/{self.repetitions})")

                # Frame is not used here due to the nature of the method, but it is kept for consistency, remember that the frame is the image with the landmarks drawn on it
                frame, results = self.processor.process_video(
                    video_path, confidence=self.confidence, transform=transform
                )
                self.counter += 1

                if results:
                    keypoints, success = self.extractor.extract(results)
                    if success:
                        print(f"Keypoints extraídos correctamente, cantidad: {len(keypoints)}")
                        self.logger.info(f"Keypoints extraídos correctamente, cantidad: {len(keypoints)}")
                    else:
                        print("Error extrayendo keypoints.")
                        self.logger.error("Error extrayendo keypoints.")
                else:
                    print("No se detectaron landmarks.")
                    self.logger.warning("No se detectaron landmarks.")

        duration = time.perf_counter() - start_time
        print(f"\nProcesados: {self.counter} videos\nDuración total: {duration:.2f}")
        self.logger.info(f"\nProcesados: {self.counter} videos\nDuración total: {duration:.2f}")

    def extract_single_path(self):
        """This extracts the keypoints
        Used when the user selects option 2 and then 1 from the main menu
        """

        # Use the static method to get video paths
        video_paths = Utilities.get_video_paths(self.directory)
        start_time = time.perf_counter()

        for video_path in video_paths:
            print(f"\n=== Procesando video: {video_path} ===")
            self.logger.info(f"\n=== Procesando video: {video_path} ===")
            # Pass the flip horizontal transformation to the process_video method, used to create some augmentation
            self.data_extractor.process_video(
                video_path, transform=Utilities.flip_horizontal, confidence=self.confidence
            )

        duration = time.perf_counter() - start_time
        print(f"\nExtracción completada\nDuración total: {duration:.2f} segundos")
        self.logger.info(f"\nExtracción completada\nDuración total: {duration:.2f} segundos")

    def train(self):
        """This will extract the keypoints from the videos in the directory
        Used when the user selects option 3 and then 2 from the main menu
        """
        all_videos = Utilities.get_video_by_action(self.directory)
        self.counter = 0
        start_time = time.perf_counter()

        for action_name, video_paths in all_videos.items():
            print(f"\n=== Procesando acción: {action_name} ===")
            self.logger.info(f"\n=== Procesando acción: {action_name} ===")
            for video_path in video_paths:
                for i in range(self.repetitions):
                    transform = Utilities.flip_horizontal if i % 2 == 0 else None
                    print(f"Procesando: {video_path} (repetición {i + 1}/{self.repetitions})")
                    self.logger.info(f"Procesando: {video_path} (repetición {i + 1}/{self.repetitions})")

                    # Frame is not used here due to the nature of the method, but it is kept for consistency, remember that the frame is the image with the landmarks drawn on it
                    frame, results = self.processor.process_video(
                        video_path, confidence=self.confidence, transform=transform
                    )
                    self.counter += 1

                    if results:
                        keypoints, success = self.extractor.extract(results)
                        if success:
                            print(f"Keypoints extraídos correctamente, cantidad: {len(keypoints)}")
                            self.logger.info(f"Keypoints extraídos correctamente, cantidad: {len(keypoints)}")
                        else:
                            print("Error extrayendo keypoints.")
                            self.logger.error("Error extrayendo keypoints.")
                    else:
                        print("No se detectaron landmarks.")
                        self.logger.warning("No se detectaron landmarks.")

        # print(self.extractor.extract(results))
        duration = time.perf_counter() - start_time
        print(f"\nProcesados: {self.counter} videos\nDuración total: {duration:.2f}")
        self.logger.info(f"\nProcesados: {self.counter} videos\nDuración total: {duration:.2f}")

    def extract_parent_path(self):
        """Processes all videos in the parent directory, assuming they are organized by action.
        Used when the user selects option 2 and then 2 from the main menu
        """
        action_video_dict = Utilities.get_video_by_action(self.directory)
        start_time = time.perf_counter()

        for action, video_paths in action_video_dict.items():
            print(f"\n=== Procesando acción: {action} ===")
            self.logger.info(f"\n=== Procesando acción: {action} ===")
            action_folder_path = os.path.dirname(video_paths[0])  # All in the same action folder

            repetition = 0
            if repetition < self.repetitions / 2:
                transform = Utilities.flip_horizontal
                print(f"Procesando acción: {action} (repetición {repetition + 1})")
                self.logger.info(f"Procesando acción: {action} (repetición {repetition + 1})")

                # Process the video with the current transformation
                self.data_extractor.process_video(action_folder_path, transform=transform, confidence=self.confidence)

                repetition += 1
            else:
                transform = None
                print(f"Procesando acción: {action} (repetición {repetition + 1})")
                self.logger.info(f"Procesando acción: {action} (repetición {repetition + 1})")

                # Process the video without transformation
                self.data_extractor.process_video(action_folder_path, transform=transform, confidence=self.confidence)
                repetition += 1

        duration = time.perf_counter() - start_time
        print(f"\nExtracción completada\nDuración total: {duration:.2f} segundos")
        self.logger.info(f"\nExtracción completada\nDuración total: {duration:.2f} segundos")
