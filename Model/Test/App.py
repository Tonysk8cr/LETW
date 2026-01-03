# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 23/09/2025

from DataLabelling import DataLabelling
from RealtimePrediction import RealtimeDetection
from SetUp import SetUp
from TrainingLSTM import TrainingLSTM
from Utilities import Utilities
from VideoBatchProcessor import VideoBatchProcessor
from Strings import Strings


DEFAULT_CONFIDENCE = 0.7


def main():
    # logger
    logger = Utilities.setup_logging()
    logger.info("Programa iniciado")

    # Configuration
    repetitions = 100
    frames = 30
    signs = ["COMER", "HOY", "MAÑANA", "TOMAR"]
    logger.info(f"Configuración - Repeticiones: {repetitions}, Frames por secuencia: {frames}, Signos: {signs}")
    paths = Utilities.training_paths()
    video_paths = paths[0]
    mp_path = paths[1]
    confidence = DEFAULT_CONFIDENCE
    logger.info(f"Confianza del modelo de mediapipe establecida en: {confidence}")

    print(Strings.MainMenu.WELCOME_MESSAGE)

    menu = True
    while menu:
        print(Strings.MainMenu.HEADER)
        menu_items = [
            (Strings.MainMenu.OPTION_CREATE_DIRECTORIES, lambda: create_project_directories(repetitions, signs, logger)),
            (
                Strings.MainMenu.OPTION_EXTRACT_DATA,
                lambda: extract_data_from_videos(logger, confidence, repetitions, signs, frames, video_paths, mp_path),
            ),
            (
                Strings.MainMenu.OPTION_REVIEW_BATCH,
                lambda: review_video_batch(logger, confidence, repetitions, signs, frames, video_paths, mp_path),
            ),
            (Strings.MainMenu.OPTION_LABEL_DATA, lambda: label_and_split_data(logger, repetitions, signs, frames, mp_path)),
            (Strings.MainMenu.OPTION_TRAIN_MODEL, lambda: train_lstm_model(logger, signs, repetitions, frames, mp_path)),
            (Strings.MainMenu.OPTION_REALTIME_DETECTION, lambda: run_realtime_detection(logger, confidence, signs)),
            (Strings.MainMenu.OPTION_EXIT, lambda: exit_program(logger)),
        ]

        for i, (desc, _) in enumerate(menu_items, 1):
            print(f"{i}. {desc}")
        print()

        user_choice = input(Strings.MainMenu.INPUT_OPTION.format(1, len(menu_items)))
        logger.info(f"El usuario seleccionó {user_choice} en el menú principal")

        try:
            choice_idx = int(user_choice) - 1
            if 0 <= choice_idx < len(menu_items):
                result = menu_items[choice_idx][1]()
                if result is False:
                    menu = False
            else:
                print(Strings.MainMenu.INVALID_OPTION)
                logger.warning(f"Opción no válida seleccionada: {user_choice}")
        except ValueError:
            print(Strings.MainMenu.INVALID_OPTION)
            logger.warning(f"Opción no válida seleccionada: {user_choice}")


# Options for the main menu
# ---------------------------------

# The confidence variables is asked inside the options that require it
# This helps the developer to test different confidence values without having to restart the program


def create_project_directories(repetitions, signs, logger):
    print(Strings.CreateDirs.CREATING)
    setup = SetUp(repetitions, signs=signs)
    Data_Path, actions, video_path = setup.create_directories()
    print(Strings.CreateDirs.CREATED.format(Data_Path, actions))
    logger.info(f"Directorios creados en {Data_Path} para las acciones: {actions}")


def extract_data_from_videos(logger, confidence, repetitions, signs, frames, video_paths, mp_path):
    # log
    logger.info("El usuario seleccionó la opción 2 del menú principal")
    # Confidence config
    print(Strings.Confidence.PROMPT.format(confidence))

    user_confidence = input(Strings.Confidence.INPUT)
    try:
        confidence = float(user_confidence)
        if confidence < 0 or confidence > 1:
            print(Strings.Confidence.OUT_OF_RANGE.format(DEFAULT_CONFIDENCE))
            confidence = DEFAULT_CONFIDENCE
    except ValueError:
        print(Strings.Confidence.INVALID_INPUT)
        confidence = DEFAULT_CONFIDENCE

    print(Strings.Confidence.SET_MSG.format(confidence))
    logger.info(f"Confianza del modelo de mediapipe establecida en: {confidence}")

    # Main menu for data extraction
    print(Strings.ExtractData.MENU)
    user_choice = input(Strings.ExtractData.INPUT_OPTION)
    logger.info(f"El usuario seleccionó {user_choice} en el menú de extracción de datos")

    if user_choice == "1":
        logger.info("El usuario seleccionó la opción 1 en el menú de extracción de datos")
        print(Strings.ExtractData.EXTRACTING_SPECIFIC)
        video_path = print(Strings.ExtractData.NO_VIDEO_SPECIFIED)
        processor = VideoBatchProcessor(
            directory=video_path,
            repetitions=repetitions,
            signs=signs,
            frames=frames,
            confidence=confidence,
            mp_path=mp_path,
        )
        processor.extract_single_path()
    elif user_choice == "2":
        logger.info("El usuario seleccionó la opción 2 en el menú de extracción de datos")
        print(Strings.ExtractData.EXTRACTING_ALL)
        parent_directory = video_paths
        processor = VideoBatchProcessor(
            directory=parent_directory,
            repetitions=repetitions,
            signs=signs,
            frames=frames,
            confidence=confidence,
            mp_path=mp_path,
        )
        processor.extract_parent_path()


def review_video_batch(logger, confidence, repetitions, signs, frames, video_paths, mp_path):
    logger.info("El usuario seleccionó la opción 3 del menú principal")

    # Confidence config
    print(Strings.Confidence.PROMPT.format(confidence))
    user_confidence = input(Strings.Confidence.INPUT)
    try:
        confidence = float(user_confidence)
        if confidence < 0 or confidence > 1:
            print(Strings.Confidence.OUT_OF_RANGE.format(DEFAULT_CONFIDENCE))
            confidence = DEFAULT_CONFIDENCE
    except ValueError:
        print(Strings.Confidence.INVALID_INPUT)
        confidence = DEFAULT_CONFIDENCE

    print(Strings.Confidence.SET_MSG.format(confidence))
    logger.info(f"Confianza del modelo de mediapipe establecida en: {confidence}")

    # Menu for batch video processing
    print(Strings.ReviewBatch.MENU)
    user_choice2 = input(Strings.ReviewBatch.INPUT_OPTION)
    logger.info(f"El usuario seleccionó {user_choice2} en el menú de procesamiento de videos")

    if user_choice2 == "1":
        video_path = print(Strings.ExtractData.NO_VIDEO_SPECIFIED)
        processor = VideoBatchProcessor(
            directory=video_path,
            repetitions=repetitions,
            confidence=confidence,
            signs=signs,
            frames=frames,
            mp_path=mp_path,
        )
        processor.run()
    elif user_choice2 == "2":
        videos_directory = video_paths
        processor = VideoBatchProcessor(
            videos_directory,
            repetitions=repetitions,
            confidence=confidence,
            signs=signs,
            frames=frames,
            mp_path=mp_path,
        )
        processor.train()
    elif user_choice2 == "3":
        print(Strings.ReviewBatch.RETURNING_MAIN)
        logger.info("El usuario seleccionó la opción 3 en el menú de procesamiento de videos")
        return
    else:
        print(Strings.MainMenu.INVALID_OPTION)
        return


def label_and_split_data(logger, repetitions, signs, frames, mp_path):
    logger.info("El usuario seleccionó la opción 4 del menú principal")
    labeller = DataLabelling(repetitions=repetitions, signs=signs, frames=frames, mp_path=mp_path)
    labeller.split_data()


def train_lstm_model(logger, signs, repetitions, frames, mp_path):
    logger.info("El usuario seleccionó la option 5 del menú principal")
    training = TrainingLSTM(signs=signs, repetitions=repetitions, frames=frames, mp_path=mp_path)
    training.build_model()


def run_realtime_detection(logger, confidence, signs):
    logger.info("El usuario seleccionó la option 6 del menú principal")

    # Confidence config
    print(Strings.Confidence.PROMPT_DETECTION.format(confidence))
    user_confidence = input(Strings.Confidence.INPUT)
    try:
        confidence = float(user_confidence)
        if confidence < 0 or confidence > 1:
            print(Strings.Confidence.OUT_OF_RANGE.format(DEFAULT_CONFIDENCE))
            confidence = DEFAULT_CONFIDENCE
    except ValueError:
        print(Strings.Confidence.INVALID_INPUT)
        confidence = DEFAULT_CONFIDENCE

    print(Strings.Confidence.SET_MSG.format(confidence))


    # Real-time detection
    print(Strings.RealtimeDetection.TEST_MSG)
    deteccion = RealtimeDetection(signs=signs, confidence=confidence)
    deteccion.real_time_detection()


def exit_program(logger):
    logger.info("El usuario seleccionó la opción 7 del menú principal. Saliendo del programa.")
    print(Strings.Exit.GOODBYE)
    return False  # Leave, in order to exit the main loop


if __name__ == "__main__":
    main()
