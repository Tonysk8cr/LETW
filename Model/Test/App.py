# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 23/09/2025

from dataclasses import dataclass, field

from DataLabelling import DataLabelling
from RealtimePrediction import RealtimeDetection
from SetUp import SetUp
from Strings import Strings
from TrainingLSTM import TrainingLSTM
from Utilities import Utilities
from VideoBatchProcessor import VideoBatchProcessor

logger = Utilities.setup_logging()


@dataclass
class Context:
    DEFAULT_REPETITIONS = 100
    DEFAULT_FRAMES = 30
    # Use Tuple for immutable default configuration
    DEFAULT_SIGNS: tuple[str, ...] = ("COMER", "HOY", "MAÑANA", "TOMAR")
    DEFAULT_CONFIDENCE = 0.7

    repetitions: int = DEFAULT_REPETITIONS
    frames: int = DEFAULT_FRAMES
    # Convert tuple to list on initialization
    signs: list[str] = field(default_factory=lambda: list(Context.DEFAULT_SIGNS))
    video_paths: str = ""
    mp_path: str = ""
    confidence: float = DEFAULT_CONFIDENCE

    def __str__(self) -> str:
        return (f"Configuración - Repeticiones: {self.repetitions}, "
                f"Frames por secuencia: {self.frames}, "
                f"Signos: {self.signs}, "
                f"Confianza: {self.confidence}")


def main() -> None:
    logger.info("Programa iniciado")

    paths = Utilities.training_paths()
    ctx = Context(
        video_paths=paths[0],
        mp_path=paths[1],
    )

    logger.info(str(ctx))

    print(Strings.MainMenu.WELCOME_MESSAGE)

    menu = True
    while menu:
        print(Strings.MainMenu.HEADER)
        menu_items = [
            (Strings.MainMenu.OPTION_CREATE_DIRECTORIES, lambda: create_project_directories(ctx)),
            (
                Strings.MainMenu.OPTION_EXTRACT_DATA,
                lambda: extract_data_from_videos(ctx),
            ),
            (
                Strings.MainMenu.OPTION_REVIEW_BATCH,
                lambda: review_video_batch(ctx),
            ),
            (Strings.MainMenu.OPTION_LABEL_DATA, lambda: label_and_split_data(ctx)),
            (Strings.MainMenu.OPTION_TRAIN_MODEL, lambda: train_lstm_model(ctx)),
            (Strings.MainMenu.OPTION_REALTIME_DETECTION, lambda: run_realtime_detection(ctx)),
            (Strings.MainMenu.OPTION_EXIT, lambda: exit_program()),
        ]

        for i, (desc, _) in enumerate(menu_items, 1):
            print(f"{i}. {desc}")
        print()

        user_choice = input(Strings.MainMenu.INPUT_OPTION.format(1, len(menu_items)))
        logger.info(f"El usuario seleccionó {user_choice} en el menú principal")

        try:
            choice_idx = int(user_choice) - 1
            if 0 <= choice_idx < len(menu_items):
                # Using explicit boolean check for continuation
                if not menu_items[choice_idx][1]():
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


def create_project_directories(ctx: Context) -> bool:
    print(Strings.CreateDirs.CREATING)
    setup = SetUp(ctx.repetitions, signs=ctx.signs)
    Data_Path, actions, video_path = setup.create_directories()
    print(Strings.CreateDirs.CREATED.format(Data_Path, actions))
    logger.info(f"Directorios creados en {Data_Path} para las acciones: {actions}")
    return True


def extract_data_from_videos(ctx: Context) -> bool:
    logger.info("El usuario seleccionó la opción 2 del menú principal")
    # Confidence config
    current_confidence = ctx.confidence
    print(Strings.Confidence.PROMPT.format(current_confidence))

    user_confidence = input(Strings.Confidence.INPUT)
    try:
        current_confidence = float(user_confidence)
        if current_confidence < 0 or current_confidence > 1:
            print(Strings.Confidence.OUT_OF_RANGE.format(Context.DEFAULT_CONFIDENCE))
            current_confidence = Context.DEFAULT_CONFIDENCE
    except ValueError:
        print(Strings.Confidence.INVALID_INPUT)
        current_confidence = Context.DEFAULT_CONFIDENCE

    print(Strings.Confidence.SET_MSG.format(current_confidence))
    logger.info(f"Confianza del modelo de mediapipe establecida en: {current_confidence}")

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
            repetitions=ctx.repetitions,
            signs=ctx.signs,
            frames=ctx.frames,
            confidence=current_confidence,
            mp_path=ctx.mp_path,
        )
        processor.extract_single_path()
    elif user_choice == "2":
        logger.info("El usuario seleccionó la opción 2 en el menú de extracción de datos")
        print(Strings.ExtractData.EXTRACTING_ALL)
        parent_directory = ctx.video_paths
        processor = VideoBatchProcessor(
            directory=parent_directory,
            repetitions=ctx.repetitions,
            signs=ctx.signs,
            frames=ctx.frames,
            confidence=current_confidence,
            mp_path=ctx.mp_path,
        )
        processor.extract_parent_path()

    return True


def review_video_batch(ctx: Context) -> bool:
    logger.info("El usuario seleccionó la opción 3 del menú principal")

    # Confidence config
    current_confidence = ctx.confidence
    print(Strings.Confidence.PROMPT.format(current_confidence))
    user_confidence = input(Strings.Confidence.INPUT)
    try:
        current_confidence = float(user_confidence)
        if current_confidence < 0 or current_confidence > 1:
            print(Strings.Confidence.OUT_OF_RANGE.format(Context.DEFAULT_CONFIDENCE))
            current_confidence = Context.DEFAULT_CONFIDENCE
    except ValueError:
        print(Strings.Confidence.INVALID_INPUT)
        current_confidence = Context.DEFAULT_CONFIDENCE

    print(Strings.Confidence.SET_MSG.format(current_confidence))
    logger.info(f"Confianza del modelo de mediapipe establecida en: {current_confidence}")

    # Menu for batch video processing
    print(Strings.ReviewBatch.MENU)
    user_choice2 = input(Strings.ReviewBatch.INPUT_OPTION)
    logger.info(f"El usuario seleccionó {user_choice2} en el menú de procesamiento de videos")

    if user_choice2 == "1":
        video_path = print(Strings.ExtractData.NO_VIDEO_SPECIFIED)
        processor = VideoBatchProcessor(
            directory=video_path,
            repetitions=ctx.repetitions,
            confidence=current_confidence,
            signs=ctx.signs,
            frames=ctx.frames,
            mp_path=ctx.mp_path,
        )
        processor.run()
    elif user_choice2 == "2":
        videos_directory = ctx.video_paths
        processor = VideoBatchProcessor(
            videos_directory,
            repetitions=ctx.repetitions,
            confidence=current_confidence,
            signs=ctx.signs,
            frames=ctx.frames,
            mp_path=ctx.mp_path,
        )
        processor.train()
    elif user_choice2 == "3":
        print(Strings.ReviewBatch.RETURNING_MAIN)
        logger.info("El usuario seleccionó la opción 3 en el menú de procesamiento de videos")
    else:
        print(Strings.MainMenu.INVALID_OPTION)

    return True


def label_and_split_data(ctx: Context) -> bool:
    logger.info("El usuario seleccionó la opción 4 del menú principal")
    labeller = DataLabelling(repetitions=ctx.repetitions, signs=ctx.signs, frames=ctx.frames, mp_path=ctx.mp_path)
    labeller.split_data()
    return True


def train_lstm_model(ctx: Context) -> bool:
    logger.info("El usuario seleccionó la option 5 del menú principal")
    training = TrainingLSTM(signs=ctx.signs, repetitions=ctx.repetitions, frames=ctx.frames, mp_path=ctx.mp_path)
    training.build_model()
    return True


def run_realtime_detection(ctx: Context) -> bool:
    logger.info("El usuario seleccionó la option 6 del menú principal")

    # Confidence config
    current_confidence = ctx.confidence
    print(Strings.Confidence.PROMPT_DETECTION.format(current_confidence))
    user_confidence = input(Strings.Confidence.INPUT)
    try:
        current_confidence = float(user_confidence)
        if current_confidence < 0 or current_confidence > 1:
            print(Strings.Confidence.OUT_OF_RANGE.format(Context.DEFAULT_CONFIDENCE))
            current_confidence = Context.DEFAULT_CONFIDENCE
    except ValueError:
        print(Strings.Confidence.INVALID_INPUT)
        current_confidence = Context.DEFAULT_CONFIDENCE

    print(Strings.Confidence.SET_MSG.format(current_confidence))

    # Real-time detection
    print(Strings.RealtimeDetection.TEST_MSG)
    deteccion = RealtimeDetection(signs=ctx.signs, confidence=current_confidence)
    deteccion.real_time_detection()
    return True


def exit_program() -> bool:
    logger.info("El usuario seleccionó la opción 7 del menú principal. Saliendo del programa.")
    print(Strings.Exit.GOODBYE)
    return False  # Leave, in order to exit the main loop


if __name__ == "__main__":
    main()
