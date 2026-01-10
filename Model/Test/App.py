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
    DEFAULT_SIGNS: tuple[str, ...] = ("HOLA", "ADIOS", "GRACIAS", "POR FAVOR", "SI", "NO", "BUENO", "MAL", "COMER", "BEBER", "CASA", "TRABAJAR")
    DEFAULT_CONFIDENCE = 0.7

    repetitions: int = DEFAULT_REPETITIONS
    frames: int = DEFAULT_FRAMES
    # Convert tuple to list on initialization
    signs: list[str] = field(default_factory=lambda: list(Context.DEFAULT_SIGNS))
    video_paths: str = ""
    mp_path: str = ""
    confidence: float = DEFAULT_CONFIDENCE

    def __str__(self) -> str:
        return (
            f"Configuración - Repeticiones: {self.repetitions}, "
            f"Frames por secuencia: {self.frames}, "
            f"Signos: {self.signs}, "
            f"Confianza: {self.confidence}"
        )


# Options for the main menu
# ---------------------------------


def _get_user_confidence(current_confidence: float) -> float:
    """Helper to prompt user for confidence level and validate input."""
    print(Strings.Confidence.PROMPT.format(current_confidence))
    user_input = input(Strings.Confidence.INPUT)
    try:
        val = float(user_input)
        if 0 <= val <= 1:
            print(Strings.Confidence.SET_MSG.format(val))
            return val
        print(Strings.Confidence.OUT_OF_RANGE.format(Context.DEFAULT_CONFIDENCE))
    except ValueError:
        print(Strings.Confidence.INVALID_INPUT)

    print(Strings.Confidence.SET_MSG.format(Context.DEFAULT_CONFIDENCE))
    return Context.DEFAULT_CONFIDENCE


def _create_video_processor(ctx: Context, confidence: float, directory: str | None = None) -> VideoBatchProcessor:
    """Helper to create a VideoBatchProcessor instance with current context."""
    return VideoBatchProcessor(
        directory=directory,
        repetitions=ctx.repetitions,
        signs=ctx.signs,
        frames=ctx.frames,
        confidence=confidence,
        mp_path=ctx.mp_path,
    )


def cmd_create_project_directories(ctx: Context) -> bool:
    """Create the required project directory structure."""

    print(Strings.CreateDirs.CREATING)
    setup = SetUp(ctx.repetitions, signs=ctx.signs)
    data_path, actions, _ = setup.create_directories()
    print(Strings.CreateDirs.CREATED.format(data_path, actions))
    logger.info(f"Directorios creados en {data_path} para las acciones: {actions}")
    return True


def cmd_extract_data_from_videos(ctx: Context) -> bool:
    """Extract keypoint data from video files."""
    confidence = _get_user_confidence(ctx.confidence)
    logger.info(f"Confianza establecida en: {confidence}")

    # Main menu for data extraction
    print(Strings.ExtractData.MENU)
    user_choice = input(Strings.ExtractData.INPUT_OPTION)
    logger.info(f"El usuario seleccionó {user_choice} en el menú de extracción de datos")

    if user_choice == "1":
        print(Strings.ExtractData.EXTRACTING_SPECIFIC)
        print(Strings.ExtractData.NO_VIDEO_SPECIFIED)
        processor = _create_video_processor(ctx, confidence, directory=None)
        processor.extract_single_path()
    elif user_choice == "2":
        print(Strings.ExtractData.EXTRACTING_ALL)
        processor = _create_video_processor(ctx, confidence, directory=ctx.video_paths)
        processor.extract_parent_path()
    elif user_choice == "3":
        print(Strings.ProcessBatch.RETURNING_MAIN)
    else:
        print(Strings.MainMenu.INVALID_OPTION)

    return True


def cmd_process_video_batch(ctx: Context) -> bool:
    """Process a batch of videos."""
    confidence = _get_user_confidence(ctx.confidence)
    logger.info(f"Confianza establecida en: {confidence}")

    # Menu for batch video processing
    print(Strings.ProcessBatch.MENU)
    user_choice2 = input(Strings.ProcessBatch.INPUT_OPTION)
    logger.info(f"El usuario seleccionó {user_choice2} en el menú de procesamiento de videos")

    if user_choice2 == "1":
        print(Strings.ExtractData.NO_VIDEO_SPECIFIED)
        processor = _create_video_processor(ctx, confidence, directory=None)
        processor.run()
    elif user_choice2 == "2":
        processor = _create_video_processor(ctx, confidence, directory=ctx.video_paths)
        processor.train()
    elif user_choice2 == "3":
        print(Strings.ProcessBatch.RETURNING_MAIN)
    else:
        print(Strings.MainMenu.INVALID_OPTION)

    return True


def cmd_label_and_split_data(ctx: Context) -> bool:
    """Label the extracted data and split it into training and validation sets."""
    labeller = DataLabelling(repetitions=ctx.repetitions, signs=ctx.signs, frames=ctx.frames, mp_path=ctx.mp_path)
    labeller.split_data()
    return True


def cmd_train_lstm_model(ctx: Context) -> bool:
    """Build and train the LSTM model using the processed data."""
    training = TrainingLSTM(signs=ctx.signs, repetitions=ctx.repetitions, frames=ctx.frames, mp_path=ctx.mp_path)
    training.build_model()
    return True


def cmd_run_realtime_detection(ctx: Context) -> bool:
    """Perform real-time gesture detection using the camera."""
    confidence = _get_user_confidence(ctx.confidence)

    # Real-time detection
    print(Strings.RealtimeDetection.TEST_MSG)
    deteccion = RealtimeDetection(signs=ctx.signs, confidence=confidence)
    deteccion.real_time_detection()
    return True


def cmd_exit_program() -> bool:
    """Terminate the program execution."""
    logger.info("Saliendo del programa.")
    print(Strings.Exit.GOODBYE)
    return False


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
            (Strings.MainMenu.OPTION_CREATE_DIRECTORIES, lambda: cmd_create_project_directories(ctx)),
            (
                Strings.MainMenu.OPTION_EXTRACT_DATA,
                lambda: cmd_extract_data_from_videos(ctx),
            ),
            (
                Strings.MainMenu.OPTION_PROCESS_BATCH,
                lambda: cmd_process_video_batch(ctx),
            ),
            (Strings.MainMenu.OPTION_LABEL_DATA, lambda: cmd_label_and_split_data(ctx)),
            (Strings.MainMenu.OPTION_TRAIN_MODEL, lambda: cmd_train_lstm_model(ctx)),
            (Strings.MainMenu.OPTION_REALTIME_DETECTION, lambda: cmd_run_realtime_detection(ctx)),
            (Strings.MainMenu.OPTION_EXIT, lambda: cmd_exit_program()),
        ]

        # Display menu items starting from index 1
        for i, (desc, _) in enumerate(menu_items, 1):
            print(f"{i}. {desc}")
        print()

        user_choice = input(Strings.MainMenu.INPUT_OPTION.format(1, len(menu_items)))
        logger.info(f"El usuario seleccionó {user_choice} en el menú principal")

        if not user_choice.isdigit():
            print(Strings.MainMenu.INVALID_OPTION)
            logger.warning(f"Opción no válida seleccionada: {user_choice}")
            continue

        # Convert 1-based user input to 0-based index for list access
        choice_idx = int(user_choice) - 1
        if not (0 <= choice_idx < len(menu_items)):
            print(Strings.MainMenu.INVALID_OPTION)
            logger.warning(f"Opción no válida seleccionada: {user_choice}")
            continue

        _, command = menu_items[choice_idx]
        if not command():
            menu = False


if __name__ == "__main__":
    main()
