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


def create_project_directories(ctx: Context) -> bool:
    print(Strings.CreateDirs.CREATING)
    setup = SetUp(ctx.repetitions, signs=ctx.signs)
    data_path, actions, _ = setup.create_directories()
    print(Strings.CreateDirs.CREATED.format(data_path, actions))
    logger.info(f"Directorios creados en {data_path} para las acciones: {actions}")
    return True


def extract_data_from_videos(ctx: Context) -> bool:
    confidence = _get_user_confidence(ctx.confidence)
    logger.info(f"Confianza establecida en: {confidence}")

    # Main menu for data extraction
    print(Strings.ExtractData.MENU)
    user_choice = input(Strings.ExtractData.INPUT_OPTION)
    logger.info(f"El usuario seleccionó {user_choice} en el menú de extracción de datos")

    if user_choice == "1":
        print(Strings.ExtractData.EXTRACTING_SPECIFIC)
        # TODO: Ask the user for the video path instead of passing None
        print(Strings.ExtractData.NO_VIDEO_SPECIFIED)
        processor = VideoBatchProcessor(
            directory=None,
            repetitions=ctx.repetitions,
            signs=ctx.signs,
            frames=ctx.frames,
            confidence=confidence,
            mp_path=ctx.mp_path,
        )
        processor.extract_single_path()
    elif user_choice == "2":
        print(Strings.ExtractData.EXTRACTING_ALL)
        processor = VideoBatchProcessor(
            directory=ctx.video_paths,
            repetitions=ctx.repetitions,
            signs=ctx.signs,
            frames=ctx.frames,
            confidence=confidence,
            mp_path=ctx.mp_path,
        )
        processor.extract_parent_path()

    return True


def review_video_batch(ctx: Context) -> bool:
    confidence = _get_user_confidence(ctx.confidence)
    logger.info(f"Confianza establecida en: {confidence}")

    # Menu for batch video processing
    print(Strings.ReviewBatch.MENU)
    user_choice2 = input(Strings.ReviewBatch.INPUT_OPTION)
    logger.info(f"El usuario seleccionó {user_choice2} en el menú de procesamiento de videos")

    if user_choice2 == "1":
        # TODO: Ask the user for the video path instead of passing None
        print(Strings.ExtractData.NO_VIDEO_SPECIFIED)
        processor = VideoBatchProcessor(
            directory=None,
            repetitions=ctx.repetitions,
            confidence=confidence,
            signs=ctx.signs,
            frames=ctx.frames,
            mp_path=ctx.mp_path,
        )
        processor.run()
    elif user_choice2 == "2":
        processor = VideoBatchProcessor(
            ctx.video_paths,
            repetitions=ctx.repetitions,
            confidence=confidence,
            signs=ctx.signs,
            frames=ctx.frames,
            mp_path=ctx.mp_path,
        )
        processor.train()
    elif user_choice2 == "3":
        print(Strings.ReviewBatch.RETURNING_MAIN)
    else:
        print(Strings.MainMenu.INVALID_OPTION)

    return True


def label_and_split_data(ctx: Context) -> bool:
    labeller = DataLabelling(repetitions=ctx.repetitions, signs=ctx.signs, frames=ctx.frames, mp_path=ctx.mp_path)
    labeller.split_data()
    return True


def train_lstm_model(ctx: Context) -> bool:
    training = TrainingLSTM(signs=ctx.signs, repetitions=ctx.repetitions, frames=ctx.frames, mp_path=ctx.mp_path)
    training.build_model()
    return True


def run_realtime_detection(ctx: Context) -> bool:
    confidence = _get_user_confidence(ctx.confidence)

    # Real-time detection
    print(Strings.RealtimeDetection.TEST_MSG)
    deteccion = RealtimeDetection(signs=ctx.signs, confidence=confidence)
    deteccion.real_time_detection()
    return True


def exit_program() -> bool:
    logger.info("Saliendo del programa.")
    print(Strings.Exit.GOODBYE)
    return False


if __name__ == "__main__":
    main()