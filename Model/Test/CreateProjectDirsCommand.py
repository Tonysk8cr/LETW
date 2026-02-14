# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 23/09/2025

from pathlib import Path

from Strings import Strings
from Utilities import Utilities


class CreateProjectDirsCommand:
    """Command to create the required project directory structure."""

    def __init__(self, workspace_path: Path, repetitions: int, signs: list[str]):
        self.workspace_path = workspace_path
        self.signs = signs
        self.logger = Utilities.setup_logging()
        self.repetitions = repetitions

    @property
    def mp_data_path(self) -> Path:
        return self.workspace_path / "MP_Data"

    @property
    def video_data_path(self) -> Path:
        return self.workspace_path / "Test_Videos"
    
    @property
    def unedited_videos_path(self) -> Path:
        return self.workspace_path / "Unedited_videos"

    def execute(self) -> bool:
        """Executes the directory creation logic with CLI feedback."""
        print(Strings.CreateDirs.CREATING)
        
        self._create_mp_data_dirs()
        self._create_video_dirs()
        self._create_unedited_video_dir()
        
        print(Strings.CreateDirs.CREATED.format(self.mp_data_path, self.signs))
        self.logger.info(f"Directorios creados en {self.mp_data_path} para las acciones: {self.signs}")
        return True

    def _create_mp_data_dirs(self):
        print(Strings.CreateDirs.CREATING_MP_DATA.format(self.mp_data_path))
        self.logger.info(f"Creando folders para los numpy arrays en {self.mp_data_path}")

        for action in self.signs:
            for sequence in range(self.repetitions):
                folder_path = self.mp_data_path / action / str(sequence)
                folder_path.mkdir(parents=True, exist_ok=True)

    def _create_video_dirs(self):
        print(Strings.CreateDirs.CREATING_VIDEO_DIRS.format(self.video_data_path))
        self.logger.info(f"Creando directorio para los videos en {self.video_data_path}")

        self.video_data_path.mkdir(parents=True, exist_ok=True)
        for action in self.signs:
            action_video_path = self.video_data_path / action
            action_video_path.mkdir(parents=True, exist_ok=True)

    def _create_unedited_video_dir(self):
        print(Strings.CreateDirs.CREATING_UNEDITED_VIDEO_DIR.format(self.unedited_videos_path))
        self.logger.info(f"Creando directorio para los videos sin editar en {self.unedited_videos_path}")
        self.unedited_videos_path.mkdir(parents=True, exist_ok=True)
        for action in self.signs:
            action_video_path = self.unedited_videos_path / action
            action_video_path.mkdir(parents=True, exist_ok=True)
