"""Exercise extraction scheduling without a camera or a MediaPipe model."""

import importlib.util
import logging
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestDataExtractor(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cv2 = MagicMock()
        self.holistic = MagicMock()
        self.holistic.return_value.__enter__.return_value.process_frame.return_value = (
            np.zeros((2, 2, 3)),
            np.array([1.0, 2.0]),
            True,
        )
        modules = {
            "cv2": self.cv2,
            "MediapipeHolistic": types.SimpleNamespace(MediapipeHolistic=self.holistic),
        }
        source = Path(__file__).resolve().parents[1] / "Model" / "Test"
        with patch.dict(sys.modules, modules):
            utility_spec = importlib.util.spec_from_file_location("Utilities", source / "Utilities.py")
            utilities = importlib.util.module_from_spec(utility_spec)
            utility_spec.loader.exec_module(utilities)
            sys.modules["Utilities"] = utilities
            spec = importlib.util.spec_from_file_location("data_extraction_under_test", source / "DataExtraction.py")
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.logger = MagicMock(spec=logging.Logger)
        self.capture = self.cv2.VideoCapture.return_value
        self.capture.isOpened.return_value = True
        self.capture.get.return_value = 2
        self.capture.read.return_value = (True, np.zeros((2, 2, 3)))

    def extract(self, path, repetitions, frames=2):
        with patch.object(self.module.Utilities, "setup_logging", return_value=self.logger):
            extractor = self.module.DataExtractor(repetitions, frames, ["HELLO"], self.root / "output")
        extractor.logger = self.logger
        output = StringIO()
        with patch.object(self.module.time, "sleep"), redirect_stdout(output):
            extractor.process_video(path, confidence=0.5)
        return output.getvalue()

    def test_single_file_repeats_without_index_error(self):
        for repetitions in (1, 3):
            with self.subTest(repetitions=repetitions):
                self.cv2.reset_mock()
                video = self.root / "hello.mp4"
                self.extract(video, repetitions)
                self.assertEqual(self.cv2.VideoCapture.call_count, repetitions)
                for call in self.cv2.VideoCapture.call_args_list:
                    self.assertEqual(call.args, (str(video),))
                self.assertEqual(self.capture.release.call_count, repetitions)
                for sequence in range(repetitions):
                    for frame in range(2):
                        saved = self.root / "output" / "HELLO" / str(sequence) / f"{frame}.npy"
                        np.testing.assert_array_equal(np.load(saved), [1.0, 2.0])

    def test_directory_still_cycles_available_videos(self):
        directory = self.root / "hello"
        directory.mkdir()
        videos = [str(directory / "first.mp4"), str(directory / "second.mp4")]
        with patch.object(self.module.Utilities, "get_video_paths", return_value=videos):
            self.extract(directory, repetitions=3)
        self.assertEqual([call.args[0] for call in self.cv2.VideoCapture.call_args_list], [*videos, videos[0]])

    def test_short_video_warning_reports_requested_frame_count(self):
        output = self.extract(self.root / "hello.mp4", repetitions=1, frames=5)
        self.assertIn("menos que los 5 necesarios", output)
        self.logger.warning.assert_called_once()
        self.assertIn("menos que los 5 necesarios", self.logger.warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
