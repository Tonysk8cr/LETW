"""Exercise preview controls without loading image or machine-learning libraries."""

import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestExtractionPreview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cv2 = MagicMock()
        self.numpy = MagicMock()
        self.numpy.linspace.return_value = [0, 1]
        self.holistic = MagicMock()
        self.holistic.return_value.__enter__.return_value.process_frame.return_value = ("image", "keypoints", True)
        self.utilities = MagicMock()
        self.utilities.get_video_paths.return_value = [str(self.root / "hello.mp4")]
        source = Path(__file__).resolve().parents[1] / "Model" / "Test"
        modules = {"cv2": self.cv2, "numpy": self.numpy}
        for name, attribute, value in (
            ("Utilities", "Utilities", self.utilities),
            ("MediapipeHolistic", "MediapipeHolistic", self.holistic),
            ("CreateProjectDirsCommand", "CreateProjectDirsCommand", MagicMock()),
            ("DataLabelling", "DataLabelling", MagicMock()),
            ("RealtimePrediction", "RealtimeDetection", MagicMock()),
            ("TrainingLSTM", "TrainingLSTM", MagicMock()),
        ):
            modules[name] = types.ModuleType(name)
            setattr(modules[name], attribute, value)
        with patch.dict(sys.modules, modules):
            for name in ("Strings", "DataExtraction", "VideoBatchProcessor", "App"):
                spec = importlib.util.spec_from_file_location(name, source / f"{name}.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                setattr(self, name, module)
        capture = self.cv2.VideoCapture.return_value
        capture.isOpened.return_value = True
        capture.get.return_value = 2
        capture.read.return_value = (True, "frame")

    def extract(self, **kwargs):
        extractor = self.DataExtraction.DataExtractor(3, 2, ["HELLO"], self.root / "output")
        with redirect_stdout(StringIO()), patch("time.sleep", side_effect=AssertionError("unexpected delay")):
            extractor.process_video(self.root, confidence=0.7, **kwargs)

    def test_default_extraction_never_opens_or_closes_gui_windows(self):
        self.extract()
        self.cv2.imshow.assert_not_called()
        self.cv2.waitKey.assert_not_called()
        self.cv2.destroyAllWindows.assert_not_called()
        self.assertEqual(self.numpy.save.call_count, 6)
        self.holistic.return_value.__enter__.return_value.process_frame.assert_called_with("frame", draw_results=False)
        self.assertEqual(self.cv2.VideoCapture.return_value.release.call_count, 3)

    def test_explicit_preview_displays_frames_and_closes_windows(self):
        self.extract(preview=True)
        self.assertEqual(self.cv2.imshow.call_count, 6)
        self.assertEqual(self.cv2.waitKey.call_count, 6)
        self.cv2.destroyAllWindows.assert_called_once_with()
        self.holistic.return_value.__enter__.return_value.process_frame.assert_called_with("frame", draw_results=True)
        self.assertEqual(self.numpy.save.call_count, 6)

    def test_batch_extraction_forwards_preview_to_both_extraction_paths(self):
        for preview in (False, True):
            for method in ("extract_single_path", "extract_parent_path"):
                with self.subTest(preview=preview, method=method):
                    self.utilities.get_video_by_action.return_value = {"HELLO": [str(self.root / "hello.mp4")]}
                    processor = self.VideoBatchProcessor.VideoBatchProcessor(
                        str(self.root), 3, ["HELLO"], 2, 0.7, str(self.root / "output"), preview=preview
                    )
                    processor.data_extractor = MagicMock()
                    with redirect_stdout(StringIO()):
                        getattr(processor, method)()
                    self.assertTrue(processor.data_extractor.process_video.called)
                    self.assertEqual(processor.data_extractor.process_video.call_args.kwargs["preview"], preview)

    def test_menu_defaults_to_headless_and_allows_explicit_opt_in(self):
        context = self.App.Context(video_paths=str(self.root), mp_path=str(self.root / "output"))
        for answer, expected in (("", False), ("n", False), ("s", True), (" S ", True)):
            with (
                self.subTest(answer=answer),
                patch("builtins.input", side_effect=["0.7", "2", answer]),
                patch.object(self.App, "_create_video_processor") as create,
                redirect_stdout(StringIO()),
            ):
                self.assertTrue(self.App.cmd_extract_data_from_videos(context))
                create.assert_called_once_with(context, 0.7, directory=str(self.root), preview=expected)
                create.return_value.extract_parent_path.assert_called_once_with()

    def test_return_to_menu_does_not_ask_for_preview(self):
        with (
            patch("builtins.input", side_effect=["0.7", "3"]) as user_input,
            patch.object(self.App, "_create_video_processor") as create,
            redirect_stdout(StringIO()),
        ):
            self.assertTrue(self.App.cmd_extract_data_from_videos(self.App.Context()))
        self.assertEqual(user_input.call_count, 2)
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
