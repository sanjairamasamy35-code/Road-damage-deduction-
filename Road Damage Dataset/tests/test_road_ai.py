"""AI preprocessing, decoding, persistence and failure-path tests.

Tests using FakeSession are mocked. The optional real-model smoke check is named
separately and is skipped when CPU ONNX Runtime is unavailable.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import app
import road_ai


class FakeMeta:
    def __init__(self, name, shape, tensor_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = tensor_type


class FakeSession:
    def __init__(self, output=None, error=None):
        self.output = output if output is not None else np.zeros((1, 8, 8400), dtype=np.float32)
        self.error = error

    def get_inputs(self):
        return [FakeMeta("images", [1, 3, 640, 640])]

    def get_outputs(self):
        return [FakeMeta("output0", [1, 8, 8400])]

    def run(self, names, inputs):
        if self.error:
            raise self.error
        assert names == ["output0"]
        assert inputs["images"].shape == (1, 3, 640, 640)
        return [self.output]


def output_with_candidate(score, class_id=3, center=(320, 320), size=(100, 80)):
    output = np.zeros((1, 8, 8400), dtype=np.float32)
    output[0, 0, 0], output[0, 1, 0] = center
    output[0, 2, 0], output[0, 3, 0] = size
    output[0, 4 + class_id, 0] = score
    return output


class RoadAITests(unittest.TestCase):
    def test_mocked_preprocessing_matches_letterbox_contract(self):
        image = Image.new("RGB", (640, 320), (255, 0, 0))
        prepared = road_ai.preprocess_image(image)
        self.assertEqual(prepared.tensor.shape, (1, 3, 640, 640))
        self.assertEqual(prepared.tensor.dtype, np.float32)
        self.assertEqual((prepared.scale, prepared.pad_x, prepared.pad_y), (1.0, 0, 160))
        np.testing.assert_allclose(prepared.tensor[0, :, 0, 0], [114 / 255] * 3, atol=1e-6)
        np.testing.assert_allclose(prepared.tensor[0, :, 200, 100], [1, 0, 0], atol=1e-6)

    def test_mocked_output_decoding_maps_boxes_and_runs_per_class_nms(self):
        image = Image.new("RGB", (320, 640))
        prepared = road_ai.preprocess_image(image)
        output = output_with_candidate(0.90, class_id=3)
        output[0, 0, 1] = 321
        output[0, 1, 1] = 320
        output[0, 2, 1] = 100
        output[0, 3, 1] = 80
        output[0, 7, 1] = 0.80
        output[0, 4, 2] = 0.70  # Same box, different class is retained.
        output[0, :4, 2] = output[0, :4, 0]
        detections = road_ai.decode_output(output, prepared)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["class_name"], "D40 pothole")
        self.assertEqual(detections[0]["box"], [110.0, 280.0, 210.0, 360.0])
        self.assertEqual(detections[1]["class_name"], "D00 longitudinal crack")

    def test_mocked_empty_and_low_confidence_outputs(self):
        image = Image.new("RGB", (64, 64))
        empty = road_ai.run_inference(image, FakeSession(), "mock-model")
        low = road_ai.run_inference(image, FakeSession(output_with_candidate(0.12)), "mock-model")
        self.assertEqual(empty["inference_status"], "no_detections")
        self.assertEqual(empty["detections"], [])
        self.assertEqual(low["inference_status"], "low_confidence")
        self.assertEqual(len(low["detections"]), 1)
        self.assertAlmostEqual(low["detections"][0]["confidence"], 0.12, places=6)
        self.assertIsInstance(low["processing_duration_ms"], float)

    def test_mocked_database_persistence_stores_detections_and_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "reports.sqlite"
            app.initialize_database(database)
            report_id = app.save_report({
                "road_name": "Test Road", "district": "Chennai", "problem_type": "Pothole",
                "severity": "Low", "description": "AI persistence fixture",
            }, database_path=database)
            result = road_ai.run_inference(
                Image.new("RGB", (320, 640)), FakeSession(output_with_candidate(0.86)), "mock-model-v1"
            )
            annotated_dir = Path(temporary) / "annotated"
            detection_id = road_ai.save_detection_result(
                report_id, result, database, annotated_dir=annotated_dir
            )
            saved = road_ai.fetch_detection_history(report_id, database)
            self.assertEqual(saved[0]["detection_id"], detection_id)
            self.assertEqual(saved[0]["model_version"], "mock-model-v1")
            self.assertEqual(saved[0]["inference_status"], "detections")
            self.assertEqual(json.loads(saved[0]["classes"]), ["D40 pothole"])
            self.assertTrue((annotated_dir / f"{detection_id}.jpg").is_file())
            self.assertEqual(app.fetch_reports(database)[0]["status"], "Submitted")

    def test_missing_weights_and_invalid_image_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(road_ai.InferenceError, "missing"):
                road_ai.create_cpu_session(Path(temporary) / "missing.onnx")
            invalid_image = Path(temporary) / "not-an-image.jpg"
            invalid_image.write_bytes(b"not an image")
            with self.assertRaises(road_ai.InferenceError):
                road_ai.load_image(invalid_image)

    def test_model_python_style_class_metadata_and_order_validation(self):
        road_ai.validate_class_metadata({
            "names": "{0: 'longitudinal_crack', 1: 'transverse_crack', "
                     "2: 'alligator_crack', 3: 'pothole'}"
        })
        with self.assertRaisesRegex(road_ai.InferenceError, "class metadata does not match"):
            road_ai.validate_class_metadata({
                "names": "{0: 'pothole', 1: 'transverse_crack', "
                         "2: 'alligator_crack', 3: 'longitudinal_crack'}"
            })

    def test_mocked_inference_error_is_clear(self):
        image = Image.new("RGB", (32, 32))
        with self.assertRaisesRegex(road_ai.InferenceError, "ONNX inference failed"):
            road_ai.run_inference(image, FakeSession(error=RuntimeError("mock failure")), "mock-model")
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "reports.sqlite"
            app.initialize_database(database)
            report_id = app.save_report({
                "road_name": "Test Road", "district": "Chennai", "problem_type": "Pothole",
                "severity": "Low", "description": "AI error fixture",
            }, database_path=database)
            error_result = {
                "model_version": "mock-model", "detections": [],
                "inference_status": "error", "processing_duration_ms": None,
                "annotated_image": None, "error_message": "mock failure",
            }
            road_ai.save_detection_result(report_id, error_result, database)
            saved = road_ai.fetch_detection_history(report_id, database)[0]
            self.assertEqual(saved["inference_status"], "error")
            self.assertEqual(saved["error_message"], "mock failure")
            self.assertEqual(app.fetch_reports(database)[0]["status"], "Submitted")

    def test_real_model_smoke_if_runtime_is_installed(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("CPU ONNX Runtime is not installed; this is not a mocked real-model test.")
        session, version, _metadata = road_ai.create_cpu_session()
        image = Image.new("RGB", (640, 480))
        result = road_ai.run_inference(image, session, version)
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual((session.get_inputs()[0].name, session.get_inputs()[0].shape),
                         ("images", [1, 3, 640, 640]))
        self.assertEqual((session.get_outputs()[0].name, session.get_outputs()[0].shape),
                         ("output0", [1, 8, 8400]))
        self.assertIn(result["inference_status"], {"detections", "no_detections", "low_confidence"})
        self.assertIsInstance(result["processing_duration_ms"], float)
        self.assertEqual(result["annotated_image"].size, image.size)
        for detection in result["detections"]:
            self.assertIn(detection["class_name"], road_ai.CLASS_NAMES)
            x1, y1, x2, y2 = detection["box"]
            self.assertTrue(0 <= x1 <= x2 <= image.width)
            self.assertTrue(0 <= y1 <= y2 <= image.height)


if __name__ == "__main__":
    unittest.main()
