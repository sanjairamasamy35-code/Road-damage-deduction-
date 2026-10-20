"""Evaluation metric/parser tests. The end-to-end runner uses a mocked model session."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from PIL import Image

import road_ai
import road_ai_evaluation as evaluation


def prediction(class_id, confidence, box):
    return {
        "class_id": class_id,
        "class_name": road_ai.CLASS_NAMES[class_id],
        "confidence": confidence,
        "box": list(box),
    }


class EvaluationMetricTests(unittest.TestCase):
    def test_fixed_threshold_metrics_ap_and_missing_classes(self):
        ground_truth = {
            "image-a.jpg": [
                {"class_id": 0, "class_name": road_ai.CLASS_NAMES[0], "box": [0, 0, 10, 10]},
                {"class_id": 1, "class_name": road_ai.CLASS_NAMES[1], "box": [20, 20, 30, 30]},
            ],
            "image-b.jpg": [
                {"class_id": 0, "class_name": road_ai.CLASS_NAMES[0], "box": [0, 0, 10, 10]},
            ],
        }
        predictions = {
            "image-a.jpg": [prediction(0, .90, [0, 0, 10, 10]), prediction(1, .80, [20, 20, 30, 30])],
            "image-b.jpg": [
                prediction(0, .95, [20, 20, 30, 30]),  # ranked FP
                prediction(0, .70, [0, 0, 10, 10]),
            ],
        }
        result = evaluation.compute_metrics(ground_truth, predictions, min_class_ground_truth=2)
        d00 = result["per_class"][0]
        self.assertEqual((d00["tp"], d00["fp"], d00["fn"]), (2, 1, 0))
        self.assertAlmostEqual(d00["precision"], 2 / 3)
        self.assertAlmostEqual(d00["recall"], 1.0)
        self.assertAlmostEqual(d00["f1"], .8)
        self.assertAlmostEqual(d00["ap50"], 2 / 3)
        self.assertTrue(d00["sufficient_examples"])
        d10 = result["per_class"][1]
        self.assertEqual(d10["ap50"], 1.0)
        self.assertFalse(d10["sufficient_examples"])
        for class_metrics in result["per_class"][2:]:
            self.assertEqual(class_metrics["ground_truth_instances"], 0)
            self.assertIsNone(class_metrics["precision"])
            self.assertIsNone(class_metrics["recall"])
            self.assertIsNone(class_metrics["f1"])
            self.assertIsNone(class_metrics["ap50"])
            self.assertFalse(class_metrics["sufficient_examples"])
        self.assertEqual(result["failure_counts"]["false_positive"], 1)
        self.assertEqual(result["failure_counts"]["missed_detection"], 0)

    def test_incorrect_class_is_reported_and_counts_as_standard_fp_and_miss(self):
        ground_truth = {
            "image.jpg": [{"class_id": 3, "class_name": road_ai.CLASS_NAMES[3], "box": [0, 0, 10, 10]}]
        }
        predictions = {"image.jpg": [prediction(0, .90, [0, 0, 10, 10])]}
        result = evaluation.compute_metrics(ground_truth, predictions)
        self.assertEqual(result["failure_counts"]["false_positive"], 1)
        self.assertEqual(result["failure_counts"]["missed_detection"], 1)
        self.assertEqual(result["failure_counts"]["incorrect_class"], 1)
        self.assertEqual(result["per_class"][0]["precision"], 0.0)
        self.assertEqual(result["per_class"][0]["fp"], 1)
        self.assertAlmostEqual(result["per_class"][3]["recall"], 0)

    def test_empty_dataset_returns_unavailable_metrics(self):
        result = evaluation.compute_metrics({"negative.jpg": []}, {"negative.jpg": []})
        self.assertIsNone(result["aggregate"]["precision"])
        self.assertIsNone(result["aggregate"]["recall"])
        self.assertIsNone(result["aggregate"]["f1"])
        self.assertIsNone(result["aggregate"]["map50"])
        self.assertIsNone(result["aggregate"]["map50_95"])
        self.assertEqual(result["aggregate"]["evaluable_classes"], 0)

    def test_missing_annotation_stops_instead_of_counting_as_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            images, annotations = Path(temporary) / "images", Path(temporary) / "annotations"
            images.mkdir()
            annotations.mkdir()
            Image.new("RGB", (16, 16)).save(images / "road.jpg")
            with self.assertRaisesRegex(evaluation.EvaluationError, "no matching XML annotation"):
                evaluation.load_evaluation_dataset(images, annotations)

    def test_invalid_bounding_boxes_are_rejected(self):
        invalid_xml = """<annotation><size><width>16</width><height>16</height></size>
          <object><name>D00</name><bndbox><xmin>0</xmin><ymin>1</ymin>
          <xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.xml"
            path.write_text(invalid_xml, encoding="utf-8")
            with self.assertRaisesRegex(evaluation.EvaluationError, "invalid or out-of-bounds"):
                evaluation.parse_voc_annotation(path, 16, 16)

    def test_runner_exports_reproducible_metrics_without_touching_production_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, annotations = root / "images", root / "annotations"
            images.mkdir()
            annotations.mkdir()
            Image.new("RGB", (16, 16), "gray").save(images / "road.jpg")
            (annotations / "road.xml").write_text(
                """<annotation><size><width>16</width><height>16</height></size>
                <object><name>D40</name><bndbox><xmin>2</xmin><ymin>2</ymin>
                <xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>""",
                encoding="utf-8",
            )
            model = root / "mock-model.onnx"
            model.write_bytes(b"mock model artifact")
            metadata = {
                "name": "Test-only RDD-style fixture", "version": "fixture-v1",
                "source": "local temporary test fixture", "license": "test only",
                "split_name": "isolated-test", "split_provenance": "single test image; mocked inference",
            }
            result = {
                "detections": [prediction(3, .91, [1, 1, 10, 10])],
                "inference_status": "detections",
            }
            kwargs = dict(
                images_dir=images, annotations_dir=annotations,
                dataset_name=metadata["name"], dataset_version=metadata["version"],
                dataset_source=metadata["source"], dataset_license=metadata["license"],
                split_name=metadata["split_name"], split_provenance=metadata["split_provenance"],
                model_path=model,
            )
            with patch.object(road_ai, "create_cpu_session", return_value=(object(), "mock-version", {})), \
                 patch.object(road_ai, "run_inference", return_value=result) as infer:
                first = evaluation.run_evaluation(output_dir=root / "out-1", **kwargs)
                second = evaluation.run_evaluation(output_dir=root / "out-2", **kwargs)
            infer.assert_any_call(
                ANY, ANY, "mock-version",
                confidence_threshold=evaluation.AP_MIN_CONFIDENCE,
            )
            self.assertEqual(first["dataset"]["image_ids_sha256"], second["dataset"]["image_ids_sha256"])
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(first["class_distribution"], second["class_distribution"])
            self.assertEqual(first["metrics"]["per_class"][3]["tp"], 1)
            self.assertTrue((root / "out-1" / "metrics.json").is_file())
            self.assertTrue((root / "out-1" / "failure_cases.csv").is_file())
            self.assertEqual(json.loads((root / "out-1" / "failure_cases.json").read_text()), [])
            self.assertEqual(infer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
