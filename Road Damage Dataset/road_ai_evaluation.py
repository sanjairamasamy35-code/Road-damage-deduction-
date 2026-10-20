"""Isolated object-detection evaluation for the RoadGuard CPU ONNX model.

Dataset annotations use Pascal VOC XML. Inference delegates to road_ai.run_inference
so evaluation uses the same image preprocessing, decoding, coordinate mapping and NMS.
This module never opens the production SQLite database or writes to evidence folders.
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError

import road_ai


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_MATCH_IOU = 0.50
AP_IOU_THRESHOLDS = tuple(round(0.50 + step * 0.05, 2) for step in range(10))
AP_MIN_CONFIDENCE = 0.001
MIN_CLASS_GROUND_TRUTH = 10

CLASS_ALIASES = {
    "d00": 0, "longitudinal crack": 0, "longitudinal_crack": 0,
    "d10": 1, "transverse crack": 1, "transverse_crack": 1,
    "d20": 2, "alligator crack": 2, "alligator_crack": 2, "aligator crack": 2,
    "d40": 3, "pothole": 3,
}


class EvaluationError(ValueError):
    """Raised when evaluation data or output configuration is invalid."""


def _normalize_class_name(value):
    return " ".join(str(value).strip().casefold().replace("_", " ").split())


def class_id_for_name(name):
    normalized = _normalize_class_name(name)
    class_id = CLASS_ALIASES.get(normalized)
    if class_id is None:
        class_id = CLASS_ALIASES.get(normalized.split(" ", 1)[0])
    if class_id is None:
        raise EvaluationError(f"Unsupported annotation class {name!r}; expected one of D00, D10, D20, D40.")
    return class_id


def discover_dataset(images_dir, annotations_dir):
    """Pair every image with one VOC XML using a unique filename stem."""
    images_root = Path(images_dir).resolve()
    annotations_root = Path(annotations_dir).resolve()
    if not images_root.is_dir():
        raise EvaluationError(f"Image directory does not exist: {images_root}")
    if not annotations_root.is_dir():
        raise EvaluationError(f"Annotation directory does not exist: {annotations_root}")
    images = sorted(
        (path for path in images_root.rglob("*") if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(images_root).as_posix().casefold(),
    )
    xml_files = sorted(
        (path for path in annotations_root.rglob("*.xml") if path.is_file()),
        key=lambda path: path.relative_to(annotations_root).as_posix().casefold(),
    )
    if not images:
        raise EvaluationError(f"No supported images were found under {images_root}")
    xml_by_stem = defaultdict(list)
    for xml_path in xml_files:
        xml_by_stem[xml_path.stem.casefold()].append(xml_path)
    duplicate_stems = [stem for stem, paths in xml_by_stem.items() if len(paths) > 1]
    if duplicate_stems:
        raise EvaluationError(
            "Annotation filename stems must be unique within the selected evaluation split; "
            f"duplicates include: {', '.join(sorted(duplicate_stems)[:10])}"
        )
    image_stems = {path.stem.casefold() for path in images}
    image_stem_counts = defaultdict(int)
    for image_path in images:
        image_stem_counts[image_path.stem.casefold()] += 1
    duplicate_image_stems = [stem for stem, count in image_stem_counts.items() if count > 1]
    if duplicate_image_stems:
        raise EvaluationError(
            "Image filename stems must be unique within the selected evaluation split; "
            f"duplicates include: {', '.join(sorted(duplicate_image_stems)[:10])}"
        )
    missing = [path for path in images if path.stem.casefold() not in xml_by_stem]
    if missing:
        examples = ", ".join(path.relative_to(images_root).as_posix() for path in missing[:10])
        raise EvaluationError(
            f"{len(missing)} image(s) have no matching XML annotation. Evaluation stops rather than "
            f"treating unlabeled images as negatives. Examples: {examples}"
        )
    extras = [path for stem, paths in xml_by_stem.items() if stem not in image_stems for path in paths]
    if extras:
        examples = ", ".join(path.name for path in extras[:10])
        raise EvaluationError(f"{len(extras)} XML annotation(s) have no matching image: {examples}")
    return [(path, xml_by_stem[path.stem.casefold()][0]) for path in images]


def parse_voc_annotation(xml_path, image_width, image_height):
    """Parse Pascal VOC boxes into zero-based half-open [x1,y1,x2,y2] coordinates."""
    try:
        root = ElementTree.parse(xml_path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise EvaluationError(f"Could not parse annotation {xml_path}: {error}") from error
    size = root.find("size")
    if size is not None:
        try:
            xml_width, xml_height = int(size.findtext("width")), int(size.findtext("height"))
        except (TypeError, ValueError) as error:
            raise EvaluationError(f"Invalid image size in {xml_path}") from error
        if xml_width != image_width or xml_height != image_height:
            raise EvaluationError(
                f"Annotation dimensions {xml_width}x{xml_height} in {xml_path} do not match "
                f"the image {image_width}x{image_height}."
            )
    objects = []
    for index, obj in enumerate(root.findall("object"), start=1):
        name = obj.findtext("name")
        if not name:
            raise EvaluationError(f"Object {index} has no class name in {xml_path}")
        class_id = class_id_for_name(name)
        box_node = obj.find("bndbox")
        if box_node is None:
            raise EvaluationError(f"Object {index} has no bndbox in {xml_path}")
        try:
            xmin, ymin, xmax, ymax = (
                float(box_node.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")
            )
        except (TypeError, ValueError) as error:
            raise EvaluationError(f"Object {index} has invalid box coordinates in {xml_path}") from error
        values = (xmin, ymin, xmax, ymax)
        if (not all(value == value and abs(value) != float("inf") for value in values)
                or xmin < 1 or ymin < 1 or xmax < xmin or ymax < ymin
                or xmax > image_width or ymax > image_height):
            raise EvaluationError(
                f"Object {index} has an invalid or out-of-bounds VOC box {values} in {xml_path}"
            )
        objects.append({
            "class_id": class_id,
            "class_name": road_ai.CLASS_NAMES[class_id],
            "box": [xmin - 1.0, ymin - 1.0, xmax, ymax],
        })
    return objects


def load_evaluation_dataset(images_dir, annotations_dir):
    """Load images lazily by path and validated labels into deterministic records."""
    records = []
    image_root = Path(images_dir).resolve()
    for image_path, annotation_path in discover_dataset(images_dir, annotations_dir):
        try:
            with Image.open(image_path) as source:
                width, height = source.size
                source.verify()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            raise EvaluationError(f"Invalid evaluation image {image_path}: {error}") from error
        labels = parse_voc_annotation(annotation_path, width, height)
        try:
            annotation_filename = ElementTree.parse(annotation_path).getroot().findtext("filename")
        except (OSError, ElementTree.ParseError):
            annotation_filename = None
        if annotation_filename and Path(annotation_filename).name.casefold() != image_path.name.casefold():
            raise EvaluationError(
                f"Annotation filename {annotation_filename!r} in {annotation_path} does not match "
                f"image {image_path.name!r}."
            )
        records.append({
            "image_id": image_path.relative_to(image_root).as_posix(),
            "image_path": image_path,
            "annotation_path": annotation_path,
            "width": width,
            "height": height,
            "ground_truth": labels,
        })
    return records


def box_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _match_class(ground_truth, predictions, class_id, iou_threshold, confidence_threshold):
    gt_by_image = {
        image_id: [item for item in items if item["class_id"] == class_id]
        for image_id, items in ground_truth.items()
    }
    matched = {image_id: set() for image_id in ground_truth}
    ranked = []
    total_gt = sum(len(items) for items in gt_by_image.values())
    for image_id, items in predictions.items():
        for order, item in enumerate(items):
            if item["class_id"] == class_id and item["confidence"] >= confidence_threshold:
                ranked.append((float(item["confidence"]), image_id, order, item))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    tp, fp, matches = [], [], []
    for _score, image_id, _order, prediction in ranked:
        candidates = gt_by_image.get(image_id, [])
        overlaps = [
            (box_iou(prediction["box"], item["box"]), index)
            for index, item in enumerate(candidates) if index not in matched[image_id]
        ]
        best_iou, best_index = max(overlaps, default=(0.0, None))
        if best_index is not None and best_iou >= iou_threshold:
            matched[image_id].add(best_index)
            tp.append(1)
            fp.append(0)
            matches.append((image_id, prediction, candidates[best_index], best_iou, True))
        else:
            tp.append(0)
            fp.append(1)
            matches.append((image_id, prediction, None, best_iou, False))
    return total_gt, tp, fp, matches, matched


def _average_precision(total_gt, tp, fp):
    if total_gt <= 0:
        return None
    if not tp:
        return 0.0
    true_positive_cumulative, false_positive_cumulative = [], []
    running_tp = running_fp = 0
    for true_positive, false_positive in zip(tp, fp):
        running_tp += true_positive
        running_fp += false_positive
        true_positive_cumulative.append(running_tp)
        false_positive_cumulative.append(running_fp)
    recall = [value / total_gt for value in true_positive_cumulative]
    precision = [
        true_positive_cumulative[index]
        / (true_positive_cumulative[index] + false_positive_cumulative[index])
        for index in range(len(tp))
    ]
    recall_points = [0.0, *recall, 1.0]
    precision_points = [0.0, *precision, 0.0]
    for index in range(len(precision_points) - 2, -1, -1):
        precision_points[index] = max(precision_points[index], precision_points[index + 1])
    return sum(
        (recall_points[index + 1] - recall_points[index]) * precision_points[index + 1]
        for index in range(len(recall_points) - 1)
    )


def _point_metrics(ground_truth, predictions, class_id, confidence_threshold, iou_threshold):
    total_gt, tp, fp, _matches, matched = _match_class(
        ground_truth, predictions, class_id, iou_threshold, confidence_threshold
    )
    if total_gt == 0:
        false_positives = sum(fp)
        return {
            "precision": 0.0 if false_positives else None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "fp": false_positives,
            "fn": 0,
        }
    true_positives = sum(tp)
    false_positives = sum(fp)
    false_negatives = total_gt - sum(len(items) for items in matched.values())
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / total_gt
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": true_positives, "fp": false_positives, "fn": false_negatives,
    }


def _image_failures(ground_truth, predictions, confidence_threshold, iou_threshold):
    failures = []
    for image_id in sorted(ground_truth):
        gt_items = ground_truth[image_id]
        pred_items = [
            (index, item) for index, item in enumerate(predictions.get(image_id, []))
            if item["confidence"] >= confidence_threshold
        ]
        matched_gt, matched_pred = set(), set()
        by_class = defaultdict(list)
        for index, item in pred_items:
            by_class[item["class_id"]].append((index, item))
        for class_id in range(len(road_ai.CLASS_NAMES)):
            class_gt = [(index, item) for index, item in enumerate(gt_items) if item["class_id"] == class_id]
            class_preds = sorted(by_class[class_id], key=lambda row: (-row[1]["confidence"], row[0]))
            local_matched = set()
            for pred_index, prediction in class_preds:
                candidates = [
                    (box_iou(prediction["box"], gt["box"]), gt_index, gt)
                    for gt_index, gt in class_gt if gt_index not in local_matched
                ]
                overlap, gt_index, gt_item = max(candidates, default=(0.0, None, None), key=lambda item: item[0])
                if gt_index is not None and overlap >= iou_threshold:
                    local_matched.add(gt_index)
                    matched_gt.add(gt_index)
                    matched_pred.add(pred_index)
        mismatch_pairs = set()
        for pred_index, prediction in pred_items:
            if pred_index in matched_pred:
                continue
            candidates = [
                (box_iou(prediction["box"], gt["box"]), gt_index, gt)
                for gt_index, gt in enumerate(gt_items)
                if gt["class_id"] != prediction["class_id"]
            ]
            overlap, gt_index, gt_item = max(candidates, default=(0.0, None, None), key=lambda item: item[0])
            wrong_class = gt_index is not None and overlap >= iou_threshold
            if wrong_class:
                mismatch_pairs.add((pred_index, gt_index))
            failures.append({
                "image_id": image_id, "type": "false_positive",
                "prediction": prediction, "ground_truth": None,
                "possible_wrong_class": bool(wrong_class),
            })
        for gt_index, gt_item in enumerate(gt_items):
            if gt_index in matched_gt:
                continue
            candidates = [
                (box_iou(prediction["box"], gt_item["box"]), pred_index, prediction)
                for pred_index, prediction in pred_items
                if prediction["class_id"] != gt_item["class_id"]
            ]
            overlap, pred_index, prediction = max(candidates, default=(0.0, None, None), key=lambda item: item[0])
            failures.append({
                "image_id": image_id, "type": "missed_detection",
                "prediction": None, "ground_truth": gt_item,
                "possible_wrong_class": pred_index is not None and overlap >= iou_threshold,
            })
        for pred_index, gt_index in sorted(mismatch_pairs):
            prediction = predictions[image_id][pred_index]
            failures.append({
                "image_id": image_id, "type": "incorrect_class",
                "prediction": prediction, "ground_truth": gt_items[gt_index],
                "iou": box_iou(prediction["box"], gt_items[gt_index]["box"]),
            })
    return failures


def compute_metrics(ground_truth, predictions, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                    iou_threshold=DEFAULT_MATCH_IOU, ap_iou_thresholds=AP_IOU_THRESHOLDS,
                    ap_min_confidence=AP_MIN_CONFIDENCE,
                    min_class_ground_truth=MIN_CLASS_GROUND_TRUTH):
    """Compute fixed-threshold P/R/F1 and interpolated AP50/AP50:95 per class."""
    per_class, included = [], []
    for class_id, class_name in enumerate(road_ai.CLASS_NAMES):
        ground_truth_count = sum(
            sum(item["class_id"] == class_id for item in items)
            for items in ground_truth.values()
        )
        images_with_class = sum(
            any(item["class_id"] == class_id for item in items)
            for items in ground_truth.values()
        )
        point = _point_metrics(
            ground_truth, predictions, class_id, confidence_threshold, iou_threshold
        )
        ap_by_iou = {}
        if ground_truth_count:
            for threshold in ap_iou_thresholds:
                count, tp, fp, _matches, _matched = _match_class(
                    ground_truth, predictions, class_id, threshold, ap_min_confidence
                )
                ap_by_iou[f"{threshold:.2f}"] = _average_precision(count, tp, fp)
            included.append(class_id)
        else:
            ap_by_iou = {f"{threshold:.2f}": None for threshold in ap_iou_thresholds}
        ap50 = ap_by_iou.get("0.50")
        available_ap = [value for value in ap_by_iou.values() if value is not None]
        per_class.append({
            "class_id": class_id, "class_name": class_name,
            "ground_truth_instances": ground_truth_count,
            "images_with_class": images_with_class,
            "sufficient_examples": ground_truth_count >= min_class_ground_truth,
            **point,
            "ap50": ap50,
            "ap50_95": sum(available_ap) / len(available_ap) if available_ap else None,
            "ap_by_iou": ap_by_iou,
        })
    point_classes = [item for item in per_class if item["ground_truth_instances"] > 0]
    map50_values = [per_class[class_id]["ap50"] for class_id in included]
    map5095_values = [per_class[class_id]["ap50_95"] for class_id in included]
    aggregate = {}
    for metric in ("precision", "recall", "f1"):
        values = [item[metric] for item in point_classes if item[metric] is not None]
        aggregate[metric] = sum(values) / len(values) if values else None
    aggregate["map50"] = sum(map50_values) / len(map50_values) if map50_values else None
    aggregate["map50_95"] = sum(map5095_values) / len(map5095_values) if map5095_values else None
    aggregate["evaluable_classes"] = len(included)
    aggregate["insufficient_classes"] = [
        item["class_name"] for item in per_class
        if item["ground_truth_instances"] < min_class_ground_truth
    ]
    failure_cases = _image_failures(
        ground_truth, predictions, confidence_threshold, iou_threshold
    )
    return {
        "aggregate": aggregate,
        "per_class": per_class,
        "class_distribution": {
            item["class_name"]: {
                "ground_truth_instances": item["ground_truth_instances"],
                "images_with_class": item["images_with_class"],
            }
            for item in per_class
        },
        "failure_cases": failure_cases,
        "failure_counts": {
            failure_type: sum(item["type"] == failure_type for item in failure_cases)
            for failure_type in ("false_positive", "missed_detection", "incorrect_class")
        },
        "protocol": {
            "confidence_threshold_for_precision_recall_f1": confidence_threshold,
            "iou_threshold_for_precision_recall_f1_and_failure_cases": iou_threshold,
            "ap_iou_thresholds": list(ap_iou_thresholds),
            "ap_prediction_min_confidence": ap_min_confidence,
            "nms_iou_threshold": road_ai.NMS_IOU_THRESHOLD,
            "ap_method": "all-point interpolated precision-envelope area",
            "aggregate_method": "macro average over classes with at least one ground-truth instance",
            "minimum_ground_truth_for_sufficient_examples": min_class_ground_truth,
        },
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_manifest_hash(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["image_id"].encode("utf-8"))
        digest.update(b"\0")
        for path in (record["image_path"], record["annotation_path"]):
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_file(path).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _safe_output_dir(output_dir, input_dirs):
    output = Path(output_dir).resolve()
    if output == road_ai.BASE_DIR / "data" or (road_ai.BASE_DIR / "data").resolve() in output.parents:
        raise EvaluationError("Evaluation output must not be written under project data/ (reports and evidence).")
    if output == (road_ai.BASE_DIR / "models").resolve() or (road_ai.BASE_DIR / "models").resolve() in output.parents:
        raise EvaluationError("Evaluation output must not be written under models/.")
    for input_dir in input_dirs:
        root = Path(input_dir).resolve()
        if output == root or root in output.parents:
            raise EvaluationError("Evaluation output must not be written inside the input dataset.")
    if output.exists() and any(output.iterdir()):
        raise EvaluationError(f"Output directory must be new or empty: {output}")
    return output


def _write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary_path.replace(path)


def _environment_info():
    import numpy
    import onnxruntime
    import PIL
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "onnxruntime": onnxruntime.__version__,
        "onnx_execution_provider": "CPUExecutionProvider",
    }


def run_evaluation(images_dir, annotations_dir, output_dir, dataset_name, dataset_version,
                   dataset_source, dataset_license, split_name, split_provenance,
                   model_path=road_ai.MODEL_FILE,
                   confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                   iou_threshold=DEFAULT_MATCH_IOU):
    """Evaluate a labeled image split and export an isolated, reproducible report."""
    metadata = {
        "name": str(dataset_name).strip(), "version": str(dataset_version).strip(),
        "source": str(dataset_source).strip(), "license": str(dataset_license).strip(),
        "split_name": str(split_name).strip(), "split_provenance": str(split_provenance).strip(),
    }
    if any(not value for value in metadata.values()):
        raise EvaluationError("Dataset name/version/source/license and split name/provenance are required.")
    if not 0.0 <= confidence_threshold <= 1.0 or not 0.0 <= iou_threshold <= 1.0:
        raise EvaluationError("Confidence and IoU thresholds must be between 0 and 1.")
    output = _safe_output_dir(output_dir, (images_dir, annotations_dir))
    records = load_evaluation_dataset(images_dir, annotations_dir)
    session, version, model_metadata = road_ai.create_cpu_session(model_path)
    predictions, ground_truth = {}, {}
    for record in records:
        image = road_ai.load_image(record["image_path"])
        result = road_ai.run_inference(
            image, session, version, confidence_threshold=AP_MIN_CONFIDENCE
        )
        predictions[record["image_id"]] = result["detections"]
        ground_truth[record["image_id"]] = record["ground_truth"]
    evaluated = compute_metrics(
        ground_truth, predictions,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    class_distribution = evaluated.pop("class_distribution")
    failure_cases = evaluated.pop("failure_cases")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            **metadata,
            "image_count": len(records),
            "image_ids_sha256": _dataset_manifest_hash(records),
            "ground_truth_instances": sum(len(items) for items in ground_truth.values()),
            "negative_image_count": sum(not items for items in ground_truth.values()),
        },
        "model": {
            "file_name": Path(model_path).name,
            "model_version": version,
            "sha256": _sha256_file(model_path),
            "embedded_class_metadata": model_metadata,
        },
        "environment": _environment_info(),
        "metrics": evaluated,
        "class_distribution": class_distribution,
        "failure_case_count": len(failure_cases),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "metrics.json", summary)
    _write_json(output / "failure_cases.json", failure_cases)
    with (output / "failure_cases.csv").open("w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.DictWriter(destination, fieldnames=(
            "image_id", "type", "prediction_class", "prediction_confidence", "prediction_box",
            "ground_truth_class", "ground_truth_box", "iou", "possible_wrong_class",
        ))
        writer.writeheader()
        for item in failure_cases:
            prediction, gt = item.get("prediction"), item.get("ground_truth")
            writer.writerow({
                "image_id": item["image_id"], "type": item["type"],
                "prediction_class": prediction.get("class_name") if prediction else "",
                "prediction_confidence": prediction.get("confidence") if prediction else "",
                "prediction_box": json.dumps(prediction.get("box")) if prediction else "",
                "ground_truth_class": gt.get("class_name") if gt else "",
                "ground_truth_box": json.dumps(gt.get("box")) if gt else "",
                "iou": item.get("iou", ""), "possible_wrong_class": item.get("possible_wrong_class", ""),
            })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate RoadGuard's ONNX model on an explicitly labeled Pascal VOC image split."
    )
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="Must be a new/empty directory outside project data/ and the input dataset.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument("--dataset-license", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--split-provenance", required=True)
    parser.add_argument("--model", default=str(road_ai.MODEL_FILE))
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_MATCH_IOU)
    args = parser.parse_args(argv)
    try:
        summary = run_evaluation(
            args.images_dir, args.annotations_dir, args.output_dir,
            args.dataset_name, args.dataset_version, args.dataset_source,
            args.dataset_license, args.split_name, args.split_provenance,
            args.model, args.confidence_threshold, args.iou_threshold,
        )
    except (EvaluationError, road_ai.InferenceError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "status": summary["status"],
        "image_count": summary["dataset"]["image_count"],
        "metrics": summary["metrics"]["aggregate"],
        "output_dir": str(Path(args.output_dir).resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
