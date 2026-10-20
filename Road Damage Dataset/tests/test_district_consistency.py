"""District registry, boundary audit, spatial matching, and migration tests."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import app


SQUARE = {
    "type": "Polygon",
    "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
}


def feature(district_id="TN-DIST-TEST", name="Test District", geometry=SQUARE):
    return {"type": "Feature", "properties": {
        "district_id": district_id, "district_name": name,
    }, "geometry": geometry}


class DistrictConsistencyTests(unittest.TestCase):
    def test_registry_has_38_unique_canonical_ids_and_names(self):
        self.assertEqual(len(app.DISTRICTS), 38)
        self.assertTrue(app.validate_district_registry(app.DISTRICTS))
        self.assertEqual(len({entry["id"] for entry in app.DISTRICTS}), 38)
        self.assertEqual(len({entry["name"] for entry in app.DISTRICTS}), 38)
        self.assertEqual(app.DISTRICT_REGISTRY["retrieved_on"], "2026-10-19")

    def test_duplicate_identifier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate canonical district id"):
            app.validate_district_registry([
                {"id": "TN-DIST-X", "name": "One"},
                {"id": "TN-DIST-X", "name": "Two"},
            ])
        registry = [{"id": "TN-DIST-ONE", "name": "One", "aliases": []}]
        feature_data = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"shapeName": "One"}, "geometry": SQUARE},
            {"type": "Feature", "properties": {"shapeName": "One"}, "geometry": SQUARE},
        ]}
        _, diagnostics = app.audit_district_features(feature_data, registry)
        self.assertEqual(diagnostics["duplicate_district_ids"], ["TN-DIST-ONE"])
        self.assertEqual(diagnostics["valid_polygon_count"], 0)

    def test_existing_geojson_has_34_valid_features_and_four_missing_districts(self):
        data = json.loads(app.DATA_FILE.read_text(encoding="utf-8"))
        prepared, diagnostics = app.audit_district_features(data)
        self.assertEqual(len(data["features"]), 34)
        self.assertEqual(diagnostics["valid_polygon_count"], 34)
        self.assertEqual(diagnostics["invalid_features"], [])
        self.assertEqual(diagnostics["duplicate_district_ids"], [])
        self.assertEqual(diagnostics["unmatched_features"], [])
        self.assertEqual(
            {entry["name"] for entry in diagnostics["missing_districts"]},
            {"Chengalpattu", "Thiruvallur", "Thiruvarur", "Thoothukudi"},
        )
        self.assertEqual(len(prepared["features"]), 34)
        self.assertTrue(all("district_id" in item["properties"] for item in prepared["features"]))

    def test_missing_boundary_is_reported_not_synthesized(self):
        small_registry = [
            {"id": "TN-DIST-ONE", "name": "One", "aliases": []},
            {"id": "TN-DIST-TWO", "name": "Two", "aliases": []},
        ]
        dataset = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"shapeName": "One"}, "geometry": SQUARE},
        ]}
        result, diagnostics = app.audit_district_features(dataset, small_registry)
        self.assertEqual(len(result["features"]), 1)
        self.assertEqual([entry["name"] for entry in diagnostics["missing_districts"]], ["Two"])

    def test_invalid_self_intersecting_and_unclosed_geometry_are_rejected(self):
        bowtie = {"type": "Polygon", "coordinates": [[
            [0, 0], [2, 2], [0, 2], [2, 0], [0, 0],
        ]]}
        unclosed = {"type": "Polygon", "coordinates": [[
            [0, 0], [2, 0], [2, 2], [0, 2],
        ]]}
        self.assertFalse(app.validate_district_geometry(bowtie)[0])
        self.assertFalse(app.validate_district_geometry(unclosed)[0])
        overlapping = {"type": "MultiPolygon", "coordinates": [
            SQUARE["coordinates"],
            [[[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]],
        ]}
        self.assertFalse(app.validate_district_geometry(overlapping)[0])
        invalid_feature_collection = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"shapeName": "Ariyalur"}, "geometry": bowtie},
        ]}
        _, diagnostics = app.audit_district_features(invalid_feature_collection)
        self.assertEqual(diagnostics["valid_polygon_count"], 0)
        self.assertEqual(diagnostics["invalid_features"][0]["name"], "Ariyalur")

    def test_coordinate_matching_inside_boundary_hole_and_invalid_coverage(self):
        polygon_with_hole = {"type": "Polygon", "coordinates": [
            SQUARE["coordinates"][0],
            [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],
        ]}
        district_map = {"features": [feature(geometry=polygon_with_hole)]}
        self.assertEqual(app.match_district(0.25, 0.25, district_map)["status"], "matched")
        self.assertEqual(app.match_district(1, 1, district_map)["status"], "no_valid_coverage")
        self.assertEqual(app.match_district(0, 1, district_map)["status"], "boundary_or_overlap")
        self.assertEqual(app.match_district(3, 3, district_map)["status"], "no_valid_coverage")
        self.assertEqual(app.match_district(181, 0, district_map)["status"], "invalid_coordinate")
        invalid_map = {"features": [feature(geometry={
            "type": "Polygon", "coordinates": [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]],
        })]}
        self.assertEqual(app.match_district(0.25, 0.25, invalid_map)["status"], "no_valid_coverage")

    def test_database_additive_migration_preserves_legacy_labels_and_photo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "legacy.sqlite"
            photo = Path(temp_dir) / "existing.jpg"
            photo.write_bytes(b"existing uploaded photo")
            connection = sqlite3.connect(database)
            try:
                connection.execute("""CREATE TABLE road_reports (
                    report_id TEXT PRIMARY KEY, submitted_at TEXT NOT NULL,
                    road_name TEXT NOT NULL, location TEXT NOT NULL DEFAULT '',
                    district TEXT NOT NULL, problem_type TEXT NOT NULL,
                    severity TEXT NOT NULL, description TEXT NOT NULL,
                    latitude REAL, longitude REAL, photo_path TEXT,
                    status TEXT NOT NULL DEFAULT 'Submitted'
                )""")
                connection.execute(
                    """INSERT INTO road_reports VALUES (?, '2025-01-01', 'Old road', '',
                       'Kanniyakumari', 'Pothole', 'Low', 'Old report', NULL, NULL, ?, 'Submitted')""",
                    ("legacy-match", str(photo)),
                )
                connection.execute(
                    """INSERT INTO road_reports VALUES (?, '2025-01-02', 'Old road', '',
                       'Historical unknown label', 'Crack', 'Medium', 'Keep label', NULL, NULL, NULL, 'Submitted')""",
                    ("legacy-unknown",),
                )
                connection.commit()
            finally:
                connection.close()

            app.initialize_database(database)
            app.initialize_database(database)
            with app.database_connection(database) as connection:
                rows = [dict(row) for row in connection.execute(
                    "SELECT report_id, district, district_id, photo_path, status FROM road_reports ORDER BY report_id"
                )]
            matched, unmatched = rows
            self.assertEqual(matched["district"], "Kanniyakumari")
            self.assertEqual(matched["district_id"], "TN-DIST-KANYAKUMARI")
            self.assertEqual(matched["photo_path"], str(photo))
            self.assertTrue(photo.exists())
            self.assertEqual(matched["status"], "Submitted")
            self.assertEqual(unmatched["district"], "Historical unknown label")
            self.assertIsNone(unmatched["district_id"])
            self.assertEqual(unmatched["status"], "Submitted")

    def test_new_report_and_filter_use_canonical_district_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "reports.sqlite"
            app.initialize_database(database)
            report_id = app.save_report({
                "road_name": "Test Road", "location": "Test point",
                "district": "Kanniyakumari", "district_id": "TN-DIST-KANYAKUMARI",
                "problem_type": "Pothole", "severity": "Low", "description": "Test",
            }, database_path=database)
            matched = app.fetch_reports(database, filters={"district_id": "TN-DIST-KANYAKUMARI"})
            other = app.fetch_reports(database, filters={"district_id": "TN-DIST-CHENNAI"})
            self.assertEqual([row["report_id"] for row in matched], [report_id])
            self.assertEqual(matched[0]["district"], "Kanyakumari")
            self.assertEqual(matched[0]["district_id"], "TN-DIST-KANYAKUMARI")
            self.assertEqual(other, [])


if __name__ == "__main__":
    unittest.main()
