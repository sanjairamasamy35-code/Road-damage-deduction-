"""Database lifecycle tests use temporary SQLite files and never touch app data."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import app


class ReportLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "reports.sqlite"
        app.initialize_database(self.database)
        self.report_id = app.save_report(
            {
                "road_name": "Test Road",
                "location": "Test landmark",
                "district": "Chennai",
                "problem_type": "Pothole",
                "severity": "Low",
                "description": "Test record for lifecycle checks.",
            },
            database_path=self.database,
            photo_bytes=b"supporting photo does not count as verification",
            photo_extension=".jpg",
            upload_dir=Path(self.temp_dir.name) / "uploads",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def status(self):
        with app.database_connection(self.database) as connection:
            return connection.execute(
                "SELECT status FROM road_reports WHERE report_id = ?", (self.report_id,)
            ).fetchone()[0]

    def history(self):
        return app.fetch_report_status_history(self.report_id, self.database)

    def transition(self, new_status, reason="test transition", evidence=None):
        return app.transition_report_status(
            self.report_id, new_status, reason, actor="Test actor",
            verification_evidence=evidence, database_path=self.database,
        )

    def test_validation_history_persists_and_assignment_requires_assignment_record(self):
        self.transition("Pending Validation")
        self.transition("Validated")
        with self.assertRaisesRegex(ValueError, "authority assignment"):
            self.transition("Assigned")
        with app.database_connection(self.database) as connection:
            photo_path = connection.execute(
                "SELECT photo_path FROM road_reports WHERE report_id = ?", (self.report_id,)
            ).fetchone()[0]
        self.assertTrue(Path(photo_path).exists())
        self.assertEqual(self.status(), "Validated")
        events = self.history()
        self.assertEqual(len(events), 3)
        self.assertEqual(events[-1]["previous_status"], "Pending Validation")
        self.assertEqual(events[-1]["actor"], "Test actor")
        self.assertEqual(events[-1]["reason"], "test transition")
    def test_invalid_transition_reports_permitted_steps_and_changes_nothing(self):
        with self.assertRaises(app.InvalidStatusTransition) as caught:
            self.transition("Assigned")
        self.assertIn("Pending Validation", str(caught.exception))
        self.assertEqual(self.status(), "Submitted")
        self.assertEqual(len(self.history()), 1)

    def test_reason_required(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            self.transition("Pending Validation", reason="  ")
        self.assertEqual(self.status(), "Submitted")

    def test_history_insert_failure_rolls_back_status_update(self):
        with app.database_connection(self.database) as connection:
            connection.execute(
                """CREATE TRIGGER fail_history_insert BEFORE INSERT ON report_status_history
                   BEGIN SELECT RAISE(ABORT, 'simulated history failure'); END"""
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "simulated history failure"):
            self.transition("Pending Validation")
        self.assertEqual(self.status(), "Submitted")
        self.assertEqual(len(self.history()), 1)

    def test_legacy_migration_preserves_records_and_maps_resolved_to_pending_verification(self):
        legacy_db = Path(self.temp_dir.name) / "legacy.sqlite"
        photo = Path(self.temp_dir.name) / "existing-photo.jpg"
        photo.write_bytes(b"pre-existing photo")
        connection = sqlite3.connect(legacy_db)
        try:
            connection.execute(
                """CREATE TABLE road_reports (
                    report_id TEXT PRIMARY KEY, submitted_at TEXT NOT NULL,
                    road_name TEXT NOT NULL, location TEXT NOT NULL DEFAULT '',
                    district TEXT NOT NULL, problem_type TEXT NOT NULL,
                    severity TEXT NOT NULL, description TEXT NOT NULL,
                    latitude REAL, longitude REAL, photo_path TEXT,
                    status TEXT NOT NULL
                )"""
            )
            rows = [
                ("legacy-submitted", "Submitted", None),
                ("legacy-pending", "Pending", None),
                ("legacy-review", "Pending Review", None),
                ("legacy-resolved", "Resolved", str(photo)),
            ]
            for report_id, status, photo_path in rows:
                connection.execute(
                    """INSERT INTO road_reports VALUES (?, '2025-01-01T00:00:00+00:00',
                       'Old Road', 'Landmark', 'Chennai', 'Pothole', 'Low', 'Legacy report',
                       NULL, NULL, ?, ?)""",
                    (report_id, photo_path, status),
                )
            connection.commit()
        finally:
            connection.close()

        app.initialize_database(legacy_db)
        app.initialize_database(legacy_db)  # idempotence
        with app.database_connection(legacy_db) as connection:
            statuses = dict(connection.execute("SELECT report_id, status FROM road_reports"))
            report_count = connection.execute("SELECT COUNT(*) FROM road_reports").fetchone()[0]
            history_count = connection.execute(
                "SELECT COUNT(*) FROM report_status_history"
            ).fetchone()[0]
            resolved_history_row = connection.execute(
                "SELECT previous_status, new_status FROM report_status_history "
                "WHERE report_id = 'legacy-resolved'"
            ).fetchone()
            resolved_history = dict(zip(resolved_history_row.keys(), resolved_history_row))
            resolved_photo = connection.execute(
                "SELECT photo_path FROM road_reports WHERE report_id = 'legacy-resolved'"
            ).fetchone()[0]

        self.assertEqual(report_count, 4)
        self.assertEqual(history_count, 4)
        self.assertEqual(statuses["legacy-submitted"], "Submitted")
        self.assertEqual(statuses["legacy-pending"], "Pending Validation")
        self.assertEqual(statuses["legacy-review"], "Pending Validation")
        self.assertEqual(
            statuses["legacy-resolved"], app.PENDING_VERIFICATION_STATUS
        )
        self.assertEqual(resolved_history["previous_status"], "Resolved")
        self.assertEqual(resolved_photo, str(photo))
        self.assertTrue(photo.exists())


if __name__ == "__main__":
    unittest.main()


