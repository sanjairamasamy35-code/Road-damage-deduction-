"""Phase 5 dashboard tests. All reports and records here are temporary fixtures."""
import tempfile
import unittest
import uuid
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import app
from streamlit.testing.v1 import AppTest


class DashboardConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "phase5-test.sqlite"
        self.uploads = self.root / "uploads"
        app.initialize_database(self.db)
        self.authority_id = app.save_authority(
            "Test-only configured authority", "Test fixture", "Temporary test configuration",
            [], "No real contact", database_path=self.db,
        )

    def tearDown(self):
        self.temp.cleanup()

    def report(self, label, status="Submitted", district="Chennai", severity="Medium", submitted_at=None):
        report_id = app.save_report({
            "road_name": f"TEST FIXTURE {label}", "location": "Test-only location",
            "district": district, "problem_type": "Pothole", "severity": severity,
            "description": "Clearly labelled automated test fixture.",
        }, database_path=self.db, upload_dir=self.uploads)
        with app.database_connection(self.db) as conn:
            conn.execute("UPDATE road_reports SET status=?, submitted_at=? WHERE report_id=?",
                         (status, submitted_at or datetime.now(timezone.utc).isoformat(), report_id))
            if status in ("Verified and Closed", "Rejected"):
                conn.execute(
                    "UPDATE manual_review_queue SET queue_status='Resolved',resolved_at=?,resolved_by=?,resolution=? WHERE report_id=?",
                    (datetime.now(timezone.utc).isoformat(), "Test actor", "Test fixture terminal status", report_id),
                )
        return report_id

    def assignment(self, report_id, assigned_at=None, due_at=None, assignment_status="Active"):
        assignment_id = str(uuid.uuid4())
        now = assigned_at or datetime.now(timezone.utc).isoformat()
        with app.database_connection(self.db) as conn:
            conn.execute("""INSERT INTO assignments (
                assignment_id,report_id,authority_id,assigned_team,assigned_actor,assigned_at,due_at,
                acceptance_status,assignment_status,manual_routing_reason
            ) VALUES (?,?,?,?,?,?,?,'Awaiting response',?,?)""",
                         (assignment_id, report_id, self.authority_id, "Test-only team", "Test actor",
                          now, due_at, assignment_status, "Automated test fixture only"))
        return assignment_id

    def test_workflow_buckets_have_documented_counts_and_click_through_matches(self):
        self.report("unassigned", "Submitted")
        self.report("assigned", "Assigned")
        self.report("scheduled", "Repair Scheduled")
        self.report("repair", "Repair In Progress")
        self.report("verification", app.PENDING_VERIFICATION_STATUS)
        self.report("verified", "Verified and Closed")
        self.report("reopened", "Reopened")
        self.report("rejected", "Rejected")
        for record in app.fetch_reports(self.db):
            if record["status"] in ("Assigned", "Repair Scheduled", "Repair In Progress", app.PENDING_VERIFICATION_STATUS):
                self.assignment(record["report_id"])

        stats = app.fetch_phase4_dashboard_stats(self.db)
        self.assertEqual(stats["unassigned_reports"], 1)
        self.assertEqual(stats["assigned_cases"], 2)
        self.assertEqual(stats["repairs_in_progress"], 1)
        self.assertEqual(stats["awaiting_verification"], 1)
        self.assertEqual(stats["verified_cases"], 1)
        self.assertEqual(stats["reopened_cases"], 1)
        # Every nonterminal test report remains in manual review until explicit resolution,
        # regardless of assignment presence.
        self.assertEqual(stats["manual_authority_review"], 6)
        for key, bucket in (
            ("unassigned_reports", "unassigned"), ("assigned_cases", "assigned"),
            ("repairs_in_progress", "repair_in_progress"), ("awaiting_verification", "awaiting_verification"),
            ("verified_cases", "verified"), ("reopened_cases", "reopened"),
            ("manual_authority_review", "manual_review"),
        ):
            self.assertEqual(stats[key], len(app.fetch_reports(self.db, {"workflow_bucket": bucket})))
        reports = app.fetch_reports(self.db)
        assigned_id = next(row["report_id"] for row in reports if row["status"] == "Assigned")
        app.resolve_manual_review(assigned_id, "Explicit test review outcome", "Test actor", self.db)
        self.assertEqual(app.fetch_phase4_dashboard_stats(self.db)["manual_authority_review"], 5)

    def test_dashboard_count_button_opens_report_history_with_matching_filter(self):
        script = "\n".join((
            "import app",
            "import streamlit as st",
            "app._dashboard_metric(st.columns(1)[0], 'TEST fixture count', 7, 'fixture', {'workflow_bucket': 'open'})",
        ))
        ui = AppTest.from_string(script).run(timeout=10)
        self.assertEqual(len(ui.exception), 0)
        self.assertEqual(len(ui.button), 1)
        ui.button[0].click().run(timeout=10)
        self.assertEqual(ui.session_state["page"], "Report History")
        self.assertEqual(ui.session_state["dashboard_history_filters"], {"workflow_bucket": "open"})

    def test_overdue_counts_only_active_due_assignments_in_active_work_status(self):
        overdue = self.report("overdue", "Repair In Progress")
        future_due = self.report("future", "Repair In Progress")
        completed_status = self.report("closed-job", "Verified and Closed")
        now = datetime.now(timezone.utc)
        self.assignment(overdue, due_at=(now - timedelta(days=1)).isoformat())
        self.assignment(future_due, due_at=(now + timedelta(days=1)).isoformat())
        self.assignment(completed_status, due_at=(now - timedelta(days=1)).isoformat())
        stats = app.fetch_phase4_dashboard_stats(self.db)
        self.assertEqual(stats["overdue_jobs"], 1)
        self.assertEqual([r["report_id"] for r in app.fetch_reports(self.db, {"workflow_bucket": "overdue"})], [overdue])

    def test_open_aggregate_includes_unmapped_legacy_statuses_but_excludes_terminal_statuses(self):
        self.report("legacy-open", "Legacy status retained")
        self.report("closed", "Verified and Closed")
        self.report("rejected", "Rejected")
        _total, open_count, _verified, _categories = app.fetch_dashboard_stats(self.db)
        self.assertEqual(open_count, 1)
        self.assertEqual(len(app.fetch_reports(self.db, {"workflow_bucket": "open"})), 1)

    def test_history_filters_share_date_district_severity_status_authority_and_latest_ai_logic(self):
        first_at = "2026-10-01T12:00:00+00:00"
        matching = self.report("matching", "Repair In Progress", district="Chennai",
                               severity="High", submitted_at=first_at)
        other = self.report("other", "Submitted", district="Madurai", severity="Low",
                            submitted_at="2026-10-15T12:00:00+00:00")
        self.assignment(matching)
        with app.database_connection(self.db) as conn:
            conn.execute("""INSERT INTO ai_detections (
                detection_id,report_id,model_version,bounding_boxes,classes,confidence_scores,
                inference_status,processing_duration_ms,timestamp,annotated_image_path,error_message
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                         ("test-ai-1", matching, "TEST ONLY", "[]", "[]", "[]",
                          "detections", 1.0, first_at, None, None))
        filters = {
            "district_id": app.district_entry_for_name("Chennai")["id"],
            "severity": "High", "status": "Repair In Progress",
            "submitted_from": "2026-10-01", "submitted_to": "2026-10-10",
            "authority_id": self.authority_id, "ai_outcome": "detections",
        }
        self.assertEqual(app.fetch_reports(self.db, filters)[0]["report_id"], matching)
        self.assertEqual(app.count_reports(filters, self.db), 1)
        self.assertEqual(app.fetch_reports(self.db, {**filters, "ai_outcome": "none"}), [])
        self.assertEqual(app.fetch_reports(self.db, {"authority_id": "__unassigned__"})[0]["report_id"], other)

    def test_derived_metrics_are_unavailable_without_pairs_and_use_recorded_data(self):
        self.assertEqual(app.fetch_dashboard_derived_metrics(self.db), {
            "avg_submission_to_assignment_seconds": None, "submission_to_assignment_n": 0,
            "avg_repair_completion_seconds": None, "repair_completion_n": 0,
            "verification_failure_rate": None, "verification_decision_n": 0,
        })
        submitted = "2026-10-01T00:00:00+00:00"
        assigned = "2026-10-01T02:00:00+00:00"
        report_id = self.report("metric", "Repair In Progress", submitted_at=submitted)
        assignment_id = self.assignment(report_id, assigned_at=assigned)
        start, finish = "2026-10-02T00:00:00+00:00", "2026-10-02T03:00:00+00:00"
        with app.database_connection(self.db) as conn:
            conn.execute("""INSERT INTO repair_records (
                repair_id,report_id,assignment_id,repair_status,repair_notes,started_at,completed_at,
                recorded_actor,created_at,updated_at
            ) VALUES (?,?,?,'Completed','Test-only notes',?,?,?, ?, ?)""",
                         ("test-repair", report_id, assignment_id, start, finish, "Test actor", start, finish))
            conn.execute("""INSERT INTO verifications (
                verification_id,report_id,repair_id,reviewer_actor,decision,comments,evidence_references,created_at
            ) VALUES ('test-verification',?,'test-repair','Test actor','More Evidence','Test-only decision','[]',?)""",
                         (report_id, finish))
        metrics = app.fetch_dashboard_derived_metrics(self.db)
        self.assertEqual(metrics["avg_submission_to_assignment_seconds"], 7200)
        self.assertEqual(metrics["submission_to_assignment_n"], 1)
        self.assertEqual(metrics["avg_repair_completion_seconds"], 10800)
        self.assertEqual(metrics["repair_completion_n"], 1)
        self.assertEqual(metrics["verification_failure_rate"], 1.0)
        self.assertEqual(metrics["verification_decision_n"], 1)

    def test_dashboard_queries_preserve_existing_report_ai_and_uploaded_photo(self):
        photo = b"temporary test evidence; not a real incident"
        report_id = app.save_report({
            "road_name": "TEST FIXTURE preservation", "location": "Test-only",
            "district": "Chennai", "problem_type": "Pothole", "severity": "Low",
            "description": "Automated preservation fixture.",
        }, database_path=self.db, photo_bytes=photo, photo_extension=".jpg", upload_dir=self.uploads)
        with app.database_connection(self.db) as conn:
            conn.execute("""INSERT INTO ai_detections (
                detection_id,report_id,model_version,bounding_boxes,classes,confidence_scores,
                inference_status,processing_duration_ms,timestamp,annotated_image_path,error_message
            ) VALUES ('keep-ai',?,'TEST ONLY','[]','[]','[]','no_detections',1,?,NULL,NULL)""",
                         (report_id, datetime.now(timezone.utc).isoformat()))
            before = conn.execute("SELECT report_id,photo_path,status FROM road_reports WHERE report_id=?",
                                  (report_id,)).fetchone()
            photo_path = self.root / before["photo_path"]
            image_before = photo_path.read_bytes()
        app.fetch_dashboard_stats(self.db)
        app.fetch_phase4_dashboard_stats(self.db)
        app.fetch_dashboard_derived_metrics(self.db)
        app.fetch_reports(self.db, {"ai_outcome": "no_detections"})
        with app.database_connection(self.db) as conn:
            after = conn.execute("SELECT report_id,photo_path,status FROM road_reports WHERE report_id=?",
                                 (report_id,)).fetchone()
            self.assertEqual(tuple(before), tuple(after))
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ai_detections WHERE detection_id='keep-ai'").fetchone()[0], 1)
        self.assertEqual(photo_path.read_bytes(), image_before)

    def test_duplicate_submission_token_returns_existing_record_without_overwriting_photo(self):
        report = {
            "road_name": "TEST FIXTURE idempotency", "location": "Temporary",
            "district": "Chennai", "problem_type": "Pothole", "severity": "Low",
            "description": "Explicit automated idempotency fixture.",
        }
        token = "test-session-submission-token"
        first = app.save_report(
            report, database_path=self.db, photo_bytes=b"first test photo",
            photo_extension=".jpg", upload_dir=self.uploads, submission_token=token,
        )
        second = app.save_report(
            report, database_path=self.db, photo_bytes=b"duplicate test photo",
            photo_extension=".jpg", upload_dir=self.uploads, submission_token=token,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(app.fetch_reports(self.db)), 1)
        self.assertEqual(len(app.fetch_report_status_history(first, self.db)), 1)
        record = app.fetch_reports(self.db)[0]
        self.assertEqual((self.uploads / f"{first}.jpg").read_bytes(), b"first test photo")
        self.assertEqual(record["submission_token"], token)

    def test_additive_submission_token_migration_preserves_legacy_record_and_photo(self):
        legacy_db = self.root / "legacy.sqlite"
        legacy_photo = self.root / "legacy-upload.jpg"
        legacy_photo.write_bytes(b"legacy photo fixture")
        conn = sqlite3.connect(legacy_db)
        try:
            conn.execute("""CREATE TABLE road_reports (
                report_id TEXT PRIMARY KEY, submitted_at TEXT NOT NULL, road_name TEXT NOT NULL,
                location TEXT NOT NULL DEFAULT '', district TEXT NOT NULL, district_id TEXT,
                problem_type TEXT NOT NULL, severity TEXT NOT NULL, description TEXT NOT NULL,
                latitude REAL, longitude REAL, photo_path TEXT,
                status TEXT NOT NULL DEFAULT 'Submitted'
            )""")
            conn.execute("""INSERT INTO road_reports VALUES (
                'legacy-test-report','2026-01-01T00:00:00+00:00','TEST FIXTURE old row','',
                'Chennai',NULL,'Pothole','Low','Migration preservation fixture',NULL,NULL,?,'Submitted'
            )""", (str(legacy_photo),))
            conn.commit()
        finally:
            conn.close()
        app.initialize_database(legacy_db)
        with app.database_connection(legacy_db) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(road_reports)")}
            report = conn.execute("SELECT * FROM road_reports WHERE report_id='legacy-test-report'").fetchone()
            index = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_road_reports_submission_token'"
            ).fetchone()
        self.assertIn("submission_token", columns)
        self.assertIsNone(report["submission_token"])
        self.assertEqual(report["status"], "Submitted")
        self.assertIsNotNone(index)
        self.assertEqual(legacy_photo.read_bytes(), b"legacy photo fixture")
        self.assertEqual(len(app.fetch_report_status_history("legacy-test-report", legacy_db)), 1)


if __name__ == "__main__":
    unittest.main()
