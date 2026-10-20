"""Phase 4 assignment, repair and verification tests use temporary SQLite files."""
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import nullcontext
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app


class Upload:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


def jpeg_bytes(color=(50, 90, 130)):
    output = io.BytesIO()
    Image.new("RGB", (24, 16), color).save(output, format="JPEG")
    return output.getvalue()


class Phase4OperationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "reports.sqlite"
        self.upload_dir = self.root / "uploads"
        self.evidence_dir = self.root / "evidence"
        app.initialize_database(self.db)
        self.report_id = app.save_report({
            "road_name": "Test Road", "location": "Test junction", "district": "Chennai",
            "problem_type": "Pothole", "severity": "Medium", "description": "Test only",
        }, self.db, upload_dir=self.upload_dir)

    def tearDown(self):
        self.temp.cleanup()

    def status(self):
        with app.database_connection(self.db) as conn:
            return conn.execute("SELECT status FROM road_reports WHERE report_id=?", (self.report_id,)).fetchone()[0]

    def validate_report(self):
        app.transition_report_status(self.report_id, "Pending Validation", "review", database_path=self.db)
        app.transition_report_status(self.report_id, "Validated", "validated", database_path=self.db)

    def authority(self, district_ids=None):
        return app.save_authority("Prototype entry", "Local prototype", "Configured by test",
                                  district_ids or [], "No real contact", database_path=self.db,
                                  road_categories=["local test road"], routing_rules="Test rule: manual operator review required")

    def assign(self, scope=None):
        self.validate_report()
        authority_id = self.authority(scope if scope is not None else [app.district_entry_for_name("Chennai")["id"]])
        assignment_id = app.assign_report(self.report_id, authority_id, "Test team", date.today() + timedelta(days=1),
                                          "Manual test routing", database_path=self.db,
                                          assigned_officer="Test officer")
        return assignment_id

    def reach_in_progress(self):
        self.assign()
        app.schedule_repair(self.report_id, date.today(), "Patch", "Mix", "Scheduled for test", database_path=self.db)
        app.start_repair(self.report_id, "Patch", "Mix", "Work began", database_path=self.db)
        app.save_repair_before_photo(
            self.report_id, Upload("before.jpg", jpeg_bytes((20, 30, 40))), database_path=self.db,
            evidence_dir=self.evidence_dir,
        )

    def complete_repair(self):
        return app.complete_repair(
            self.report_id, "Patch", "Cold mix", "Repair completion notes", Upload("after.jpg", jpeg_bytes()),
            database_path=self.db, evidence_dir=self.evidence_dir,
        )

    def test_assignment_persists_history_and_acceptance_only_when_recorded(self):
        assignment_id = self.assign()
        row = app.fetch_assignments(self.report_id, self.db)[0]
        self.assertEqual(row["assignment_id"], assignment_id)
        self.assertEqual(row["acceptance_status"], "Awaiting response")
        self.assertEqual(row["assigned_officer"], "Test officer")
        authority = app.fetch_authorities(self.db)[0]
        self.assertEqual(json.loads(authority["road_categories"]), ["local test road"])
        self.assertEqual(authority["routing_rules"], "Test rule: manual operator review required")
        self.assertEqual(len(app.fetch_manual_review_queue(self.db)), 1)
        self.assertEqual(app.fetch_phase4_dashboard_stats(self.db)["manual_authority_review"], 1)
        self.assertEqual(len(app.fetch_assignment_history(assignment_id, self.db)), 1)
        app.record_assignment_response(assignment_id, "Acceptance recorded", "Call log reference", "Test actor", self.db)
        row = app.fetch_assignments(self.report_id, self.db)[0]
        self.assertEqual(row["acceptance_status"], "Acceptance recorded")
        self.assertEqual(row["acceptance_actor"], "Test actor")
        self.assertEqual(len(app.fetch_assignment_history(assignment_id, self.db)), 2)

    def test_uncertain_scope_stays_in_manual_review_queue(self):
        assignment_id = self.assign(scope=[app.district_entry_for_name("Madurai")["id"]])
        self.assertTrue(assignment_id)
        self.assertEqual(len(app.fetch_manual_review_queue(self.db)), 1)
        self.assertIn("No configured district hint match", app.fetch_assignments(self.report_id, self.db)[0]["manual_routing_reason"])
        app.resolve_manual_review(self.report_id, "Operator inspected local information; road ownership unverified", database_path=self.db)
        self.assertEqual(len(app.fetch_manual_review_queue(self.db)), 0)

    def test_declined_assignment_is_not_counted_as_an_active_assignment(self):
        assignment_id = self.assign()
        app.record_assignment_response(
            assignment_id, "Decline recorded", "Decline noted by operator", "Test actor", self.db,
        )
        stats = app.fetch_phase4_dashboard_stats(self.db)
        self.assertEqual(stats["assigned_cases"], 0)
        self.assertEqual(stats["unassigned_reports"], 1)
        self.assertEqual(app.fetch_assignments(self.report_id, self.db)[0]["assignment_status"], "Declined")

    def test_repair_after_photo_and_notes_are_required(self):
        self.reach_in_progress()
        with self.assertRaisesRegex(ValueError, "Completion notes"):
            app.complete_repair(self.report_id, "Patch", "Mix", "", Upload("after.jpg", jpeg_bytes()),
                                database_path=self.db, evidence_dir=self.evidence_dir)
        with self.assertRaisesRegex(ValueError, "repair-after photo is required"):
            app.complete_repair(self.report_id, "Patch", "Mix", "Notes", None,
                                database_path=self.db, evidence_dir=self.evidence_dir)
        self.assertEqual(self.status(), "Repair In Progress")
        app.complete_repair(self.report_id, "Patch", "Mix", "Notes", Upload("after.jpg", jpeg_bytes()),
                            database_path=self.db, evidence_dir=self.evidence_dir)
        self.assertEqual(self.status(), app.PENDING_VERIFICATION_STATUS)

    def test_successful_verification_closes_only_after_evidence(self):
        self.reach_in_progress()
        self.complete_repair()
        verification_id = app.record_verification_decision(self.report_id, "Approved", "After photo inspected", "Inspector metadata", self.db)
        self.assertTrue(verification_id)
        self.assertEqual(self.status(), "Verified and Closed")
        verification = app.fetch_verifications(self.report_id, self.db)[0]
        self.assertEqual(verification["report_id"], self.report_id)
        self.assertEqual(verification["decision"], "Approved")
        self.assertEqual(verification["reviewer_actor"], "Inspector metadata")
        self.assertTrue(json.loads(verification["evidence_references"]))
        status_event = app.fetch_report_status_history(self.report_id, self.db)[-1]
        self.assertEqual(status_event["new_status"], "Verified and Closed")
        self.assertIn(verification["verification_id"], status_event["verification_evidence"])
        evidence = app.fetch_report_evidence(self.report_id, self.db)
        types = {item["image_type"] for item in evidence}
        self.assertIn("Repair Before", types)
        self.assertIn("Repair After", types)
        before_path = next(item["image_path"] for item in evidence if item["image_type"] == "Repair Before")
        after_path = next(item["image_path"] for item in evidence if item["image_type"] == "Repair After")
        self.assertNotEqual(before_path, after_path)
        with self.assertRaises(app.InvalidStatusTransition):
            app.record_verification_decision(self.report_id, "Approved", "Duplicate", "Inspector metadata", self.db)

    def test_approval_rejects_missing_or_corrupt_after_image_file(self):
        self.reach_in_progress()
        self.complete_repair()
        after = next(
            item for item in app.fetch_report_evidence(self.report_id, self.db)
            if item["image_type"] == "Repair After"
        )
        image_path = Path(after["image_path"])
        image_path.unlink()
        history_count = len(app.fetch_report_status_history(self.report_id, self.db))
        with self.assertRaisesRegex(ValueError, "valid saved repair-after photo"):
            app.record_verification_decision(self.report_id, "Approved", "Test approval", "Test reviewer", self.db)
        self.assertEqual(self.status(), app.PENDING_VERIFICATION_STATUS)
        self.assertEqual(app.fetch_verifications(self.report_id, self.db), [])
        self.assertEqual(len(app.fetch_report_status_history(self.report_id, self.db)), history_count)
        with app.database_connection(self.db) as conn:
            conn.execute("DELETE FROM report_evidence WHERE evidence_id=?", (after["evidence_id"],))
        with self.assertRaisesRegex(ValueError, "repair-after photo is required"):
            app.record_verification_decision(self.report_id, "Approved", "Test approval", "Test reviewer", self.db)
        self.assertEqual(self.status(), app.PENDING_VERIFICATION_STATUS)
        self.assertEqual(app.fetch_verifications(self.report_id, self.db), [])

    def test_additional_evidence_decision_returns_case_to_repair(self):
        self.reach_in_progress()
        self.complete_repair()
        verification_id = app.record_verification_decision(
            self.report_id, "More Evidence", "Please provide a clearer repair image", "Test reviewer", self.db,
        )
        self.assertTrue(verification_id)
        self.assertEqual(self.status(), "Repair In Progress")
        self.assertEqual(app.fetch_verifications(self.report_id, self.db)[0]["decision"], "More Evidence")
        history = app.fetch_report_status_history(self.report_id, self.db)
        self.assertEqual(history[-1]["new_status"], "Repair In Progress")

    def test_verification_controls_are_visible_but_disabled_without_evidence(self):
        record = next(
            item for item in app.fetch_reports(database_path=self.db)
            if item["report_id"] == self.report_id
        )
        record["status"] = "Repair In Progress"
        with patch.object(app, "fetch_report_evidence", return_value=[]), \
             patch.object(app, "fetch_repair_records", return_value=[]), \
             patch.object(app, "fetch_verifications", return_value=[]), \
             patch.object(app.st, "columns", return_value=[nullcontext(), nullcontext()]), \
             patch.object(app.st, "markdown"), patch.object(app.st, "write"), \
             patch.object(app.st, "caption"), patch.object(app.st, "info") as info, \
             patch.object(app.st, "radio") as radio, patch.object(app.st, "text_area") as text_area, \
             patch.object(app.st, "button") as button:
            app._render_verification_section(record)
        self.assertEqual(radio.call_args.kwargs["disabled"], True)
        self.assertEqual(text_area.call_args.kwargs["disabled"], True)
        self.assertEqual(button.call_args.kwargs["disabled"], True)
        self.assertEqual(set(radio.call_args.args[1]), {
            "Approve repair", "Reject repair and return to repair", "Request additional evidence",
        })
        self.assertIn("completed repair record is required", info.call_args.args[0])

    def test_failed_verification_returns_to_repair_and_can_be_reopened(self):
        self.reach_in_progress()
        self.complete_repair()
        app.record_verification_decision(self.report_id, "Rejected", "Repair not adequate", "Inspector metadata", self.db)
        self.assertEqual(self.status(), "Repair In Progress")
        self.assertEqual(app.fetch_repair_records(self.report_id, self.db)[0]["repair_status"], "In Progress")
        history = app.fetch_report_status_history(self.report_id, self.db)
        self.assertEqual(history[-1]["new_status"], "Repair In Progress")
        self.assertIn("returned for follow-up", history[-1]["reason"])
        # Reopening a successfully closed case uses the existing lifecycle transition.
        self.complete_repair()
        app.record_verification_decision(self.report_id, "Approved", "Follow-up passed", "Inspector metadata", self.db)
        app.transition_report_status(self.report_id, "Reopened", "New issue reported", database_path=self.db)
        self.assertEqual(self.status(), "Reopened")
        app.transition_report_status(self.report_id, "Pending Validation", "Revalidation", database_path=self.db)
        self.assertEqual(self.status(), "Pending Validation")

    def test_direct_assignment_and_closure_are_blocked_without_records(self):
        with self.assertRaises(app.InvalidStatusTransition):
            app.transition_report_status(self.report_id, "Assigned", "Skip assignment", database_path=self.db)
        self.validate_report()
        with self.assertRaisesRegex(app.InvalidStatusTransition, "assignment must be recorded"):
            app.transition_report_status(self.report_id, "Assigned", "Skip assignment record", database_path=self.db)
        self.assertEqual(self.status(), "Validated")

    def test_migration_preserves_existing_report_ai_and_upload_and_backfills_evidence(self):
        original = self.root / "old-photo.jpg"
        original.write_bytes(jpeg_bytes())
        report_id = app.save_report({
            "road_name": "Legacy Road", "district": "Chennai", "problem_type": "Crack",
            "severity": "Low", "description": "Keep this row",
        }, self.db, photo_bytes=jpeg_bytes(), photo_extension=".jpg", upload_dir=self.upload_dir)
        with app.database_connection(self.db) as conn:
            conn.execute("""INSERT INTO ai_detections VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                ("ai-legacy", report_id, "v-test", "[]", "[]", "[]", "no_detections", 1.0,
                 "2025-01-01T00:00:00+00:00", None, None))
            before_reports = conn.execute("SELECT COUNT(*) FROM road_reports").fetchone()[0]
            before_ai = conn.execute("SELECT COUNT(*) FROM ai_detections").fetchone()[0]
            before_history = conn.execute("SELECT COUNT(*) FROM report_status_history").fetchone()[0]
        saved_report = next(item for item in app.fetch_reports(database_path=self.db) if item["report_id"] == report_id)
        uploaded = Path(saved_report["photo_path"])
        if not uploaded.is_absolute():
            uploaded = app.BASE_DIR / uploaded
        self.assertTrue(uploaded.exists())
        app.initialize_database(self.db)
        app.initialize_database(self.db)
        with app.database_connection(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM road_reports").fetchone()[0], before_reports)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ai_detections").fetchone()[0], before_ai)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM report_status_history").fetchone()[0], before_history)
            self.assertEqual(conn.execute("SELECT detection_id FROM ai_detections WHERE detection_id='ai-legacy'").fetchone()[0], "ai-legacy")
        evidence = app.fetch_report_evidence(report_id, self.db)
        self.assertTrue(any(item["image_type"] == "Original Evidence" for item in evidence))
        self.assertTrue(uploaded.exists())


if __name__ == "__main__":
    unittest.main()
