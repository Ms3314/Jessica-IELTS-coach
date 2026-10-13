"""Profile (saved mock-test scores) tests — temp files only.

    python tests/test_profile.py
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import profile


def _db():
    d = tempfile.TemporaryDirectory()
    return d, Path(d.name) / "data" / "profile.db"


def _overview(path):
    """overview() against a temp database AND an empty exams folder (never the real one)."""
    return profile.overview(path, path.parent / "no-exams")


def _log(exam_id="e1", session_id="s1", started_at=1000.0, length="full", finished=True):
    return SimpleNamespace(session_id=session_id, exam_id=exam_id, started_at=started_at,
                           length=length, finished=finished)


def _report(overall=6.5, f=7.0, l=6.0, g=6.0, p=6.5, reliability="medium", finished=True):
    crit = lambda b: {"band": b, "summary": "", "strengths": [], "improvements": [], "evidence": []}  # noqa: E731
    return {"overall_band": overall, "fluency_coherence": crit(f), "lexical_resource": crit(l),
            "grammatical_range_accuracy": crit(g), "pronunciation": crit(p),
            "reliability": reliability, "finished": finished}


def test_a_tests_scores_are_saved_with_every_criterion():
    d, path = _db()
    assert profile.record_exam(_report(), _log(), path) is True
    [row] = profile.list_scores(path)
    assert (row["overall"], row["fluency"], row["lexical"], row["grammar"], row["pronunciation"]) == (6.5, 7.0, 6.0, 6.0, 6.5)
    assert row["length"] == "full" and row["finished"] is True and row["reliability"] == "medium"
    assert row["session_id"] == "s1" and row["exam_id"] == "e1" and row["taken_at"] == 1000.0
    d.cleanup()


def test_a_report_without_scores_is_not_saved():
    d, path = _db()
    assert profile.record_exam({"insufficient": True, "message": "too few answers"}, _log(), path) is False
    assert profile.record_exam({"error": "api down"}, _log(), path) is False
    assert profile.list_scores(path) == []
    d.cleanup()


def test_the_database_is_created_on_first_use_and_listing_without_one_is_empty():
    d, path = _db()
    assert not path.exists() and profile.list_scores(path) == []
    profile.record_exam(_report(), _log(), path)
    assert path.exists()
    d.cleanup()


def test_analysing_the_same_test_again_replaces_its_scores():
    d, path = _db()
    profile.record_exam(_report(overall=6.0), _log(), path)
    profile.record_exam(_report(overall=7.0), _log(), path)
    rows = profile.list_scores(path)
    assert len(rows) == 1 and rows[0]["overall"] == 7.0
    d.cleanup()


def test_tests_are_listed_newest_first():
    d, path = _db()
    for i, (when, overall) in enumerate([(100.0, 5.5), (300.0, 6.5), (200.0, 6.0)]):
        profile.record_exam(_report(overall=overall), _log(exam_id=f"e{i}", started_at=when), path)
    assert [r["overall"] for r in profile.list_scores(path)] == [6.5, 6.0, 5.5]
    d.cleanup()


def test_the_summary_gives_latest_best_average_change_and_a_trend():
    d, path = _db()
    for i, (when, overall) in enumerate([(100.0, 5.5), (200.0, 6.0), (300.0, 7.0)]):
        profile.record_exam(_report(overall=overall), _log(exam_id=f"e{i}", started_at=when), path)
    s = _overview(path)["summary"]
    assert (s["tests"], s["latest"], s["best"], s["average"], s["change"]) == (3, 7.0, 7.0, 6.2, 1.0)
    assert s["trend"] == [5.5, 6.0, 7.0], "oldest to newest, ready for a chart"
    assert s["criteria_average"]["fluency"] == 7.0
    d.cleanup()


def test_a_first_test_has_no_change_to_report_and_no_tests_means_an_empty_summary():
    d, path = _db()
    assert profile.summarize([])["tests"] == 0 and profile.summarize([])["latest"] is None
    profile.record_exam(_report(overall=6.0), _log(), path)
    assert _overview(path)["summary"]["change"] is None
    d.cleanup()


def test_a_missing_criterion_does_not_break_the_averages():
    d, path = _db()
    partial = _report()
    del partial["pronunciation"]
    profile.record_exam(partial, _log(), path)
    s = _overview(path)["summary"]
    assert s["criteria_average"]["pronunciation"] is None and s["criteria_average"]["fluency"] == 7.0
    d.cleanup()


def test_deleting_a_session_removes_only_its_scores():
    d, path = _db()
    profile.record_exam(_report(), _log(session_id="keep", exam_id="a"), path)
    profile.record_exam(_report(), _log(session_id="drop", exam_id="b"), path)
    profile.delete_session("drop", path)
    assert [r["session_id"] for r in profile.list_scores(path)] == ["keep"]
    profile.delete_session("never-existed", path)             # harmless
    d.cleanup()


def test_the_short_test_is_recorded_as_short():
    d, path = _db()
    profile.record_exam(_report(), _log(length="short"), path)
    assert profile.list_scores(path)[0]["length"] == "short"
    d.cleanup()


def _write_exam(root, session, exam, report, meta=None):
    import json

    d = root / session / exam
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps(report))
    if meta is not None:
        (d / "exam.json").write_text(json.dumps(meta))


def test_tests_taken_before_the_profile_existed_are_imported_from_their_reports():
    d, path = _db()
    exams = Path(d.name) / "exams"
    _write_exam(exams, "s1", "20261013-233700", _report(overall=6.0),
                {"started_at": 1700000000.0, "finished": True, "length": "short"})
    _write_exam(exams, "s2", "20261013-234800", _report(overall=6.5), {"started_at": 1700005000.0, "finished": False})
    assert profile.sync_from_exams(path, exams) == 2
    rows = {r["session_id"]: r for r in profile.list_scores(path)}
    assert rows["s1"]["overall"] == 6.0 and rows["s1"]["length"] == "short" and rows["s1"]["taken_at"] == 1700000000.0
    assert rows["s2"]["length"] == "full" and rows["s2"]["finished"] is False, "older tests count as full"
    d.cleanup()


def test_importing_skips_reports_without_scores_and_never_duplicates_or_overwrites():
    d, path = _db()
    exams = Path(d.name) / "exams"
    _write_exam(exams, "s1", "e1", {"insufficient": True, "message": "too few answers"}, {})
    _write_exam(exams, "s2", "e2", _report(overall=6.0), {"started_at": 1.0})
    assert profile.sync_from_exams(path, exams) == 1
    profile.record_exam(_report(overall=7.0), _log(session_id="s2", exam_id="e2"), path)   # a newer score for it
    assert profile.sync_from_exams(path, exams) == 0, "already known: nothing added"
    assert [r["overall"] for r in profile.list_scores(path)] == [7.0], "and the saved score was left alone"
    d.cleanup()


def test_a_broken_report_file_does_not_stop_the_import():
    d, path = _db()
    exams = Path(d.name) / "exams"
    bad = exams / "s1" / "e1"
    bad.mkdir(parents=True)
    (bad / "report.json").write_text("{not json")
    _write_exam(exams, "s2", "e2", _report(overall=6.5), {"started_at": 5.0})
    assert profile.sync_from_exams(path, exams) == 1
    d.cleanup()


def test_a_missing_exams_folder_is_simply_nothing_to_import():
    d, path = _db()
    assert profile.sync_from_exams(path, Path(d.name) / "nope") == 0
    d.cleanup()


def test_opening_the_profile_imports_old_tests_first():
    d, path = _db()
    exams = Path(d.name) / "exams"
    _write_exam(exams, "s1", "e1", _report(overall=6.0), {"started_at": 10.0})
    result = profile.overview(path, exams)
    assert result["summary"]["tests"] == 1 and result["summary"]["latest"] == 6.0
    d.cleanup()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} profile tests passed")
