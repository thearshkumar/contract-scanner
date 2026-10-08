import json
import logging

from shared.jsonlog import JsonFormatter, fingerprint, setup


def _record(**extra):
    rec = logging.makeLogRecord({"name": "eval", "levelno": logging.INFO, "levelname": "INFO", "msg": "run %s", "args": ("x",)})
    rec.__dict__.update(extra)
    return rec


def test_one_json_object_per_line_with_extras():
    line = JsonFormatter().format(_record(neurons=12.5, model="m"))
    assert "\n" not in line
    row = json.loads(line)
    assert row["msg"] == "run x" and row["level"] == "info" and row["logger"] == "eval"
    assert row["neurons"] == 12.5 and row["model"] == "m"


def test_exceptions_are_captured():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = _record()
        rec.exc_info = sys.exc_info()
    assert "ValueError" in json.loads(JsonFormatter().format(rec))["exc"]


def test_fingerprint_hides_text():
    fp = fingerprint("CONFIDENTIAL clause")
    assert fp["len"] == 19 and len(fp["sha256"]) == 16
    assert "CONFIDENTIAL" not in json.dumps(fp)


def test_setup_writes_jsonl(tmp_path, capsys):
    path = tmp_path / "run.jsonl"
    setup(path=str(path))
    logging.getLogger("t").info("hello", extra={"step": 1})
    logging.shutdown()
    assert json.loads(path.read_text().strip())["step"] == 1
