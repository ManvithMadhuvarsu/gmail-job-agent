import json

from tools.atomic_io import append_jsonl, atomic_write_json, atomic_write_text


def test_atomic_write_text_overwrites(tmp_path):
    p = tmp_path / "a.txt"
    atomic_write_text(p, "first\n")
    atomic_write_text(p, "second\n")
    assert p.read_text(encoding="utf-8") == "second\n"


def test_atomic_write_text_does_not_leave_tmp_files(tmp_path):
    p = tmp_path / "b.txt"
    atomic_write_text(p, "data")
    leftover = [child for child in tmp_path.iterdir() if child.name.endswith(".tmp")]
    assert leftover == []


def test_atomic_write_json_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    atomic_write_json(p, {"x": [1, 2, 3]})
    assert json.loads(p.read_text(encoding="utf-8")) == {"x": [1, 2, 3]}


def test_append_jsonl_one_record_per_line(tmp_path):
    p = tmp_path / "events.jsonl"
    append_jsonl(p, {"i": 1})
    append_jsonl(p, {"i": 2})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["i"] for line in lines] == [1, 2]
