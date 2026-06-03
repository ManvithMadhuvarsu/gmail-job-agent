from tools import runtime_state


def test_manual_run_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_state, "STATE_PATH", tmp_path / "runtime_state.json")

    runtime_state.record_manual_run_started(dry_run=True, source="web")
    state = runtime_state.snapshot()

    assert state["manual_run"]["status"] == "running"
    assert state["manual_run"]["dry_run"] is True
    assert state["manual_run"]["source"] == "web"
    assert state["manual_run"]["started_at"]
    assert state["manual_run"]["finished_at"] is None

    runtime_state.record_manual_run_finished(dry_run=True, ok=False, error="boom")
    state = runtime_state.snapshot()

    assert state["manual_run"]["status"] == "failed"
    assert state["manual_run"]["error"] == "boom"
    assert state["manual_run"]["finished_at"]
