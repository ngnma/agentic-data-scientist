import json
from agents.memory import JSONMemory


def test_memory_upsert_and_get(tmp_path):
    memory_file = tmp_path / "memory.json"
    mem = JSONMemory(str(memory_file))

    mem.upsert_dataset_record("fp_123", {"best_model": "RandomForest"})
    record = mem.get_dataset_record("fp_123")

    assert record == {"best_model": "RandomForest"}
    assert memory_file.exists()


def test_memory_add_note(tmp_path):
    memory_file = tmp_path / "memory.json"
    mem = JSONMemory(str(memory_file))

    mem.add_note("hello")

    data = json.loads(memory_file.read_text())
    assert len(data["notes"]) == 1
    assert data["notes"][0]["msg"] == "hello"


def test_memory_recovers_from_corrupt_file(tmp_path):
    memory_file = tmp_path / "memory.json"
    memory_file.write_text("{ bad json")

    mem = JSONMemory(str(memory_file))

    assert "datasets" in mem.data
    assert "notes" in mem.data
    assert (tmp_path / "memory.json.bak").exists()