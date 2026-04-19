import json
import pytest


from agents.memory import (
    JSONMemory,
    build_reflection_entry,
    get_relevant_reflections,
    prioritize_suggestions_from_memory,
)


def test_json_memory_upsert_and_get(tmp_path):
    path = tmp_path / "memory.json"
    mem = JSONMemory(str(path))

    mem.upsert_dataset_record("fp_1", {"best_model": "RandomForest"})
    record = mem.get_dataset_record("fp_1")

    assert record["best_model"] == "RandomForest"
    assert path.exists()


def test_json_memory_add_note(tmp_path):
    path = tmp_path / "memory.json"
    mem = JSONMemory(str(path))

    mem.add_note("hello")

    data = json.loads(path.read_text())
    assert data["notes"][0]["msg"] == "hello"


def test_json_memory_recovers_from_corrupt_file(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ bad json")

    mem = JSONMemory(str(path))

    assert "datasets" in mem.data
    assert "notes" in mem.data
    assert (tmp_path / "memory.json.bak").exists()


def test_build_reflection_entry_computes_outcome():
    entry = build_reflection_entry(
        run_id="run1",
        dataset_profile={
            "shape": {"rows": 10, "cols": 4},
            "imbalance_ratio": 1.0,
            "noise_ratio": 0.1,
            "missing_pct": {"a": 0.0},
        },
        best_metrics={"model": "LogisticRegression"},
        reflection={"issues": ["overfitting"], "suggestions": [["P3A4_regularization"]]},
        actions_applied=["P3A4_regularization"],
        before_metrics={"model": "LogisticRegression", "f1_macro": 0.5, "balanced_accuracy": 0.6},
        after_metrics={"model": "LogisticRegression", "f1_macro": 0.7, "balanced_accuracy": 0.8},
    )

    assert entry["outcome"]["improved"] is True
    assert entry["outcome"]["delta_f1_macro"] == pytest.approx(0.2)
    assert entry["actions_applied"] == ["P3A4_regularization"]


def test_get_relevant_reflections_ranks_matching_issue():
    history = {
        "history": [
            {
                "run_id": "a",
                "issues": ["overfitting"],
                "context": {"best_model": "RandomForest", "rows": 100, "cols": 10},
                "outcome": {"delta_f1_macro": 0.10},
            },
            {
                "run_id": "b",
                "issues": ["underfitting"],
                "context": {"best_model": "LogisticRegression", "rows": 100, "cols": 10},
                "outcome": {"delta_f1_macro": 0.20},
            },
        ]
    }

    results = get_relevant_reflections(
        reflection_memory=history,
        issues=["overfitting"],
        dataset_profile={"shape": {"rows": 100, "cols": 10}},
        best_model="RandomForest",
        top_k=2,
    )

    assert len(results) == 1
    assert results[0]["run_id"] == "a"


def test_prioritize_suggestions_from_memory_uses_successful_actions():
    suggestions = [
        ["P4A_tune_hyperparameters", "P3A4_regularization"],
        ["P3A_SMOTE"],
    ]
    memories = [
        {
            "actions_applied": ["P3A4_regularization"],
            "outcome": {"delta_f1_macro": 0.08, "improved": True},
        },
        {
            "actions_applied": ["P4A_tune_hyperparameters"],
            "outcome": {"delta_f1_macro": 0.0, "improved": False},
        },
    ]

    reordered = prioritize_suggestions_from_memory(suggestions, memories)

    assert reordered[0][0] == "P3A4_regularization"