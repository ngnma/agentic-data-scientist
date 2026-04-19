from types import SimpleNamespace

import pandas as pd

from tools.evaluation import evaluate_best, save_json, write_markdown_report


def test_save_json_writes_file(tmp_path):
    path = tmp_path / "data.json"
    save_json(path, {"a": 1})

    assert path.exists()
    assert '"a": 1' in path.read_text()


def test_evaluate_best_creates_confusion_matrix(tmp_path):
    payload = {
        "best": {
            "name": "DummyMostFrequent",
            "metrics": {
                "model": "DummyMostFrequent",
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "f1_macro": 0.5,
                "precision_macro": 0.5,
                "recall_macro": 0.5,
            },
            "y_test": pd.Series([0, 1, 0, 1]),
            "y_pred": [0, 1, 1, 1],
        },
        "all_metrics": [{"model": "DummyMostFrequent", "f1_macro": 0.5}],
    }

    result = evaluate_best(payload, str(tmp_path))

    assert result["best_metrics"]["model"] == "DummyMostFrequent"
    assert "classification_report" in result
    assert (tmp_path / "confusion_matrix.png").exists()


def test_write_markdown_report_writes_expected_sections(tmp_path):
    out_path = tmp_path / "report.md"
    ctx = SimpleNamespace(
        run_id="run1",
        started_at="2026-01-01T00:00:00Z",
        data_path="data/example_dataset.csv",
        target="target",
    )

    write_markdown_report(
        out_path=out_path,
        ctx=ctx,
        fingerprint="fp_1",
        dataset_profile={
            "shape": {"rows": 10, "cols": 4},
            "is_classification": True,
            "imbalance_ratio": 1.0,
            "feature_types": {"numeric": ["x1"], "categorical": ["x2"]},
            "notes": [],
        },
        plan=["P3B_select_models"],
        eval_payload={
            "best_metrics": {
                "model": "LogisticRegression",
                "accuracy": 0.8,
                "balanced_accuracy": 0.8,
                "f1_macro": 0.8,
                "precision_macro": 0.8,
                "recall_macro": 0.8,
            },
            "all_metrics": [],
            "confusion_matrix_path": "confusion_matrix.png",
        },
        reflection={"suggestions": ["Tune hyperparameters"]},
    )

    text = out_path.read_text()
    assert "# Agentic Data Scientist Report" in text
    assert "LogisticRegression" in text
    assert "confusion_matrix.png" in text