import subprocess
import sys


def test_smoke_run_creates_outputs(tmp_path):
    out_dir = tmp_path / "agt_outputs"
    cmd = [
        sys.executable,
        "run_agent.py",
        "--data",
        "data/example_dataset.csv",
        "--target",
        "auto",
        "--output_root",
        str(out_dir),
        "--quiet",
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, res.stderr
    assert out_dir.exists()