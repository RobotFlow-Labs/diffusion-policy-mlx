"""Tests for examples/ scripts.

Each test runs the corresponding example as a subprocess and verifies
it exits successfully (return code 0).
"""

import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _run_example(script_name: str, timeout: int = 60):
    """Run an example script and assert it succeeds."""
    script_path = EXAMPLES_DIR / script_name
    assert script_path.exists(), f"Example not found: {script_path}"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(EXAMPLES_DIR.parent),  # run from project root
    )
    assert result.returncode == 0, (
        f"{script_name} failed (exit code {result.returncode}).\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )
    return result.stdout


class TestExamples:
    def test_01_quickstart(self):
        """01_quickstart.py should run inference and print shapes."""
        output = _run_example("01_quickstart.py", timeout=30)
        assert "action shape:" in output
        assert "action_pred shape:" in output
        assert "Done!" in output

    def test_02_noise_visualization(self):
        """02_noise_visualization.py should show forward and reverse diffusion."""
        output = _run_example("02_noise_visualization.py", timeout=30)
        assert "Forward process" in output
        assert "Reverse process" in output
        assert "Done!" in output

    def test_03_train_synthetic(self):
        """03_train_synthetic.py should train and show decreasing loss."""
        output = _run_example("03_train_synthetic.py", timeout=120)
        assert "Training complete" in output
        assert "Loss decreased: True" in output
        assert "Done!" in output

    def test_04_scheduler_comparison(self):
        """04_scheduler_comparison.py should compare DDPM and DDIM timing."""
        output = _run_example("04_scheduler_comparison.py", timeout=60)
        assert "DDPM" in output
        assert "DDIM" in output
        assert "speedup" in output
        assert "Done!" in output

    def test_05_weight_conversion(self):
        """05_weight_conversion.py should convert weights (or skip gracefully)."""
        output = _run_example("05_weight_conversion.py", timeout=30)
        # Either succeeds with conversion or skips due to missing torch
        assert "Done!" in output or "Skipping example" in output

    def test_06_resnet_features(self):
        """06_resnet_features.py should extract features and print stats."""
        output = _run_example("06_resnet_features.py", timeout=30)
        assert "Feature dim:" in output
        assert "512" in output
        assert "Done!" in output
