#!/usr/bin/env python
"""Download the PushT dataset for Diffusion Policy.

Usage::

    python scripts/download_pusht.py --output data/

Downloads the zarr replay buffer from the official Diffusion Policy
data server hosted at Columbia University.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht_cchi_v7_replay.zarr.zip"

# Expected output directory name after extraction
ZARR_DIR_NAME = "pusht_cchi_v7_replay.zarr"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Simple download progress indicator."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100.0)
        mb_down = downloaded / 1e6
        mb_total = total_size / 1e6
        print(
            f"\r  {mb_down:.1f} / {mb_total:.1f} MB ({pct:.1f}%)",
            end="",
            flush=True,
        )


def download_pusht(output_dir: str = "data") -> str:
    """Download and extract the PushT zarr dataset.

    Parameters
    ----------
    output_dir : str
        Parent directory where the zarr archive will be placed.

    Returns
    -------
    str
        Path to the extracted zarr directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    zarr_path = output_path / ZARR_DIR_NAME

    if zarr_path.exists():
        print(f"Dataset already exists at {zarr_path}")
        return str(zarr_path)

    zip_path = output_path / f"{ZARR_DIR_NAME}.zip"

    print(f"Downloading PushT dataset from:\n  {PUSHT_URL}")
    print(f"Saving to: {zip_path}")
    urlretrieve(PUSHT_URL, str(zip_path), reporthook=_progress_hook)
    print()  # newline after progress

    print("Extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        # Validate against path traversal (Zip Slip)
        for member in zf.namelist():
            member_path = os.path.realpath(os.path.join(str(output_path), member))
            if not member_path.startswith(os.path.realpath(str(output_path))):
                raise ValueError(
                    f"Zip entry {member!r} would escape target directory — aborting"
                )
        zf.extractall(str(output_path))

    # Clean up zip
    zip_path.unlink()

    # Verify basic structure
    _verify(zarr_path)

    print(f"Dataset ready at {zarr_path}")
    return str(zarr_path)


def _verify(zarr_path: Path) -> None:
    """Quick integrity check on the extracted zarr."""
    import zarr

    try:
        root = zarr.open(str(zarr_path), mode="r")
        episode_ends = root["meta/episode_ends"][:]
        n_steps = int(episode_ends[-1])
        n_episodes = len(episode_ends)

        img_shape = root["data/img"].shape
        state_shape = root["data/state"].shape
        action_shape = root["data/action"].shape

        print(f"  Episodes:  {n_episodes}")
        print(f"  Steps:     {n_steps}")
        print(f"  Images:    {img_shape}")
        print(f"  States:    {state_shape}")
        print(f"  Actions:   {action_shape}")

        assert img_shape[0] == n_steps, "img length mismatch"
        assert state_shape[0] == n_steps, "state length mismatch"
        assert action_shape[0] == n_steps, "action length mismatch"
        print("  Integrity check passed.")
    except Exception as exc:
        print(f"  WARNING: Integrity check failed: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PushT dataset for Diffusion Policy")
    parser.add_argument(
        "--output",
        default="data",
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()
    download_pusht(args.output)


if __name__ == "__main__":
    main()
