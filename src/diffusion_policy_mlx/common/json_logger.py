"""JSON Lines logger for training metrics.

Upstream: diffusion_policy/common/json_logger.py

Simplified compared to upstream — uses append-only JSONL format without
the complex seek/truncate logic. Each line is a self-contained JSON object
with a ``step`` field.
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class JsonLogger:
    """Log training metrics to a JSON Lines (.jsonl) file.

    Each call to :meth:`log` writes one JSON object per line. All values
    are filtered to numeric types by default (non-numeric values like
    strings are silently dropped unless a custom filter is provided).

    The logger supports context-manager usage::

        with JsonLogger("metrics.jsonl") as logger:
            logger.log({"loss": 0.5, "lr": 1e-4}, step=100)

    Or explicit open/close::

        logger = JsonLogger("metrics.jsonl")
        logger.start()
        logger.log({"loss": 0.5}, step=100)
        logger.stop()

    Args:
        path: Path to the JSONL output file (created if it doesn't exist,
            appended to if it does).
        filter_fn: Optional callable ``(key, value) -> bool`` that decides
            which entries to keep. Defaults to keeping only numeric values.
    """

    def __init__(
        self,
        path: Union[str, Path],
        filter_fn: Optional[Any] = None,
    ):
        self.path = Path(path)
        self.filter_fn = filter_fn or (lambda k, v: isinstance(v, numbers.Number))
        self._file = None
        self._last_log: Optional[Dict[str, Any]] = None

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> "JsonLogger":
        """Open the log file for appending.

        Returns self for chaining.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", buffering=1)  # line-buffered
        return self

    def stop(self) -> None:
        """Flush and close the log file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> "JsonLogger":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # -- Logging -------------------------------------------------------------

    def log(self, data: Dict[str, Any], step: int) -> None:
        """Write one log entry.

        The *step* is always included. Other entries are filtered through
        ``filter_fn`` and numeric values are cast to Python int/float for
        JSON serialization (e.g., numpy scalars).

        Args:
            data: Dictionary of metric names to values.
            step: The global training step number.
        """
        # Filter to numeric values
        filtered: Dict[str, Any] = {"step": int(step)}
        for k, v in data.items():
            if self.filter_fn(k, v):
                if isinstance(v, numbers.Integral):
                    filtered[k] = int(v)
                elif isinstance(v, numbers.Number):
                    filtered[k] = float(v)
                else:
                    filtered[k] = v

        self._last_log = filtered

        # Write as single JSON line
        line = json.dumps(filtered, separators=(",", ":")) + "\n"
        if self._file is not None:
            self._file.write(line)
        else:
            # Auto-open in append mode if not started
            with open(self.path, "a") as f:
                f.write(line)

    @property
    def last_log(self) -> Optional[Dict[str, Any]]:
        """Return the most recent log entry, or None."""
        return self._last_log

    # -- Reading -------------------------------------------------------------

    @staticmethod
    def read(path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Read all entries from a JSONL log file.

        Incomplete (non-newline-terminated) trailing lines are skipped.

        Args:
            path: Path to the JSONL file.

        Returns:
            List of dicts, one per log entry.
        """
        entries: List[Dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            return entries
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines (e.g., incomplete writes)
                    continue
        return entries
