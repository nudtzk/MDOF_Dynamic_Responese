from __future__ import annotations

from pathlib import Path

import numpy as np

G = 9.81
EL_CENTRO_DT = 0.02


def load_el_centro_record(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the El Centro NS acceleration record from a whitespace-delimited text file."""
    data_path = Path(path)
    raw_text = data_path.read_text(encoding="utf-8")
    values = np.fromiter((float(token) for token in raw_text.split()), dtype=float)
    time = np.arange(values.size, dtype=float) * EL_CENTRO_DT
    return time, values


def extend_ground_motion(
    time: np.ndarray,
    accel_g: np.ndarray,
    total_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad the record with zeros so the response can continue after shaking stops."""
    if total_duration <= time[-1]:
        return time.copy(), accel_g.copy()

    dt = float(time[1] - time[0])
    n_total = int(round(total_duration / dt)) + 1
    extended_accel = np.zeros(n_total, dtype=float)
    extended_accel[: accel_g.size] = accel_g
    extended_time = np.arange(n_total, dtype=float) * dt
    return extended_time, extended_accel
