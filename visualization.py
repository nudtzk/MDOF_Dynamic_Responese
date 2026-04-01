from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from building_model import BuildingModel
from solver import ResponseHistory


def save_displacement_envelope(
    model: BuildingModel,
    response: ResponseHistory,
    output_path: str | Path,
) -> Path:
    out_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 9))
    floor_ids = np.arange(1, model.config.n_floors + 1, dtype=int)
    ax.plot(response.max_abs_displacement_m * 1000.0, floor_ids, color="#0b6e4f", linewidth=2.5)
    ax.fill_betweenx(
        floor_ids,
        0.0,
        response.max_abs_displacement_m * 1000.0,
        color="#90d1c2",
        alpha=0.45,
    )
    ax.set_xlabel("Maximum absolute lateral displacement (mm)")
    ax.set_ylabel("Floor")
    ax.set_title("30-story bending-shear coupled building\npeak floor displacement envelope")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylim(1, model.config.n_floors)
    ax.set_yticks(np.arange(1, model.config.n_floors + 1, 2))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_response_gif(
    model: BuildingModel,
    response: ResponseHistory,
    output_path: str | Path,
    frame_stride: int = 3,
) -> Path:
    out_path = Path(output_path)
    heights = np.concatenate(([0.0], model.floor_heights_m))
    displacements = np.hstack([np.zeros((response.displacements_m.shape[0], 1)), response.displacements_m])

    max_disp = max(float(np.max(np.abs(response.displacements_m))), 1e-6)
    x_limit = max_disp * 1.35
    selected = np.arange(0, response.time_s.size, frame_stride, dtype=int)
    if selected[-1] != response.time_s.size - 1:
        selected = np.append(selected, response.time_s.size - 1)

    fig, (ax_building, ax_ground) = plt.subplots(
        1,
        2,
        figsize=(11, 7),
        gridspec_kw={"width_ratios": [1.1, 1.4]},
    )

    building_line, = ax_building.plot([], [], color="#174a7e", linewidth=3)
    floor_dots = ax_building.scatter([], [], s=22, color="#d1495b", zorder=3)
    time_marker = ax_ground.axvline(response.time_s[0], color="#d1495b", linewidth=2)

    ax_building.set_xlim(-x_limit, x_limit)
    ax_building.set_ylim(0.0, model.config.total_height_m * 1.03)
    ax_building.set_xlabel("Lateral displacement (m)")
    ax_building.set_ylabel("Height (m)")
    ax_building.set_title("Building lateral response")
    ax_building.grid(True, linestyle="--", alpha=0.3)
    ax_building.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)

    ax_ground.plot(response.time_s, response.ground_accel_g, color="#2f4858", linewidth=1.3)
    ax_ground.set_xlabel("Time (s)")
    ax_ground.set_ylabel("Ground acceleration (g)")
    ax_ground.set_title("El Centro input with extended free vibration")
    ax_ground.grid(True, linestyle="--", alpha=0.3)

    caption = ax_building.text(
        0.03,
        0.97,
        "",
        transform=ax_building.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#888"},
    )

    def update(frame_number: int):
        step = int(selected[frame_number])
        shape = displacements[step]
        building_line.set_data(shape, heights)
        floor_dots.set_offsets(np.column_stack([shape[1:], model.floor_heights_m]))
        time_marker.set_xdata([response.time_s[step], response.time_s[step]])
        caption.set_text(
            "\n".join(
                [
                    f"t = {response.time_s[step]:6.2f} s",
                    f"roof = {response.displacements_m[step, -1] * 1000.0:7.2f} mm",
                    f"ag = {response.ground_accel_g[step]:6.3f} g",
                ]
            )
        )
        return building_line, floor_dots, time_marker, caption

    animation = FuncAnimation(
        fig,
        update,
        frames=selected.size,
        interval=40,
        blit=False,
    )
    fps = max(1, round(1.0 / (frame_stride * (response.time_s[1] - response.time_s[0]))))
    animation.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def save_mode_shapes_plot(
    model: BuildingModel,
    output_path: str | Path,
    n_modes: int = 5,
) -> Path:
    out_path = Path(output_path)
    heights = np.concatenate(([0.0], model.floor_heights_m))
    mode_count = min(n_modes, model.mode_shapes.shape[1])

    fig, axes = plt.subplots(1, mode_count, figsize=(3.2 * mode_count, 8), sharey=True)
    if mode_count == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        mode = model.mode_shapes[:, idx]
        mode_scaled = mode / max(np.max(np.abs(mode)), 1e-12)
        shape = np.concatenate(([0.0], mode_scaled))
        ax.plot(shape, heights, color="#174a7e", linewidth=2.5)
        ax.scatter(shape[1:], model.floor_heights_m, color="#d1495b", s=18, zorder=3)
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(-1.15, 1.15)
        ax.set_xlabel("Normalized mode shape")
        ax.set_title(
            "\n".join(
                [
                    f"Mode {idx + 1}",
                    f"T = {2.0 * np.pi / model.natural_frequencies_rad_s[idx]:.3f} s",
                    f"Meff = {model.effective_modal_mass_ratios[idx] * 100.0:.1f}%",
                ]
            )
        )

    axes[0].set_ylabel("Height (m)")
    axes[0].set_ylim(0.0, model.config.total_height_m * 1.03)
    fig.suptitle("First five lateral mode shapes", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
