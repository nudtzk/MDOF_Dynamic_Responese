from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.building_model import BuildingConfig, build_bending_shear_model
from src.earthquake_data import EL_CENTRO_DT, extend_ground_motion, load_el_centro_record
from src.paths import DATA_DIR, RESULTS_DIR, ensure_project_dirs
from src.solver import solve_linear_response
from src.visualization import save_displacement_envelope, save_mode_shapes_plot, save_response_gif


def main() -> None:
    ensure_project_dirs()
    record_path = DATA_DIR / "el_centro_ns_1940.txt"

    record_time_s, ground_accel_g = load_el_centro_record(record_path)
    record_duration_s = record_time_s[-1]
    total_duration_s = 2.0 * record_duration_s
    time_s, extended_ground_accel_g = extend_ground_motion(
        record_time_s,
        ground_accel_g,
        total_duration=total_duration_s,
    )

    config = BuildingConfig()
    model = build_bending_shear_model(config)
    response = solve_linear_response(model, time_s, extended_ground_accel_g)

    gif_path = save_response_gif(model, response, RESULTS_DIR / "mdof_el_centro_response.gif")
    envelope_path = save_displacement_envelope(
        model,
        response,
        RESULTS_DIR / "max_floor_displacement_envelope.png",
    )
    mode_shapes_path = save_mode_shapes_plot(model, RESULTS_DIR / "first_five_mode_shapes.png")
    _save_ground_motion_plot(time_s, extended_ground_accel_g, RESULTS_DIR / "el_centro_input_extended.png")
    _write_summary(model, response, RESULTS_DIR / "results_summary.md")

    print("Run complete.")
    print(f"Input dt: {EL_CENTRO_DT:.3f} s")
    print(f"Original record duration: {record_duration_s:.2f} s")
    print(f"Simulated duration: {time_s[-1]:.2f} s")
    print("\nFirst five modes:")
    for mode_idx in range(min(5, model.natural_frequencies_rad_s.size)):
        period_s = 2.0 * np.pi / model.natural_frequencies_rad_s[mode_idx]
        gamma = model.modal_participation_factors[mode_idx]
        mass_ratio = model.effective_modal_mass_ratios[mode_idx]
        print(
            "  "
            f"Mode {mode_idx + 1}: "
            f"T = {period_s:.3f} s, "
            f"Gamma = {gamma:.4f}, "
            f"Effective mass ratio = {mass_ratio * 100.0:.2f}%"
        )
    print(f"GIF: {gif_path}")
    print(f"Envelope image: {envelope_path}")
    print(f"Mode shapes image: {mode_shapes_path}")
    print(f"Peak roof displacement: {response.peak_roof_displacement_m * 1000.0:.2f} mm")


def _save_ground_motion_plot(time_s: np.ndarray, accel_g: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_s, accel_g, color="#313638", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ground acceleration (g)")
    ax.set_title("El Centro NS ground motion with zero-padded free-vibration tail")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_summary(model, response, output_path: Path) -> None:
    first_three_hz = model.natural_frequencies_rad_s[:3] / (2.0 * np.pi)
    text = "\n".join(
        [
            "# MDOF Response Summary",
            "",
            f"- Floors: {model.config.n_floors}",
            f"- Story height: {model.config.story_height_m:.2f} m",
            f"- Total height: {model.config.total_height_m:.2f} m",
            f"- Plan dimensions: {model.config.plan_width_m:.2f} m x {model.config.plan_depth_m:.2f} m",
            f"- Gross density used for lumped masses: {model.config.gross_density_kg_m3:.1f} kg/m^3",
            f"- Floor mass: {model.config.floor_mass_kg:,.0f} kg",
            (
                f"- Target periods from literature-calibrated reference: "
                f"T1={model.target_periods_s[0]:.3f} s, T2={model.target_periods_s[1]:.3f} s"
            ),
            (
                f"- Achieved discrete-model periods: "
                f"T1={model.achieved_periods_s[0]:.3f} s, T2={model.achieved_periods_s[1]:.3f} s"
            ),
            (
                f"- Calibrated equivalent shear rigidity kGA: "
                f"{model.equivalent_shear_rigidity_n:.3e} N"
            ),
            (
                f"- Calibrated equivalent flexural rigidity: "
                f"{model.equivalent_flexural_rigidity_n_m2:.3e} N*m^2"
            ),
            f"- Peak roof displacement: {response.peak_roof_displacement_m * 1000.0:.2f} mm",
            (
                "- First three natural frequencies (Hz): "
                + ", ".join(f"{value:.3f}" for value in first_three_hz)
            ),
            "- First five modal participation data:",
            *[
                (
                    f"  - Mode {idx + 1}: "
                    f"T={2.0 * np.pi / model.natural_frequencies_rad_s[idx]:.3f} s, "
                    f"Gamma={model.modal_participation_factors[idx]:.4f}, "
                    f"Meff={model.effective_modal_mass_ratios[idx] * 100.0:.2f}%"
                )
                for idx in range(min(5, model.natural_frequencies_rad_s.size))
            ],
            f"- Literature reference: {model.literature_reference.source_label}",
        ]
    )
    output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
