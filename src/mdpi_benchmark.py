from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.building_model import BuildingConfig, build_bending_shear_model
from src.paths import DATA_DIR, RESULTS_DIR, ensure_project_dirs
from src.solver import solve_linear_response


PAPER_METRICS = {
    "T1_s": 1.827,
    "f1_hz": 0.547,
    "meff_first3": 0.7195,
    "meff_all": 0.9187,
    "top_accel_g": 0.25,
    "top_accel_ratio_pct": 71.6,
    "max_story_drift_m": 0.0084,
    "drift_ratio_pct": 0.28,
    "top_base_disp_ratio": 20.66,
}

# Digitized from Figure 14(a), RP-Mat curve, using the extracted page image.
_PAPER_DRIFT_X_SAMPLES = [
    162.0,
    162.0,
    211.0,
    260.0,
    265.5,
    325.0,
    360.0,
    379.0,
    431.0,
    436.0,
    472.0,
    478.0,
    495.0,
    512.0,
    519.0,
    525.5,
    530.0,
    531.5,
    530.0,
    529.0,
    526.0,
    521.0,
    516.0,
    511.0,
    507.0,
]


def load_digitized_motion(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(Path(path), delimiter=",", skiprows=1)
    time_s = data[:, 0]
    accel_g = data[:, 1]
    accel_g = accel_g - np.mean(accel_g)
    peak = np.max(np.abs(accel_g))
    if peak > 0.0:
        accel_g = accel_g * (0.349 / peak)
    return time_s, accel_g


def build_mdpi_case() -> tuple[BuildingConfig, object]:
    config = BuildingConfig(
        n_floors=25,
        story_height_m=3.0,
        plan_width_m=13.0,
        plan_depth_m=13.0,
        gross_density_kg_m3=300.0,
        damping_ratio=0.05,
    )
    model = build_bending_shear_model(config)
    return config, model


def compute_metrics(model, response, input_pga_g: float) -> dict[str, float]:
    periods = 2.0 * np.pi / model.natural_frequencies_rad_s
    floor_abs_accel_g = np.abs(response.absolute_accel_m_s2) / 9.81
    top_accel_g = float(np.max(floor_abs_accel_g[:, -1]))
    floor_disp = np.abs(response.displacements_m)
    top_base_disp_ratio = float(np.max(floor_disp[:, -1]) / max(np.max(floor_disp[:, 0]), 1.0e-12))

    story_h = model.config.story_height_m
    drift = np.empty_like(response.displacements_m)
    drift[:, 0] = response.displacements_m[:, 0]
    drift[:, 1:] = response.displacements_m[:, 1:] - response.displacements_m[:, :-1]
    max_story_drift_m = float(np.max(np.abs(drift)))
    drift_ratio_pct = max_story_drift_m / story_h * 100.0

    return {
        "T1_s": float(periods[0]),
        "f1_hz": float(model.natural_frequencies_rad_s[0] / (2.0 * np.pi)),
        "meff_first3": float(np.sum(model.effective_modal_mass_ratios[:3])),
        "meff_all": float(np.sum(model.effective_modal_mass_ratios)),
        "top_accel_g": top_accel_g,
        "top_accel_ratio_pct": top_accel_g / input_pga_g * 100.0,
        "max_story_drift_m": max_story_drift_m,
        "drift_ratio_pct": drift_ratio_pct,
        "top_base_disp_ratio": top_base_disp_ratio,
    }


def compute_story_drift_profile_pct(model, response) -> np.ndarray:
    drift = np.empty_like(response.displacements_m)
    drift[:, 0] = response.displacements_m[:, 0]
    drift[:, 1:] = response.displacements_m[:, 1:] - response.displacements_m[:, :-1]
    return np.max(np.abs(drift), axis=0) / model.config.story_height_m * 100.0


def get_digitized_paper_story_drift_pct() -> np.ndarray:
    x0 = 162.0
    x1 = 531.5
    d0 = 0.162
    d1 = 0.277
    scale = (d1 - d0) / (x1 - x0)
    return np.array([d0 + (x - x0) * scale for x in _PAPER_DRIFT_X_SAMPLES], dtype=float)


def save_metric_plot(metrics: dict[str, float], output_path: str | Path) -> Path:
    out_path = Path(output_path)
    labels = [
        "T1 (s)",
        "f1 (Hz)",
        "Meff 1-3",
        "Meff all",
        "Top accel (g)",
        "Top accel / PGA (%)",
        "Max story drift (m)",
        "Drift ratio (%)",
        "Top/base disp ratio",
    ]
    paper = np.array(
        [
            PAPER_METRICS["T1_s"],
            PAPER_METRICS["f1_hz"],
            PAPER_METRICS["meff_first3"],
            PAPER_METRICS["meff_all"],
            PAPER_METRICS["top_accel_g"],
            PAPER_METRICS["top_accel_ratio_pct"],
            PAPER_METRICS["max_story_drift_m"],
            PAPER_METRICS["drift_ratio_pct"],
            PAPER_METRICS["top_base_disp_ratio"],
        ]
    )
    model = np.array(
        [
            metrics["T1_s"],
            metrics["f1_hz"],
            metrics["meff_first3"],
            metrics["meff_all"],
            metrics["top_accel_g"],
            metrics["top_accel_ratio_pct"],
            metrics["max_story_drift_m"],
            metrics["drift_ratio_pct"],
            metrics["top_base_disp_ratio"],
        ]
    )

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - width / 2, paper, width=width, color="#9fb3c8", label="Paper")
    ax.bar(x + width / 2, model, width=width, color="#24577a", label="Reduced model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("MDPI benchmark: paper values vs reduced-model values")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_response_envelopes(model, response, output_path: str | Path) -> Path:
    out_path = Path(output_path)
    stories = np.arange(1, model.config.n_floors + 1)
    peak_accel_g = np.max(np.abs(response.absolute_accel_m_s2), axis=0) / 9.81
    peak_disp_mm = np.max(np.abs(response.displacements_m), axis=0) * 1000.0
    peak_drift_pct = compute_story_drift_profile_pct(model, response)

    fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
    axes[0].plot(peak_accel_g, stories, color="#7a1f1f", linewidth=2.2)
    axes[0].set_xlabel("Peak accel (g)")
    axes[0].set_ylabel("Story")
    axes[0].set_title("Model peak acceleration")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(peak_disp_mm, stories, color="#0b6e4f", linewidth=2.2)
    axes[1].set_xlabel("Peak displacement (mm)")
    axes[1].set_title("Model peak displacement")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].plot(peak_drift_pct, stories, color="#1f4e79", linewidth=2.2)
    axes[2].set_xlabel("Inter-story drift (%)")
    axes[2].set_title("Model peak drift ratio")
    axes[2].grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylim(1, model.config.n_floors)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_story_drift_comparison(model, response, output_path: str | Path) -> Path:
    out_path = Path(output_path)
    stories = np.arange(1, model.config.n_floors + 1)
    model_drift_pct = compute_story_drift_profile_pct(model, response)
    paper_drift_pct = get_digitized_paper_story_drift_pct()

    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.plot(paper_drift_pct, stories, color="#a33b20", linewidth=2.4, marker="o", markersize=3.8, label="Paper Figure 14(a) digitized")
    ax.plot(model_drift_pct, stories, color="#24577a", linewidth=2.4, marker="s", markersize=3.2, label="Reduced model")
    ax.set_xlabel("Inter-story drift (%)")
    ax.set_ylabel("Story")
    ax.set_title("Story-wise inter-story drift: paper vs reduced model")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(1, model.config.n_floors)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_error_plot(metrics: dict[str, float], output_path: str | Path) -> Path:
    out_path = Path(output_path)
    keys = [
        "T1_s",
        "f1_hz",
        "meff_first3",
        "meff_all",
        "top_accel_g",
        "top_accel_ratio_pct",
        "max_story_drift_m",
        "drift_ratio_pct",
        "top_base_disp_ratio",
    ]
    labels = [
        "T1",
        "f1",
        "Meff 1-3",
        "Meff all",
        "Top accel",
        "Top accel/PGA",
        "Max drift",
        "Drift ratio",
        "Top/base disp",
    ]
    errors = [100.0 * (metrics[k] - PAPER_METRICS[k]) / PAPER_METRICS[k] for k in keys]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#24577a" if err >= 0.0 else "#a33b20" for err in errors]
    ax.barh(y, errors, color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Error relative to paper (%)")
    ax.set_title("MDPI benchmark metric errors")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_report(metrics: dict[str, float], output_path: str | Path) -> Path:
    out_path = Path(output_path)
    lines = [
        "# MDPI Benchmark Comparison",
        "",
        "- Paper: Dynamic Soil Structure Interaction of a High-Rise Building Resting over a Finned Pile Mat",
        "- Earthquake input: digitized from paper Figure 2 and rescaled to PGA = 0.349 g",
        "- Building used in reduced model: 25 stories, 3 m/story, 13 m x 13 m plan",
        "",
        "## Metrics",
        "",
    ]
    mapping = [
        ("T1_s", "T1 (s)"),
        ("f1_hz", "f1 (Hz)"),
        ("meff_first3", "Effective mass ratio of first three modes"),
        ("meff_all", "Cumulative effective mass ratio"),
        ("top_accel_g", "Top-floor peak acceleration (g)"),
        ("top_accel_ratio_pct", "Top-floor acceleration / PGA (%)"),
        ("max_story_drift_m", "Maximum story drift (m)"),
        ("drift_ratio_pct", "Maximum drift ratio (%)"),
        ("top_base_disp_ratio", "Top/base displacement ratio"),
    ]
    for key, label in mapping:
        paper_value = PAPER_METRICS[key]
        model_value = metrics[key]
        error_pct = (model_value - paper_value) / paper_value * 100.0 if paper_value != 0 else float("nan")
        lines.append(f"- {label}: paper = {paper_value:.6g}, model = {model_value:.6g}, error = {error_pct:+.2f}%")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    ensure_project_dirs()
    motion_path = DATA_DIR / "mdpi_elcentro_digitized.csv"
    time_s, accel_g = load_digitized_motion(motion_path)
    _, model = build_mdpi_case()
    response = solve_linear_response(model, time_s, accel_g)
    metrics = compute_metrics(model, response, input_pga_g=float(np.max(np.abs(accel_g))))

    report_path = write_report(metrics, RESULTS_DIR / "mdpi_benchmark_results.md")
    metrics_plot = save_metric_plot(metrics, RESULTS_DIR / "mdpi_benchmark_metrics.png")
    envelope_plot = save_response_envelopes(model, response, RESULTS_DIR / "mdpi_benchmark_envelopes.png")
    error_plot = save_error_plot(metrics, RESULTS_DIR / "mdpi_benchmark_errors.png")
    drift_plot = save_story_drift_comparison(model, response, RESULTS_DIR / "mdpi_story_drift_comparison.png")

    print("MDPI benchmark complete.")
    print(f"Report: {report_path}")
    print(f"Metric plot: {metrics_plot}")
    print(f"Envelope plot: {envelope_plot}")
    print(f"Error plot: {error_plot}")
    print(f"Story-drift comparison plot: {drift_plot}")


if __name__ == "__main__":
    main()
