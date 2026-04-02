# Bending-Shear Coupled MDOF Building Under Earthquake Excitation

This repository contains a reduced-order structural dynamics workflow for high-rise buildings subjected to earthquake loading. The implementation uses a bending-shear coupled multi-degree-of-freedom (MDOF) model, solves the base-excited response in the time domain, and generates response plots and animations.

The project currently supports two main workflows:

- a default `30`-story benchmark driven by an El Centro acceleration record;
- a paper-based validation case derived from an open-access MDPI study of a `25`-story building under El Centro excitation.

## Features

- Bending-shear coupled reduced MDOF building model
- Geometry-based estimation of target modal periods using a literature reference building
- Time-history response analysis under ground acceleration input
- Modal output including periods and effective modal mass ratios
- Response visualization:
  - animated lateral deformation GIF
  - floor displacement envelope
  - mode-shape plots
  - paper-vs-model benchmark comparison plots
- PDF-based extraction workflow for open-access benchmark papers

## Repository Layout

```text
airport_runway/
|-- src/                  # Model, solver, benchmark, and path utilities
|-- data/                 # Input earthquake records and downloaded papers
|   `-- papers/           # Benchmark paper files
|-- results/              # Generated plots, reports, and extracted figures
|-- main.py               # Default 30-story benchmark entry point
`-- requirements.txt      # Minimal runtime requirements
```

Key source files:

- `src/building_model.py`  
  Reduced bending-shear coupled building model and stiffness calibration logic

- `src/earthquake_data.py`  
  Earthquake record loading and zero-padding utilities

- `src/solver.py`  
  Newmark-beta time-history solver for base excitation

- `src/visualization.py`  
  GIF generation and structural response plotting

- `src/mdpi_benchmark.py`  
  Paper-based validation case for the extracted MDPI benchmark

- `src/paths.py`  
  Shared canonical paths for `src/`, `data/`, and `results/`

## Model Formulation

The structural model is a cantilever-type bending-shear coupled MDOF system. Each story contributes translational and rotational inertia, and the stiffness is assembled from Timoshenko-type beam elements.

The governing equation is:

```math
M \ddot{u} + C \dot{u} + K u = - M r a_g(t)
```

where:

- `M` is the mass matrix
- `C` is the damping matrix
- `K` is the lateral stiffness matrix
- `u` is the floor displacement vector
- `r` is the influence vector
- `a_g(t)` is the ground acceleration input

Time integration uses the average-acceleration Newmark-beta method.

## Parameter Estimation Strategy

This repository does not model the building as a solid cantilever with a full-section reinforced-concrete `EI`. Instead, it estimates target periods from coarse exterior building parameters and then calibrates equivalent:

- flexural rigidity `EI`
- shear rigidity `kGA`

The current reference building is based on:

- Suwansaya, P.; Warnitchai, P. (2023), `Buildings`, 13(3), 670

Reference building values used by the estimator:

- stories: `30`
- story height: `3.5 m`
- height: `105 m`
- plan width: `64 m`
- plan depth: `16 m`
- reported modal periods:
  - `T1 = 4.420 s`
  - `T2 = 1.088 s`

For the default case, the repository uses the same geometry as the reference building, so the estimated target periods reduce to the published values. For other envelopes, the target periods are scaled from the same reference before stiffness calibration.

## Default Benchmark

The default benchmark in `main.py` uses:

- stories: `30`
- story height: `3.5 m`
- plan: `64 m x 16 m`
- ground motion: El Centro NS record from `data/el_centro_ns_1940.txt`

Outputs are written to `results/`:

- `results/mdof_el_centro_response.gif`
- `results/max_floor_displacement_envelope.png`
- `results/first_five_mode_shapes.png`
- `results/el_centro_input_extended.png`
- `results/results_summary.md`

## MDPI Validation Case

The repository also includes a validation workflow based on:

- `Dynamic Soil Structure Interaction of a High-Rise Building Resting over a Finned Pile Mat`
- `Infrastructures` 2022, 7(10), 142
- DOI: `10.3390/infrastructures7100142`

Extracted benchmark properties:

- stories: `25`
- story height: `3 m`
- total height: `75 m`
- plan: `13 m x 13 m`
- El Centro input from the paper:
  - `Mw = 6.9`
  - duration `53.74 s`
  - `PGA = 0.349 g`
- paper modal target:
  - `f1 = 0.547 Hz`
  - `T1 = 1.827 s`
- paper response targets:
  - top-floor peak acceleration `0.25 g`
  - maximum story drift `0.0084 m`
  - maximum drift ratio `0.28%`

Generated validation outputs include:

- `results/mdpi_benchmark_results.md`
- `results/mdpi_benchmark_metrics.png`
- `results/mdpi_benchmark_errors.png`
- `results/mdpi_benchmark_envelopes.png`
- `results/mdpi_story_drift_comparison.png`
- `results/mdpi_extracted_validation.md`

## Environment

This project has been verified in the following conda environment:

- interpreter: `D:\software\anaconda\envs\nudtzk\python.exe`
- Python: `3.11.14`

Verified key packages in that environment:

- `numpy 1.26.4`
- `matplotlib 3.10.8`
- `pillow 10.4.0`
- `PyMuPDF 1.26.5`

## Installation

If you want to use the same conda environment directly:

```powershell
D:\software\anaconda\envs\nudtzk\python.exe -m pip install -r requirements.txt
```

If you prefer your own environment, install at least:

- `numpy`
- `matplotlib`
- `Pillow`
- `PyMuPDF`

## Usage

Run the default benchmark:

```powershell
D:\software\anaconda\envs\nudtzk\python.exe main.py
```

Run the MDPI validation case:

```powershell
D:\software\anaconda\envs\nudtzk\python.exe src\mdpi_benchmark.py
```

## Data Sources

- Default El Centro record:
  - `data/el_centro_ns_1940.txt`

- MDPI benchmark paper:
  - `data/papers/mdpi_finned_pile_mat_2022.pdf`

- Additional screened paper page:
  - `data/papers/dual_frame_shear_wall_sciencedirect.html`

## Current Validation Status

The current reduced model performs well on first-mode modal matching for the MDPI benchmark:

- paper `T1 = 1.827 s`
- model `T1 ~= 1.859 s`

However, the dynamic response does not yet match the published SSI response closely:

- top-floor peak acceleration is over-predicted
- inter-story drift is over-predicted
- story-wise drift shape differs from the paper's `RP-Mat` curve

This is expected because the paper response corresponds to an SSI system with piled-mat and finned-pile-mat foundation behavior, while the current repository models the superstructure only as a reduced bending-shear cantilever.

## Limitations

- The reduced model is currently a superstructure model; it does not explicitly model soil-structure interaction.
- Paper-response comparisons based on extracted figures are approximate when the original paper does not provide raw numeric tables.
- The digitized El Centro record used for the MDPI benchmark is reconstructed from the published figure and then rescaled to the reported PGA.

## License

This repository is released under the MIT License. See the `LICENSE` file for details.
