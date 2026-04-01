# 30-Story MDOF Building Under El Centro Excitation

This project simulates the lateral dynamic response of a 30-story building under the El Centro NS earthquake record using a bending-shear coupled MDOF model.

## What the code does

- Reads the El Centro acceleration record from `el_centro_ns_1940.txt`
- Builds a 30-floor lumped-mass model with:
  - `3.5 m` story height
  - `64 m x 16 m` tower plan dimensions
  - coupled shear and bending stiffness contributions
  - equivalent stiffness calibrated from literature-based first and second modal periods instead of treating the building as a solid concrete cantilever
- Solves the linear equation of motion under base acceleration
- Extends the response to `2 x` the earthquake record duration so the free-vibration decay remains visible
- Exports:
  - `mdof_el_centro_response.gif`
  - `max_floor_displacement_envelope.png`
  - `el_centro_input_extended.png`
  - `first_five_mode_shapes.png`
  - `results_summary.md`

## Model notes

The coupling is represented as:

`K_total = K_shear + K_bending`

where:

- `K_shear` comes from inter-story shear springs
- `K_bending` comes from a discrete curvature-energy bending model

The dynamic equilibrium equation is:

`M u_ddot + C u_dot + K u = -M r a_g(t)`

with:

- `u` as the floor lateral displacement vector
- `r` as the influence vector of ones
- `a_g(t)` as the ground acceleration input

Time integration uses the average-acceleration Newmark-beta method.

## Literature-based calibration

The current version does not estimate stiffness from a full solid rectangular concrete section. Instead, it follows the coupled shear-flexural cantilever concept summarized by Suwansaya and Warnitchai (2023), who note that if the first and second modal periods are known, the effective `EI` and `GA` of the coupled model can be estimated from those periods and their ratio.

For the regular 30-story RC shear-wall building `S1` in the paper (`x` direction), the reported properties include:

- Tower width `B = 64.0 m`
- Tower depth `D = 16.0 m`
- Height `H = 105.0 m`
- Number of stories `= 30`
- Typical story height `h = 3.5 m`

The reported modal periods are:

- `T1 = 4.420 s`
- `T2 = 1.088 s`
- `T1/T2 = 4.063`

This project now uses those `S1` dimensions directly as the default building geometry. The code calibrates the equivalent shear stiffness and flexural rigidity so that the discrete 30-DOF model matches the paper's first and second modal periods.

Reference:

- [Suwansaya & Warnitchai (2023), Buildings 13(3), 670](https://www.mdpi.com/2075-5309/13/3/670)

## Files

- `main.py`: entry point
- `earthquake_data.py`: earthquake record loading and duration extension
- `building_model.py`: 30-story bending-shear coupled model assembly
- `solver.py`: MDOF response solver
- `visualization.py`: GIF and envelope plotting
- `requirements.txt`: minimal runtime dependencies

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Output meaning

- The GIF shows the building deformation together with the El Centro time history marker.
- The envelope plot shows the maximum absolute lateral displacement reached by each floor.
- The mode-shape plot shows the first five normalized lateral mode shapes and their effective modal mass percentages.
- The simulation length is twice the original record duration because the second half is zero-padded input, which lets the structural free vibration continue after the earthquake stops.

## Data source

The El Centro acceleration file in this project was downloaded from this public text record:

- [El Centro Ground Motion gist](https://gist.github.com/terjehaukaas/60ed4b634d22b14a1bf6a86461d39caf)
