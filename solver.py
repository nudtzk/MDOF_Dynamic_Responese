from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building_model import BuildingModel


@dataclass(frozen=True)
class ResponseHistory:
    time_s: np.ndarray
    ground_accel_g: np.ndarray
    displacements_m: np.ndarray
    velocities_m_s: np.ndarray
    absolute_accel_m_s2: np.ndarray
    max_abs_displacement_m: np.ndarray
    peak_roof_displacement_m: float


def solve_linear_response(
    model: BuildingModel,
    time_s: np.ndarray,
    ground_accel_g: np.ndarray,
    gravity_ms2: float = 9.81,
) -> ResponseHistory:
    mass = model.mass_matrix
    damping = model.damping_matrix
    stiffness = model.stiffness_matrix
    influence = model.influence_vector
    dt = float(time_s[1] - time_s[0])

    n_steps = time_s.size
    n_dof = mass.shape[0]

    u = np.zeros((n_steps, n_dof), dtype=float)
    v = np.zeros_like(u)
    a_rel = np.zeros_like(u)

    beta = 0.25
    gamma = 0.5
    ag = ground_accel_g * gravity_ms2

    effective_force = -mass @ influence
    a_rel[0] = np.linalg.solve(mass, effective_force * ag[0] - damping @ v[0] - stiffness @ u[0])

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    k_eff = stiffness + a0 * mass + a1 * damping

    for step in range(n_steps - 1):
        rhs = (
            effective_force * ag[step + 1]
            + mass @ (a0 * u[step] + a2 * v[step] + a3 * a_rel[step])
            + damping @ (a1 * u[step] + a4 * v[step] + a5 * a_rel[step])
        )
        u[step + 1] = np.linalg.solve(k_eff, rhs)
        a_rel[step + 1] = (
            a0 * (u[step + 1] - u[step]) - a2 * v[step] - a3 * a_rel[step]
        )
        v[step + 1] = v[step] + dt * ((1.0 - gamma) * a_rel[step] + gamma * a_rel[step + 1])

    absolute_accel = a_rel + ag[:, None]
    floor_u = u[:, model.response_dof_indices]
    floor_v = v[:, model.response_dof_indices]
    floor_abs_accel = absolute_accel[:, model.response_dof_indices]
    max_abs_displacement = np.max(np.abs(floor_u), axis=0)
    peak_roof = float(np.max(np.abs(floor_u[:, -1])))

    return ResponseHistory(
        time_s=time_s,
        ground_accel_g=ground_accel_g,
        displacements_m=floor_u,
        velocities_m_s=floor_v,
        absolute_accel_m_s2=floor_abs_accel,
        max_abs_displacement_m=max_abs_displacement,
        peak_roof_displacement_m=peak_roof,
    )
