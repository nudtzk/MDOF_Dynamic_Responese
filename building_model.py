from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiteratureReference:
    source_label: str
    reference_height_m: float
    first_mode_period_s: float
    second_mode_period_s: float


REFERENCE_RC_SHEAR_WALL = LiteratureReference(
    source_label="Suwansaya & Warnitchai (2023), Building S1, x direction",
    reference_height_m=105.0,
    first_mode_period_s=4.420,
    second_mode_period_s=1.088,
)


@dataclass(frozen=True)
class BuildingConfig:
    n_floors: int = 30
    story_height_m: float = 3.5
    plan_width_m: float = 64.0
    plan_depth_m: float = 16.0
    gross_density_kg_m3: float = 300.0
    damping_ratio: float = 0.025
    target_first_mode_period_s: float | None = None
    target_second_mode_period_s: float | None = None
    shear_correction_factor: float = 1.2
    rotational_inertia_factor: float = 1.0 / 12.0

    @property
    def total_height_m(self) -> float:
        return self.n_floors * self.story_height_m

    @property
    def footprint_area_m2(self) -> float:
        return self.plan_width_m * self.plan_depth_m

    @property
    def story_volume_m3(self) -> float:
        return self.footprint_area_m2 * self.story_height_m

    @property
    def floor_mass_kg(self) -> float:
        return self.gross_density_kg_m3 * self.story_volume_m3


@dataclass(frozen=True)
class BuildingModel:
    config: BuildingConfig
    floor_heights_m: np.ndarray
    masses_kg: np.ndarray
    mass_matrix: np.ndarray
    stiffness_matrix: np.ndarray
    damping_matrix: np.ndarray
    influence_vector: np.ndarray
    response_dof_indices: np.ndarray
    mode_shapes: np.ndarray
    modal_vectors: np.ndarray
    natural_frequencies_rad_s: np.ndarray
    modal_participation_factors: np.ndarray
    effective_modal_mass_ratios: np.ndarray
    equivalent_flexural_rigidity_n_m2: float
    equivalent_shear_rigidity_n: float
    target_periods_s: tuple[float, float]
    achieved_periods_s: tuple[float, float]
    literature_reference: LiteratureReference


def estimate_target_periods(config: BuildingConfig) -> tuple[float, float]:
    if config.target_first_mode_period_s is not None and config.target_second_mode_period_s is not None:
        return config.target_first_mode_period_s, config.target_second_mode_period_s
    return (
        REFERENCE_RC_SHEAR_WALL.first_mode_period_s,
        REFERENCE_RC_SHEAR_WALL.second_mode_period_s,
    )


def build_bending_shear_model(config: BuildingConfig) -> BuildingModel:
    target_periods = estimate_target_periods(config)
    floor_heights = np.arange(1, config.n_floors + 1, dtype=float) * config.story_height_m
    masses = np.full(config.n_floors, config.floor_mass_kg, dtype=float)
    response_dof_indices = np.arange(0, 2 * config.n_floors, 2, dtype=int)

    mass_matrix = _assemble_mass_matrix(config, masses)
    flexural_rigidity, shear_rigidity = calibrate_bending_shear_stiffness(config, mass_matrix, target_periods)
    stiffness_matrix = _assemble_timoshenko_stiffness(config, flexural_rigidity, shear_rigidity)

    modal_vectors, omegas = _solve_modes(mass_matrix, stiffness_matrix)
    mode_shapes = _extract_floor_mode_shapes(modal_vectors, response_dof_indices)
    modal_participation_factors, effective_modal_mass_ratios = _compute_modal_participation(
        mass_matrix=mass_matrix,
        modal_vectors=modal_vectors,
        influence_vector=_build_influence_vector(config.n_floors),
    )
    damping_matrix = _build_rayleigh_damping(
        mass_matrix=mass_matrix,
        stiffness_matrix=stiffness_matrix,
        omegas=omegas,
        damping_ratio=config.damping_ratio,
    )

    periods = 2.0 * np.pi / omegas[:2]
    return BuildingModel(
        config=config,
        floor_heights_m=floor_heights,
        masses_kg=masses,
        mass_matrix=mass_matrix,
        stiffness_matrix=stiffness_matrix,
        damping_matrix=damping_matrix,
        influence_vector=_build_influence_vector(config.n_floors),
        response_dof_indices=response_dof_indices,
        mode_shapes=mode_shapes,
        modal_vectors=modal_vectors,
        natural_frequencies_rad_s=omegas,
        modal_participation_factors=modal_participation_factors,
        effective_modal_mass_ratios=effective_modal_mass_ratios,
        equivalent_flexural_rigidity_n_m2=flexural_rigidity,
        equivalent_shear_rigidity_n=shear_rigidity,
        target_periods_s=target_periods,
        achieved_periods_s=(float(periods[0]), float(periods[1])),
        literature_reference=REFERENCE_RC_SHEAR_WALL,
    )


def calibrate_bending_shear_stiffness(
    config: BuildingConfig,
    mass_matrix: np.ndarray,
    target_periods_s: tuple[float, float],
) -> tuple[float, float]:
    target_ratio = target_periods_s[0] / target_periods_s[1]
    response_dof_indices = np.arange(0, 2 * config.n_floors, 2, dtype=int)
    best_error = float("inf")
    best_eta = None
    best_first_period = None
    reference_ei = 1.0e12

    for eta in np.logspace(-2, 4, 800):
        stiffness_unit = _assemble_timoshenko_stiffness(
            config=config,
            flexural_rigidity_n_m2=reference_ei,
            shear_rigidity_n=eta * reference_ei / config.total_height_m**2,
        )
        modal_vectors, omegas = _solve_modes(mass_matrix, stiffness_unit)
        if omegas[0] <= 1.0e-12 or omegas[1] <= 1.0e-12:
            continue
        mode_shapes = _extract_floor_mode_shapes(modal_vectors, response_dof_indices)
        if not _looks_like_cantilever_first_mode(mode_shapes[:, 0]):
            continue
        periods = 2.0 * np.pi / omegas[:2]
        error = abs(np.log((periods[0] / periods[1]) / target_ratio))
        if error < best_error:
            best_error = error
            best_eta = eta
            best_first_period = float(periods[0])

    if best_eta is None or best_first_period is None:
        raise RuntimeError("Failed to calibrate a cantilever-consistent bending-shear model.")

    stiffness_scale = (best_first_period / target_periods_s[0]) ** 2
    flexural_rigidity = stiffness_scale * reference_ei
    shear_rigidity = best_eta * flexural_rigidity / config.total_height_m**2
    return float(flexural_rigidity), float(shear_rigidity)


def _assemble_mass_matrix(config: BuildingConfig, masses_kg: np.ndarray) -> np.ndarray:
    n_dof = 2 * config.n_floors
    mass_matrix = np.zeros((n_dof, n_dof), dtype=float)
    rotary_inertia = masses_kg * config.story_height_m**2 * config.rotational_inertia_factor

    for floor in range(config.n_floors):
        trans_dof = 2 * floor
        rot_dof = trans_dof + 1
        mass_matrix[trans_dof, trans_dof] = masses_kg[floor]
        mass_matrix[rot_dof, rot_dof] = rotary_inertia[floor]

    return mass_matrix


def _assemble_timoshenko_stiffness(
    config: BuildingConfig,
    flexural_rigidity_n_m2: float,
    shear_rigidity_n: float,
) -> np.ndarray:
    n_active_dof = 2 * config.n_floors
    global_stiffness = np.zeros((n_active_dof, n_active_dof), dtype=float)
    story_length = config.story_height_m

    for element in range(config.n_floors):
        k_local = _timoshenko_element_stiffness(
            length_m=story_length,
            flexural_rigidity_n_m2=flexural_rigidity_n_m2,
            shear_rigidity_n=shear_rigidity_n,
            shear_correction_factor=config.shear_correction_factor,
        )

        node_i = element
        node_j = element + 1
        dof_map_full = np.array(
            [2 * node_i, 2 * node_i + 1, 2 * node_j, 2 * node_j + 1],
            dtype=int,
        )

        for a in range(4):
            ia = dof_map_full[a]
            if ia < 2:
                continue
            for b in range(4):
                ib = dof_map_full[b]
                if ib < 2:
                    continue
                global_stiffness[ia - 2, ib - 2] += k_local[a, b]

    return global_stiffness


def _timoshenko_element_stiffness(
    length_m: float,
    flexural_rigidity_n_m2: float,
    shear_rigidity_n: float,
    shear_correction_factor: float,
) -> np.ndarray:
    effective_shear = shear_correction_factor * shear_rigidity_n
    phi = 12.0 * flexural_rigidity_n_m2 / (effective_shear * length_m**2)
    factor = flexural_rigidity_n_m2 / (length_m**3 * (1.0 + phi))
    l = length_m
    return factor * np.array(
        [
            [12.0, 6.0 * l, -12.0, 6.0 * l],
            [6.0 * l, (4.0 + phi) * l**2, -6.0 * l, (2.0 - phi) * l**2],
            [-12.0, -6.0 * l, 12.0, -6.0 * l],
            [6.0 * l, (2.0 - phi) * l**2, -6.0 * l, (4.0 + phi) * l**2],
        ],
        dtype=float,
    )


def _build_influence_vector(n_floors: int) -> np.ndarray:
    influence = np.zeros(2 * n_floors, dtype=float)
    influence[0::2] = 1.0
    return influence


def _solve_modes(mass_matrix: np.ndarray, stiffness_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(np.linalg.solve(mass_matrix, stiffness_matrix))
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    omegas = np.sqrt(np.clip(eigvals, 0.0, None))
    modal_vectors = _mass_normalize_modes(eigvecs, mass_matrix)
    return modal_vectors, omegas


def _mass_normalize_modes(vectors: np.ndarray, mass_matrix: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(vectors)
    for idx in range(vectors.shape[1]):
        vector = vectors[:, idx]
        modal_mass = float(vector.T @ mass_matrix @ vector)
        normalized[:, idx] = vector / np.sqrt(modal_mass)
    return normalized


def _extract_floor_mode_shapes(modal_vectors: np.ndarray, response_dof_indices: np.ndarray) -> np.ndarray:
    shapes = modal_vectors[response_dof_indices, :].copy()
    for idx in range(shapes.shape[1]):
        roof_value = shapes[-1, idx]
        if abs(roof_value) > 1.0e-14:
            shapes[:, idx] /= abs(roof_value)
            if shapes[-1, idx] < 0.0:
                shapes[:, idx] *= -1.0
    return shapes


def _looks_like_cantilever_first_mode(mode_shape: np.ndarray) -> bool:
    if mode_shape[-1] == 0.0:
        return False
    normalized = mode_shape / mode_shape[-1]
    if normalized[-1] < 0.0:
        normalized *= -1.0
    diffs = np.diff(normalized)
    return bool(np.all(diffs >= -1.0e-6) and normalized[0] >= 0.0 and normalized[-1] > 0.9)


def _compute_modal_participation(
    mass_matrix: np.ndarray,
    modal_vectors: np.ndarray,
    influence_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_mass = float(influence_vector.T @ mass_matrix @ influence_vector)
    participation_factors = np.zeros(modal_vectors.shape[1], dtype=float)
    effective_mass_ratios = np.zeros(modal_vectors.shape[1], dtype=float)

    for idx in range(modal_vectors.shape[1]):
        phi = modal_vectors[:, idx]
        numerator = float(phi.T @ mass_matrix @ influence_vector)
        denominator = float(phi.T @ mass_matrix @ phi)
        gamma = numerator / denominator
        effective_mass = numerator**2 / denominator
        participation_factors[idx] = gamma
        effective_mass_ratios[idx] = effective_mass / total_mass

    return participation_factors, effective_mass_ratios


def _build_rayleigh_damping(
    mass_matrix: np.ndarray,
    stiffness_matrix: np.ndarray,
    omegas: np.ndarray,
    damping_ratio: float,
) -> np.ndarray:
    omega_1 = omegas[0]
    omega_2 = omegas[1]
    system = np.array(
        [
            [1.0 / (2.0 * omega_1), omega_1 / 2.0],
            [1.0 / (2.0 * omega_2), omega_2 / 2.0],
        ],
        dtype=float,
    )
    alpha_m, beta_k = np.linalg.solve(system, np.array([damping_ratio, damping_ratio], dtype=float))
    return alpha_m * mass_matrix + beta_k * stiffness_matrix
