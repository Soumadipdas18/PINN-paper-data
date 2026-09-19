"""Metrics and case-level tables. No training or test-set access occurs on import."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .config import *


def rmse(a, b):
    return float(np.sqrt(np.mean(np.square(np.asarray(a) - np.asarray(b)))))

def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def r2(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(1.0 - np.sum((a - b) ** 2) / (np.sum((a - a.mean()) ** 2) + 1.0e-12))

def detailed_metrics(pred, config: ModelConfig, norm: Normalization) -> Dict[str, float]:
    t_true = pred["true_temp"]
    t_pred = pred["temp"]
    lf_true = pred["true_logf"].reshape(-1)
    lf_head = pred["logf"].reshape(-1)
    lf_phys = pred["logf_phys"].reshape(-1)
    q_true = pred["true_q"].reshape(-1)
    q_pred = pred["q"].reshape(-1)
    f_true = np.power(10.0, lf_true)
    f_head = np.power(10.0, lf_head)
    f_phys = np.power(10.0, lf_phys)
    jump = t_pred[:, config.n_heat, 0] - t_pred[:, config.n_heat - 1, 0]
    near = np.abs(f_true - F_SAFETY_THRESHOLD_MIN) <= (
        NEAR_THRESHOLD_FRACTION * F_SAFETY_THRESHOLD_MIN
    )
    if not np.any(near):
        near = np.ones_like(f_true, dtype=bool)

    false_safe = (f_head >= F_SAFETY_THRESHOLD_MIN) & (f_true < F_SAFETY_THRESHOLD_MIN)
    false_unsafe = (f_head < F_SAFETY_THRESHOLD_MIN) & (f_true >= F_SAFETY_THRESHOLD_MIN)
    f_abs = np.abs(f_head - f_true)
    signed = f_head - f_true
    near_underprediction = np.maximum(f_true[near] - f_head[near], 0.0)
    score = (
        rmse(t_true, t_pred) / norm.temperature_std
        + rmse(lf_true, lf_head) / norm.logF_std
        + rmse(q_true, q_pred) / norm.retention_std
        + rmse(lf_true, lf_phys) / norm.logF_std
        + CONSISTENCY_SCORE_WEIGHT * rmse(lf_head, lf_phys) / norm.logF_std
    )
    return {
        "validation_score": float(score),
        "trajectory_RMSE_C": rmse(t_true, t_pred),
        "trajectory_MAE_C": mae(t_true, t_pred),
        "direct_logF_RMSE": rmse(lf_true, lf_head),
        "direct_logF_MAE": mae(lf_true, lf_head),
        "direct_logF_R2": r2(lf_true, lf_head),
        "trajectory_logF_RMSE": rmse(lf_true, lf_phys),
        "direct_vs_trajectory_logF_RMSE": rmse(lf_head, lf_phys),
        "direct_F_MAE_min": mae(f_true, f_head),
        "trajectory_F_MAE_min": mae(f_true, f_phys),
        "retention_RMSE": rmse(q_true, q_pred),
        "retention_MAE": mae(q_true, q_pred),
        "retention_R2": r2(q_true, q_pred),
        "retention_out_of_range_count": int(np.sum((q_pred < 0.0) | (q_pred > 1.0))),
        "stage_jump_mean_abs_C": float(np.mean(np.abs(jump))),
        "stage_jump_p95_abs_C": float(np.percentile(np.abs(jump), 95)),
        "signed_F_error_mean_min": float(np.mean(signed)),
        "absolute_F_error_p95_min": float(np.percentile(f_abs, 95)),
        "absolute_F_error_p99_min": float(np.percentile(f_abs, 99)),
        "near_target_underprediction_p95_min": float(
            np.percentile(near_underprediction, 95)
        ),
        "false_safe_rate_all": float(np.mean(false_safe)),
        "false_unsafe_rate_all": float(np.mean(false_unsafe)),
        "false_safe_rate_near_target": float(np.mean(false_safe[near])),
        "false_unsafe_rate_near_target": float(np.mean(false_unsafe[near])),
        "near_target_count": int(np.sum(near)),
        "inference_ms_per_case": 1000.0 * pred["inference_seconds"] / len(f_true),
    }

def final_test_metrics(
    prediction: Dict[str, np.ndarray],
    config: ModelConfig,
    normalization: Normalization,
) -> Dict[str, float]:
    """Compute final predictive, consistency, continuity, and safety metrics."""
    t_true = prediction["true_temp"]
    t_pred = prediction["temp"]
    logf_true = prediction["true_logf"].reshape(-1)
    logf_direct = prediction["logf"].reshape(-1)
    logf_trajectory = prediction["logf_phys"].reshape(-1)
    q_true = prediction["true_q"].reshape(-1)
    q_pred = prediction["q"].reshape(-1)
    f_true = np.power(10.0, logf_true)
    f_direct = np.power(10.0, logf_direct)
    f_trajectory = np.power(10.0, logf_trajectory)
    f_conservative = np.minimum(f_direct, f_trajectory)
    f_abs = np.abs(f_direct - f_true)
    near = np.abs(f_true - F_SAFETY_THRESHOLD_MIN) <= (
        NEAR_THRESHOLD_FRACTION * F_SAFETY_THRESHOLD_MIN
    )

    direct_false_safe = (f_direct >= F_SAFETY_THRESHOLD_MIN) & (
        f_true < F_SAFETY_THRESHOLD_MIN
    )
    direct_false_unsafe = (f_direct < F_SAFETY_THRESHOLD_MIN) & (
        f_true >= F_SAFETY_THRESHOLD_MIN
    )
    conservative_false_safe = (f_conservative >= F_SAFETY_THRESHOLD_MIN) & (
        f_true < F_SAFETY_THRESHOLD_MIN
    )
    conservative_false_unsafe = (f_conservative < F_SAFETY_THRESHOLD_MIN) & (
        f_true >= F_SAFETY_THRESHOLD_MIN
    )
    jump = t_pred[:, config.n_heat, 0] - t_pred[:, config.n_heat - 1, 0]

    def masked_rate(mask: np.ndarray, subset: np.ndarray) -> float:
        if not np.any(subset):
            return float("nan")
        return float(np.mean(mask[subset]))

    mape = float(
        np.mean(np.abs(f_direct - f_true) / np.maximum(np.abs(f_true), 1.0e-12))
        * 100.0
    )
    composite = (
        rmse(t_true, t_pred) / normalization.temperature_std
        + rmse(logf_true, logf_direct) / normalization.logF_std
        + rmse(q_true, q_pred) / normalization.retention_std
        + rmse(logf_true, logf_trajectory) / normalization.logF_std
        + CONSISTENCY_SCORE_WEIGHT
        * rmse(logf_direct, logf_trajectory)
        / normalization.logF_std
    )
    return {
        "test_cases": int(len(f_true)),
        "composite_test_score": float(composite),
        "trajectory_RMSE_C": rmse(t_true, t_pred),
        "trajectory_MAE_C": mae(t_true, t_pred),
        "direct_logF_RMSE": rmse(logf_true, logf_direct),
        "direct_logF_MAE": mae(logf_true, logf_direct),
        "direct_logF_R2": r2(logf_true, logf_direct),
        "direct_F_MAPE_percent": mape,
        "direct_F_MAE_min": mae(f_true, f_direct),
        "trajectory_logF_RMSE": rmse(logf_true, logf_trajectory),
        "trajectory_logF_MAE": mae(logf_true, logf_trajectory),
        "trajectory_logF_R2": r2(logf_true, logf_trajectory),
        "trajectory_F_MAE_min": mae(f_true, f_trajectory),
        "direct_vs_trajectory_logF_RMSE": rmse(
            logf_direct, logf_trajectory
        ),
        "direct_vs_trajectory_logF_mean_residual": float(
            np.mean(logf_direct - logf_trajectory)
        ),
        "direct_vs_trajectory_logF_std_residual": float(
            np.std(logf_direct - logf_trajectory, ddof=0)
        ),
        "retention_RMSE": rmse(q_true, q_pred),
        "retention_MAE": mae(q_true, q_pred),
        "retention_R2": r2(q_true, q_pred),
        "retention_out_of_range_count": int(
            np.sum((q_pred < 0.0) | (q_pred > 1.0))
        ),
        "stage_jump_mean_abs_C": float(np.mean(np.abs(jump))),
        "stage_jump_p95_abs_C": float(np.percentile(np.abs(jump), 95)),
        "signed_direct_F_error_mean_min": float(np.mean(f_direct - f_true)),
        "absolute_direct_F_error_p95_min": float(np.percentile(f_abs, 95)),
        "absolute_direct_F_error_p99_min": float(np.percentile(f_abs, 99)),
        "near_target_count": int(np.sum(near)),
        "direct_false_safe_rate_all": float(np.mean(direct_false_safe)),
        "direct_false_unsafe_rate_all": float(np.mean(direct_false_unsafe)),
        "direct_false_safe_rate_near_target": masked_rate(direct_false_safe, near),
        "direct_false_unsafe_rate_near_target": masked_rate(
            direct_false_unsafe, near
        ),
        "conservative_false_safe_rate_all": float(
            np.mean(conservative_false_safe)
        ),
        "conservative_false_unsafe_rate_all": float(
            np.mean(conservative_false_unsafe)
        ),
        "conservative_false_safe_rate_near_target": masked_rate(
            conservative_false_safe, near
        ),
        "conservative_false_unsafe_rate_near_target": masked_rate(
            conservative_false_unsafe, near
        ),
        "inference_seconds_total_test_set": float(
            prediction["inference_seconds"]
        ),
        "inference_ms_per_case": float(
            1000.0 * prediction["inference_seconds"] / len(f_true)
        ),
    }

def nearest_training_points(
    x: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance in the four-dimensional input space scaled by domain spans."""
    scaled = (np.asarray(x, dtype=np.float64) - INPUT_LOWER_BOUNDS) / (
        INPUT_UPPER_BOUNDS - INPUT_LOWER_BOUNDS
    )
    search = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    search.fit(scaled[train_idx])
    distances, positions = search.kneighbors(scaled[test_idx])
    nearest_global_indices = train_idx[positions[:, 0]]
    return distances[:, 0].astype(np.float64), nearest_global_indices.astype(np.int64)

def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation without an additional SciPy dependency."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.sum(finite) < 3:
        return float("nan")
    rank_a = pd.Series(a[finite]).rank(method="average").to_numpy()
    rank_b = pd.Series(b[finite]).rank(method="average").to_numpy()
    if np.std(rank_a) == 0.0 or np.std(rank_b) == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])

def build_case_prediction_table(
    protocol: str,
    seed: int,
    test_idx: np.ndarray,
    simulation_ids: np.ndarray,
    x: np.ndarray,
    prediction: Dict[str, np.ndarray],
    nearest_distance: np.ndarray,
    nearest_indices: np.ndarray,
    config: ModelConfig,
) -> pd.DataFrame:
    true_temperature = prediction["true_temp"]
    predicted_temperature = prediction["temp"]
    true_logf = prediction["true_logf"].reshape(-1)
    direct_logf = prediction["logf"].reshape(-1)
    trajectory_logf = prediction["logf_phys"].reshape(-1)
    true_f = np.power(10.0, true_logf)
    direct_f = np.power(10.0, direct_logf)
    trajectory_f = np.power(10.0, trajectory_logf)
    true_q = prediction["true_q"].reshape(-1)
    predicted_q = prediction["q"].reshape(-1)
    temperature_rmse = np.sqrt(
        np.mean(np.square(predicted_temperature - true_temperature), axis=(1, 2))
    )
    stage_jump = (
        predicted_temperature[:, config.n_heat, 0]
        - predicted_temperature[:, config.n_heat - 1, 0]
    )
    direct_f_error = direct_f - true_f
    trajectory_f_error = trajectory_f - true_f
    direct_logf_error = direct_logf - true_logf
    trajectory_logf_error = trajectory_logf - true_logf
    q_error = predicted_q - true_q
    distance_percentile = pd.Series(nearest_distance).rank(
        method="first", pct=True
    ).to_numpy()
    distance_quartile = np.minimum(
        np.ceil(distance_percentile * 4.0).astype(int), 4
    )

    return pd.DataFrame(
        {
            "protocol": protocol,
            "protocol_label": PROTOCOL_LABELS[protocol],
            "seed": int(seed),
            "row_index": test_idx,
            "simulation_id": simulation_ids[test_idx],
            "CUT_min": x[test_idx, 0],
            "heating_time_min": x[test_idx, 1],
            "cooling_time_min": x[test_idx, 2],
            "retort_temperature_C": x[test_idx, 3],
            "nearest_training_distance": nearest_distance,
            "nearest_training_row_index": nearest_indices,
            "nearest_training_simulation_id": simulation_ids[nearest_indices],
            "distance_quartile": distance_quartile,
            "temperature_trajectory_RMSE_C": temperature_rmse,
            "true_log10_F": true_logf,
            "predicted_direct_log10_F": direct_logf,
            "predicted_trajectory_log10_F": trajectory_logf,
            "direct_log10_F_error": direct_logf_error,
            "absolute_direct_log10_F_error": np.abs(direct_logf_error),
            "trajectory_log10_F_error": trajectory_logf_error,
            "absolute_trajectory_log10_F_error": np.abs(trajectory_logf_error),
            "true_F_min": true_f,
            "predicted_direct_F_min": direct_f,
            "predicted_trajectory_F_min": trajectory_f,
            "direct_F_error_min": direct_f_error,
            "absolute_direct_F_error_min": np.abs(direct_f_error),
            "direct_F_relative_error_percent": 100.0 * direct_f_error / true_f,
            "trajectory_F_error_min": trajectory_f_error,
            "absolute_trajectory_F_error_min": np.abs(trajectory_f_error),
            "trajectory_F_relative_error_percent": 100.0 * trajectory_f_error / true_f,
            "true_retention_index": true_q,
            "predicted_retention_index": predicted_q,
            "retention_error": q_error,
            "absolute_retention_error": np.abs(q_error),
            "stage_transition_jump_C": stage_jump,
            "false_safe": (
                (direct_f >= F_SAFETY_THRESHOLD_MIN)
                & (true_f < F_SAFETY_THRESHOLD_MIN)
            ).astype(int),
            "false_unsafe": (
                (direct_f < F_SAFETY_THRESHOLD_MIN)
                & (true_f >= F_SAFETY_THRESHOLD_MIN)
            ).astype(int),
        }
    )

def metrics_from_structured_prediction(
    case_table: pd.DataFrame,
    prediction: Dict[str, np.ndarray],
    normalization: Normalization,
    config: ModelConfig,
) -> Dict[str, object]:
    true_temperature = prediction["true_temp"]
    predicted_temperature = prediction["temp"]
    true_logf = case_table["true_log10_F"].to_numpy()
    direct_logf = case_table["predicted_direct_log10_F"].to_numpy()
    trajectory_logf = case_table["predicted_trajectory_log10_F"].to_numpy()
    true_f = case_table["true_F_min"].to_numpy()
    direct_f = case_table["predicted_direct_F_min"].to_numpy()
    trajectory_f = case_table["predicted_trajectory_F_min"].to_numpy()
    true_q = case_table["true_retention_index"].to_numpy()
    predicted_q = case_table["predicted_retention_index"].to_numpy()
    distance = case_table["nearest_training_distance"].to_numpy()
    direct_absolute_logf_error = case_table[
        "absolute_direct_log10_F_error"
    ].to_numpy()
    trajectory_absolute_logf_error = case_table[
        "absolute_trajectory_log10_F_error"
    ].to_numpy()
    temperature_case_rmse = case_table[
        "temperature_trajectory_RMSE_C"
    ].to_numpy()
    q_absolute_error = case_table["absolute_retention_error"].to_numpy()
    near = np.abs(true_f - F_SAFETY_THRESHOLD_MIN) <= (
        NEAR_THRESHOLD_FRACTION * F_SAFETY_THRESHOLD_MIN
    )

    composite_score = (
        rmse(true_temperature, predicted_temperature) / normalization.temperature_std
        + rmse(true_logf, direct_logf) / normalization.logF_std
        + rmse(true_q, predicted_q) / normalization.retention_std
        + rmse(true_logf, trajectory_logf) / normalization.logF_std
        + CONSISTENCY_SCORE_WEIGHT
        * rmse(direct_logf, trajectory_logf)
        / normalization.logF_std
    )
    metrics: Dict[str, object] = {
        "structured_test_score": float(composite_score),
        "trajectory_RMSE_C": rmse(true_temperature, predicted_temperature),
        "trajectory_MAE_C": mae(true_temperature, predicted_temperature),
        "direct_logF_RMSE": rmse(true_logf, direct_logf),
        "direct_logF_MAE": mae(true_logf, direct_logf),
        "direct_logF_R2": r2(true_logf, direct_logf),
        "trajectory_logF_RMSE": rmse(true_logf, trajectory_logf),
        "trajectory_logF_MAE": mae(true_logf, trajectory_logf),
        "direct_vs_trajectory_logF_RMSE": rmse(direct_logf, trajectory_logf),
        "direct_F_RMSE_min": rmse(true_f, direct_f),
        "direct_F_MAE_min": mae(true_f, direct_f),
        "trajectory_F_RMSE_min": rmse(true_f, trajectory_f),
        "trajectory_F_MAE_min": mae(true_f, trajectory_f),
        "signed_F_error_mean_min": float(np.mean(direct_f - true_f)),
        "absolute_F_error_p95_min": float(
            np.percentile(np.abs(direct_f - true_f), 95)
        ),
        "absolute_F_error_p99_min": float(
            np.percentile(np.abs(direct_f - true_f), 99)
        ),
        "direct_F_absolute_percentage_error_median": float(
            np.median(100.0 * np.abs(direct_f - true_f) / true_f)
        ),
        "direct_F_absolute_percentage_error_p95": float(
            np.percentile(100.0 * np.abs(direct_f - true_f) / true_f, 95)
        ),
        "retention_RMSE": rmse(true_q, predicted_q),
        "retention_MAE": mae(true_q, predicted_q),
        "retention_R2": r2(true_q, predicted_q),
        "retention_out_of_range_count": int(
            np.sum((predicted_q < 0.0) | (predicted_q > 1.0))
        ),
        "stage_jump_mean_abs_C": float(
            np.mean(np.abs(case_table["stage_transition_jump_C"]))
        ),
        "stage_jump_p95_abs_C": float(
            np.percentile(np.abs(case_table["stage_transition_jump_C"]), 95)
        ),
        "false_safe_rate_all": float(case_table["false_safe"].mean()),
        "false_unsafe_rate_all": float(case_table["false_unsafe"].mean()),
        "near_target_count": int(np.sum(near)),
        "nearest_training_distance_min": float(np.min(distance)),
        "nearest_training_distance_median": float(np.median(distance)),
        "nearest_training_distance_p95": float(np.percentile(distance, 95)),
        "nearest_training_distance_max": float(np.max(distance)),
        "distance_vs_temperature_error_spearman": spearman_correlation(
            distance, temperature_case_rmse
        ),
        "distance_vs_direct_logF_error_spearman": spearman_correlation(
            distance, direct_absolute_logf_error
        ),
        "distance_vs_trajectory_logF_error_spearman": spearman_correlation(
            distance, trajectory_absolute_logf_error
        ),
        "distance_vs_retention_error_spearman": spearman_correlation(
            distance, q_absolute_error
        ),
        "inference_ms_per_case": 1000.0
        * float(prediction["inference_seconds"])
        / len(case_table),
    }
    if np.any(near):
        metrics.update(
            {
                "false_safe_rate_near_target": float(
                    case_table.loc[near, "false_safe"].mean()
                ),
                "false_unsafe_rate_near_target": float(
                    case_table.loc[near, "false_unsafe"].mean()
                ),
                "near_target_underprediction_p95_min": float(
                    np.percentile(np.maximum(true_f[near] - direct_f[near], 0.0), 95)
                ),
            }
        )
    else:
        metrics.update(
            {
                "false_safe_rate_near_target": float("nan"),
                "false_unsafe_rate_near_target": float("nan"),
                "near_target_underprediction_p95_min": float("nan"),
            }
        )
    return metrics

def aggregate_run_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in run_metrics.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    summary = run_metrics.groupby(
        ["protocol", "protocol_label"], sort=False
    )[numeric_columns].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    order = {name: position for position, name in enumerate(PROTOCOL_ORDER)}
    summary["protocol_order"] = summary["protocol"].map(order)
    return summary.sort_values("protocol_order").drop(
        columns="protocol_order"
    ).reset_index(drop=True)

def build_distance_bin_tables(
    all_case_predictions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = all_case_predictions.groupby(
        ["protocol", "protocol_label", "seed", "distance_quartile"],
        sort=False,
        observed=True,
    )
    by_seed = grouped.agg(
        cases=("simulation_id", "size"),
        nearest_distance_min=("nearest_training_distance", "min"),
        nearest_distance_mean=("nearest_training_distance", "mean"),
        nearest_distance_median=("nearest_training_distance", "median"),
        nearest_distance_max=("nearest_training_distance", "max"),
        temperature_RMSE_C_mean=("temperature_trajectory_RMSE_C", "mean"),
        direct_absolute_logF_error_mean=("absolute_direct_log10_F_error", "mean"),
        trajectory_absolute_logF_error_mean=(
            "absolute_trajectory_log10_F_error",
            "mean",
        ),
        direct_absolute_F_error_mean_min=("absolute_direct_F_error_min", "mean"),
        trajectory_absolute_F_error_mean_min=(
            "absolute_trajectory_F_error_min",
            "mean",
        ),
        retention_absolute_error_mean=("absolute_retention_error", "mean"),
        false_safe_rate=("false_safe", "mean"),
        false_unsafe_rate=("false_unsafe", "mean"),
    ).reset_index()

    numeric_columns = [
        column
        for column in by_seed.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "distance_quartile"}
    ]
    across_seeds = by_seed.groupby(
        ["protocol", "protocol_label", "distance_quartile"],
        sort=False,
        observed=True,
    )[numeric_columns].agg(["mean", "std"])
    across_seeds.columns = [
        f"{name}_{stat}" for name, stat in across_seeds.columns
    ]
    across_seeds = across_seeds.reset_index()
    return by_seed, across_seeds

def validation_safety_margin(predictions):
    """Margin for optimistic errors, not underpredictions. This is not a safety certificate."""
    overs = []
    counts = []
    for p in predictions:
        truth = np.power(10.0, p["true_logf"].reshape(-1))
        estimate = np.minimum(np.power(10.0,p["logf"].reshape(-1)), np.power(10.0,p["logf_phys"].reshape(-1)))
        near = np.abs(truth - F_SAFETY_THRESHOLD_MIN) <= F_SAFETY_THRESHOLD_MIN * NEAR_THRESHOLD_FRACTION
        counts.append(int(near.sum()))
        if near.any():
            overs.append(float(np.percentile(np.maximum(estimate[near]-truth[near],0.0),95)))
    return max([SAFETY_MARGIN_FLOOR_MIN]+overs), {"near_target_counts_by_seed":counts,"basis":"maximum across seeds of validation near-target p95 positive overprediction of min(F_direct,F_traj)","fallback_floor_used":not bool(overs)}
