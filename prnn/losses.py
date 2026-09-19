"""Training losses and Kaggle-matched batch aggregation."""
from __future__ import annotations
import numpy as np
from .config import CONSISTENCY_SCORE_WEIGHT
from .physics import logf_from_temperature_keras


def per_case_losses(model, batch, config, norm, training):
    """Return per-case loss components.

    This helper is convenient for the lightweight testing backend. Production
    TensorFlow training reduces these values to batch means before computing
    the composite selection score, matching the Kaggle implementation.
    """
    import keras
    ops = keras.ops
    xb, seqb, hm, cm, tb, lfb, qb = batch
    that, lfhat, qhat = model([xb, seqb], training=training)
    that, lfhat, qhat, tb, lfb, qb = [
        ops.cast(v, "float32") for v in (that, lfhat, qhat, tb, lfb, qb)
    ]
    lfphys = logf_from_temperature_keras(that, hm, cm, config.n_heat, config.n_cool)

    def scalar_mse(x):
        return ops.mean(ops.square(x), axis=1)

    t = ops.mean(ops.square((tb - that) / norm.temperature_std), axis=(1, 2))
    f = scalar_mse((lfb - lfhat) / norm.logF_std)
    q = scalar_mse((qb - qhat) / norm.retention_std)
    physics = scalar_mse((lfhat - lfphys) / norm.logF_std)
    ftraj = scalar_mse((lfb - lfphys) / norm.logF_std)
    initial = scalar_mse((that[:, 0, :] - tb[:, 0, :]) / norm.temperature_std)
    continuity = scalar_mse(
        (that[:, config.n_heat, :] - that[:, config.n_heat - 1, :])
        / norm.temperature_std
    )
    total = (
        config.temperature_weight * t
        + config.logF_weight * f
        + config.retention_weight * q
        + config.lambda_phys * physics
        + config.initial_condition_weight * initial
        + config.stage_continuity_weight * continuity
    )
    return {
        "total": total,
        "temperature": t,
        "logF": f,
        "retention": q,
        "physics": physics,
        "trajectory_F": ftraj,
        "initial": initial,
        "continuity": continuity,
    }


def batch_losses(model, batch, config, norm, training):
    """Return Kaggle-equivalent scalar losses for one local batch.

    The selection score is formed from the square roots of the batch MSEs
    *before* epoch averaging. This intentionally reproduces the selection
    arithmetic used in the original Stage A-E/final/holdout scripts.
    """
    import keras
    ops = keras.ops
    case = per_case_losses(model, batch, config, norm, training)
    out = {k: ops.mean(v) for k, v in case.items()}
    eps = ops.cast(1.0e-12, "float32")
    out["selection_score"] = (
        ops.sqrt(out["temperature"] + eps)
        + ops.sqrt(out["logF"] + eps)
        + ops.sqrt(out["retention"] + eps)
        + ops.sqrt(out["trajectory_F"] + eps)
        + CONSISTENCY_SCORE_WEIGHT * ops.sqrt(out["physics"] + eps)
    )
    return out


def aggregate_epoch(sums: dict, batch_count: int) -> dict:
    """Average already-computed batch scalar losses with equal batch weight."""
    if batch_count <= 0:
        raise ValueError("An epoch cannot contain zero batches.")
    result = {k: float(v) / batch_count for k, v in sums.items()}
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError(
            "Nonfinite training/validation loss. The run was not marked complete."
        )
    return result


def weighted_loss_history(history, config):
    out = history[["epoch"]].copy()
    weights = {
        "temperature": config.temperature_weight,
        "logF": config.logF_weight,
        "retention": config.retention_weight,
        "physics": config.lambda_phys,
        "initial": config.initial_condition_weight,
        "continuity": config.stage_continuity_weight,
    }
    for role in ("train", "validation"):
        for name, weight in weights.items():
            out[f"{role}_weighted_{name}"] = weight * history[f"{role}_{name}"]
    return out
