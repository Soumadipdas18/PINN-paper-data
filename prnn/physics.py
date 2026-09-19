"""Segment-wise trapzeoidal lethality in physical minutes."""
from __future__ import annotations
import numpy as np
from .config import T_REF_C, Z_F_C


def logf_from_temperature_np(temperature, heating_minutes, cooling_minutes, n_heat, n_cool):
    """Kaggle-equivalent NumPy lethality integration."""
    t = np.clip(np.asarray(temperature, np.float64), -20.0, 180.0)
    if t.ndim == 2:
        t = t[:, :, None]
    if t.ndim != 3 or t.shape[1] != n_heat + n_cool:
        raise ValueError("Temperature array and stage-grid dimensions do not match.")
    h = t[:, :n_heat, 0]
    c = t[:, n_heat:, 0]
    hm = np.asarray(heating_minutes).reshape(-1)
    cm = np.asarray(cooling_minutes).reshape(-1)
    if len(hm) != len(t) or len(cm) != len(t):
        raise ValueError("Stage durations must match the number of trajectories.")
    lh = np.power(10.0, (h - T_REF_C) / Z_F_C)
    lc = np.power(10.0, (c - T_REF_C) / Z_F_C)
    fh = np.trapz(lh, dx=1.0, axis=1) * hm / (n_heat - 1)
    fc = np.trapz(lc, dx=1.0, axis=1) * cm / (n_cool - 1)
    return np.log10(np.maximum(fh + fc, 1.0e-12)).astype(np.float32).reshape(-1, 1)


def logf_from_temperature_keras(temperature, heating_minutes, cooling_minutes, n_heat, n_cool):
    """Differentiable Kaggle-equivalent lethality integration."""
    import keras
    ops = keras.ops
    t = ops.cast(temperature, "float32")
    t = ops.clip(t, -20.0, 180.0)
    h = t[:, :n_heat, :]
    c = t[:, n_heat:, :]
    hm = ops.reshape(ops.cast(heating_minutes, "float32"), (-1, 1))
    cm = ops.reshape(ops.cast(cooling_minutes, "float32"), (-1, 1))
    dt_h = hm / float(n_heat - 1)
    dt_c = cm / float(n_cool - 1)
    l_h = ops.power(ops.cast(10.0, "float32"), (h - T_REF_C) / Z_F_C)
    l_c = ops.power(ops.cast(10.0, "float32"), (c - T_REF_C) / Z_F_C)
    f_h = dt_h * (
        0.5 * (l_h[:, 0, :] + l_h[:, -1, :])
        + ops.sum(l_h[:, 1:-1, :], axis=1)
    )
    f_c = dt_c * (
        0.5 * (l_c[:, 0, :] + l_c[:, -1, :])
        + ops.sum(l_c[:, 1:-1, :], axis=1)
    )
    total = ops.maximum(f_h + f_c, ops.cast(1.0e-12, "float32"))
    return ops.log(total) / ops.log(ops.cast(10.0, "float32"))
