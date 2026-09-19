"""Production TensorFlow backend matching the Kaggle training arithmetic."""
from __future__ import annotations
import os
import time
import platform
import sys
import numpy as np
import pandas as pd
from .losses import batch_losses
from .physics import logf_from_temperature_np


class TensorFlowBackend:
    def __init__(self, options):
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        if options.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import tensorflow as tf
            import keras
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc
        self.tf = tf
        self.keras = keras
        self.options = options
        if tf.config.threading.get_intra_op_parallelism_threads() != options.threads:
            tf.config.threading.set_intra_op_parallelism_threads(options.threads)
        if tf.config.threading.get_inter_op_parallelism_threads() != options.threads:
            tf.config.threading.set_inter_op_parallelism_threads(options.threads)
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except (RuntimeError, ValueError):
                pass
        if options.device == "gpu" and not gpus:
            raise RuntimeError("GPU requested but TensorFlow found no GPU.")
        if options.precision == "mixed_float16" and not gpus:
            raise RuntimeError(
                "Mixed precision was requested without a GPU. Use --precision float32 on CPU."
            )
        keras.mixed_precision.set_global_policy(options.precision)
        self.strategy = (
            tf.distribute.MirroredStrategy()
            if len(gpus) >= 2
            else tf.distribute.get_strategy()
        )
        self.replicas = int(self.strategy.num_replicas_in_sync)
        if 64 % self.replicas != 0:
            raise ValueError(
                "The global batch size must be divisible by the number of distributed replicas."
            )
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        self.environment = {
            "backend": "tensorflow",
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "gpu_devices": [g.name for g in gpus],
            "replicas": self.replicas,
            "precision": options.precision,
        }

    def make_model(self, config, norm, seed):
        self.keras.backend.clear_session()
        self.keras.utils.set_random_seed(seed)
        from .model import build_surrogate
        with self.strategy.scope():
            return build_surrogate(config, norm)

    def load_model(self, path):
        with self.strategy.scope():
            return self.keras.models.load_model(path, compile=False, safe_mode=True)

    def _dataset(self, arrays, batch_size, shuffle, seed):
        """Match the Kaggle tf.data pipeline: shuffle, batch, prefetch."""
        tf = self.tf
        ds = tf.data.Dataset.from_tensor_slices(arrays)
        if shuffle:
            ds = ds.shuffle(len(arrays[0]), seed=seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def fit(self, model, prepared, splits, config, seed, epochs, restore_best):
        tf = self.tf
        strategy = self.strategy
        with strategy.scope():
            base = tf.keras.optimizers.Adam(config.learning_rate, clipnorm=1.0)
            mixed = self.options.precision == "mixed_float16"
            opt = tf.keras.mixed_precision.LossScaleOptimizer(base) if mixed else base
            opt.build(model.trainable_variables)

        train = self._dataset(
            prepared.arrays(splits["train"]), config.batch_size, True, seed
        )
        val = self._dataset(
            prepared.arrays(splits["validation"]), config.batch_size, False, seed
        )
        train = strategy.experimental_distribute_dataset(train)
        val = strategy.experimental_distribute_dataset(val)

        def replica_train_step(batch):
            with tf.GradientTape() as tape:
                losses = batch_losses(model, batch, config, prepared.norm, training=True)
                distributed_loss = losses["total"] / tf.cast(
                    strategy.num_replicas_in_sync, tf.float32
                )
                optimized_loss = opt.scale_loss(distributed_loss) if mixed else distributed_loss
            grads = tape.gradient(optimized_loss, model.trainable_variables)
            pairs = [
                (g, v)
                for g, v in zip(grads, model.trainable_variables)
                if g is not None
            ]
            if len(pairs) != len(model.trainable_variables):
                raise RuntimeError("A trainable model variable has no loss gradient.")
            opt.apply_gradients(pairs)
            return losses

        def replica_validation_step(batch):
            return batch_losses(model, batch, config, prepared.norm, training=False)

        @tf.function(reduce_retracing=True)
        def distributed_train_step(batch):
            per_replica = strategy.run(replica_train_step, args=(batch,))
            return {
                key: strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
                for key, value in per_replica.items()
            }

        @tf.function(reduce_retracing=True)
        def distributed_validation_step(batch):
            per_replica = strategy.run(replica_validation_step, args=(batch,))
            return {
                key: strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
                for key, value in per_replica.items()
            }

        best_score = np.inf
        best_epoch = 0
        best_weights = None
        lr_wait = 0
        rows = []
        training_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            learning_rate_used = float(base.learning_rate.numpy())
            summaries = {}
            for role, dataset, step in (
                ("train", train, distributed_train_step),
                ("validation", val, distributed_validation_step),
            ):
                sums = None
                batches = 0
                for batch in dataset:
                    out = step(batch)
                    if sums is None:
                        sums = {k: tf.identity(v) for k, v in out.items()}
                    else:
                        sums = {k: sums[k] + v for k, v in out.items()}
                    batches += 1
                if sums is None or batches == 0:
                    raise RuntimeError(f"{role} dataset must not be empty.")
                summaries[role] = {
                    k: float((v / batches).numpy()) for k, v in sums.items()
                }

            score = summaries["validation"]["selection_score"]
            if score < best_score - 1.0e-5:
                best_score = score
                best_epoch = epoch
                if restore_best:
                    best_weights = model.get_weights()
                lr_wait = 0
            else:
                lr_wait += 1
                if lr_wait >= 4:
                    base.learning_rate.assign(
                        max(float(base.learning_rate.numpy()) * 0.5, 1.0e-5)
                    )
                    lr_wait = 0

            rows.append(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate_used,
                    "epoch_seconds": time.perf_counter() - epoch_start,
                    "elapsed_seconds": time.perf_counter() - training_start,
                    **{
                        f"{role}_{key}": value
                        for role, losses in summaries.items()
                        for key, value in losses.items()
                    },
                }
            )
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(
                    f"  epoch {epoch}/{epochs}; validation score={score:.8g}",
                    flush=True,
                )

        if best_epoch == 0:
            raise RuntimeError("Training did not produce a finite validation score.")
        if restore_best:
            if best_weights is None:
                raise RuntimeError("Best validation weights were not retained.")
            model.set_weights(best_weights)
        return pd.DataFrame(rows), best_epoch, time.perf_counter() - training_start

    def predict(self, model, prepared, indices, config, warmup_indices=None):
        """Collect predictions and time only the model forward passes.

        The final main-model test may pass validation indices for the same
        pre-test warm-up used in the Kaggle final-training script. Stage E and
        structured holdouts pass no explicit warm-up.
        """
        if warmup_indices is not None and len(warmup_indices):
            ii = np.asarray(warmup_indices)[: config.batch_size]
            outputs = model(
                [prepared.x_scaled[ii], prepared.seq[ii]], training=False
            )
            for value in outputs:
                value.numpy()

        parts = [[], [], []]
        start = time.perf_counter()
        for offset in range(0, len(indices), config.batch_size):
            ii = indices[offset : offset + config.batch_size]
            outputs = model(
                [prepared.x_scaled[ii], prepared.seq[ii]], training=False
            )
            for part, value in zip(parts, outputs):
                part.append(np.asarray(value.numpy(), dtype=np.float32))
        elapsed = time.perf_counter() - start
        temp, logf, q = [np.concatenate(part) for part in parts]
        lfphys = logf_from_temperature_np(
            temp,
            prepared.heat_minutes[indices],
            prepared.cool_minutes[indices],
            config.n_heat,
            config.n_cool,
        )
        return {
            "temp": temp,
            "logf": logf,
            "q": q,
            "logf_phys": lfphys,
            "true_temp": prepared.temperature[indices],
            "true_logf": prepared.logf[indices],
            "true_q": prepared.retention[indices],
            "inference_seconds": elapsed,
        }

    def raw_predict(self, model, x_scaled, seq, batch_size=64):
        parts = [[], [], []]
        for off in range(0, len(x_scaled), batch_size):
            outputs = model(
                [x_scaled[off : off + batch_size], seq[off : off + batch_size]],
                training=False,
            )
            for part, value in zip(parts, outputs):
                part.append(np.asarray(value.numpy(), dtype=np.float32))
        return [np.concatenate(part) for part in parts]

    def attention(self, model, prepared, indices, config):
        partial = self.keras.Model(
            model.inputs, model.get_layer("temporal_attention").output
        )
        values = []
        for off in range(0, len(indices), config.batch_size):
            ii = indices[off : off + config.batch_size]
            values.append(
                np.asarray(
                    partial(
                        [prepared.x_scaled[ii], prepared.seq[ii]], training=False
                    ).numpy(),
                    dtype=np.float32,
                )
            )
        return np.concatenate(values)
