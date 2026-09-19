"""Stage-coupled recurrent surrogate with attention-based scalar heads."""
from __future__ import annotations
import numpy as np
import keras
from keras import Model, layers
from .config import ModelConfig, Normalization, INITIAL_TEMPERATURE_C

def build_surrogate(config: ModelConfig, norm: Normalization) -> Model:
    """Three-module conditional surrogate with stage coupling and attention."""
    total = config.n_heat + config.n_cool
    x_in = layers.Input((4,), name="operating_inputs_scaled")
    seq_in = layers.Input((total, 6), name="time_and_boundary_features")

    # Module 1: operating-condition encoder.
    ctx = layers.Dense(config.encoder_units, activation="swish", name="encoder_dense_1")(x_in)
    ctx = layers.LayerNormalization(name="encoder_norm")(ctx)
    ctx = layers.Dense(config.encoder_units, activation="swish", name="encoder_dense_2")(ctx)
    if config.dropout > 0:
        ctx = layers.Dropout(config.dropout, name="encoder_dropout")(ctx)

    ctx_seq = layers.RepeatVector(total, name="repeat_context")(ctx)
    decoder_input = layers.Concatenate(name="decoder_features")([ctx_seq, seq_in])
    heat_input = layers.Cropping1D(
        cropping=(0, config.n_cool), name="heating_features"
    )(decoder_input)
    cool_input = layers.Cropping1D(
        cropping=(config.n_heat, 0), name="cooling_features"
    )(decoder_input)

    # Module 2: sequential temperature decoder.  Heating states initialize the
    # cooling decoder, so the decoder carries thermal memory across the stage.
    if config.model_kind == "lstm":
        h0 = layers.Dense(config.units, activation="tanh", name="initial_hidden")(ctx)
        c0 = layers.Dense(config.units, activation="tanh", name="initial_cell")(ctx)
        heat_layer = layers.LSTM(
            config.units,
            return_sequences=True,
            return_state=True,
            dropout=config.dropout,
            name="heating_lstm",
        )
        cool_layer = layers.LSTM(
            config.units,
            return_sequences=True,
            return_state=True,
            dropout=config.dropout,
            name="cooling_lstm",
        )
        heat_hidden, h_heat, c_heat = heat_layer(heat_input, initial_state=[h0, c0])
        cool_hidden, _, _ = cool_layer(cool_input, initial_state=[h_heat, c_heat])
    elif config.model_kind == "gru":
        h0 = layers.Dense(config.units, activation="tanh", name="initial_hidden")(ctx)
        # Two GRUCell decoders transfer the final heating state to cooling.
        heat_layer = layers.RNN(
            layers.GRUCell(
                config.units,
                dropout=config.dropout,
                reset_after=True,
                name="heating_gru_cell",
            ),
            return_sequences=True,
            return_state=True,
            name="heating_gru",
        )
        cool_layer = layers.RNN(
            layers.GRUCell(
                config.units,
                dropout=config.dropout,
                reset_after=True,
                name="cooling_gru_cell",
            ),
            return_sequences=True,
            return_state=True,
            name="cooling_gru",
        )
        heat_hidden, h_heat = heat_layer(heat_input, initial_state=[h0])
        cool_hidden, _ = cool_layer(cool_input, initial_state=[h_heat])
    elif config.model_kind == "mlp":
        # A controlled time-conditioned baseline.  It sees exactly the same
        # operating and time/boundary features but has no recurrent state.
        hidden_all = layers.TimeDistributed(
            layers.Dense(config.units, activation="swish"), name="time_mlp_1"
        )(decoder_input)
        hidden_all = layers.TimeDistributed(
            layers.Dense(config.units, activation="swish"), name="time_mlp_2"
        )(hidden_all)
        if config.dropout > 0:
            hidden_all = layers.Dropout(config.dropout, name="time_mlp_dropout")(hidden_all)
        heat_hidden = layers.Cropping1D(
            cropping=(0, config.n_cool), name="mlp_heating_hidden"
        )(hidden_all)
        cool_hidden = layers.Cropping1D(
            cropping=(config.n_heat, 0), name="mlp_cooling_hidden"
        )(hidden_all)
    else:
        raise ValueError(f"Unknown model_kind: {config.model_kind}")

    hidden = layers.Concatenate(axis=1, name="joined_stage_states")(
        [heat_hidden, cool_hidden]
    )
    initial_temp_z = (INITIAL_TEMPERATURE_C - norm.temperature_mean) / norm.temperature_std
    temp_z = layers.TimeDistributed(
        layers.Dense(
            1,
            bias_initializer=keras.initializers.Constant(initial_temp_z),
            dtype="float32",
        ),
        name="temperature_z",
    )(hidden)
    temp_out = layers.Rescaling(
        scale=norm.temperature_std,
        offset=norm.temperature_mean,
        name="temperature_C",
        dtype="float32",
    )(temp_z)

    # Module 3: attention-pooled scalar heads.
    attn_hidden = layers.TimeDistributed(
        layers.Dense(max(config.units // 2, 32), activation="tanh"), name="attention_hidden"
    )(hidden)
    attn_logits = layers.TimeDistributed(
        layers.Dense(1, dtype="float32"), name="attention_logits"
    )(attn_hidden)
    attn_weights = layers.Softmax(
        axis=1, name="temporal_attention", dtype="float32"
    )(attn_logits)
    pooled = layers.Dot(
        axes=1, name="attention_weighted_sum", dtype="float32"
    )([attn_weights, hidden])
    pooled = layers.Flatten(name="attention_pool")(pooled)
    scalar_features = layers.Concatenate(name="scalar_features")([ctx, pooled])
    scalar_features = layers.Dense(config.units, activation="swish", name="scalar_dense")(scalar_features)
    if config.dropout > 0:
        scalar_features = layers.Dropout(config.dropout, name="scalar_dropout")(scalar_features)

    logf_z = layers.Dense(1, name="logF_z", dtype="float32")(scalar_features)
    logf_out = layers.Rescaling(
        scale=norm.logF_std,
        offset=norm.logF_mean,
        name="log10_F_direct",
        dtype="float32",
    )(logf_z)
    q0 = float(np.clip(norm.retention_mean, 1.0e-5, 1.0 - 1.0e-5))
    retention_out = layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=keras.initializers.Constant(np.log(q0 / (1.0 - q0))),
        name="retention_index",
        dtype="float32",
    )(scalar_features)
    return Model([x_in, seq_in], [temp_out, logf_out, retention_out], name=config.model_kind)
