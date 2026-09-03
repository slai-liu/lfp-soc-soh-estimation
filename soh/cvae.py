"""Optional TensorFlow CVAE for auxiliary U1-U21 feature generation.

This module is not imported by the default Random Forest path. It fits feature
and condition scalers only on the supplied training rows.
"""
from __future__ import annotations

import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler

try:
    from . import FEATURE_COLUMNS
except ImportError:  # direct script execution
    from __init__ import FEATURE_COLUMNS


def generate_auxiliary_features(train_frame, target_soc: float, *, epochs: int = 50,
                                batch_size: int = 32, seed: int = 0):
    try:
        import tensorflow as tf
        from tensorflow.keras import backend as K
        from tensorflow.keras.layers import Dense, Input, Lambda, MultiHeadAttention
        from tensorflow.keras.losses import mse
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise RuntimeError("CVAE is optional; install soh/requirements-cvae.txt") from exc
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    x = train_frame[FEATURE_COLUMNS].to_numpy(float)
    conditions = train_frame[["SOC", "SOH"]].to_numpy(float)
    feature_scaler = MinMaxScaler().fit(x)
    condition_scaler = MinMaxScaler().fit(conditions)
    x_scaled = feature_scaler.transform(x); c_scaled = condition_scaler.transform(conditions)

    condition_input = Input((2,), name="condition")
    embedding = Dense(64, activation="relu")(condition_input)
    expanded = tf.expand_dims(embedding, 2)
    features = Input((21,), name="features")
    hidden = Dense(64, activation="relu")(features)
    attended = MultiHeadAttention(num_heads=1, key_dim=64)(
        tf.expand_dims(hidden, 2), expanded, expanded)
    attended = tf.squeeze(attended, 2)
    z_mean = Dense(2, name="z_mean")(attended)
    z_log_var = Dense(2, name="z_log_var")(attended)

    def sampling(values):
        mean, log_var = values
        epsilon = K.random_normal((K.shape(mean)[0], 2), seed=seed)
        return mean + K.exp(.5 * log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])
    latent_input = Input((2,), name="latent")
    decoded_hidden = Dense(64, activation="relu")(latent_input)
    decoded_attended = MultiHeadAttention(num_heads=1, key_dim=64)(
        tf.expand_dims(decoded_hidden, 2), expanded, expanded)
    decoded = Dense(21, activation="sigmoid")(tf.squeeze(decoded_attended, 2))
    decoder = Model([latent_input, condition_input], decoded)
    output = decoder([z, condition_input])
    vae = Model([features, condition_input], output)
    reconstruction = 21 * mse(features, output)
    kl = -.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae.add_loss(K.mean(.5 * reconstruction + .5 * kl)); vae.compile(optimizer=Adam())
    vae.fit([x_scaled, c_scaled], x_scaled, epochs=epochs, batch_size=batch_size,
            shuffle=False, verbose=2)

    batteries = train_frame[["battery_id", "SOH"]].drop_duplicates("battery_id").sort_values("battery_id")
    generation_conditions = np.column_stack([
        np.full(len(batteries), target_soc), batteries.SOH.to_numpy(float)])
    generation_scaled = condition_scaler.transform(generation_conditions)
    latent = tf.random.stateless_normal((len(batteries), 2), seed=(seed, seed))
    generated = feature_scaler.inverse_transform(
        decoder.predict([latent, generation_scaled], batch_size=batch_size, verbose=0))
    result = batteries.copy(); result["SOC"] = target_soc
    result.loc[:, FEATURE_COLUMNS] = generated
    return result[["battery_id", "SOC", "SOH", *FEATURE_COLUMNS]]
