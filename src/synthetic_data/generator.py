# src/synthetic_data/generator.py

import numpy as np
import pandas as pd


def generate_synthetic_test_data(
    df,
    model,
    axis,
    min_c,
    max_c,
    n_normal=20,
    n_alert=8,
    n_gap=6,
    n_error=8,
    n_recovery=20,
    seconds_per_step=1,
    seed=42,
):
    rng = np.random.default_rng(seed)

    last_time = pd.to_datetime(df["Time"], utc=True).max()
    start_time = last_time + pd.Timedelta(seconds=seconds_per_step)

    n_total = n_normal + n_alert + n_gap + n_error + n_recovery
    timestamps = [
        start_time + pd.Timedelta(seconds=i * seconds_per_step)
        for i in range(n_total)
    ]

    t_last = df["time_seconds"].max()
    time_seconds = np.array(
        [t_last + (i + 1) * seconds_per_step for i in range(n_total)]
    )

    slope = model.coef_[0]
    intercept = model.intercept_
    baseline = intercept + slope * time_seconds

    residuals = np.zeros(n_total)

    # 1. normal
    residuals[:n_normal] = rng.normal(0, min_c / 4, size=n_normal)

    # 2. alert
    a0 = n_normal
    a1 = a0 + n_alert
    alert_center = (min_c + max_c) / 2
    residuals[a0:a1] = rng.normal(alert_center, (max_c - min_c) / 10,
                                  size=n_alert)

    # 3. gap — residual falls back near baseline so detector re-arms
    g0 = a1
    g1 = g0 + n_gap
    residuals[g0:g1] = rng.normal(0, min_c / 4, size=n_gap)

    # 4. error
    e0 = g1
    e1 = e0 + n_error
    error_center = max_c * 1.2
    residuals[e0:e1] = rng.normal(error_center, max_c * 0.05,
                                  size=n_error)

    # 5. recovery
    residuals[e1:] = rng.normal(0, min_c / 4, size=n_recovery)

    actual = baseline + residuals
    prediction = baseline

    segment = (
        ["normal"]     * n_normal
        + ["alert"]    * n_alert
        + ["gap"]      * n_gap
        + ["error"]    * n_error
        + ["recovery"] * n_recovery
    )

    return pd.DataFrame({
        "Time": timestamps,
        "time_seconds": time_seconds,
        axis: actual,
        f"{axis}_prediction": prediction,
        f"{axis}_residual": residuals,
        "segment": segment,
    })