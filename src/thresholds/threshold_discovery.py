# src/thresholds/threshold_discovery.py

import pandas as pd


def calculate_residual_summary(df, axes):
    """
    Compute per-axis residual statistics.

    Returns a DataFrame with one row per axis, including:
    Mean, Std, Min, Q90, Q95, Q99, Q995, Max
    """
    results = []
    for axis in axes:
        residuals = df[f"{axis}_residual"]
        results.append({
            "Axis": axis,
            "Mean": residuals.mean(),
            "Std": residuals.std(),
            "Min": residuals.min(),
            "Q90": residuals.quantile(0.90),
            "Q95": residuals.quantile(0.95),
            "Q99": residuals.quantile(0.99),
            "Q995": residuals.quantile(0.995),
            "Max": residuals.max(),
        })
    return pd.DataFrame(results)


def discover_thresholds(df, axes, alert_percentile=0.95,
                        error_percentile=0.99, time_threshold=5):
    """
    Derive MinC / MaxC / T per axis from the residual distribution.

    Parameters
    ----------
    df : DataFrame with '{axis}_residual' columns
    axes : list of axis names, e.g. ['Axis #1', ..., 'Axis #8']
    alert_percentile : float, e.g. 0.95 -> MinC = 95th percentile of residuals
    error_percentile : float, e.g. 0.99 -> MaxC = 99th percentile of residuals
    time_threshold : int/float, seconds the deviation must persist

    Returns
    -------
    dict[axis] -> {"MinC": float, "MaxC": float, "T": float}
    """
    thresholds = {}
    for axis in axes:
        residuals = df[f"{axis}_residual"]
        thresholds[axis] = {
            "MinC": residuals.quantile(alert_percentile),
            "MaxC": residuals.quantile(error_percentile),
            "T": time_threshold,
        }
    return thresholds