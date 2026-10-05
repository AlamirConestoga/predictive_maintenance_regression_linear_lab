"""
Regression module for the Predictive Maintenance lab.

Fits univariate linear regressions (Time -> Axis values) for each axis,
then produces predictions and residuals for anomaly detection.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def prepare_time_column(df):
    """
    Convert the 'Time' column to datetime and add 'time_seconds'.

    'time_seconds' is the number of seconds elapsed since the first
    timestamp in the dataframe. It is the x-variable for regression.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a column named 'Time'.

    Returns
    -------
    pandas.DataFrame
        A copy of df with:
        - 'Time' cast to datetime (UTC)
        - a new 'time_seconds' column (float)
    """
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df["time_seconds"] = (
        df["Time"] - df["Time"].min()
    ).dt.total_seconds()
    return df


def train_regression_models(df, axes):
    """
    Fit one LinearRegression per axis: time_seconds -> axis value.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'time_seconds' (from prepare_time_column) and
        one column per name in `axes`.
    axes : list[str]
        e.g. ['Axis #1', 'Axis #2', ..., 'Axis #8'].

    Returns
    -------
    models : dict[str, LinearRegression]
        Trained model per axis.
    results : pandas.DataFrame
        One row per axis with columns:
        ['Axis', 'Slope', 'Intercept', 'R2'].
    """
    models = {}
    results = []

    # sklearn expects a 2D array -> double brackets
    X = df[["time_seconds"]]

    for axis in axes:
        y = df[axis]

        model = LinearRegression()
        model.fit(X, y)

        prediction = model.predict(X)

        results.append({
            "Axis": axis,
            "Slope": model.coef_[0],
            "Intercept": model.intercept_,
            "R2": r2_score(y, prediction),
        })

        models[axis] = model

    return models, pd.DataFrame(results)


def add_predictions_and_residuals(df, models, axes):
    """
    Add per-axis prediction and residual columns to df.

    For each axis, this adds:
      - '{axis}_prediction' : model prediction
      - '{axis}_residual'   : actual - prediction

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'time_seconds' and one column per name in `axes`.
    models : dict[str, LinearRegression]
        Output of train_regression_models.
    axes : list[str]

    Returns
    -------
    pandas.DataFrame
        A copy of df with 2 * len(axes) new columns.
    """
    df = df.copy()
    X = df[["time_seconds"]]

    for axis in axes:
        pred_col = f"{axis}_prediction"
        res_col  = f"{axis}_residual"

        df[pred_col] = models[axis].predict(X)
        df[res_col]  = df[axis] - df[pred_col]

    return df