# src/visualization/regression_plots.py

"""
Visualization helpers for the Predictive Maintenance lab.

Three public functions:

- plot_regression(df, model, axis, ...)
    Regression line over the raw scatter of one axis.

- plot_residual_distribution(df, axis, ...)
    Histogram of residuals with MinC/MaxC thresholds marked.

- plot_alerts_and_errors(stream_df, events_df, axis, thresholds, ...)
    Annotated regression plot with Alert/Error windows shaded and
    each event labeled with its duration. Saves a PNG.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_regression(df, model, axis, save_path=None, show=True):
    """
    Regression line over the raw scatter of one axis.

    Parameters
    ----------
    df : DataFrame with 'Time' and 'time_seconds' columns (see
         regression_model.prepare_time_column).
    model : fitted sklearn LinearRegression for this axis.
    axis : str, e.g. 'Axis #2'.
    save_path : str or Path, optional. If given, saves the PNG.
    show : bool. Whether to call plt.show().
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(df["time_seconds"], df[axis],
               s=4, alpha=0.4, label=f"{axis} (raw)")

    prediction = model.predict(df[["time_seconds"]])
    ax.plot(df["time_seconds"], prediction,
            color="red", linewidth=1.5, label="regression line")

    ax.set_title(f"{axis} — Time vs current")
    ax.set_xlabel("time_seconds")
    ax.set_ylabel(axis)
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_residual_distribution(df, axis, min_c=None, max_c=None,
                               save_path=None, show=True):
    """
    Histogram of residuals for one axis with MinC/MaxC thresholds marked.
    """
    residuals = df[f"{axis}_residual"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(residuals, bins=60, color="steelblue", alpha=0.7,
            edgecolor="black", linewidth=0.3)

    if min_c is not None:
        ax.axvline(min_c, color="orange", linestyle="--",
                   label=f"MinC = {min_c:.2f}")
    if max_c is not None:
        ax.axvline(max_c, color="red", linestyle="--",
                   label=f"MaxC = {max_c:.2f}")

    ax.set_title(f"{axis} — Residual distribution")
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_alerts_and_errors(stream_df, events_df, axis, thresholds,
                           save_path=None, show=True):
    """
    Annotated regression plot with Alert/Error windows shaded.

    Parameters
    ----------
    stream_df : DataFrame from Neon / synthetic stream. Must contain:
                'time', 'axis_2', 'axis_2_prediction'.
    events_df : DataFrame of detected events. Must contain:
                'event_type', 'start_time', 'end_time',
                'duration_seconds', 'max_deviation'.
    axis : str, e.g. 'Axis #2'. Column names in stream_df use lowercase
           with underscores (axis_2).
    thresholds : dict like {'MinC': float, 'MaxC': float, 'T': float}.
    save_path : str or Path, optional.
    show : bool.

    Returns
    -------
    (fig, ax) : matplotlib figure and axes.
    """
    # --- Data prep ------------------------------------------------
    plot_df = stream_df.sort_values("time").reset_index(drop=True)
    events = events_df.copy()
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
    events["end_time"]   = pd.to_datetime(events["end_time"],   utc=True)

    # Map "Axis #2" -> "axis_2"
    axis_col = axis.lower().replace(" #", "_").replace(" ", "_")
    pred_col = f"{axis_col}_prediction"

    minc = thresholds["MinC"]
    maxc = thresholds["MaxC"]

    baseline_mean = plot_df[pred_col].mean()
    alert_line    = baseline_mean + minc
    error_line    = baseline_mean + maxc

    # --- Figure ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(plot_df["time"], plot_df[pred_col],
            linestyle="--", color="gray", linewidth=1,
            label="Regression baseline")

    ax.axhline(alert_line, color="orange", linestyle=":", linewidth=1,
               label=f"MinC (Alert) = baseline + {minc:.2f}")
    ax.axhline(error_line, color="red", linestyle=":", linewidth=1,
               label=f"MaxC (Error) = baseline + {maxc:.2f}")

    ax.plot(plot_df["time"], plot_df[axis_col],
            color="steelblue", linewidth=1.4, marker="o", markersize=3,
            label=f"{axis} (actual)")

    color_map = {"alert": "orange", "error": "red"}

    # Shade event windows
    for _, ev in events.iterrows():
        c = color_map.get(ev["event_type"], "gray")
        ax.axvspan(ev["start_time"], ev["end_time"],
                   alpha=0.18, color=c, zorder=0)

    # Annotate each event with duration
    y_top = plot_df[axis_col].max()
    y_span = y_top - plot_df[axis_col].min()

    for i, (_, ev) in enumerate(events.iterrows()):
        mid = ev["start_time"] + (ev["end_time"] - ev["start_time"]) / 2
        c = color_map.get(ev["event_type"], "gray")

        ax.axvline(mid, color=c, linewidth=1, linestyle="-", alpha=0.5)

        label = (
            f"{ev['event_type'].upper()}\n"
            f"duration = {ev['duration_seconds']:.1f} s\n"
            f"max dev = {ev['max_deviation']:.2f}"
        )
        ax.annotate(
            label,
            xy=(mid, y_top + y_span * 0.02),
            xytext=(mid, y_top + y_span * (0.15 + 0.18 * i)),
            ha="center", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=c, linewidth=1.2),
            arrowprops=dict(arrowstyle="-", color=c, lw=1),
        )

    ax.set_title(
        f"Synthetic stream ({axis}) — Regression baseline "
        f"with detected Alert / Error events",
        fontsize=12,
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(f"{axis} current")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(
        plot_df[axis_col].min() - y_span * 0.05,
        y_top + y_span * 0.75,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig, ax