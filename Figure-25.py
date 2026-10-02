
# Lagged carbon–ecosystem relationships with sea-ice variability

# Right panels:
# Dot = median bootstrap best lag
# Thick vertical line = interquartile range, 25–75%
#
# Correction in this version:
# Sea labels are shown in Spring, Summer, and Early autumn panels.
# Output: 1080 dpi

#Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 0) Install
!pip -q install pandas numpy matplotlib


# 1) Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


# 2) File paths
lag_mean_csv = "/content/drive/MyDrive/SAM_Thesis/Processed/Fig25_lag_correlation_mean.csv"

summary_csv = "/content/drive/MyDrive/SAM_Thesis/Processed/Fig25_best_lag_summary_by_sea_season_variable.csv"

fig_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(fig_dir, exist_ok=True)

OUT_FIG = os.path.join(
    fig_dir,
    "Fig25_Lagged_Carbon_Ecosystem_Relationships_FINAL_CLEAN_1080dpi.png"
)

for path in [lag_mean_csv, summary_csv]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found:\n{path}")

print("All input files found.")


# 3) Load data
lag_mean_df = pd.read_csv(lag_mean_csv)
summary_df = pd.read_csv(summary_csv)

print("Lag mean columns:")
print(lag_mean_df.columns.tolist())

print("\nSummary columns:")
print(summary_df.columns.tolist())


# 4) Basic settings
SAVE_DPI = 1080

sea_codes = [
    "WED", "KHV", "RLS", "LAZ", "COS", "COO", "DAV",
    "MAW", "DUR", "SOM", "ROS", "AMU", "BEL"
]

season_order = ["Spring", "Summer", "Early autumn"]
variables = ["POC", "CHL", "NPPint"]

summary_df["season"] = pd.Categorical(
    summary_df["season"],
    categories=season_order,
    ordered=True
)

summary_df["sea"] = pd.Categorical(
    summary_df["sea"],
    categories=sea_codes,
    ordered=True
)

summary_df["variable"] = pd.Categorical(
    summary_df["variable"],
    categories=variables,
    ordered=True
)

lag_mean_df["variable"] = pd.Categorical(
    lag_mean_df["variable"],
    categories=variables,
    ordered=True
)

summary_df = summary_df.sort_values(["season", "sea", "variable"]).reset_index(drop=True)
lag_mean_df = lag_mean_df.sort_values(["variable", "lag"]).reset_index(drop=True)


# 5) Style
var_style = {
    "POC": {
        "label": "POC",
        "title": "SIC variability → POC",
        "color": "#1f5fbf",
        "shade": "#b8cbee",
    },
    "CHL": {
        "label": "Chlorophyll-a",
        "title": "SIC variability → Chlorophyll-a",
        "color": "#1b8a2f",
        "shade": "#bfe5bf",
    },
    "NPPint": {
        "label": r"NPP$_{int}$",
        "title": r"SIC variability → NPP$_{int}$",
        "color": "#f07f13",
        "shade": "#ffd0a6",
    },
}

panel_labels = {
    "POC": "a",
    "CHL": "c",
    "NPPint": "e",
    "Spring": "b",
    "Summer": "d",
    "Early autumn": "f",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.85,
    "savefig.dpi": SAVE_DPI,
})


# 6) Helper functions
def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def style_spines(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(0.85)


def panel_label(ax, label, x=-0.10, y=1.05):
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold"
    )


def plot_lag_curve(ax, var_key):
    style = var_style[var_key]

    sub = lag_mean_df[lag_mean_df["variable"] == var_key].copy()

    sub["lag"] = clean_numeric(sub["lag"])
    sub["mean_r"] = clean_numeric(sub["mean_r"])
    sub["ci95"] = clean_numeric(sub["ci95"])

    sub = sub.dropna(subset=["lag", "mean_r"]).sort_values("lag")

    lags = sub["lag"].values.astype(float)
    mean_r = sub["mean_r"].values.astype(float)
    ci95 = sub["ci95"].fillna(0).values.astype(float)

    color = style["color"]
    shade = style["shade"]

    ax.axhline(
        0,
        color="0.25",
        linestyle="--",
        linewidth=0.8,
        zorder=1
    )

    ax.axvline(
        0,
        color="0.25",
        linestyle="--",
        linewidth=0.8,
        zorder=1
    )

    ax.fill_between(
        lags,
        mean_r - ci95,
        mean_r + ci95,
        color=shade,
        alpha=0.55,
        linewidth=0,
        zorder=2
    )

    ax.plot(
        lags,
        mean_r,
        color=color,
        linewidth=2.0,
        zorder=4
    )

    ax.scatter(
        lags,
        mean_r,
        color=color,
        edgecolor="black",
        linewidth=0.25,
        s=22,
        zorder=5
    )

    if "best_lag_mean_curve" in sub.columns and "best_mean_r" in sub.columns:
        best_lag = sub["best_lag_mean_curve"].dropna().iloc[0]
        best_r = sub["best_mean_r"].dropna().iloc[0]
    else:
        idx = np.nanargmax(mean_r)
        best_lag = lags[idx]
        best_r = mean_r[idx]

    ax.axvspan(
        best_lag - 0.35,
        best_lag + 0.35,
        color=shade,
        alpha=0.35,
        zorder=1
    )

    ax.text(
        0.95,
        0.10,
        f"Peak lag ≈ {int(best_lag):+d} months\nmean r = {best_r:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=color,
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="white",
            edgecolor=color,
            linewidth=0.7,
            alpha=0.92
        )
    )

    ax.set_title(
        style["title"],
        fontsize=11.4,
        fontweight="bold",
        color=color,
        pad=4
    )

    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-0.8, 0.8)

    ax.set_xticks(np.arange(-3, 4, 1))
    ax.set_yticks(np.arange(-0.8, 0.81, 0.4))

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.55,
        color="0.72",
        alpha=0.85
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        labelsize=8.0,
        width=0.8,
        length=3.5
    )

    ax.set_ylabel(
        "Lag correlation (r)",
        fontsize=9.2,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Lag (months)",
        fontsize=9.2,
        fontweight="bold"
    )

    style_spines(ax)
    panel_label(ax, panel_labels[var_key], x=0.04, y=0.88)


def interval_plot_for_season(ax, season_name, show_xlabel=False):
    """
    Clean sea-wise best-lag interval plot.

    Dot = median bootstrap best lag.
    Thick vertical line = interquartile range, 25–75%.
    Sea tick labels are shown in all seasonal panels.
    """
    season_df = summary_df[summary_df["season"] == season_name].copy()

    base_positions = np.arange(len(sea_codes))

    offsets = {
        "POC": -0.23,
        "CHL": 0.00,
        "NPPint": 0.23,
    }

    for var_key in variables:
        color = var_style[var_key]["color"]

        for i, sea in enumerate(sea_codes):
            sub = season_df[
                (season_df["sea"] == sea)
                & (season_df["variable"] == var_key)
            ]

            if sub.empty:
                continue

            row = sub.iloc[0]

            q25 = pd.to_numeric(row.get("q25_boot_lag", np.nan), errors="coerce")
            med = pd.to_numeric(row.get("median_boot_lag", np.nan), errors="coerce")
            q75 = pd.to_numeric(row.get("q75_boot_lag", np.nan), errors="coerce")

            x = base_positions[i] + offsets[var_key]

            # IQR range: thick vertical line
            if np.isfinite(q25) and np.isfinite(q75):
                ax.vlines(
                    x,
                    q25,
                    q75,
                    color=color,
                    linewidth=4.8,
                    alpha=0.52,
                    zorder=3
                )

            # Median dot
            if np.isfinite(med):
                ax.scatter(
                    x,
                    med,
                    s=30,
                    color=color,
                    edgecolor="black",
                    linewidth=0.35,
                    zorder=4
                )

    ax.axhline(
        0,
        color="0.25",
        linestyle="--",
        linewidth=0.8,
        zorder=1
    )

    ax.set_xlim(-0.75, len(sea_codes) - 0.25)

    # Slight breathing space, while keeping actual lag ticks at -3 to +3
    ax.set_ylim(-3.2, 3.2)
    ax.set_yticks(np.arange(-3, 4, 1))

    ax.set_title(
        season_name,
        fontsize=11.4,
        fontweight="bold",
        pad=3
    )

    ax.set_ylabel(
        "Best lag (months)",
        fontsize=9.2,
        fontweight="bold"
    )

    ax.set_xticks(base_positions)

    # Show sea labels in all seasonal panels
    ax.set_xticklabels(
        sea_codes,
        rotation=35,
        ha="right",
        fontweight="bold",
        fontsize=8.0
    )

    # Only bottom row gets x-axis title
    if show_xlabel:
        ax.set_xlabel(
            "Antarctic Shelf Seas",
            fontsize=9.4,
            fontweight="bold"
        )
    else:
        ax.set_xlabel("")

    ax.grid(
        True,
        axis="both",
        linestyle=":",
        linewidth=0.55,
        color="0.72",
        alpha=0.85
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        labelsize=8.0,
        width=0.8,
        length=3.5
    )

    style_spines(ax)
    panel_label(ax, panel_labels[season_name], x=-0.07, y=1.04)


# 7) Create figure
fig = plt.figure(figsize=(15.2, 9.5))

gs = fig.add_gridspec(
    nrows=4,
    ncols=2,
    height_ratios=[1.0, 1.0, 1.0, 0.18],
    width_ratios=[0.80, 2.10],
    hspace=0.56,
    wspace=0.18
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

ax_e = fig.add_subplot(gs[2, 0])
ax_f = fig.add_subplot(gs[2, 1])

ax_leg = fig.add_subplot(gs[3, :])
ax_leg.axis("off")


# 8) Plot panels
plot_lag_curve(ax_a, "POC")
plot_lag_curve(ax_c, "CHL")
plot_lag_curve(ax_e, "NPPint")

interval_plot_for_season(ax_b, "Spring", show_xlabel=False)
interval_plot_for_season(ax_d, "Summer", show_xlabel=False)
interval_plot_for_season(ax_f, "Early autumn", show_xlabel=True)


# 9) Legend
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color=var_style["POC"]["color"],
        markerfacecolor=var_style["POC"]["color"],
        markeredgecolor="black",
        markersize=6,
        linewidth=4,
        alpha=0.65,
        label="POC"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color=var_style["CHL"]["color"],
        markerfacecolor=var_style["CHL"]["color"],
        markeredgecolor="black",
        markersize=6,
        linewidth=4,
        alpha=0.65,
        label="Chlorophyll-a"
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color=var_style["NPPint"]["color"],
        markerfacecolor=var_style["NPPint"]["color"],
        markeredgecolor="black",
        markersize=6,
        linewidth=4,
        alpha=0.65,
        label=r"NPP$_{int}$"
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Zero lag / zero correlation"
    ),
    Line2D(
        [0],
        [0],
        color="0.35",
        linewidth=4,
        alpha=0.55,
        label="Thick range = 25–75%; dot = median"
    ),
]

ax_leg.legend(
    handles=legend_handles,
    loc="center",
    ncol=5,
    frameon=True,
    fontsize=8.4,
    handlelength=2.1,
    columnspacing=1.4,
    borderpad=0.7
)


# 11) Save
plt.savefig(
    OUT_FIG,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    pad_inches=0.12
)

plt.show()

print("\nSaved final Fig. 25 at 1080 dpi:")
print(OUT_FIG)
