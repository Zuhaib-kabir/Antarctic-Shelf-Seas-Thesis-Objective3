# Figure:
# Sea-ice retreat and particulate carbon response
# across Antarctic marginal seas


# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 0) Install
!pip -q install adjustText

# 1) Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from adjustText import adjust_text


# 2) Input / output paths
csv_file = "/content/drive/MyDrive/SAM_Thesis/Processed/Fig22_seawise_metrics.csv"

out_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(out_dir, exist_ok=True)

OUT_FIG = os.path.join(
    out_dir,
    "Fig22_SeaIce_Carbon_Bloom_Response_FINAL_1080dpi.png"
)

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file not found: {csv_file}")

print("Using CSV:")
print(csv_file)


# 3) Load metrics
metrics = pd.read_csv(csv_file)

required_cols = [
    "sea",
    "sic_active_mean_percent",
    "poc_active_mean",
    "pic_active_mean",
    "retreat_day_mean",
    "bloom_day_mean",
    "lag_days_mean",
]

for c in required_cols:
    if c not in metrics.columns:
        raise ValueError(f"Missing column in CSV: {c}")

sea_order = [
    "WED", "KHV", "RLS", "LAZ", "COS", "COO", "DAV",
    "MAW", "DUR", "SOM", "ROS", "AMU", "BEL"
]

metrics["sea"] = pd.Categorical(
    metrics["sea"],
    categories=sea_order,
    ordered=True
)

metrics = metrics.sort_values("sea").reset_index(drop=True)

print("\nLoaded metrics:")
print(metrics)


# 4) Settings
SAVE_DPI = 1080

# 1080 dpi with a smaller figure size avoids huge memory use.
# If Colab crashes during save, change SAVE_DPI to 600.
FIGSIZE = (11.4, 8.4)

sea_colors = {
    "WED": "#1f77b4",
    "KHV": "#ff7f0e",
    "RLS": "#2ca02c",
    "LAZ": "#9467bd",
    "COS": "#d62728",
    "COO": "#8c564b",
    "DAV": "#e377c2",
    "MAW": "#7f7f7f",
    "DUR": "#bcbd22",
    "SOM": "#17becf",
    "ROS": "#001f77",
    "AMU": "#ff7f0e",
    "BEL": "#008000",
}

legend_items = [
    ("WED", "Weddell Sea"),
    ("KHV", "King Haakon VII Sea"),
    ("RLS", "Riiser-Larsen Sea"),
    ("LAZ", "Lazarev Sea"),
    ("COS", "Cosmonauts Sea"),
    ("COO", "Cooperation Sea"),
    ("DAV", "Davis Sea"),
    ("MAW", "Mawson Sea"),
    ("DUR", "D'Urville Sea"),
    ("SOM", "Somov Sea"),
    ("ROS", "Ross Sea"),
    ("AMU", "Amundsen Sea"),
    ("BEL", "Bellingshausen Sea"),
]


# 5) Unit and timing preparation
# POC remains mg C m^-3
metrics["poc_plot"] = metrics["poc_active_mean"]

# PIC is very small in mg C m^-3, so plot in µg C m^-3
metrics["pic_plot"] = metrics["pic_active_mean"] * 1000.0


def doy_to_seasonal_day_since_sep1(doy):
    """
    Convert day-of-year to days since 1 September.
    This keeps Antarctic spring-summer bloom timing continuous.

    Sep 1 ≈ day 244.
    Sep 1 = 0.
    Jan/Feb/Mar/Apr are shifted after December.
    """
    doy = np.asarray(doy, dtype=float)

    # Convert values >365 to normal calendar day
    d = np.where(doy > 365, doy - 365, doy)

    sep1 = 244.0

    seasonal_day = np.where(
        d >= sep1,
        d - sep1,
        d + (365.0 - sep1)
    )

    return seasonal_day


metrics["retreat_season_day"] = doy_to_seasonal_day_since_sep1(
    metrics["retreat_day_mean"]
)

metrics["bloom_season_day"] = doy_to_seasonal_day_since_sep1(
    metrics["bloom_day_mean"]
)

# If bloom appears before retreat due to year crossing, shift bloom forward.
mask = (
    np.isfinite(metrics["retreat_season_day"])
    & np.isfinite(metrics["bloom_season_day"])
    & (metrics["bloom_season_day"] < metrics["retreat_season_day"])
)

metrics.loc[mask, "bloom_season_day"] = (
    metrics.loc[mask, "bloom_season_day"] + 365.0
)


# 6) Helper functions
def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return None

    slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])

    return slope, intercept, r, p, mask.sum()


def add_fit(ax, x, y, color="black", lw=1.4):
    res = fit_line(x, y)

    if res is None:
        return None

    slope, intercept, r, p, n = res

    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)

    xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
    yfit = slope * xfit + intercept

    ax.plot(xfit, yfit, color=color, lw=lw, zorder=2)

    return r, p, n


def format_stat(stat):
    if stat is None:
        return "n/a"

    r, p, n = stat

    if p < 0.001:
        return f"r = {r:.2f}, p < 0.001, n = {n}"
    else:
        return f"r = {r:.2f}, p = {p:.3f}, n = {n}"


def style_ax(ax):
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.45,
        color="0.86",
        alpha=0.95
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        labelsize=8.5,
        width=0.8,
        length=3.5
    )

    for sp in ax.spines.values():
        sp.set_linewidth(0.85)


def panel_label(ax, label):
    ax.text(
        -0.115,
        1.065,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold"
    )


def scatter_seas(ax, x_col, y_col):
    for _, row in metrics.iterrows():
        sea = str(row["sea"])
        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        ax.scatter(
            x,
            y,
            s=36,
            color=sea_colors[sea],
            edgecolor="black",
            linewidth=0.35,
            zorder=3
        )


def add_labels_adjusted(ax, x_col, y_col, fontsize=6.8):
    texts = []

    for _, row in metrics.iterrows():
        sea = str(row["sea"])
        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        texts.append(
            ax.text(
                x,
                y,
                sea,
                fontsize=fontsize,
                fontweight="bold",
                zorder=4
            )
        )

    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.35, 1.45),
        expand_text=(1.20, 1.30),
        force_points=0.30,
        force_text=0.45,
        lim=300,
        arrowprops=dict(
            arrowstyle="-",
            color="0.35",
            lw=0.35,
            alpha=0.65
        )
    )

    return texts


def add_stat_box(ax, stat):
    ax.text(
        0.975,
        0.955,
        format_stat(stat),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.65,
            alpha=0.95
        )
    )


def add_note_box(ax, text, xy, ha="left"):
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        fontsize=7.8,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.25",
            linewidth=0.65,
            alpha=0.95
        )
    )


def set_padded_ylim(ax, values, lower_zero=False, pad_frac=0.18):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return

    vmin = vals.min()
    vmax = vals.max()
    pad = (vmax - vmin) * pad_frac

    if pad == 0:
        pad = max(abs(vmax) * 0.1, 1)

    ymin = vmin - pad
    ymax = vmax + pad

    if lower_zero:
        ymin = max(0, ymin)

    ax.set_ylim(ymin, ymax)


# 7) Make figure
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.85,
    "savefig.dpi": SAVE_DPI,
})

fig = plt.figure(figsize=FIGSIZE)

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    height_ratios=[1.0, 1.0, 0.26],
    hspace=0.42,
    wspace=0.24
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
ax_leg = fig.add_subplot(gs[2, :])
ax_leg.axis("off")


# Panel (a): POC vs SIC
scatter_seas(ax_a, "sic_active_mean_percent", "poc_plot")

stat_a = add_fit(
    ax_a,
    metrics["sic_active_mean_percent"].values,
    metrics["poc_plot"].values
)

ax_a.set_title("POC vs SIC", fontsize=12, fontweight="bold", pad=5)
ax_a.set_xlabel("Sea-ice concentration, SIC (%)", fontsize=9.5, fontweight="bold")
ax_a.set_ylabel(r"POC (mg C m$^{-3}$)", fontsize=9.5, fontweight="bold")

ax_a.set_xlim(0, 100)
set_padded_ylim(ax_a, metrics["poc_plot"].values, lower_zero=False, pad_frac=0.20)

add_labels_adjusted(ax_a, "sic_active_mean_percent", "poc_plot", fontsize=6.8)

# NOTE BOX MOVED LITTLE LEFT FROM EDGE
add_note_box(
    ax_a,
    "Lower SIC = post-melt condition\nHigher POC after melt",
    (0.50, 0.06),
    ha="left"
)

add_stat_box(ax_a, stat_a)

style_ax(ax_a)
panel_label(ax_a, "a")


# Panel (b): PIC vs SIC
scatter_seas(ax_b, "sic_active_mean_percent", "pic_plot")

stat_b = add_fit(
    ax_b,
    metrics["sic_active_mean_percent"].values,
    metrics["pic_plot"].values
)

ax_b.set_title("PIC vs SIC", fontsize=12, fontweight="bold", pad=5)
ax_b.set_xlabel("Sea-ice concentration, SIC (%)", fontsize=9.5, fontweight="bold")
ax_b.set_ylabel(r"PIC ($\mu$g C m$^{-3}$)", fontsize=9.5, fontweight="bold")

ax_b.set_xlim(0, 100)
set_padded_ylim(ax_b, metrics["pic_plot"].values, lower_zero=False, pad_frac=0.24)

add_labels_adjusted(ax_b, "sic_active_mean_percent", "pic_plot", fontsize=6.8)

add_note_box(
    ax_b,
    "Weaker PIC response",
    (0.62, 0.06),
    ha="left"
)

add_stat_box(ax_b, stat_b)

style_ax(ax_b)
panel_label(ax_b, "b")


# Panel (c): SIC retreat vs bloom timing
scatter_seas(ax_c, "retreat_season_day", "bloom_season_day")

stat_c = add_fit(
    ax_c,
    metrics["retreat_season_day"].values,
    metrics["bloom_season_day"].values
)

valid = (
    np.isfinite(metrics["retreat_season_day"].values)
    & np.isfinite(metrics["bloom_season_day"].values)
)

if valid.sum() > 0:
    x_min = np.nanmin(metrics.loc[valid, "retreat_season_day"])
    x_max = np.nanmax(metrics.loc[valid, "retreat_season_day"])
    y_min = np.nanmin(metrics.loc[valid, "bloom_season_day"])
    y_max = np.nanmax(metrics.loc[valid, "bloom_season_day"])

    line_min = max(0, min(x_min, y_min) - 8)
    line_max = max(x_max, y_max) + 10
else:
    line_min, line_max = 0, 180

ax_c.plot(
    [line_min, line_max],
    [line_min, line_max],
    linestyle=(0, (5, 4)),
    color="black",
    linewidth=0.95,
    zorder=1
)

ax_c.text(
    line_max - 20,
    line_max - 15,
    "No lag",
    fontsize=7.6,
    fontstyle="italic"
)

ax_c.set_title("SIC retreat vs bloom timing", fontsize=12, fontweight="bold", pad=5)
ax_c.set_xlabel(
    "Ice-edge opening / SIC retreat timing\n(days since 1 September)",
    fontsize=9.5,
    fontweight="bold"
)
ax_c.set_ylabel(
    "Bloom peak timing\n(days since 1 September)",
    fontsize=9.5,
    fontweight="bold"
)

ax_c.set_xlim(line_min, line_max)
ax_c.set_ylim(line_min, line_max)

add_labels_adjusted(ax_c, "retreat_season_day", "bloom_season_day", fontsize=6.8)

add_note_box(
    ax_c,
    "Above dashed line =\ndelayed bloom",
    (0.63, 0.08)
)

add_stat_box(ax_c, stat_c)

style_ax(ax_c)
panel_label(ax_c, "c")


# Panel (d): Bloom lag by sea
df_lag = metrics.copy()
df_lag = df_lag[np.isfinite(df_lag["lag_days_mean"])]
df_lag = df_lag.sort_values("lag_days_mean", ascending=False)

ypos = np.arange(len(df_lag))

for i, (_, row) in enumerate(df_lag.iterrows()):
    sea = str(row["sea"])
    lag = row["lag_days_mean"]

    ax_d.hlines(
        y=i,
        xmin=0,
        xmax=lag,
        color="0.25",
        linewidth=0.75,
        zorder=1
    )

    ax_d.scatter(
        lag,
        i,
        s=36,
        color=sea_colors[sea],
        edgecolor="black",
        linewidth=0.35,
        zorder=3
    )

ax_d.set_yticks(ypos)
ax_d.set_yticklabels(df_lag["sea"].values, fontsize=8.5, fontweight="bold")
ax_d.invert_yaxis()

ax_d.set_title("Bloom lag by sea", fontsize=12, fontweight="bold", pad=5)
ax_d.set_xlabel("Lag after retreat (days)", fontsize=9.5, fontweight="bold")

if len(df_lag) > 0:
    max_lag = np.nanmax(df_lag["lag_days_mean"].values)
else:
    max_lag = 30

ax_d.set_xlim(0, max(35, max_lag + 5))

style_ax(ax_d)
panel_label(ax_d, "d")


# 8) Legend / abbreviation key
ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, 1)

ax_leg.plot([0.00, 1.00], [0.96, 0.96], color="0.55", lw=0.65)

ax_leg.text(
    0.00,
    0.80,
    "Sea abbreviations:",
    fontsize=9.2,
    fontweight="bold",
    ha="left",
    va="center"
)

cols = 5
rows = int(np.ceil(len(legend_items) / cols))

x_positions = np.linspace(0.035, 0.855, cols)
y_positions = [0.55, 0.31, 0.08]

for idx, (code, name) in enumerate(legend_items):
    col = idx // rows
    row = idx % rows

    x0 = x_positions[col]
    y0 = y_positions[row]

    ax_leg.scatter(
        x0,
        y0,
        s=42,
        color=sea_colors[code],
        edgecolor="black",
        linewidth=0.28
    )

    ax_leg.text(
        x0 + 0.016,
        y0,
        f"{code} – {name}",
        fontsize=7.7,
        ha="left",
        va="center"
    )


# 9) Main title and save

plt.savefig(
    OUT_FIG,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    pad_inches=0.12
)

plt.show()

print("\nSaved final 1080 dpi figure:")
print(OUT_FIG)
