# Stratification and biological carbon pump relationships
# layout:
# (a) POC vs MLD
# (b) Chlorophyll-a vs MLD
# (c) Stratification vs productivity / bloom proxy
# (d) Conceptual biological carbon pump pathway


# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# 0) Install
!pip -q install pandas numpy matplotlib scipy adjustText


# 1) Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from adjustText import adjust_text

from matplotlib.patches import (
    Rectangle,
    FancyBboxPatch,
    Circle,
    FancyArrowPatch
)


# 2) Input / output paths
summary_csv = "/content/drive/MyDrive/SAM_Thesis/Processed/Fig24_BCP_seawise_summary_ACTIVE_LOW_RAM.csv"

out_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(out_dir, exist_ok=True)

OUT_FIG = os.path.join(
    out_dir,
    "Fig24_Stratification_BCP_DUMMY_STYLE_FINAL_1080dpi.png"
)

if not os.path.exists(summary_csv):
    raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

print("Using summary CSV:")
print(summary_csv)


# 3) Load data
df = pd.read_csv(summary_csv)

required_cols = [
    "sea",
    "poc_active_mean",
    "chl_active_mean",
    "mld_active_mean",
    "strat_active_mean",
]

for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

sea_order = [
    "WED", "KHV", "RLS", "LAZ", "COS", "COO", "DAV",
    "MAW", "DUR", "SOM", "ROS", "AMU", "BEL"
]

df["sea"] = pd.Categorical(
    df["sea"],
    categories=sea_order,
    ordered=True
)

df = df.sort_values("sea").reset_index(drop=True)

print("\nLoaded summary data:")
print(df)


# 4) Colors and sea names
sea_colors = {
    "WED": "#1f77b4",
    "KHV": "#ff7f0e",
    "RLS": "#2ca02c",
    "LAZ": "#d62728",
    "COS": "#9467bd",
    "COO": "#8c564b",
    "DAV": "#e377c2",
    "MAW": "#7f7f7f",
    "DUR": "#bcbd22",
    "SOM": "#17becf",
    "ROS": "#006d6f",
    "AMU": "#f39c12",
    "BEL": "#0b4f8a",
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


# 5) Scatter plot helpers
def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return None

    slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
    return slope, intercept, r, p, mask.sum()


def add_regression(ax, x, y):
    result = fit_line(x, y)

    if result is None:
        return None

    slope, intercept, r, p, n = result

    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)

    xfit = np.linspace(
        np.nanmin(x[mask]),
        np.nanmax(x[mask]),
        100
    )

    yfit = slope * xfit + intercept

    ax.plot(
        xfit,
        yfit,
        color="black",
        linewidth=1.25,
        zorder=2
    )

    return r, p, n


def stat_text(stat):
    if stat is None:
        return "n/a"

    r, p, n = stat

    if p < 0.001:
        return f"r = {r:.2f}, p < 0.001, n = {n}"
    else:
        return f"r = {r:.2f}, p = {p:.3f}, n = {n}"


def add_stat_box(ax, stat):
    ax.text(
        0.97,
        0.95,
        stat_text(stat),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="0.45",
            linewidth=0.55,
            alpha=0.95
        ),
        zorder=8
    )


def style_ax(ax):
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
        labelsize=8.1,
        width=0.8,
        length=3.5
    )

    for sp in ax.spines.values():
        sp.set_linewidth(0.85)


def panel_label(ax, label):
    ax.text(
        -0.13,
        1.07,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        fontweight="bold"
    )


def scatter_seas(ax, x_col, y_col):
    for _, row in df.iterrows():
        sea = str(row["sea"])
        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        ax.scatter(
            x,
            y,
            s=42,
            color=sea_colors[sea],
            edgecolor="black",
            linewidth=0.35,
            zorder=4
        )


def add_labels_adjusted(ax, x_col, y_col, fontsize=6.9):
    texts = []

    for _, row in df.iterrows():
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
                color="black",
                zorder=5
            )
        )

    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.55, 1.70),
        expand_text=(1.35, 1.55),
        force_points=0.35,
        force_text=0.55,
        lim=400,
        arrowprops=dict(
            arrowstyle="-",
            color="0.35",
            lw=0.35,
            alpha=0.65
        )
    )

    return texts


def padded_limits(values, pad_frac=0.18, lower_zero=False):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return 0, 1

    vmin = vals.min()
    vmax = vals.max()

    pad = (vmax - vmin) * pad_frac

    if pad == 0:
        pad = max(abs(vmax) * 0.1, 1)

    ymin = vmin - pad
    ymax = vmax + pad

    if lower_zero:
        ymin = max(0, ymin)

    return ymin, ymax


def add_note_box(ax, text, xy, ha="center"):
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va="center",
        fontsize=7.5,
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="0.35",
            linewidth=0.65,
            alpha=0.96
        ),
        zorder=8
    )


# 6) Conceptual pathway helpers
def draw_arrow(ax, x1, y1, x2, y2, color):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.35,
        color=color
    )
    ax.add_patch(arr)


def rounded_box(ax, x, y, w, h, text, face, edge, fontsize=6.4):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.016",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.85
    )

    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        linespacing=1.0
    )


def water_column(ax, x, y, w, h, shallow=True):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="#d9eef8",
            edgecolor="0.35",
            linewidth=0.7
        )
    )

    xs = np.linspace(x, x + w, 80)
    ys = y + h - 0.018 + 0.0045 * np.sin(np.linspace(0, 6 * np.pi, 80))

    ax.plot(xs, ys, color="#1f77b4", lw=0.8)

    ml_y = y + h * 0.70 if shallow else y + h * 0.35

    ax.plot(
        [x, x + w],
        [ml_y, ml_y],
        color="#1f77b4",
        lw=0.9,
        linestyle="--"
    )

    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            0.012,
            facecolor="#8c6d31",
            edgecolor="none",
            alpha=0.65
        )
    )

    if not shallow:
        circ1 = FancyArrowPatch(
            (x + w * 0.35, y + h * 0.60),
            (x + w * 0.67, y + h * 0.60),
            connectionstyle="arc3,rad=0.65",
            arrowstyle="-|>",
            mutation_scale=9,
            color="white",
            lw=1.0
        )

        circ2 = FancyArrowPatch(
            (x + w * 0.67, y + h * 0.42),
            (x + w * 0.35, y + h * 0.42),
            connectionstyle="arc3,rad=0.65",
            arrowstyle="-|>",
            mutation_scale=9,
            color="white",
            lw=1.0
        )

        ax.add_patch(circ1)
        ax.add_patch(circ2)


def sun_icon(ax, x, y):
    ax.add_patch(
        Circle(
            (x, y),
            0.025,
            facecolor="#ffc928",
            edgecolor="#c28700",
            linewidth=0.6
        )
    )

    for ang in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        ax.plot(
            [x + 0.034 * np.cos(ang), x + 0.052 * np.cos(ang)],
            [y + 0.034 * np.sin(ang), y + 0.052 * np.sin(ang)],
            color="#f4b400",
            lw=0.75
        )


def cloud_icon(ax, x, y):
    ax.add_patch(Circle((x - 0.020, y), 0.023, facecolor="0.75", edgecolor="0.45", lw=0.6))
    ax.add_patch(Circle((x + 0.010, y + 0.012), 0.030, facecolor="0.75", edgecolor="0.45", lw=0.6))
    ax.add_patch(Circle((x + 0.040, y), 0.021, facecolor="0.75", edgecolor="0.45", lw=0.6))

    for dx in [-0.02, 0.01, 0.04]:
        ax.plot(
            [x + dx, x + dx - 0.01],
            [y - 0.045, y - 0.078],
            color="0.5",
            lw=0.75
        )


def bloom_icon(ax, x, y, dense=True):
    rng = np.random.default_rng(7 if dense else 3)
    n = 14 if dense else 5

    for _ in range(n):
        dx = rng.uniform(-0.050, 0.050)
        dy = rng.uniform(-0.036, 0.036)

        ax.add_patch(
            Circle(
                (x + dx, y + dy),
                0.0075 if dense else 0.006,
                facecolor="#8fd19e",
                edgecolor="#2ca25f",
                linewidth=0.55
            )
        )


def carbon_export_icon(ax, x, y, strong=True):
    rng = np.random.default_rng(12 if strong else 5)
    n = 20 if strong else 7

    for _ in range(n):
        dx = rng.uniform(-0.048, 0.048)
        dy = rng.uniform(0.000, 0.072)

        ax.add_patch(
            Circle(
                (x + dx, y + dy),
                0.0045,
                facecolor="#7a4a21",
                edgecolor="none",
                alpha=0.9
            )
        )

    ax.annotate(
        "",
        xy=(x, y - 0.035),
        xytext=(x, y + 0.062),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#7a4a21",
            lw=1.5 if strong else 1.0
        )
    )


def draw_conceptual_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title(
        "Conceptual biological carbon pump pathway",
        fontsize=11.3,
        fontweight="bold",
        pad=5
    )

    panel_label(ax, "d")

    headings = [
        (0.13, "MLD & stratification"),
        (0.37, "Light environment"),
        (0.60, "Bloom response"),
        (0.82, "Carbon export"),
    ]

    for x, text in headings:
        ax.text(
            x,
            0.925,
            text,
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold"
        )

    # Big row backgrounds
    ax.add_patch(
        FancyBboxPatch(
            (-0.020, 0.480),
            1.040,
            0.385,
            boxstyle="round,pad=0.014,rounding_size=0.024",
            facecolor="#f2f8ec",
            edgecolor="#3b7f2a",
            linewidth=0.85
        )
    )

    ax.add_patch(
        FancyBboxPatch(
            (-0.020, 0.045),
            1.040,
            0.385,
            boxstyle="round,pad=0.014,rounding_size=0.024",
            facecolor="#edf4fb",
            edgecolor="#2b6ca3",
            linewidth=0.85
        )
    )

    # Top row

    rounded_box(
        ax,
        0.020,
        0.575,
        0.145,
        0.190,
        "Shallow MLD\n+\nstrong\nstratification",
        "#e5f5df",
        "#3b7f2a",
        fontsize=6.45
    )

    draw_arrow(ax, 0.185, 0.670, 0.240, 0.670, "#3b7f2a")
    water_column(ax, 0.255, 0.565, 0.115, 0.215, shallow=True)

    draw_arrow(ax, 0.388, 0.670, 0.445, 0.670, "#3b7f2a")
    sun_icon(ax, 0.490, 0.695)
    ax.text(0.490, 0.565, "High light\nexposure", ha="center", va="center",fontweight="bold", fontsize=6)

    draw_arrow(ax, 0.540, 0.670, 0.595, 0.670, "#3b7f2a")
    bloom_icon(ax, 0.650, 0.695, dense=True)
    ax.text(0.650, 0.565, "Bloom\nenhancement", ha="center", va="center",fontweight="bold", fontsize=6)

    draw_arrow(ax, 0.715, 0.670, 0.765, 0.670, "#3b7f2a")
    carbon_export_icon(ax, 0.805, 0.695, strong=True)
    ax.text(0.805, 0.555, "Higher POC /\nstronger export", ha="center", va="center",fontweight="bold", fontsize=6)

    draw_arrow(ax, 0.855, 0.670, 0.890, 0.670, "#3b7f2a")

    # Wider and longer final box
    rounded_box(
        ax,
        0.895,
        0.585,
        0.115,
        0.180,
        "Strong\ncarbon\npump",
        "#e5f5df",
        "#3b7f2a",
        fontsize=6.85
    )

    # Bottom row
    rounded_box(
        ax,
        0.020,
        0.145,
        0.145,
        0.190,
        "Deep MLD\n+\nweak\nstratification",
        "#e2eff9",
        "#2b6ca3",
        fontsize=6.45
    )

    draw_arrow(ax, 0.185, 0.240, 0.240, 0.240, "#2b6ca3")
    water_column(ax, 0.255, 0.135, 0.115, 0.215, shallow=False)

    draw_arrow(ax, 0.388, 0.240, 0.445, 0.240, "#2b6ca3")
    cloud_icon(ax, 0.490, 0.270)
    ax.text(0.490, 0.135, "Low light /\ndilution", ha="center", va="center",fontweight="bold", fontsize=6)

    draw_arrow(ax, 0.540, 0.240, 0.595, 0.240, "#2b6ca3")
    bloom_icon(ax, 0.650, 0.270, dense=False)
    ax.text(0.650, 0.135, "Reduced\nbloom", ha="center", va="center",fontweight="bold",  fontsize=6)

    draw_arrow(ax, 0.715, 0.240, 0.765, 0.240, "#2b6ca3")
    carbon_export_icon(ax, 0.805, 0.270, strong=False)
    ax.text(0.805, 0.125, "Reduced POC /\nweaker export", ha="center", va="center", fontweight="bold", fontsize=6)

    draw_arrow(ax, 0.855, 0.240, 0.890, 0.240, "#2b6ca3")

    # Wider and longer final box
    rounded_box(
        ax,
        0.895,
        0.155,
        0.115,
        0.180,
        "Weak\ncarbon\npump",
        "#e2eff9",
        "#2b6ca3",
        fontsize=6.85
    )


# 7) Figure setup
SAVE_DPI = 1080

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.85,
    "savefig.dpi": SAVE_DPI,
})

# Wider figure to give panel (d) more horizontal length
fig = plt.figure(figsize=(14.1, 9.1))

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    height_ratios=[1.0, 1.05, 0.40],
    width_ratios=[1.00, 1.10],
    hspace=0.42,
    wspace=0.24
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
ax_leg = fig.add_subplot(gs[2, :])
ax_leg.axis("off")


# 8) Panel (a): POC vs MLD
scatter_seas(ax_a, "mld_active_mean", "poc_active_mean")

stat_a = add_regression(
    ax_a,
    df["mld_active_mean"].values,
    df["poc_active_mean"].values
)

add_labels_adjusted(ax_a, "mld_active_mean", "poc_active_mean", fontsize=6.9)

ax_a.set_title("POC vs MLD", fontsize=12, fontweight="bold", pad=5)
ax_a.set_xlabel("Mixed-layer depth, MLD (m)", fontsize=9.5, fontweight="bold")
ax_a.set_ylabel(r"POC (mg C m$^{-3}$)", fontsize=9.5, fontweight="bold")

xmin, xmax = padded_limits(df["mld_active_mean"], pad_frac=0.27, lower_zero=False)
ymin, ymax = padded_limits(df["poc_active_mean"], pad_frac=0.27, lower_zero=False)

ax_a.set_xlim(xmin, xmax)
ax_a.set_ylim(ymin, ymax)

add_note_box(
    ax_a,
    "Shallow MLD favors\nhigher POC accumulation",
    (0.70, 0.82)
)

add_stat_box(ax_a, stat_a)
style_ax(ax_a)
panel_label(ax_a, "a")


# 9) Panel (b): CHL vs MLD
scatter_seas(ax_b, "mld_active_mean", "chl_active_mean")

stat_b = add_regression(
    ax_b,
    df["mld_active_mean"].values,
    df["chl_active_mean"].values
)

add_labels_adjusted(ax_b, "mld_active_mean", "chl_active_mean", fontsize=6.9)

ax_b.set_title("Chlorophyll-a vs MLD", fontsize=12, fontweight="bold", pad=5)
ax_b.set_xlabel("Mixed-layer depth, MLD (m)", fontsize=9.5, fontweight="bold")
ax_b.set_ylabel(r"Chlorophyll-a (mg m$^{-3}$)", fontsize=9.5, fontweight="bold")

xmin, xmax = padded_limits(df["mld_active_mean"], pad_frac=0.27, lower_zero=False)
ymin, ymax = padded_limits(df["chl_active_mean"], pad_frac=0.33, lower_zero=True)

ax_b.set_xlim(xmin, xmax)
ax_b.set_ylim(ymin, ymax)

add_note_box(
    ax_b,
    "Deep mixing suppresses\nbloom biomass",
    (0.83, 0.78)
)

add_stat_box(ax_b, stat_b)
style_ax(ax_b)
panel_label(ax_b, "b")


# 10) Panel (c): Stratification vs productivity / bloom proxy
scatter_seas(ax_c, "strat_active_mean", "chl_active_mean")

stat_c = add_regression(
    ax_c,
    df["strat_active_mean"].values,
    df["chl_active_mean"].values
)

add_labels_adjusted(ax_c, "strat_active_mean", "chl_active_mean", fontsize=6.9)

ax_c.set_title("Stratification vs bloom proxy", fontsize=12, fontweight="bold", pad=5)

ax_c.set_xlabel(
    r"Stratification index, $\Delta\sigma_{\theta}$ (kg m$^{-3}$)",
    fontsize=9.5,
    fontweight="bold"
)

ax_c.set_ylabel(
    r"Bloom proxy , CHL (mg m$^{-3}$)",
    fontsize=9.5,
    fontweight="bold"
)

xmin, xmax = padded_limits(df["strat_active_mean"], pad_frac=0.30, lower_zero=False)
ymin, ymax = padded_limits(df["chl_active_mean"], pad_frac=0.33, lower_zero=True)

ax_c.set_xlim(xmin, xmax)
ax_c.set_ylim(ymin, ymax)

add_note_box(
    ax_c,
    "Enhanced stratification\npromotes bloom development",
    (0.22, 0.88)
)

add_stat_box(ax_c, stat_c)
style_ax(ax_c)
panel_label(ax_c, "c")


# 11) Panel (d): Conceptual pathway
draw_conceptual_panel(ax_d)


# 12) Bottom legend band
ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, 1)

legend_box = FancyBboxPatch(
    (0.005, 0.08),
    0.99,
    0.84,
    boxstyle="round,pad=0.006,rounding_size=0.015",
    fill=False,
    edgecolor="black",
    linewidth=0.80
)

ax_leg.add_patch(legend_box)

ax_leg.text(
    0.020,
    0.75,
    "Sea abbreviations:",
    fontsize=9.2,
    fontweight="bold",
    ha="left",
    va="center"
)

positions = [
    (0.18, 0.75), (0.35, 0.75), (0.55, 0.75), (0.72, 0.75), (0.87, 0.75),
    (0.10, 0.47), (0.28, 0.47), (0.45, 0.47), (0.62, 0.47), (0.80, 0.47),
    (0.24, 0.20), (0.43, 0.20), (0.62, 0.20),
]

for (code, name), (x, y) in zip(legend_items, positions):
    ax_leg.scatter(
        x,
        y,
        s=48,
        color=sea_colors[code],
        edgecolor="black",
        linewidth=0.30,
        zorder=3
    )

    ax_leg.text(
        x + 0.018,
        y,
        f"{code} – {name}",
        fontsize=7.45,
        ha="left",
        va="center"
    )


# 13) Main title and save


plt.savefig(
    OUT_FIG,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    pad_inches=0.12
)

plt.show()

print("\nSaved final dummy-style Fig. 24 at 1080 dpi:")
print(OUT_FIG)
