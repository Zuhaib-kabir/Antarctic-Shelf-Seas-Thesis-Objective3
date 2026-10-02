# Seasonal PFT community composition
# Left  column = stacked bar plots
# Right column = heatmap plots

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 1) Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# 2) File paths
out_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"

rel_csv = os.path.join(out_dir, "Obj2Fig21_PFT_relative_contribution.csv")

OUT_PATH = os.path.join(out_dir, "Obj2Fig21_SO.png")

SAVE_DPI = 1080

if not os.path.exists(rel_csv):
    raise FileNotFoundError(f"CSV not found: {rel_csv}")

rel_df = pd.read_csv(rel_csv)

print("CSV loaded:")
print(rel_csv)
print(rel_df.head())


# 3) Figure settings
sea_codes = [
    "WED", "KHV", "RLS", "LAZ", "COS", "COO", "DAV",
    "MAW", "DUR", "SOM", "ROS", "AMU", "BEL"
]

SEASON_ORDER = ["Spring", "Summer", "Early autumn"]

PFT_ORDER = ["DIATO", "HAPTO", "PICO"]

PFT_LABELS = {
    "DIATO": "Diatom",
    "HAPTO": "Haptophyte",
    "PICO":  "Picophytoplankton",
}

PFT_COLORS = {
    "DIATO": "#00864B",
    "HAPTO": "#FF8500",
    "PICO":  "#5E3C99",
}


# 4) Convert CSV to plotting dictionary
rel = {}

for pft in PFT_ORDER:
    temp = rel_df[rel_df["PFT"] == pft].copy()
    temp = temp.set_index("Sea")
    temp = temp.loc[sea_codes, SEASON_ORDER]
    rel[pft] = temp.astype(float)


# 5) Plot style
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": SAVE_DPI,
})


# 6) Heatmap colormaps
cmap_diato = LinearSegmentedColormap.from_list(
    "diatom_cmap",
    ["#f7fcf5", PFT_COLORS["DIATO"]]
)

cmap_hapto = LinearSegmentedColormap.from_list(
    "hapto_cmap",
    ["#fff5eb", PFT_COLORS["HAPTO"]]
)

cmap_pico = LinearSegmentedColormap.from_list(
    "pico_cmap",
    ["#f2f0f7", PFT_COLORS["PICO"]]
)

HEAT_CMAPS = {
    "DIATO": cmap_diato,
    "HAPTO": cmap_hapto,
    "PICO":  cmap_pico,
}


# 7) Better colorbar limits
def nice_limits(data, step=5):
    """
    Uses the real value range of each PFT heatmap.
    This prevents the heatmap from looking like one flat color.
    """
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    vmin = np.floor(vmin / step) * step
    vmax = np.ceil(vmax / step) * step

    if vmin == vmax:
        vmin = max(0, vmin - step)
        vmax = min(100, vmax + step)

    return max(0, vmin), min(100, vmax)


# 8) Create figure
fig = plt.figure(figsize=(14.5, 9.2))

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=[1.08, 1.00],
    height_ratios=[1, 1, 1],
    left=0.08,
    right=0.94,
    top=0.91,
    bottom=0.10,
    wspace=0.34,
    hspace=0.42
)

axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

panel_letters = {
    (0, 0): "(a)", (0, 1): "(b)",
    (1, 0): "(c)", (1, 1): "(d)",
    (2, 0): "(e)", (2, 1): "(f)",
}


# 9) Column headings
fig.text(
    0.28,
    0.965,
    "Stacked bar plots",
    ha="center",
    va="center",
    fontsize=19,
    fontweight="bold"
)

fig.text(
    0.735,
    0.965,
    "Heatmap plots",
    ha="center",
    va="center",
    fontsize=19,
    fontweight="bold"
)

# Note:
# The subtitle underline lines are intentionally removed
# for a cleaner thesis-style figure.


# 10) Left column: stacked bar plots
for row, season_name in enumerate(SEASON_ORDER):

    ax = axes[row][0]

    y = np.arange(len(sea_codes))
    left = np.zeros(len(sea_codes))

    for pft in PFT_ORDER:
        vals = rel[pft].loc[sea_codes, season_name].values.astype(float)

        ax.barh(
            y,
            vals,
            left=left,
            height=0.72,
            color=PFT_COLORS[pft],
            edgecolor="white",
            linewidth=0.45,
            label=PFT_LABELS[pft]
        )

        left += np.nan_to_num(vals, nan=0.0)

    ax.set_yticks(y)
    ax.set_yticklabels(sea_codes)
    ax.invert_yaxis()

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))

    ax.set_title(season_name, pad=7, fontweight="bold")
    ax.set_xlabel("Relative contribution (%)")

    # Required y-axis label
    ax.set_ylabel("Antarctic Shelf Seas", labelpad=12)

    ax.grid(axis="x", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel label outside subplot
    ax.text(
        -0.22,
        1.06,
        panel_letters[(row, 0)],
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom"
    )


# 11) Right column: heatmap plots
for row, pft in enumerate(PFT_ORDER):

    ax = axes[row][1]

    data = rel[pft].loc[sea_codes, SEASON_ORDER].astype(float).values

    # Improved cbar range based on actual data
    vmin, vmax = nice_limits(data, step=5)

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=HEAT_CMAPS[pft],
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(
        f"{PFT_LABELS[pft]} contribution (%)",
        pad=7,
        fontweight="bold"
    )

    ax.set_xticks(np.arange(len(SEASON_ORDER)))
    ax.set_xticklabels(SEASON_ORDER)

    ax.set_yticks(np.arange(len(sea_codes)))
    ax.set_yticklabels(sea_codes)

    # White grid lines inside heatmap
    ax.set_xticks(np.arange(-0.5, len(SEASON_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sea_codes), 1), minor=True)

    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add values inside heatmap cells
    for i in range(len(sea_codes)):
        for j in range(len(SEASON_ORDER)):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8.5
                )

    # Panel label outside subplot
    ax.text(
        -0.25,
        1.06,
        panel_letters[(row, 1)],
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    # Individual colorbar
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.045,
        pad=0.04
    )

    cbar.set_label("%")

    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.0f}" for t in ticks])


# 12) Bottom legend only
handles = [
    mpl.patches.Patch(
        color=PFT_COLORS[pft],
        label=PFT_LABELS[pft]
    )
    for pft in PFT_ORDER
]

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.035),
    columnspacing=2.0,
    handlelength=2.5
)


# 13) Save figure at 1080 DPI
plt.savefig(
    OUT_PATH,
    dpi=1080,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nSaved figure:")
print(OUT_PATH)
