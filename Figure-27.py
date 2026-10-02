# Fig-9
# Seasonal distribution of phytoplankton functional types
#
# Columns:
#   Spring, Summer, Early autumn
#
# Rows:
#   Diatoms
#   Haptophytes
#   Picophytoplankton
#
# Unit:
#   mg Chl m^-3




#Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 0) Install
!pip -q install xarray netCDF4 h5netcdf dask cartopy matplotlib numpy


# 1) Imports
import os
import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# 2) Input / output paths
files = {
    "DIATO": "/content/drive/MyDrive/SAM_Thesis/Data/DIATO_monthly_2008_2025_SO.nc",
    "HAPTO": "/content/drive/MyDrive/SAM_Thesis/Data/HAPTO_monthly_2008_2025_SO.nc",
    "PICO":  "/content/drive/MyDrive/SAM_Thesis/Data/PICO_monthly_2008_2025_SO.nc.nc",
}

# Safety check for accidental ".nc.nc"
if not os.path.exists(files["PICO"]):
    alt_pico = files["PICO"].replace(".nc.nc", ".nc")
    if os.path.exists(alt_pico):
        files["PICO"] = alt_pico

out_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(out_dir, exist_ok=True)

OUT_PATH = os.path.join(out_dir, "Obj2Fig20_SO.png")

for key, f in files.items():
    if not os.path.exists(f):
        raise FileNotFoundError(f"{key} file not found: {f}")

print("All input files found.")
print(files)


# 3) Settings
CLIM_START_YEAR = 2008
CLIM_END_YEAR   = 2025

SEASONS = {
    "Spring": [9, 10, 11],       # SON
    "Summer": [12, 1, 2],        # DJF
    "Early autumn": [3, 4],      # Mar-Apr
}

COLUMN_ORDER = ["Spring", "Summer", "Early autumn"]
ROW_ORDER = ["DIATO", "HAPTO", "PICO"]

ROW_LABELS = {
    "DIATO": "Diatoms",
    "HAPTO": "Haptophytes",
    "PICO": "Picophytoplankton",
}

# Low-RAM plotting coarsening
PLOT_COARSEN = 10

# Unit label
UNIT_LABEL = r"mg Chl m$^{-3}$"

# Colormaps
CMAPS = {
    "DIATO": "PuBuGn",
    "HAPTO": "RdPu",
    "PICO": "YlGn",
}

# Make NaN / no retrieval white
CMAP_OBJECTS = {}
for key, cmap_name in CMAPS.items():
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")
    CMAP_OBJECTS[key] = cmap


# Manual colorbar ranges
MANUAL_RANGES = {
    "DIATO": (0.10, 0.50),   # requested: 0.1 to 0.5
    "HAPTO": (0.01, 0.10),
    "PICO":  (0.01, 0.10),
}

MANUAL_TICKS = {
    "DIATO": np.round(np.arange(0.10, 0.50 + 0.05, 0.05), 2).tolist(),
    "HAPTO": np.round(np.arange(0.01, 0.10 + 0.01, 0.01), 2).tolist(),
    "PICO":  np.round(np.arange(0.01, 0.10 + 0.01, 0.01), 2).tolist(),
}

MANUAL_TICKLABELS = {
    "DIATO": [f"{x:.2f}" for x in MANUAL_TICKS["DIATO"]],
    "HAPTO": [f"{x:.2f}" for x in MANUAL_TICKS["HAPTO"]],
    "PICO":  [f"{x:.2f}" for x in MANUAL_TICKS["PICO"]],
}


# Figure layout
FIGSIZE = (14.5, 12.5)

LEFT   = 0.10
RIGHT  = 0.97
TOP    = 0.94
BOTTOM = 0.05
WSPACE = 0.18
HSPACE = 0.16

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "savefig.dpi": 300,
})


# 4) Helper functions
def standardize_coords(ds):
    rename_dict = {}

    for c in list(ds.coords):
        cl = c.lower()
        if cl in ["latitude", "nav_lat", "y"]:
            rename_dict[c] = "lat"
        elif cl in ["longitude", "nav_lon", "x"]:
            rename_dict[c] = "lon"
        elif cl in ["valid_time", "time_counter"]:
            rename_dict[c] = "time"

    for d in list(ds.dims):
        dl = d.lower()
        if dl in ["latitude", "nav_lat", "y"] and d not in rename_dict:
            rename_dict[d] = "lat"
        elif dl in ["longitude", "nav_lon", "x"] and d not in rename_dict:
            rename_dict[d] = "lon"
        elif dl in ["valid_time", "time_counter"] and d not in rename_dict:
            rename_dict[d] = "time"

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "lon" in ds.coords and float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180)

    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    if "time" in ds.coords:
        ds = ds.sortby("time")

    return ds


def prepare_ds(ds):
    ds = standardize_coords(ds)

    ds = ds.sel(
        time=slice(f"{CLIM_START_YEAR}-01-01", f"{CLIM_END_YEAR}-12-31")
    )

    ds = ds.where(ds["lat"] <= -60, drop=True)

    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    if "time" in ds.coords:
        ds = ds.sortby("time")

    return ds


def find_var(ds, key):
    possible = {
        "DIATO": [
            "DIATO", "Diato", "diato",
            "DIATOM", "diatom", "diatoms", "Diatoms"
        ],
        "HAPTO": [
            "HAPTO", "Hapto", "hapto",
            "HAPTOPHYTE", "haptophyte", "haptophytes", "Haptophytes"
        ],
        "PICO": [
            "PICO", "Pico", "pico",
            "PICOPHYTO", "picophyto",
            "picophytoplankton", "Picophytoplankton"
        ],
    }

    for name in possible[key]:
        if name in ds.data_vars:
            return name

    vars_list = [v for v in ds.data_vars if "time" in ds[v].dims]

    if len(vars_list) == 1:
        print(f"Using only time-dependent variable for {key}: {vars_list[0]}")
        return vars_list[0]

    raise ValueError(
        f"Could not detect variable for {key}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def open_dataset_low_ram(path):
    return xr.open_dataset(
        path,
        decode_times=True,
        engine="h5netcdf",
        chunks={"time": 12}
    )


def squeeze_depth_if_present(da):
    for d in list(da.dims):
        if d.lower() in ["depth", "lev", "level", "deptht"]:
            da = da.isel({d: 0}, drop=True)
    return da


def get_season_year(time_da, months):
    year = time_da.dt.year

    # DJF rule: December belongs to following summer year
    if 12 in months and 1 in months:
        year = xr.where(time_da.dt.month == 12, year + 1, year)

    return year.rename("season_year")


def seasonal_mean_by_year(da, months, min_months=2):
    da_s = da.where(da["time"].dt.month.isin(months), drop=True)

    if da_s.sizes.get("time", 0) == 0:
        return None

    sy = get_season_year(da_s["time"], months)
    da_s = da_s.assign_coords(season_year=sy)

    yearly = da_s.groupby("season_year").mean("time", skipna=True)

    years = sy.values
    valid_years = []

    for y in np.unique(years):
        n_months = np.sum(years == y)
        if n_months >= min_months:
            valid_years.append(int(y))

    if len(valid_years) == 0:
        return None

    yearly = yearly.sel(season_year=valid_years)

    return yearly


def seasonal_climatology(da, months, season_name):
    if season_name == "Early autumn":
        min_months = 1
    else:
        min_months = 2

    yearly = seasonal_mean_by_year(da, months, min_months=min_months)

    if yearly is None:
        raise ValueError(f"No valid seasonal data found for {season_name}")

    clim = yearly.sel(
        season_year=slice(CLIM_START_YEAR, CLIM_END_YEAR)
    ).mean("season_year", skipna=True)

    return clim


def coarsen_for_plot(da, factor=PLOT_COARSEN):
    if factor is None or factor <= 1:
        return da.compute()

    da2 = da.coarsen(
        lat=factor,
        lon=factor,
        boundary="trim"
    ).mean(skipna=True)

    return da2.compute()


def mesh_from_da(da):
    lon2, lat2 = np.meshgrid(da["lon"].values, da["lat"].values)
    return lon2, lat2


def polar_ax(fig, nrows, ncols, idx):
    proj = ccrs.SouthPolarStereo()
    ax = fig.add_subplot(nrows, ncols, idx, projection=proj)

    theta = np.linspace(0, 2 * np.pi, 240)
    center = [0.5, 0.5]
    radius = 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    ax.add_feature(
        cfeature.LAND,
        facecolor="0.88",
        edgecolor="black",
        linewidth=0.45,
        zorder=3
    )

    ax.coastlines(linewidth=0.55, zorder=4)

    lon_grid = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 120, 150]
    lat_grid = [-60, -70, -80]

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.35,
        linestyle=":",
        color="0.55",
        alpha=0.60,
        zorder=5
    )

    gl.xlocator = mticker.FixedLocator(lon_grid)
    gl.ylocator = mticker.FixedLocator(lat_grid)

    edge_lat = -57.5

    for lo in lon_grid:
        if lo < 0:
            label = f"{abs(lo)}°W"
        elif lo > 0:
            label = f"{lo}°E"
        else:
            label = "0°"

        ax.text(
            lo,
            edge_lat,
            label,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=10
        )

    for la in [-70, -80]:
        ax.text(
            0,
            la,
            f"{abs(la)}°S",
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=10
        )

    return ax


def add_panel_letter(ax, letter):
    ax.text(
        -0.08,
        1.02,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold"
    )


def add_cbar(fig, pcm, ax, label, ticks=None, ticklabels=None, extend="neither"):
    cb = fig.colorbar(
        pcm,
        ax=ax,
        shrink=0.82,
        pad=0.02,
        ticks=ticks,
        extend=extend
    )

    cb.set_label(label, fontsize=11, fontweight="bold")
    cb.ax.tick_params(labelsize=8)

    if ticklabels is not None:
        cb.ax.set_yticklabels(ticklabels)

    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")

    return cb


# 5) Open datasets
datasets = {}

for key, path in files.items():
    print(f"\nOpening {key}: {path}")
    ds = open_dataset_low_ram(path)
    ds = prepare_ds(ds)
    datasets[key] = ds
    print(ds)


# 6) Detect variables
data_vars = {}

for key, ds in datasets.items():
    varname = find_var(ds, key)

    da = ds[varname].astype("float32")
    da = squeeze_depth_if_present(da)
    da = da.where(np.isfinite(da))

    data_vars[key] = da

    print("\nDetected variable:")
    print(key, "->", varname)
    print("dims:", da.dims)
    print("shape:", da.shape)
    print("units:", da.attrs.get("units", "unknown"))


# 7) Seasonal climatology maps
maps = {key: {} for key in ROW_ORDER}

for key in ROW_ORDER:
    da = data_vars[key]

    print("\n" + "="*70)
    print(f"Calculating seasonal maps for {key}")
    print("="*70)

    for season_name, months in SEASONS.items():
        print(f"  {season_name}")

        clim = seasonal_climatology(
            da,
            months,
            season_name
        )

        maps[key][season_name] = coarsen_for_plot(clim, PLOT_COARSEN)

        valid_cells = int(np.isfinite(maps[key][season_name]).sum())
        print("    valid cells:", valid_cells)

        gc.collect()


# 8) Plot ranges
plot_ranges = MANUAL_RANGES

print("\nPlot ranges:")
for key in ROW_ORDER:
    print(key, ":", plot_ranges[key])
    print("ticks:", MANUAL_TICKS[key])


# 9) Plot figure
fig = plt.figure(figsize=FIGSIZE)

letters = list("abcdefghi")
letter_i = 0

axs = np.empty((3, 3), dtype=object)

for r, key in enumerate(ROW_ORDER):

    vmin, vmax = plot_ranges[key]

    for c, season_name in enumerate(COLUMN_ORDER):

        ax = polar_ax(fig, 3, 3, r * 3 + c + 1)
        axs[r, c] = ax

        da_plot = maps[key][season_name]
        lon2, lat2 = mesh_from_da(da_plot)

        pcm = ax.pcolormesh(
            lon2,
            lat2,
            da_plot,
            transform=ccrs.PlateCarree(),
            cmap=CMAP_OBJECTS[key],
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            zorder=1
        )

        add_cbar(
            fig,
            pcm,
            ax,
            UNIT_LABEL,
            ticks=MANUAL_TICKS[key],
            ticklabels=MANUAL_TICKLABELS[key],
            extend="both"
        )

        add_panel_letter(ax, letters[letter_i])
        letter_i += 1


# 10) Layout labels
plt.subplots_adjust(
    left=LEFT,
    right=RIGHT,
    top=TOP,
    bottom=BOTTOM,
    wspace=WSPACE,
    hspace=HSPACE
)

# Column titles
for j, season_name in enumerate(COLUMN_ORDER):
    pos = axs[0, j].get_position()
    x_center = 0.5 * (pos.x0 + pos.x1)
    y_top = pos.y1

    fig.text(
        x_center,
        y_top + 0.025,
        season_name,
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="bold"
    )

# Row labels
for r, key in enumerate(ROW_ORDER):
    row_label = ROW_LABELS[key]

    pos0 = axs[r, 0].get_position()
    pos2 = axs[r, 2].get_position()

    y_center = 0.5 * (min(pos0.y0, pos2.y0) + max(pos0.y1, pos2.y1))
    x_left = pos0.x0

    fig.text(
        x_left - 0.065,
        y_center,
        row_label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold"
    )


# 11) Save output
plt.savefig(
    OUT_PATH,
    dpi=1080,
    bbox_inches="tight"
)

plt.show()

print("\nSaved figure:")
print(OUT_PATH)


# 12) Close datasets
for ds in datasets.values():
    ds.close()

del datasets
gc.collect()

print("\nDone.")
