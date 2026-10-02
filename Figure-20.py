# Fig 2
# Seasonal variability of PIC and POC across Antarctic shelf seas
#
# Columns:
#   Spring, Summer, Early autumn
#
# Rows:
#   PIC
#   POC
#   PIC anomaly
#   POC anomaly
#
# Unit handling:
#   PIC original unit = mol m^-3
#   PIC converted to mg C m^-3 using:
#       PIC_mg_C_m3 = PIC_mol_m3 * 12.0107 * 1000
#
#   POC original unit = mg m^-3
#   Interpreted and labeled here as mg C m^-3
#
# Final plotted unit for all rows:
#   mg C m^-3
#
# Output:
#   /content/drive/MyDrive/SAM_Thesis/Fig/Obj2Fig17_SO.png

# 1) Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# 0) Install
!pip -q install xarray netCDF4 h5netcdf dask cartopy


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
    "PIC": "/content/drive/MyDrive/SAM_Thesis/Data/PIC_monthly_2008_2025_SO.nc",
    "POC": "/content/drive/MyDrive/SAM_Thesis/Data/POC_monthly_2008_2025_SO.nc",
}

out_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(out_dir, exist_ok=True)

OUT_PATH = os.path.join(out_dir, "Obj2Fig17_SO.png")

for f in files.values():
    if not os.path.exists(f):
        raise FileNotFoundError(f"File not found: {f}")

print("All input files found.")


# 3) Settings
CLIM_START_YEAR = 2008
CLIM_END_YEAR   = 2025

SEASONS = {
    "Spring": [9, 10, 11],        # SON
    "Summer": [12, 1, 2],         # DJF
    "Early autumn": [3, 4],       # Mar-Apr only, because May is missing
}

COLUMN_ORDER = ["Spring", "Summer", "Early autumn"]
ROW_ORDER = ["PIC", "POC", "PIC anomaly", "POC anomaly"]

# Original grid is very large, so downsample only for plotting
PLOT_COARSEN = 6

# Unit conversion
MOLAR_MASS_C_G_MOL = 12.0107
PIC_MOL_TO_MG_C = MOLAR_MASS_C_G_MOL * 1000.0

COMMON_UNIT_LABEL = r"mg C m$^{-3}$"

# Colormaps
PIC_CMAP = "YlGnBu"
POC_CMAP = "PuBuGn"
PIC_ANOM_CMAP = "RdBu_r"
POC_ANOM_CMAP = "PuOr_r"

# Manual plot ranges (keep None for automatic)
PIC_VMIN = None
PIC_VMAX = None

POC_VMIN = None
POC_VMAX = None

PIC_ANOM_LIMIT = None
POC_ANOM_LIMIT = None

# Percentile-based automatic limits
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98

# Optional manual ticks
PIC_TICKS = None
POC_TICKS = None
PIC_ANOM_TICKS = None
POC_ANOM_TICKS = None

# Figure layout
FIGSIZE = (14.5, 16.0)

LEFT   = 0.10
RIGHT  = 0.97
TOP    = 0.95
BOTTOM = 0.04
WSPACE = 0.18
HSPACE = 0.15

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "savefig.dpi": 300
})


# 4) Helper functions
def standardize_coords(ds):
    rename_dict = {}

    for c in list(ds.coords):
        cl = c.lower()

        if cl in ["latitude", "nav_lat"]:
            rename_dict[c] = "lat"
        elif cl in ["longitude", "nav_lon"]:
            rename_dict[c] = "lon"
        elif cl in ["valid_time", "time_counter"]:
            rename_dict[c] = "time"

    for d in list(ds.dims):
        dl = d.lower()

        if dl in ["latitude", "nav_lat"] and d not in rename_dict:
            rename_dict[d] = "lat"
        elif dl in ["longitude", "nav_lon"] and d not in rename_dict:
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
    ds = ds.sortby("lat")
    ds = ds.sortby("lon")
    ds = ds.sortby("time")

    return ds


def find_var(ds, possible_names):
    for name in possible_names:
        if name in ds.data_vars:
            return name

    vars_list = [v for v in ds.data_vars if "time" in ds[v].dims]

    if len(vars_list) == 1:
        print(f"Using only time-dependent variable found: {vars_list[0]}")
        return vars_list[0]

    raise ValueError(
        f"Could not find variable from {possible_names}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def get_season_year(time_da, months):
    year = time_da.dt.year

    # DJF rule: December belongs to following summer year
    if 12 in months and 1 in months:
        year = xr.where(time_da.dt.month == 12, year + 1, year)

    return year.rename("season_year")


def seasonal_mean_by_year(da, months, min_months=2):
    da_season = da.where(da["time"].dt.month.isin(months), drop=True)

    if da_season.sizes.get("time", 0) == 0:
        return None

    sy = get_season_year(da_season["time"], months)
    da_season = da_season.assign_coords(season_year=sy)

    mean_da = da_season.groupby("season_year").mean("time", skipna=True)

    years = sy.values
    unique_years = np.unique(years)

    valid_years = []

    for y in unique_years:
        n_months = np.sum(years == y)
        if n_months >= min_months:
            valid_years.append(int(y))

    if len(valid_years) == 0:
        return None

    mean_da = mean_da.sel(season_year=valid_years)

    return mean_da


def seasonal_climatology(da, months, season_name):
    # Early autumn has only Mar-Apr; April missing in some years
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


def full_period_mean(da):
    return da.mean("time", skipna=True)


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


def robust_range(data_list, p_low=2, p_high=98):
    vals = []

    for da in data_list:
        arr = da.values
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals.append(arr)

    if len(vals) == 0:
        return 0, 1

    vals = np.concatenate(vals)

    vmin = np.nanpercentile(vals, p_low)
    vmax = np.nanpercentile(vals, p_high)

    return float(vmin), float(vmax)


def robust_anom_limit(data_list, p=98):
    vals = []

    for da in data_list:
        arr = da.values
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals.append(arr)

    if len(vals) == 0:
        return 1.0

    vals = np.concatenate(vals)
    lim = np.nanpercentile(np.abs(vals), p)

    if not np.isfinite(lim) or lim == 0:
        lim = 1.0

    return float(lim)


def pretty_ticks(vmin, vmax, n=5):
    return np.linspace(vmin, vmax, n)


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


def add_cbar(fig, pcm, ax, label, ticks=None, extend="neither"):
    cb = fig.colorbar(
        pcm,
        ax=ax,
        shrink=0.82,
        pad=0.02,
        ticks=ticks,
        extend=extend
    )

    cb.set_label(label, fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=9)

    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")

    return cb


# 5) Open datasets
ds_pic = xr.open_dataset(
    files["PIC"],
    decode_times=True,
    engine="h5netcdf",
    chunks={"time": 12}
)

ds_poc = xr.open_dataset(
    files["POC"],
    decode_times=True,
    engine="h5netcdf",
    chunks={"time": 12}
)

ds_pic = prepare_ds(ds_pic)
ds_poc = prepare_ds(ds_poc)

print("\nPIC dataset:")
print(ds_pic)

print("\nPOC dataset:")
print(ds_poc)


# 6) Detect variables
pic_var = find_var(ds_pic, ["PIC", "pic"])
poc_var = find_var(ds_poc, ["POC", "poc"])

print("\nDetected variables:")
print("PIC:", pic_var)
print("POC:", poc_var)


# 7) Prepare variables and convert units
pic_raw = ds_pic[pic_var].astype("float32")
poc_raw = ds_poc[poc_var].astype("float32")

pic_raw = pic_raw.where(np.isfinite(pic_raw))
poc_raw = poc_raw.where(np.isfinite(poc_raw))

print("\nOriginal units:")
print("PIC units:", pic_raw.attrs.get("units", "unknown"))
print("POC units:", poc_raw.attrs.get("units", "unknown"))

# Convert PIC from mol m^-3 to mg C m^-3
pic = pic_raw * PIC_MOL_TO_MG_C
pic.attrs["units"] = "mg C m^-3"
pic.attrs["conversion_note"] = (
    "Converted from mol m^-3 to mg C m^-3 using carbon molar mass 12.0107 g mol^-1"
)

# POC: label as mg C m^-3 for figure consistency
poc = poc_raw
poc.attrs["units"] = "mg C m^-3"
poc.attrs["label_note"] = "Plotted as mg C m^-3"

print("\nConverted plotting units:")
print("PIC plotted as: mg C m^-3")
print("POC plotted as: mg C m^-3")
print("PIC conversion factor:", PIC_MOL_TO_MG_C)


# 8) Seasonal climatology and anomaly
maps_pic = {}
maps_poc = {}
maps_pic_anom = {}
maps_poc_anom = {}

print("\nCalculating full-period means for anomaly reference...")
pic_mean = full_period_mean(pic)
poc_mean = full_period_mean(poc)

for season_name, months in SEASONS.items():
    print(f"\nCalculating {season_name}")

    pic_season = seasonal_climatology(pic, months, season_name)
    poc_season = seasonal_climatology(poc, months, season_name)

    pic_anom = pic_season - pic_mean
    poc_anom = poc_season - poc_mean

    maps_pic[season_name] = coarsen_for_plot(pic_season)
    maps_poc[season_name] = coarsen_for_plot(poc_season)
    maps_pic_anom[season_name] = coarsen_for_plot(pic_anom)
    maps_poc_anom[season_name] = coarsen_for_plot(poc_anom)

    print("  PIC valid cells      :", int(np.isfinite(maps_pic[season_name]).sum()))
    print("  POC valid cells      :", int(np.isfinite(maps_poc[season_name]).sum()))
    print("  PIC anomaly valid    :", int(np.isfinite(maps_pic_anom[season_name]).sum()))
    print("  POC anomaly valid    :", int(np.isfinite(maps_poc_anom[season_name]).sum()))

    gc.collect()


# 9) Plot ranges
if PIC_VMIN is None or PIC_VMAX is None:
    PIC_VMIN_AUTO, PIC_VMAX_AUTO = robust_range(
        [maps_pic[s] for s in COLUMN_ORDER],
        PERCENTILE_LOW,
        PERCENTILE_HIGH
    )
    PIC_VMIN = PIC_VMIN_AUTO if PIC_VMIN is None else PIC_VMIN
    PIC_VMAX = PIC_VMAX_AUTO if PIC_VMAX is None else PIC_VMAX

if POC_VMIN is None or POC_VMAX is None:
    POC_VMIN_AUTO, POC_VMAX_AUTO = robust_range(
        [maps_poc[s] for s in COLUMN_ORDER],
        PERCENTILE_LOW,
        PERCENTILE_HIGH
    )
    POC_VMIN = POC_VMIN_AUTO if POC_VMIN is None else POC_VMIN
    POC_VMAX = POC_VMAX_AUTO if POC_VMAX is None else POC_VMAX

if PIC_ANOM_LIMIT is None:
    PIC_ANOM_LIMIT = robust_anom_limit(
        [maps_pic_anom[s] for s in COLUMN_ORDER],
        PERCENTILE_HIGH
    )

if POC_ANOM_LIMIT is None:
    POC_ANOM_LIMIT = robust_anom_limit(
        [maps_poc_anom[s] for s in COLUMN_ORDER],
        PERCENTILE_HIGH
    )

if PIC_TICKS is None:
    PIC_TICKS = pretty_ticks(PIC_VMIN, PIC_VMAX, 5)

if POC_TICKS is None:
    POC_TICKS = pretty_ticks(POC_VMIN, POC_VMAX, 5)

if PIC_ANOM_TICKS is None:
    PIC_ANOM_TICKS = pretty_ticks(-PIC_ANOM_LIMIT, PIC_ANOM_LIMIT, 5)

if POC_ANOM_TICKS is None:
    POC_ANOM_TICKS = pretty_ticks(-POC_ANOM_LIMIT, POC_ANOM_LIMIT, 5)

print("\nPlot ranges in mg C m^-3:")
print("PIC:", PIC_VMIN, PIC_VMAX)
print("POC:", POC_VMIN, POC_VMAX)
print("PIC anomaly:", -PIC_ANOM_LIMIT, PIC_ANOM_LIMIT)
print("POC anomaly:", -POC_ANOM_LIMIT, POC_ANOM_LIMIT)


# 10) Plot figure
fig = plt.figure(figsize=FIGSIZE)

letters = list("abcdefghijkl")
letter_i = 0

axs = np.empty((4, 3), dtype=object)

for r, row_name in enumerate(ROW_ORDER):

    for c, season_name in enumerate(COLUMN_ORDER):

        ax = polar_ax(fig, 4, 3, r * 3 + c + 1)
        axs[r, c] = ax

        if row_name == "PIC":
            da_plot = maps_pic[season_name]
            lon2, lat2 = mesh_from_da(da_plot)

            pcm = ax.pcolormesh(
                lon2,
                lat2,
                da_plot,
                transform=ccrs.PlateCarree(),
                cmap=PIC_CMAP,
                vmin=PIC_VMIN,
                vmax=PIC_VMAX,
                shading="auto",
                zorder=1
            )

            add_cbar(
                fig,
                pcm,
                ax,
                COMMON_UNIT_LABEL,
                ticks=PIC_TICKS,
                extend="both"
            )

        elif row_name == "POC":
            da_plot = maps_poc[season_name]
            lon2, lat2 = mesh_from_da(da_plot)

            pcm = ax.pcolormesh(
                lon2,
                lat2,
                da_plot,
                transform=ccrs.PlateCarree(),
                cmap=POC_CMAP,
                vmin=POC_VMIN,
                vmax=POC_VMAX,
                shading="auto",
                zorder=1
            )

            add_cbar(
                fig,
                pcm,
                ax,
                COMMON_UNIT_LABEL,
                ticks=POC_TICKS,
                extend="both"
            )

        elif row_name == "PIC anomaly":
            da_plot = maps_pic_anom[season_name]
            lon2, lat2 = mesh_from_da(da_plot)

            pcm = ax.pcolormesh(
                lon2,
                lat2,
                da_plot,
                transform=ccrs.PlateCarree(),
                cmap=PIC_ANOM_CMAP,
                vmin=-PIC_ANOM_LIMIT,
                vmax=PIC_ANOM_LIMIT,
                shading="auto",
                zorder=1
            )

            add_cbar(
                fig,
                pcm,
                ax,
                COMMON_UNIT_LABEL,
                ticks=PIC_ANOM_TICKS,
                extend="both"
            )

        elif row_name == "POC anomaly":
            da_plot = maps_poc_anom[season_name]
            lon2, lat2 = mesh_from_da(da_plot)

            pcm = ax.pcolormesh(
                lon2,
                lat2,
                da_plot,
                transform=ccrs.PlateCarree(),
                cmap=POC_ANOM_CMAP,
                vmin=-POC_ANOM_LIMIT,
                vmax=POC_ANOM_LIMIT,
                shading="auto",
                zorder=1
            )

            add_cbar(
                fig,
                pcm,
                ax,
                COMMON_UNIT_LABEL,
                ticks=POC_ANOM_TICKS,
                extend="both"
            )

        add_panel_letter(ax, letters[letter_i])
        letter_i += 1


# 11) Layout labels
plt.subplots_adjust(
    left=LEFT,
    right=RIGHT,
    top=TOP,
    bottom=BOTTOM,
    wspace=WSPACE,
    hspace=HSPACE
)

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

for r, row_name in enumerate(ROW_ORDER):
    pos0 = axs[r, 0].get_position()
    pos2 = axs[r, 2].get_position()

    y_center = 0.5 * (min(pos0.y0, pos2.y0) + max(pos0.y1, pos2.y1))
    x_left = pos0.x0

    fig.text(
        x_left - 0.065,
        y_center,
        row_name,
        rotation=90,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold"
    )


# 12) Save output
plt.savefig(
    OUT_PATH,
    dpi=1080,
    bbox_inches="tight"
)

plt.show()

print("\nSaved figure:")
print(OUT_PATH)


# 13) Close datasets
ds_pic.close()
ds_poc.close()

print("\nDone.")
