# Fig-8
# Sea-wise PIC/POC boxplots
#
# Panels:
#   (a) Sea-wise PIC boxplots
#   (b) Sea-wise POC boxplots
#
# Layout:
#   1 column × 2 rows
#
# Seasons:
#   Spring       = SON
#   Summer       = DJF
#   Early autumn = Mar-Apr
#
# Units:
#   PIC original = mol m^-3
#   PIC plotted  = mg C m^-3
#   POC plotted  = mg C m^-3


# 1) Mount Google Drive
from google.colab import drive

drive.mount('/content/drive')


# 0) Install
!pip -q install xarray netCDF4 h5netcdf dask matplotlib numpy


# 1) Imports
import os
import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# 2) Input / output paths
files = {
    "PIC": "/content/drive/MyDrive/SAM_Thesis/Data/PIC_monthly_2008_2025_SO.nc",
    "POC": "/content/drive/MyDrive/SAM_Thesis/Data/POC_monthly_2008_2025_SO.nc",
}

out_dir = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(out_dir, exist_ok=True)

OUT_PATH = os.path.join(out_dir, "Obj2Fig18_SO2.png")

sea_info = [
    ("WED", "Weddell Sea", "60°W–20°W", -60, -20),
    ("KHV", "King Haakon VII Sea", "20°W–0°", -20, 0),
    ("RLS", "Riiser-Larsen Sea", "0°–10°E", 0, 10),
    ("LAZ", "Lazarev Sea", "10°E–30°E", 10, 30),
    ("COS", "Cosmonauts Sea", "30°E–50°E", 30, 50),
    ("COO", "Cooperation Sea", "50°E–70°E", 50, 70),
    ("DAV", "Davis Sea", "70°E–90°E", 70, 90),
    ("MAW", "Mawson Sea", "90°E–130°E", 90, 130),
    ("DUR", "D'Urville Sea", "130°E–150°E", 130, 150),
    ("SOM", "Somov Sea", "150°E–170°E", 150, 170),
    ("ROS", "Ross Sea", "170°E–130°W", 170, -130),
    ("AMU", "Amundsen Sea", "130°W–100°W", -130, -100),
    ("BEL", "Bellingshausen Sea", "100°W–60°W", -100, -60),
]

for f in files.values():
    if not os.path.exists(f):
        raise FileNotFoundError(f"File not found: {f}")

print("All input files found.")


# 3) Settings
CLIM_START_YEAR = 2008
CLIM_END_YEAR   = 2025

SEASONS = {
    "Spring": [9, 10, 11],      # SON
    "Summer": [12, 1, 2],       # DJF
    "Early autumn": [3, 4],     # Mar-Apr only
}

SEASON_ORDER = ["Spring", "Summer", "Early autumn"]

# Low-RAM settings
BOX_COARSEN = 10
MAX_SAMPLES_PER_BOX = 2000
RANDOM_SEED = 42

# Hide outliers for cleaner figure
SHOW_FLIERS = False

# PIC conversion: mol C m^-3 to mg C m^-3
MOLAR_MASS_C_G_MOL = 12.0107
PIC_MOL_TO_MG_C = MOLAR_MASS_C_G_MOL * 1000.0

UNIT_LABEL = r"mg C m$^{-3}$"

SEASON_COLORS = {
    "Spring": "#54A24B",
    "Summer": "#F58518",
    "Early autumn": "#7E57C2",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "savefig.dpi": 300,
})

FIGSIZE = (14, 12)


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

    return ds


def find_var(ds, possible_names, require_time=False):
    for name in possible_names:
        if name in ds.data_vars:
            return name

    if require_time:
        candidates = [v for v in ds.data_vars if "time" in ds[v].dims]
        if len(candidates) == 1:
            return candidates[0]

    candidates = list(ds.data_vars)
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"Could not find variable from {possible_names}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def get_season_year(time_da, months):
    year = time_da.dt.year

    # DJF rule: December belongs to following year
    if 12 in months and 1 in months:
        year = xr.where(time_da.dt.month == 12, year + 1, year)

    return year.rename("season_year")


def coarsen_spatial_lazy(da, factor=BOX_COARSEN):
    if factor is None or factor <= 1:
        return da

    return da.coarsen(
        lat=factor,
        lon=factor,
        boundary="trim"
    ).mean(skipna=True)


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

    return clim.compute()


def lonlat_mesh(da):
    return np.meshgrid(da["lon"].values, da["lat"].values)


def make_sector_mask(lon2, lat2, lon_min, lon_max):
    if lon_min <= lon_max:
        lon_mask = (lon2 >= lon_min) & (lon2 < lon_max)
    else:
        lon_mask = (lon2 >= lon_min) | (lon2 < lon_max)

    lat_mask = lat2 <= -60
    return lon_mask & lat_mask


def sample_values(arr, max_n=MAX_SAMPLES_PER_BOX, seed=RANDOM_SEED):
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return np.array([np.nan])

    if vals.size <= max_n:
        return vals

    rng = np.random.default_rng(seed)
    idx = rng.choice(vals.size, size=max_n, replace=False)
    return vals[idx]


def upper_limit(data_groups, q=99.5, pad=0.10):
    vals = []

    for arr in data_groups:
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if a.size > 0:
            vals.append(a)

    if len(vals) == 0:
        return 1.0

    vals = np.concatenate(vals)
    vmax = np.nanpercentile(vals, q)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(vals)

    return float(vmax * (1 + pad))


def style_boxplot(bp, color):
    for box in bp["boxes"]:
        box.set(facecolor=color, edgecolor=color, alpha=0.45, linewidth=1.2)

    for median in bp["medians"]:
        median.set(color=color, linewidth=1.8)

    for whisker in bp["whiskers"]:
        whisker.set(color=color, linewidth=1.0)

    for cap in bp["caps"]:
        cap.set(color=color, linewidth=1.0)


# 5) Open datasets
ds_pic = xr.open_dataset(
    files["PIC"],
    decode_times=True,
    engine="h5netcdf",
    chunks={"time": 12, "lat": 120, "lon": 1440}
)

ds_poc = xr.open_dataset(
    files["POC"],
    decode_times=True,
    engine="h5netcdf",
    chunks={"time": 12, "lat": 120, "lon": 1440}
)

ds_pic = prepare_ds(ds_pic)
ds_poc = prepare_ds(ds_poc)

print("\nPIC dataset:")
print(ds_pic)

print("\nPOC dataset:")
print(ds_poc)


# 6) Detect variables
pic_var = find_var(ds_pic, ["PIC", "pic"], require_time=True)
poc_var = find_var(ds_poc, ["POC", "poc"], require_time=True)

print("\nDetected variables:")
print("PIC:", pic_var)
print("POC:", poc_var)


# 7) Prepare carbon variables
pic_raw = ds_pic[pic_var].astype("float32").where(np.isfinite(ds_pic[pic_var]))
poc_raw = ds_poc[poc_var].astype("float32").where(np.isfinite(ds_poc[poc_var]))

print("\nOriginal units:")
print("PIC:", pic_raw.attrs.get("units", "unknown"))
print("POC:", poc_raw.attrs.get("units", "unknown"))

# Convert PIC to mg C m^-3
pic = pic_raw * PIC_MOL_TO_MG_C
pic.attrs["units"] = "mg C m^-3"

# POC plotted as mg C m^-3
poc = poc_raw
poc.attrs["units"] = "mg C m^-3"

print("\nPlotting units:")
print("PIC:", pic.attrs["units"])
print("POC:", poc.attrs["units"])
print("PIC conversion factor:", PIC_MOL_TO_MG_C)


# 8) Coarsen first for low RAM
print("\nApplying spatial coarsening before seasonal calculations...")

pic_c = coarsen_spatial_lazy(pic, BOX_COARSEN)
poc_c = coarsen_spatial_lazy(poc, BOX_COARSEN)

print("Coarsened PIC shape:", pic_c.shape)
print("Coarsened POC shape:", poc_c.shape)


# 9) Seasonal climatology on coarsened grid
maps_pic = {}
maps_poc = {}

for season_name, months in SEASONS.items():
    print(f"\nCalculating seasonal climatology: {season_name}")

    maps_pic[season_name] = seasonal_climatology(pic_c, months, season_name)
    maps_poc[season_name] = seasonal_climatology(poc_c, months, season_name)

    print("  PIC valid cells:", int(np.isfinite(maps_pic[season_name]).sum()))
    print("  POC valid cells:", int(np.isfinite(maps_poc[season_name]).sum()))

    gc.collect()


# 10) Build marginal-sea masks
ref_da = maps_pic["Spring"]
lon2, lat2 = lonlat_mesh(ref_da)

sea_masks = {}
sea_labels = []

for abbr, full_name, lon_text, lon_min, lon_max in sea_info:
    smask = make_sector_mask(lon2, lat2, lon_min, lon_max)
    sea_masks[abbr] = smask
    sea_labels.append(abbr)

print("\nTotal number of seas:", len(sea_labels))


# 11) Extract sea-wise boxplot samples
pic_box_data = {s: [] for s in SEASON_ORDER}
poc_box_data = {s: [] for s in SEASON_ORDER}

for season_name in SEASON_ORDER:
    print(f"\nExtracting samples: {season_name}")

    pic_arr = maps_pic[season_name].values
    poc_arr = maps_poc[season_name].values

    for abbr, full_name, lon_text, lon_min, lon_max in sea_info:
        mask = sea_masks[abbr]

        pic_vals = sample_values(pic_arr[mask])
        poc_vals = sample_values(poc_arr[mask])

        pic_box_data[season_name].append(pic_vals)
        poc_box_data[season_name].append(poc_vals)

        print(
            f"{abbr}: "
            f"PIC n={np.sum(np.isfinite(pic_vals))}, "
            f"POC n={np.sum(np.isfinite(poc_vals))}"
        )

    gc.collect()


# 12) Axis ranges
pic_all = []
poc_all = []

for s in SEASON_ORDER:
    pic_all.extend(pic_box_data[s])
    poc_all.extend(poc_box_data[s])

PIC_YMAX = upper_limit(pic_all, q=99.5, pad=0.10)
POC_YMAX = upper_limit(poc_all, q=99.5, pad=0.10)

print("\nAxis maxima:")
print("PIC y max:", PIC_YMAX)
print("POC y max:", POC_YMAX)


# Optional manual limit for cleaner POC look
# Uncomment if you want fixed limit
# POC_YMAX = 180

# Manual POC limit for cleaner publication-style figure
POC_YMAX = 200

print("\nFinal axis maxima used:")
print("PIC y max:", PIC_YMAX)
print("POC y max:", POC_YMAX)


# 13) Plot
fig, axs = plt.subplots(2, 1, figsize=FIGSIZE)

ax1 = axs[0]
ax2 = axs[1]

n_seas = len(sea_labels)
base_positions = np.arange(1, n_seas + 1)

offsets = {
    "Spring": -0.25,
    "Summer": 0.00,
    "Early autumn": 0.25
}

box_width = 0.22


# (a) Sea-wise PIC
for season_name in SEASON_ORDER:
    pos = base_positions + offsets[season_name]

    bp = ax1.boxplot(
        pic_box_data[season_name],
        positions=pos,
        widths=box_width,
        patch_artist=True,
        showfliers=SHOW_FLIERS,
        whis=1.5
    )

    style_boxplot(bp, SEASON_COLORS[season_name])

ax1.set_title("Sea-wise PIC boxplots", pad=12, fontweight="bold")
ax1.set_ylabel(f"PIC ({UNIT_LABEL})", fontweight="bold")
ax1.set_xlabel("Antarctic Shelf Seas", fontweight="bold")
ax1.set_xticks(base_positions)
ax1.set_xticklabels(sea_labels, rotation=45, ha="right")
ax1.set_xlim(0.3, n_seas + 0.7)
ax1.set_ylim(0, PIC_YMAX)
ax1.grid(axis="y", linestyle=":", color="0.75", linewidth=1.0)
ax1.text(-0.08, 1.04, "(a)", transform=ax1.transAxes, fontsize=16, fontweight="bold")


# (b) Sea-wise POC
for season_name in SEASON_ORDER:
    pos = base_positions + offsets[season_name]

    bp = ax2.boxplot(
        poc_box_data[season_name],
        positions=pos,
        widths=box_width,
        patch_artist=True,
        showfliers=SHOW_FLIERS,
        whis=1.5
    )

    style_boxplot(bp, SEASON_COLORS[season_name])

ax2.set_title("Sea-wise POC boxplots", pad=12, fontweight="bold")
ax2.set_ylabel(f"POC ({UNIT_LABEL})", fontweight="bold")
ax2.set_xlabel("Antarctic Shelf Seas", fontweight="bold")
ax2.set_xticks(base_positions)
ax2.set_xticklabels(sea_labels, rotation=45, ha="right")
ax2.set_xlim(0.3, n_seas + 0.7)
ax2.set_ylim(0, POC_YMAX)
ax2.grid(axis="y", linestyle=":", color="0.75", linewidth=1.0)
ax2.text(-0.08, 1.04, "(b)", transform=ax2.transAxes, fontsize=16, fontweight="bold")


# 14) Final formatting
for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

legend_handles = [
    Patch(
        facecolor=SEASON_COLORS["Spring"],
        edgecolor=SEASON_COLORS["Spring"],
        alpha=0.45,
        label="Spring"
    ),
    Patch(
        facecolor=SEASON_COLORS["Summer"],
        edgecolor=SEASON_COLORS["Summer"],
        alpha=0.45,
        label="Summer"
    ),
    Patch(
        facecolor=SEASON_COLORS["Early autumn"],
        edgecolor=SEASON_COLORS["Early autumn"],
        alpha=0.45,
        label="Early autumn"
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    frameon=True,
    fancybox=False,
    edgecolor="0.4",
    fontsize=11,
    bbox_to_anchor=(0.5, 0.02)
)

plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.96,
    bottom=0.12,
    hspace=0.40
)



# 15) Save
plt.savefig(
    OUT_PATH,
    dpi=1080,
    bbox_inches="tight"
)

plt.show()

print("\nSaved figure:")
print(OUT_PATH)


# 16) Close datasets
ds_pic.close()
ds_poc.close()

del ds_pic, ds_poc
gc.collect()

print("\nDone.")
