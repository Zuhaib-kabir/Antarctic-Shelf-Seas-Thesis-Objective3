# using existing SIC CSV + processing only POC and CHL


#Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# 0) Install
!pip -q install xarray netCDF4 h5netcdf dask pandas numpy

# 1) Imports
import os
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

# 2) Input / output paths
files = {
    "SIC": "/content/drive/MyDrive/SAM_Thesis/Data/SIC_seawise_monthly_2008_2025_LOW_RAM.csv",
    "POC": "/content/drive/MyDrive/SAM_Thesis/Data/POC_monthly_2008_2025_SO.nc",
    "CHL": "/content/drive/MyDrive/SAM_Thesis/Data/CHL_monthly_2008_2025_SO.nc",
}

out_dir = "/content/drive/MyDrive/SAM_Thesis/Processed"
os.makedirs(out_dir, exist_ok=True)

OUT_CSV = os.path.join(
    out_dir,
    "Fig23_seawise_monthly_timeseries_LOW_RAM.csv"
)

for key, path in files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"{key} file not found: {path}")

print("All input files found.")

# 3) Settings
START_DATE = "2008-01-01"
END_DATE   = "2025-12-31"

OPEN_CHUNKS = {
    "time": 1,
    "lat": 120,
    "lon": 720,
}

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

sea_codes = [x[0] for x in sea_info]

# 4) Helper functions
def month_start_datetime(values):
    """
    Safe conversion to month-start timestamps.
    Works for pandas Series, DatetimeIndex, numpy arrays, and lists.
    """
    s = pd.Series(pd.to_datetime(values))
    return s.dt.to_period("M").dt.to_timestamp()


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

    if "time" not in ds.coords:
        raise ValueError("Dataset has no time coordinate.")

    ds = ds.sel(time=slice(START_DATE, END_DATE))

    ds = ds.where((ds["lat"] >= -90) & (ds["lat"] <= -60), drop=True)

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")
    ds = ds.sortby("time")

    return ds


def find_var(ds, label):
    possible = {
        "POC": [
            "POC", "poc", "particulate_organic_carbon"
        ],
        "CHL": [
            "CHL", "chl", "chlor_a",
            "chlorophyll", "chlorophyll_a"
        ],
    }

    for name in possible[label]:
        if name in ds.data_vars:
            return name

    candidates = []
    for v in ds.data_vars:
        dims = ds[v].dims
        if "time" in dims and "lat" in dims and "lon" in dims:
            candidates.append(v)

    if len(candidates) == 1:
        print(f"Using detected variable for {label}: {candidates[0]}")
        return candidates[0]

    raise ValueError(
        f"Could not detect variable for {label}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def squeeze_depth_if_present(da):
    for d in list(da.dims):
        if d.lower() in ["depth", "lev", "level", "deptht"]:
            print(f"Squeezing depth dimension: {d}")
            da = da.isel({d: 0}, drop=True)
    return da


def make_monthly_if_needed(da):
    """
    If daily/irregular, convert to monthly mean.
    If already monthly, keep as monthly.
    """
    if da.sizes.get("time", 0) > 216:
        print("More than 216 time steps detected. Resampling to monthly mean...")
        da = da.resample(time="1MS").mean(skipna=True)

    da = da.assign_coords(
        time=month_start_datetime(da["time"].values).values
    )

    return da


def lon_mask(lon, lon_min, lon_max):
    if lon_min < lon_max:
        return (lon >= lon_min) & (lon < lon_max)
    else:
        return (lon >= lon_min) | (lon < lon_max)


def area_weighted_sea_mean(da, lon_min, lon_max):
    mask = lon_mask(da["lon"], lon_min, lon_max)
    sub = da.where(mask, drop=True)

    weights = np.cos(np.deg2rad(sub["lat"]))

    ts = sub.weighted(weights).mean(
        dim=("lat", "lon"),
        skipna=True
    )

    return ts

# 5) Load existing SIC CSV
print("\nLoading existing SIC CSV...")

sic_df = pd.read_csv(files["SIC"])

possible_time_cols = ["time", "date", "month", "Time", "DATE"]
time_col = None

for c in possible_time_cols:
    if c in sic_df.columns:
        time_col = c
        break

if time_col is None:
    raise ValueError(
        f"No time column found in SIC CSV. Columns are: {list(sic_df.columns)}"
    )

sic_df[time_col] = month_start_datetime(sic_df[time_col])
sic_df = sic_df.rename(columns={time_col: "time"})
sic_df = sic_df.sort_values("time").reset_index(drop=True)

sic_df = sic_df[
    (sic_df["time"] >= pd.Timestamp("2008-01-01"))
    & (sic_df["time"] <= pd.Timestamp("2025-12-01"))
].copy()

missing_sic = [sea for sea in sea_codes if sea not in sic_df.columns]

if len(missing_sic) > 0:
    raise ValueError(
        "Missing sea columns in SIC CSV: "
        + ", ".join(missing_sic)
        + f"\nAvailable columns: {list(sic_df.columns)}"
    )

wide_df = pd.DataFrame()
wide_df["time"] = sic_df["time"]

for sea in sea_codes:
    wide_df[f"SIC_{sea}"] = sic_df[sea].astype("float32")

print("SIC loaded and renamed.")
print(wide_df.head())

# 6) Process POC and CHL only
for var_key in ["POC", "CHL"]:

    path = files[var_key]

    print("\n" + "=" * 80)
    print(f"Processing variable: {var_key}")
    print("=" * 80)

    ds = xr.open_dataset(
        path,
        engine="h5netcdf",
        decode_times=True,
        chunks=OPEN_CHUNKS
    )

    ds = prepare_ds(ds)

    varname = find_var(ds, var_key)
    da = ds[varname].astype("float32")

    da = squeeze_depth_if_present(da)
    da = da.where(np.isfinite(da))
    da = make_monthly_if_needed(da)

    print("Variable:", varname)
    print("Shape:", da.shape)
    print(
        "Time:",
        str(da["time"].values[0])[:10],
        "to",
        str(da["time"].values[-1])[:10]
    )

    for sea_code, sea_name, lon_label, lon_min, lon_max in sea_info:
        print(f"  {var_key} | {sea_code}: {sea_name} ({lon_label})")

        ts = area_weighted_sea_mean(
            da,
            lon_min,
            lon_max
        )

        with ProgressBar():
            ts_loaded = ts.compute()

        temp = pd.DataFrame({
            "time": month_start_datetime(ts_loaded["time"].values),
            f"{var_key}_{sea_code}": ts_loaded.values.astype("float32")
        })

        wide_df = wide_df.merge(
            temp,
            on="time",
            how="left"
        )

        del ts, ts_loaded, temp
        gc.collect()

    ds.close()
    del ds, da
    gc.collect()

    print(f"Finished variable: {var_key}")

# 7) Force complete monthly range
full_time = pd.date_range(
    "2008-01-01",
    "2025-12-01",
    freq="MS"
)

wide_df = (
    wide_df
    .set_index("time")
    .reindex(full_time)
    .rename_axis("time")
    .reset_index()
)

ordered_cols = ["time"]

for var in ["SIC", "POC", "CHL"]:
    for sea in sea_codes:
        col = f"{var}_{sea}"
        if col in wide_df.columns:
            ordered_cols.append(col)

wide_df = wide_df[ordered_cols]

# 8) Save final CSV
print("\nFinal CSV shape:")
print(wide_df.shape)

print("\nPreview:")
print(wide_df.head())

print("\nMissing values summary:")
print(wide_df.isna().sum())

wide_df.to_csv(OUT_CSV, index=False)

print("\nSaved final Fig. 23 monthly CSV:")
print(OUT_CSV)

print("\nDone.")
