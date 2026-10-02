# Antarctic Shelf Seas Thesis — Objective 3

This repository contains the Python workflows used for **Objective 3 of the Antarctic shelf-seas thesis**, focusing on **Southern Ocean carbon-cycle, phytoplankton, sea-ice–ecosystem, and biological-carbon-pump variability across the Antarctic shelf seas**.

The repository includes both **figure-generation scripts** and **analysis/preprocessing scripts** used to create intermediate products for later Objective 3 analyses. Most workflows cover **2008–2025** and are designed primarily for execution in **Google Colab**, with input and output files referenced from Google Drive.

---

## Objective 3 analysis scope

The code examines Antarctic carbon and ecosystem variability using:

- particulate inorganic carbon (PIC);
- particulate organic carbon (POC);
- chlorophyll-a;
- volumetric and vertically integrated net primary production;
- photosynthetically available radiation (PAR);
- euphotic depth;
- phytoplankton functional types;
- sea-ice concentration (SIC);
- sea-ice retreat timing;
- bloom timing;
- lagged SIC–ecosystem relationships;
- mixed-layer depth;
- upper-ocean stratification;
- biological-carbon-pump proxy relationships.

Most sea-wise analyses use the same **13 Antarctic shelf-sea sectors**:

1. Weddell Sea (WED)
2. King Haakon VII Sea (KHV)
3. Riiser-Larsen Sea (RLS)
4. Lazarev Sea (LAZ)
5. Cosmonauts Sea (COS)
6. Cooperation Sea (COO)
7. Davis Sea (DAV)
8. Mawson Sea (MAW)
9. D'Urville Sea (DUR)
10. Somov Sea (SOM)
11. Ross Sea (ROS)
12. Amundsen Sea (AMU)
13. Bellingshausen Sea (BEL)

The Ross Sea is treated as a date-line-crossing longitude sector where required.

---

# Repository contents

| Script | Main purpose |
|---|---|
| [`Figure-20.py`](./Figure-20.py) | Seasonal PIC and POC distributions and anomalies |
| [`Figure-21.py`](./Figure-21.py) | Seasonal chlorophyll-a, integrated NPP, PAR, and euphotic depth |
| [`Figure-22.py`](./Figure-22.py) | Seasonal phytoplankton functional-type community composition |
| [`Figure-23.py`](./Figure-23.py) | Sea-ice retreat and particulate-carbon / bloom response relationships |
| [`Figure-24.py`](./Figure-24.py) | Low-RAM creation of combined sea-wise SIC, POC, and chlorophyll-a monthly time series |
| [`Figure-25.py`](./Figure-25.py) | Lagged SIC–carbon/ecosystem relationships and bootstrap best-lag summaries |
| [`Figure-26.py`](./Figure-26.py) | Sea-wise seasonal PIC and POC boxplots |
| [`Figure-27.py`](./Figure-27.py) | Seasonal distributions of diatoms, haptophytes, and picophytoplankton |
| [`Figure-28.py`](./Figure-28.py) | MLD, stratification, carbon/biological relationships, and biological-carbon-pump conceptual pathway |

---

# Main workflows

## `Figure-20.py` — Seasonal PIC and POC variability

This script maps seasonal particulate-carbon conditions across the Southern Ocean.

### Variables

- particulate inorganic carbon (PIC);
- particulate organic carbon (POC);
- PIC anomaly;
- POC anomaly.

### Seasonal structure

```text
Spring       = September–November (SON)
Summer       = December–February (DJF)
Early autumn = March–April
```

The source workflow uses **March–April only** for early autumn because May is unavailable in the relevant source product.

### Unit treatment

PIC is converted from:

```text
mol m⁻³
```

to:

```text
mg C m⁻³
```

using:

```text
PIC_mg_C_m3 = PIC_mol_m3 × 12.0107 × 1000
```

POC is plotted as:

```text
mg C m⁻³
```

### Main inputs

```text
PIC_monthly_2008_2025_SO.nc
POC_monthly_2008_2025_SO.nc
```

### Processing

The workflow:

- standardizes latitude, longitude, and time coordinates;
- normalizes longitude to `[-180°, 180°)`;
- restricts the domain south of 60°S;
- selects the 2008–2025 period;
- calculates seasonal climatological fields;
- calculates the corresponding seasonal anomalies;
- applies spatial coarsening only for plotting the large native grid.

White regions in the plotted products represent land or missing optical retrievals rather than zero carbon concentration.

---

## `Figure-21.py` — Phytoplankton biomass, productivity, and light environment

This workflow analyzes:

1. chlorophyll-a;
2. approximate vertically integrated net primary production;
3. photosynthetically available radiation;
4. euphotic depth.

### Integrated NPP

The script defines:

```text
NPPINT = NPPv × Zeu
```

with units:

```text
mg C m⁻² d⁻¹
```

### Variable units

```text
Chlorophyll-a   = mg m⁻³
Integrated NPP  = mg C m⁻² d⁻¹
PAR             = mol photons m⁻² d⁻¹
Euphotic depth  = m
```

### Main inputs

```text
CHL_monthly_2008_2025_SO.nc
nppv.nc
PAR_monthly_2008_2025_SO.nc
ZEU_monthly_2008_2025_SO.nc
```

The workflow uses variable-specific spatial coarsening to reduce memory demand during plotting.

---

## `Figure-22.py` — Phytoplankton functional-type composition

This script visualizes seasonal community composition for three phytoplankton functional types:

- diatoms (`DIATO`);
- haptophytes (`HAPTO`);
- picophytoplankton (`PICO`).

### Input table

```text
Obj2Fig21_PFT_relative_contribution.csv
```

### Figure design

The left column contains stacked bar plots and the right column contains heatmaps.

The three seasons are:

```text
Spring
Summer
Early autumn
```

for all 13 Antarctic shelf seas.

Relative contributions are expressed as percentages.

The plotting code calculates PFT-specific heatmap limits from the actual data range so spatial/seasonal variability remains visible.

---

## `Figure-23.py` — Sea-ice retreat and particulate-carbon response

This workflow evaluates sea-wise relationships among:

- active-season SIC;
- POC;
- PIC;
- sea-ice retreat timing;
- chlorophyll-a bloom timing;
- retreat-to-bloom lag.

### Main input

```text
Fig22_seawise_metrics.csv
```

Required fields include:

```text
sea
sic_active_mean_percent
poc_active_mean
pic_active_mean
retreat_day_mean
bloom_day_mean
lag_days_mean
```

### Seasonal timing treatment

Retreat and bloom day-of-year values are transformed to:

```text
days since 1 September
```

to keep the Antarctic spring–summer seasonal cycle continuous through the December–January calendar boundary.

If a bloom date appears before the corresponding retreat date only because of year crossing, the bloom timing is shifted forward by 365 days.

### Carbon units

POC is plotted as:

```text
mg C m⁻³
```

PIC is multiplied by 1000 for the figure and shown at a finer plotting scale.

### Statistical treatment

The script uses `scipy.stats.linregress` and reports:

- correlation coefficient `r`;
- p-value;
- number of valid Antarctic seas.

---

## `Figure-24.py` — Sea-wise SIC, POC, and chlorophyll-a monthly time-series product

**This is currently a preprocessing/data-product script rather than a plotting workflow.**

The current repository version uses an already prepared sea-wise SIC CSV and processes only the gridded POC and chlorophyll-a datasets.

### Main inputs

```text
SIC_seawise_monthly_2008_2025_LOW_RAM.csv
POC_monthly_2008_2025_SO.nc
CHL_monthly_2008_2025_SO.nc
```

### Main output

```text
Fig23_seawise_monthly_timeseries_LOW_RAM.csv
```

saved under:

```text
/content/drive/MyDrive/SAM_Thesis/Processed/
```

### Analysis period

```text
2008-01-01 to 2025-12-31
```

The output is forced onto the complete monthly sequence from:

```text
January 2008 through December 2025
```

### Processing strategy

The script:

- loads the existing SIC sea-wise time series rather than recalculating SIC;
- standardizes NetCDF latitude, longitude, and time coordinates;
- restricts POC and CHL to 90°S–60°S;
- converts irregular/high-frequency data to monthly means when more than 216 time steps are detected;
- squeezes a depth dimension to its first level if a depth-like dimension is present;
- uses Dask/chunked NetCDF reading;
- calculates area-weighted means using cosine-latitude weights;
- processes all 13 Antarctic shelf-sea longitude sectors;
- merges SIC, POC, and CHL into one wide monthly table;
- preserves missing values rather than filling them with zeros;
- reindexes the final product to the complete 216-month study period.

### Output column structure

The resulting CSV contains:

```text
time

SIC_WED ... SIC_BEL
POC_WED ... POC_BEL
CHL_WED ... CHL_BEL
```

for the 13 Antarctic shelf seas.

### Low-RAM settings

The current code opens POC and CHL using chunking approximately equivalent to:

```text
time = 1
lat  = 120
lon  = 720
```

and computes one sea-wise time series at a time.

This current `Figure-24.py` therefore serves as an **analysis-ready monthly data-preparation step for later sea-ice/carbon/ecosystem analyses**.

---

## `Figure-25.py` — Lagged SIC–ecosystem relationships

This workflow examines lagged relationships between SIC variability and:

- POC;
- chlorophyll-a;
- integrated NPP (`NPPint`).

### Main inputs

```text
Fig25_lag_correlation_mean.csv
Fig25_best_lag_summary_by_sea_season_variable.csv
```

### Mean lag-correlation panels

The workflow displays:

- lag;
- mean correlation;
- an uncertainty envelope;
- best lag from the mean correlation curve.

### Sea-wise bootstrap best-lag panels

For each sea and season:

```text
dot                 = median bootstrap best lag
thick vertical line = 25th–75th percentile interval
```

The seasonal panels are:

```text
Spring
Summer
Early autumn
```

The current code explicitly keeps sea labels visible in all three seasonal panels.

---

## `Figure-26.py` — Sea-wise PIC and POC distributions

This script compares seasonal PIC and POC distributions across the 13 shelf seas using boxplots.

### Panels

```text
(a) PIC
(b) POC
```

### Seasons

```text
Spring       = SON
Summer       = DJF
Early autumn = March–April
```

### Units

PIC is converted to:

```text
mg C m⁻³
```

using the carbon molar mass:

```text
12.0107 g mol⁻¹
```

POC is plotted in:

```text
mg C m⁻³
```

### Low-RAM / reproducible sampling settings

```text
BOX_COARSEN            = 10
MAX_SAMPLES_PER_BOX    = 2000
RANDOM_SEED            = 42
SHOW_FLIERS            = False
```

The random seed makes subsampling reproducible.

---

## `Figure-27.py` — Seasonal phytoplankton functional-type distributions

This workflow maps the seasonal spatial distribution of:

- diatoms;
- haptophytes;
- picophytoplankton.

### Units

```text
mg Chl m⁻³
```

### Main inputs

```text
DIATO_monthly_2008_2025_SO.nc
HAPTO_monthly_2008_2025_SO.nc
PICO_monthly_2008_2025_SO.nc
```

The code also includes a safety check for an accidental:

```text
.nc.nc
```

PICO filename and falls back to the corresponding `.nc` file if it exists.

### Figure structure

Columns:

```text
Spring
Summer
Early autumn
```

Rows:

```text
Diatoms
Haptophytes
Picophytoplankton
```

The workflow uses low-RAM spatial coarsening and separate display ranges for the three PFT products.

---

## `Figure-28.py` — Upper-ocean structure and biological-carbon-pump relationships

This script examines sea-wise relationships among POC, chlorophyll-a, mixed-layer depth, and upper-ocean stratification.

### Main input

```text
Fig24_BCP_seawise_summary_ACTIVE_LOW_RAM.csv
```

Required columns include:

```text
sea
poc_active_mean
chl_active_mean
mld_active_mean
strat_active_mean
```

### Figure structure

```text
(a) POC vs MLD
(b) Chlorophyll-a vs MLD
(c) Stratification vs productivity / bloom proxy
(d) Conceptual biological carbon pump pathway
```

### Statistical treatment

The quantitative panels use linear regression and report:

- `r`;
- p-value;
- number of valid seas.

The fourth panel is a **conceptual schematic**, constructed using Matplotlib shapes and arrows rather than a fitted statistical/causal model.

Accordingly, the conceptual panel should be interpreted as a schematic representation of proposed or proxy-based biological-carbon-pump conditions, not as a direct measurement of carbon export or sequestration.

---

# Seasonal definitions

The main Objective 3 workflows generally use:

```text
Spring       = September–November
Summer       = December–February
Early autumn = March–April
```

The use of **March–April** instead of the complete MAM season is retained from the source workflows where May is unavailable.

Where seasonal years are calculated from monthly fields, December is assigned to the following DJF season-year.

---

# Principal datasets

The Objective 3 scripts reference a combination of satellite-derived, processed, and physical-ocean datasets, including:

- PIC;
- POC;
- chlorophyll-a;
- volumetric NPP;
- euphotic depth;
- PAR;
- diatom chlorophyll;
- haptophyte chlorophyll;
- picophytoplankton chlorophyll;
- OSTIA sea-ice concentration / fraction;
- sea-wise SIC time series;
- mixed-layer depth;
- upper-ocean stratification;
- derived bloom and retreat timing products.

Large source datasets are referenced through Google Drive and are not stored in this repository.

---

# Analysis-ready supporting products

Several figure scripts use processed CSV products created elsewhere in the broader Objective 3 workflow.

Examples include:

```text
Obj2Fig21_PFT_relative_contribution.csv

Fig22_seawise_metrics.csv

Fig23_seawise_monthly_timeseries_LOW_RAM.csv

Fig25_lag_correlation_mean.csv
Fig25_best_lag_summary_by_sea_season_variable.csv

Fig24_BCP_seawise_summary_ACTIVE_LOW_RAM.csv
```

The current `Figure-24.py` directly creates:

```text
Fig23_seawise_monthly_timeseries_LOW_RAM.csv
```

from the SIC, POC, and CHL source products.

---

# Google Drive directory structure

The scripts mainly reference:

```text
/content/drive/MyDrive/SAM_Thesis/Data/
/content/drive/MyDrive/SAM_Thesis/Processed/
/content/drive/MyDrive/SAM_Thesis/Fig/
```

Before running a workflow, verify the relevant input paths near the beginning of the script.

---

# Python environment

The repository is designed primarily for **Google Colab**.

Packages used across the workflows include:

```text
numpy
pandas
xarray
dask
netCDF4
h5netcdf
scipy
matplotlib
cartopy
adjustText
```

Some scripts use Colab/Jupyter package installation syntax such as:

```text
!pip
```

so they should be run in Google Colab or another compatible notebook environment unless those commands are converted to standard Python package-installation calls.

---

# Typical workflow

1. Open the required `.py` file in Google Colab.
2. Mount Google Drive.
3. Confirm that all required source and processed files exist at the configured paths.
4. Install any missing packages.
5. Run the workflow from top to bottom.
6. Check the configured output directory for the generated CSV or figure.

For scripts that depend on processed tables, generate those data products first.

A practical dependency example is:

```text
existing sea-wise SIC CSV
        +
gridded POC
        +
gridded chlorophyll-a
        ↓
Figure-24.py
        ↓
Fig23_seawise_monthly_timeseries_LOW_RAM.csv
        ↓
later sea-wise carbon / ecosystem analyses
```

---

# Reproducibility notes

The current repository preserves important analysis choices directly in the code, including:

- the main **2008–2025** analysis period;
- the same 13 Antarctic shelf-sea sectors;
- Southern Hemisphere seasonal definitions;
- explicit DJF handling;
- March–April early-autumn definition where used;
- longitude normalization to `[-180°, 180°)`;
- cosine-latitude weighting for sea-wise gridded averages;
- PIC conversion using **12.0107 g mol⁻¹** as the carbon molar mass;
- integrated NPP estimated as `NPPv × Zeu`;
- reproducible random subsampling in the PIC/POC boxplot workflow;
- low-RAM Dask/chunked processing for large gridded fields;
- preservation of missing observations as missing rather than assigning zero;
- bootstrap-based best-lag summaries in the SIC–ecosystem workflow.

Users reproducing the analysis should preserve these settings unless intentionally performing sensitivity tests.

---

# Repository structure

```text
.
├── Figure-20.py
├── Figure-21.py
├── Figure-22.py
├── Figure-23.py
├── Figure-24.py
├── Figure-25.py
├── Figure-26.py
├── Figure-27.py
├── Figure-28.py
├── .gitignore
├── LICENSE
└── README.md
```

---

# License

This repository is distributed under the license included in [`LICENSE`](./LICENSE).

---

# Repository scope

This repository provides the Python workflows for **Objective 3**, documenting seasonal particulate-carbon variability, phytoplankton biomass and functional-type structure, productivity and light conditions, sea-ice–carbon timing, lagged ecosystem responses, low-RAM sea-wise SIC/POC/chlorophyll data preparation, and upper-ocean controls on biological-carbon-pump proxies across the Antarctic shelf seas.
