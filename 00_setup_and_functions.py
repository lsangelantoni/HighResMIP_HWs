"""
00_setup_and_functions.py

Shared imports, settings, model metadata, and utility functions for the
SOM-based HighResMIP heatwave analysis. This module is imported by every
other script in this pipeline (01-07) and is not meant to be run directly.

Usage in each downstream script:
    from importlib import import_module
    setup = import_module("00_setup_and_functions")
    # or, if renamed to a valid module name (e.g. som_common.py):
    from som_common import *
"""

import numpy as np
import xarray as xr
import os
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.ticker as mticker
import pandas as pd
from scipy.signal import detrend
from scipy import stats
import scipy
from minisom import MiniSom
from scipy import signal
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

from functions import crop_dom
os.environ["ESMFMKFILE"] = "/users_home/cmcc/ls21622/.conda/envs/DEVELOP/lib/esmf.mk"
import xesmf as xe
from functions import regrid as regrid_func  # renamed to avoid clashing with local `regrid` below
from functions import shift_lon
from functions import detect_hw_grid_numba
from functions import extract_domain_events
from functions import compute_metrics
from functions import process_file_models
from functions import process_file_era5
import pymannkendall as mk
from joblib import Parallel, delayed

# ================================================================
# Main settings
# ================================================================
normalize = False

ystart = 1975
yend = 2014

VAR = 'zg'
VAR_TIT = 'Z500' if VAR == 'zg' else 'SST'

som_grid_rows = 3
som_grid_cols = 5
iterations = 10000
learning_rate = 0.1
sigma = 1.0
res = '125'

som_path = '/work/cmcc/ls21622/HighResMIP/ERA5/som/claude'

# NOTE: lon/lat bounds were reassigned twice in the original notebook
# (first to a wider domain, then narrowed) -- only the FINAL, actually
# effective values are kept here.
lon_min = -50
lon_max = 40
lat_min = 25
lat_max = 70

models = ['CMCC-CM2_r1', 'EC-Earth3P_r1', 'EC-Earth3P_r2', 'EC-Earth3P_r3',
          'ECMWF-IFS_r1', 'ECMWF-IFS_r2', 'ECMWF-IFS_r3',
          'MPI-ESM1-2_r1']

models_lr = ['CMCC-CM2-HR4', 'EC-Earth3P', 'EC-Earth3P', 'EC-Earth3P',
             'ECMWF-IFS-LR', 'ECMWF-IFS-LR', 'ECMWF-IFS-LR',
             'MPI-ESM1-2-HR']

models_hr = ['CMCC-CM2-VHR4', 'EC-Earth3P-HR', 'EC-Earth3P-HR', 'EC-Earth3P-HR',
             'ECMWF-IFS-HR', 'ECMWF-IFS-HR', 'ECMWF-IFS-HR',
             'MPI-ESM1-2-XR']

realizations = ['r1i1p1f1', 'r1i1p2f1', 'r2i1p2f1', 'r3i1p2f1',
                'r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1',
                'r1i1p1f1']

metrics = ['mean_tmax', 'magnitude', 'evap_fraction', 'evap_deficit', 'evap_deficit_delta']
grids = ["gn",
         "gr", "gr", "gr",
         "gr", "gr", "gr",
         "gn"]
cmap_fill = np.array([255, 239, 219]) / 255

n_iter = 5000  # Monte Carlo iterations (some sections locally override to 2000)
n_nodes = som_grid_rows * som_grid_cols

# Convenience index groups into `models` for the two multi-realization
# ensembles (used by the internal-variability / ensemble-spread analyses)
ec_earth_idx = [1, 2, 3]   # EC-Earth3P r1, r2, r3
ecmwf_idx = [4, 5, 6]      # ECMWF-IFS  r1, r2, r3

# ================================================================
# Regrid target grid (shared across all preprocessing calls)
# ================================================================
ds_grid_out = xr.open_dataset(
    '/work/cmcc/ls21622/HighResMIP/CMCC-CM2-HR4/r1i1p1f1/zg/'
    'zg_day_CMCC-CM2-HR4_hist-1950_r1i1p1f1_gn_1951-2014_AMJJASO.nc'
)
method = 'bilinear'


def regrid(ds, ds_out, method):
    regridder = xe.Regridder(ds, ds_out, method=method)
    return regridder(ds, ds_out, method)


# ================================================================
# Shared helper functions
# ================================================================
def node_number(row, col, ncols=som_grid_cols):
    """Convert (row, col) SOM grid coordinates to a 1-indexed node number.
    NOTE: consolidates what was previously redefined locally (identically)
    in ~7 separate places throughout the original monolithic notebook."""
    return row * ncols + col + 1


def create_triangle(x, y, orientation, color, edgecolor='black', linewidth=0.5):
    """Create a triangle patch for the split-cell (LR/HR) heatmap figures.
    orientation: 'lower' for bottom-left triangle (LR), 'upper' for
    top-right triangle (HR).
    NOTE: consolidates what was previously redefined locally (identically)
    in ~7 separate places throughout the original monolithic notebook."""
    if orientation == 'lower':
        vertices = [(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x - 0.5, y + 0.5)]
    else:
        vertices = [(x + 0.5, y + 0.5), (x + 0.5, y - 0.5), (x - 0.5, y + 0.5)]
    return plt.Polygon(vertices, facecolor=color, edgecolor=edgecolor,
                       linewidth=linewidth, closed=True)


def build_hw_intensity_df(tasmax_anom, ds_som):
    """
    Build DataFrame of HW-day Tmax anomaly intensity per SOM node.

    Returns
    -------
    df : pandas.DataFrame
        columns = ['time', 'node_id', 'tmax_anom']
    """
    ncol = ds_som.som_col.size
    node_id = (ds_som.bmu_row * ncol + ds_som.bmu_col).rename("node_id")
    hw_intensity = tasmax_anom.mean(dim=("lat", "lon"), skipna=True)

    df = (
        xr.Dataset({
            "tmax_anom": hw_intensity,
            "node_id": ("time", node_id.values),
        })
        .to_dataframe()
        .reset_index()
    )
    return df, node_id


def build_combined_intensity_df(tasmax_anom_lr, tasmax_anom_hr, ds_som_lr, ds_som_hr,
                                 models, models_lr, models_hr):
    """Build combined DataFrame for both resolutions with resolution labels."""
    df_lr_list, df_hr_list = [], []

    for model_name, model_name_lr, model_name_hr, realization in zip(
            models, models_lr, models_hr, realizations):

        def build_single_df(tasmax_anom, ds_som, resolution):
            ncol = ds_som.som_col.size
            node_id = (ds_som.bmu_row * ncol + ds_som.bmu_col).rename("node_id")
            hw_intensity = tasmax_anom.mean(dim=("lat", "lon"), skipna=True)
            df = xr.Dataset({
                "tmax_anom": hw_intensity,
                "node_id": ("time", node_id.values),
            }).to_dataframe().reset_index()
            df['resolution'] = resolution
            df['model'] = model_name
            return df

        print(f"Processing: {model_name_lr} -> {model_name_lr}_{realization[0:2]}")
        df_lr_list.append(build_single_df(
            tasmax_anom_lr[f"{model_name_lr}_{realization[0:2]}"],
            ds_som_lr[f"{model_name_lr}_{realization[0:2]}"], 'LowRes'))
        print(f"Processing: {model_name_hr} -> {model_name_hr}_{realization[0:2]}")
        df_hr_list.append(build_single_df(
            tasmax_anom_hr[f"{model_name_hr}_{realization[0:2]}"],
            ds_som_hr[f"{model_name_hr}_{realization[0:2]}"], 'HighRes'))

    df_lr_combined = pd.concat(df_lr_list, ignore_index=True)
    df_hr_combined = pd.concat(df_hr_list, ignore_index=True)
    df_combined = pd.concat([df_lr_combined, df_hr_combined], ignore_index=True)

    return df_lr_combined, df_hr_combined, df_combined


def calculate_daily_climatology(ds, var_name, window=31):
    ds = ds.sel(time=slice(f"{ystart}-01-01", f"{ystart + 30}-12-31"))
    clim = (
        ds[var_name]
        .groupby("time.dayofyear")
        .mean("time")
        .rolling(dayofyear=window, center=True, min_periods=1)
        .mean()
    )
    clim = clim.sel(dayofyear=clim.dayofyear != 366)
    return clim


def preprocess_for_som(data_all, var_name):
    """Load, regrid, crop, compute daily climatology and detrended,
    latitude-weighted anomalies for a single variable/model file."""
    ds = np.squeeze(xr.open_dataset(data_all))

    if "valid_time" in ds:
        ds = ds.rename({"valid_time": "time"})
    if "longitude" in ds:
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    if "z" in ds:
        ds['z'] = ds['z'] / 10
        ds = ds.rename({"z": VAR})
    if "tos" in ds:
        ds['tos'] = ds['tos'] - 273.16
    if "sst" in ds:
        ds = ds.rename({"sst": "tos"})
        ds['tos'] = ds['tos'] - 273.16

    ds = regrid(ds, ds_grid_out, method)

    ds = ds.sel(time=slice(f"{ystart}-01-01", f"{yend}-12-31"))
    ds = ds.sel(time=ds.time.dt.month.isin([5, 6, 7, 8, 9]))
    ds = ds.sortby("lat")
    ds = crop_dom(ds, lon_min, lon_max, lat_min, lat_max)

    field = ds[var_name]

    clim = calculate_daily_climatology(ds, var_name)
    anomaly = field.groupby("time.dayofyear") - clim

    def detrend_1d(x):
        valid = np.isfinite(x)
        if valid.sum() < 2:
            return x
        out = np.full_like(x, np.nan)
        out[valid] = signal.detrend(x[valid])
        return out

    anomaly_dt = xr.apply_ufunc(
        detrend_1d, anomaly,
        input_core_dims=[["time"]], output_core_dims=[["time"]],
        vectorize=True, dask="allowed",
    )

    std = anomaly_dt.std("time", skipna=True)
    std = std.where(std > 0)

    weights = np.sqrt(np.cos(np.deg2rad(anomaly_dt.lat)))
    if normalize:
        anomaly_std = anomaly_dt / std
        X_weighted = anomaly_std * weights.broadcast_like(anomaly_dt)
    else:
        X_weighted = anomaly_dt * weights.broadcast_like(anomaly_dt)

    print('preprocessing done')
    return anomaly_dt, X_weighted, clim
