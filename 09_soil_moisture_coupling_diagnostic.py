"""
09_soil_moisture_coupling_diagnostic.py

Diagnostic for Reviewer 2, Sec 3.4(1) ("Attribution Uncertainty in
Blocking Regime Frequency Changes").

The Reviewer's alternative explanation has two distinct parts, and this
script tests them separately:

  (A) STATIC COUPLING CONTRAST -- is soil moisture-temperature coupling
      stronger on the HW days of regime 11 than on all MJJAS days
      assigned to regime 11?

  (B) TREND CONFOUND (the actual claim) -- are regime-11 days occurring
      under progressively DRIER conditions over the study period, such
      that "identical circulation regimes produce stronger heatwaves"
      without any change in the circulation-regime relationship itself?

Regime 11 is the target because it is the one regime showing both a
statistically significant increasing ERA5 trend (Section 3.2.5) and
HW-promoting status (Section 3.2.3) -- i.e. the regime whose
"efficiency increase" claim (lines 767-770) this comment questions.

Coupling is quantified via the Terrestrial Coupling Index (TCI; Guo
et al. 2006; Dirmeyer 2011), TCI = std(response) * Corr(SM, response),
here applied with temperature as the response variable (an adaptation
of Dirmeyer's flux-based original, in the same spirit as Seneviratne
et al. 2006's adaptation of Koster et al.'s 2004 precipitation-based
metric). A plain Pearson correlation is also reported for reference.

--------------------------------------------------------------------
METHODOLOGICAL NOTES (differences from the first draft of this script)
--------------------------------------------------------------------
1. ANOMALIES, NOT RAW VALUES. Across MJJAS the domain-mean Tmax rises
   ~10 K while soil moisture dries down, so a correlation of raw daily
   values is dominated by the seasonal cycle and returns a strongly
   negative r with zero land-atmosphere feedback. Worse, it biases the
   headline contrast in a known direction: regime-11 HW days cluster in
   Jul-Aug, a narrower seasonal window carrying less seasonal-cycle
   variance, so they would show a *weaker* raw correlation for purely
   calendrical reasons. Both fields are therefore converted to
   anomalies from a smoothed day-of-year climatology, per grid point,
   before any spatial averaging. A linearly detrended variant is also
   reported, so that (A) measures day-to-day covariability and cannot
   be inflated by the co-trending that (B) tests explicitly.

2. MASKING: BOTH GROUPS GET A WHOLE-DOMAIN VERSION AND A MASKED
   VERSION, COMPUTED SYMMETRICALLY. An earlier draft of this script
   assumed the HW footprint mask (sub_hw) could not be applied to the
   all-days baseline, since sub_hw was assumed to be 0 everywhere on
   days that are not official HW days. This assumption was WRONG:
   sub_hw is a per-pixel, per-day exceedance field (tmax > p90 at that
   grid cell, that day), computed on every summer day, not only on
   days meeting the domain-wide 30% threshold that defines an official
   "HW day". The mask is therefore valid and meaningful on every day,
   and is applied symmetrically to both groups (see diagnostic A2
   below), in addition to the whole-domain comparison (A), which is
   retained since it answers a related but distinct question (the
   day-to-day covariability of domain-mean SM and Tmax, rather than
   the covariability specifically within each day's locally
   hot-exceeding footprint).

3. THE DIFFERENCE IS TESTED, not read off an ordering. Fisher z on the
   two correlations (with an autocorrelation-adjusted effective sample
   size) and a moving-block bootstrap on the TCI difference.

4. THE TREND HAS A CONTROL. A drying trend on regime-11 days is only
   evidence for the Reviewer's confound if it exceeds the background
   drying trend over all MJJAS days. Both are computed, plus their
   difference. Trends use Theil-Sen slopes with a Mann-Kendall test
   whose variance is corrected for serial correlation (Hamed & Rao
   1998); ordinary least squares on yearly means gives optimistic
   p-values here.

5. THE SOIL MOISTURE PERIOD IS DISCOVERED, NOT ASSUMED. Actual file
   coverage is reported and used in every label, rather than assumed
   to match the full study period.

6. Latitude-weighted spatial means; loud failure instead of a silent
   fallback variable; explicit grid-consistency check; duplicate time
   labels dropped.

7. TWO SOIL LAYERS MERGED, NOT TREATED SEPARATELY. ERA5 layer 1
   (swvl1, 0-7 cm) and layer 2 (swvl2, 7-28 cm) are now both available
   for the full study period and are combined into a single 0-28 cm
   volumetric soil moisture field via a depth-weighted average,
   SM = (7*swvl1 + 21*swvl2) / 28, rather than using layer 2 alone.
   This is the standard way to combine ERA5 layers of different
   thickness into one physically meaningful profile-average moisture
   value (a simple unweighted average of the two layers would
   implicitly treat a 7 cm layer as equally representative of the
   profile as a 21 cm layer, which is not correct).

Requires: 00_setup_and_functions.py (import * or run first)
"""

import os
import warnings

from importlib import import_module

setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress, norm, theilslopes

# ================================================================
# Configuration
# ================================================================
TARGET_REGIME = 11  # 1-indexed, matches manuscript regime numbering
node_target = TARGET_REGIME - 1

min_lon, max_lon = -10, 15
min_lat, max_lat = 42, 58

# ERA5 soil moisture layers to merge (see methodological note 7).
SM_VARNAME_L1 = "swvl1"   # 0-7 cm
SM_VARNAME_L2 = "swvl2"   # 7-28 cm
L1_DEPTH_CM = 7.0
L2_DEPTH_CM = 21.0
TOTAL_DEPTH_CM = L1_DEPTH_CM + L2_DEPTH_CM  # 28 cm
SM_LAYER1_DIR = "/data/cmcc/ls21622/ERA5/scripts_download/lev_1"
SM_LAYER2_DIR = "/data/cmcc/ls21622/ERA5/scripts_download/lev_2"

CLIM_SMOOTH_WINDOW = 15   # days, for the day-of-year climatology
MIN_DAYS_PER_YEAR = 3     # skip years with too few regime days for a stable mean
BOOTSTRAP_N = 5000        # moving-block bootstrap replicates
BOOTSTRAP_BLOCK = 5       # days per block
RANDOM_SEED = 20240517

# What to do if SM / Tmax / HW-mask are not on the same grid after
# cropping. "raise" is the default because xarray would otherwise
# inner-join the coordinates silently and leave very few (or zero)
# shared points. Set to "nearest" only if you know the grids are
# compatible and nearest-neighbour sampling is acceptable.
GRID_MISMATCH_ACTION = "raise"   # "raise" | "nearest"

# Set True only when running interactively.
SHOW_FIGURE = False

OUT_PREFIX = f"regime{TARGET_REGIME}_soil_moisture_diagnostic"

rng = np.random.default_rng(RANDOM_SEED)

# Categorical colours (fixed entity -> colour mapping, never cycled)
C_ALLDAYS = "#2a78d6"   # regime-11, all MJJAS days
C_HWDAYS = "#eb6834"    # regime-11, HW days
C_CONTROL = "#1baf7a"   # all MJJAS days, every regime (background control)
C_INK = "#0b0b0b"
C_INK_2 = "#52514e"
C_GRID = "#dcdbd6"


# ================================================================
# Helpers
# ================================================================
def dedup_time(ds):
    """Drop duplicated time labels (overlapping input files) so that
    later .reindex() calls do not raise on a non-unique index."""
    if "time" not in ds.dims:
        return ds
    try:
        return ds.drop_duplicates("time")
    except AttributeError:  # xarray < 2022.03
        _, keep = np.unique(ds.time.values, return_index=True)
        return ds.isel(time=np.sort(keep))


def normalise_time(ds):
    ds = ds.copy()
    ds["time"] = pd.to_datetime(ds.time.astype("datetime64[ns]")).normalize()
    return ds


def standard_names(ds):
    """Rename ERA5's longitude/latitude/valid_time to lon/lat/time, each
    only if actually present (the first draft assumed valid_time existed
    whenever longitude did)."""
    mapping = {}
    for src, dst in (("longitude", "lon"), ("latitude", "lat"), ("valid_time", "time")):
        if src in ds.variables and dst not in ds.variables:
            mapping[src] = dst
    return ds.rename(mapping) if mapping else ds


def grids_match(da_a, da_b, tol=1e-4):
    for dim in ("lat", "lon"):
        a, b = da_a[dim].values, da_b[dim].values
        if a.shape != b.shape or np.max(np.abs(a - b)) > tol:
            return False
    return True


def reconcile_grid(da, reference, name):
    """Return `da` on `reference`'s horizontal grid, or raise."""
    if grids_match(da, reference):
        return da
    msg = (f"'{name}' is not on the reference grid after cropping "
           f"(lat {da.lat.size} vs {reference.lat.size}, "
           f"lon {da.lon.size} vs {reference.lon.size}). xarray would "
           f"silently inner-join these coordinates and could leave almost "
           f"no shared grid points.")
    if GRID_MISMATCH_ACTION == "nearest":
        print(f"[grid] {msg}\n[grid] Regridding '{name}' by nearest neighbour.")
        return da.reindex(lat=reference.lat, lon=reference.lon, method="nearest")
    raise ValueError(msg + " Set GRID_MISMATCH_ACTION='nearest' to sample it "
                           "onto the reference grid, or regrid the inputs "
                           "properly (conservative remapping) beforehand.")


def area_weighted_mean(da):
    """cos(lat)-weighted spatial mean. Weights renormalise over the
    non-NaN points, so this also does the right thing for the
    HW-footprint-masked fields."""
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(dim=["lat", "lon"], skipna=True)


def daily_anomalies(da, window=CLIM_SMOOTH_WINDOW):
    """Per-grid-point anomalies from a smoothed day-of-year climatology.

    The climatology is built from the full available MJJAS record. Leap
    years shift the day-of-year index by one after February; with a
    `window`-day running mean that offset is negligible here.
    """
    clim = da.groupby("time.dayofyear").mean("time")
    clim = clim.rolling(dayofyear=window, center=True, min_periods=1).mean()
    anom = da.groupby("time.dayofyear") - clim
    return anom.drop_vars("dayofyear", errors="ignore")


def detrend_series(s):
    """Remove the least-squares linear trend in time from a daily series."""
    if len(s) < 3:
        return s
    x = s.index.year.values + (s.index.dayofyear.values - 1) / 366.0
    fit = linregress(x, s.values)
    return pd.Series(s.values - (fit.slope * x + fit.intercept), index=s.index)


def lag1_autocorr(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x[:-1] * x[1:]) / denom)


def effective_n(x, y):
    """Effective sample size for a correlation between two serially
    correlated series (Bretherton et al. 1999)."""
    n = len(x)
    r1 = lag1_autocorr(x) * lag1_autocorr(y)
    if r1 <= 0:
        return float(n)
    return float(np.clip(n * (1 - r1) / (1 + r1), 4, n))


def fisher_z_difference(r1, n1, r2, n2):
    """Two-sided test that two independent correlations differ."""
    if min(n1, n2) <= 3:
        return np.nan, np.nan
    z1, z2 = np.arctanh(np.clip(r1, -0.999999, 0.999999)), np.arctanh(np.clip(r2, -0.999999, 0.999999))
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z1 - z2) / se
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def tci(sm, response):
    """Terrestrial Coupling Index with temperature as the response."""
    r = np.corrcoef(sm, response)[0, 1]
    return float(np.std(response, ddof=1) * r), float(r)


def moving_block_bootstrap_tci(sm, response, n_boot=BOOTSTRAP_N, block=BOOTSTRAP_BLOCK):
    """Bootstrap distribution of TCI.

    Blocks are taken over the subset's own ordering. Because a regime
    subset is a set of non-contiguous calendar days, a 'block' groups
    days that are usually but not always consecutive; this is an
    approximation that keeps most of the within-spell dependence.
    """
    sm = np.asarray(sm, dtype=float)
    response = np.asarray(response, dtype=float)
    n = len(sm)
    if n < 2 * block:
        return np.full(n_boot, np.nan)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out[i] = tci(sm[idx], response[idx])[0]
    return out


def mann_kendall(values, autocorr_correction=True):
    """Mann-Kendall trend test with the Hamed & Rao (1998) variance
    correction for serial correlation, plus a Theil-Sen slope.

    `values` must be evenly spaced in time (one value per year here).
    Returns (theil_sen_slope, p_value, correction_factor).
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan

    t = np.arange(n, dtype=float)
    s = sum(np.sum(np.sign(x[k + 1:] - x[k])) for k in range(n - 1))

    _, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0

    correction = 1.0
    if autocorr_correction and n >= 10:
        ts_slope = theilslopes(x, t)[0]
        ranks = stats.rankdata(x - ts_slope * t)
        rk = ranks - ranks.mean()
        denom = np.sum(rk ** 2)
        if denom > 0:
            acc = 0.0
            for lag in range(1, n - 2):
                r = np.sum(rk[:n - lag] * rk[lag:]) / denom
                m = n - lag
                lo = (-1 - 1.645 * np.sqrt(m - 1)) / m
                hi = (-1 + 1.645 * np.sqrt(m - 1)) / m
                if r <= lo or r >= hi:
                    acc += m * (m - 1) * (m - 2) * r
            correction = 1 + (2.0 / (n * (n - 1) * (n - 2))) * acc
            correction = float(max(correction, 0.05))  # guard pathological cases
        var_s *= correction

    if var_s <= 0:
        return np.nan, np.nan, correction
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(theilslopes(x, t)[0]), float(p), float(correction)


def yearly_mean_with_threshold(series, min_days=MIN_DAYS_PER_YEAR):
    """Yearly means, dropping years with too few days for a stable mean."""
    years = series.index.year
    counts = series.groupby(years).size()
    means = series.groupby(years).mean()
    return means.loc[counts[counts >= min_days].index]


def stars(p):
    if not np.isfinite(p):
        return "n/a"
    return "significant (p<0.05)" if p < 0.05 else "not significant"


# ================================================================
# Load ERA5 soil moisture, BOTH layers, and MERGE them into a single
# depth-weighted 0-28 cm volumetric soil moisture field (methodological
# note 7). A given (year, month) is only used if BOTH layers' files
# exist for it, so the two layers are guaranteed to cover the same
# days before combining.
# ================================================================
requested_years = list(range(ystart, yend + 1))

paired_files = []
for yr in requested_years:
    for mo in [5, 6, 7, 8, 9]:
        f1 = f"{SM_LAYER1_DIR}/ERA5_SM_daily_{yr}_{mo}.nc"
        f2 = f"{SM_LAYER2_DIR}/ERA5_SM_daily_{yr}_{mo}.nc"
        if os.path.exists(f1) and os.path.exists(f2):
            paired_files.append((yr, f1, f2))

if not paired_files:
    raise FileNotFoundError(
        f"No matched layer-1/layer-2 soil-moisture file pairs found for "
        f"{ystart}-{yend}, months 5-9, under {SM_LAYER1_DIR} and {SM_LAYER2_DIR}.")

sm_years_present = sorted({p[0] for p in paired_files})
SM_YSTART, SM_YEND = min(sm_years_present), max(sm_years_present)
SM_PERIOD = f"{SM_YSTART}-{SM_YEND}"

l1_files = [p[1] for p in paired_files]
l2_files = [p[2] for p in paired_files]

print(f"Loading {len(l1_files)} matched layer-1/layer-2 soil moisture file pairs "
      f"({SM_PERIOD}, months 5-9)")
if (SM_YSTART, SM_YEND) != (ystart, yend):
    warnings.warn(
        f"\n*** Soil-moisture record (both layers matched) covers {SM_PERIOD}, but "
        f"the analysis period is {ystart}-{yend}. Every soil-moisture result below "
        f"is for {SM_PERIOD} ONLY. Extend the input files before quoting these "
        f"numbers in the response. ***\n", RuntimeWarning)

ds_sm_l1 = xr.open_mfdataset(l1_files, combine="by_coords")
ds_sm_l1 = standard_names(ds_sm_l1)
ds_sm_l2 = xr.open_mfdataset(l2_files, combine="by_coords")
ds_sm_l2 = standard_names(ds_sm_l2)

if SM_VARNAME_L1 not in ds_sm_l1.data_vars:
    raise KeyError(
        f"'{SM_VARNAME_L1}' not found in the layer-1 soil-moisture files. Available: "
        f"{list(ds_sm_l1.data_vars)}.")
if SM_VARNAME_L2 not in ds_sm_l2.data_vars:
    raise KeyError(
        f"'{SM_VARNAME_L2}' not found in the layer-2 soil-moisture files. Available: "
        f"{list(ds_sm_l2.data_vars)}.")

ds_sm_l1 = ds_sm_l1.sortby("lat")
ds_sm_l1 = crop_dom(ds_sm_l1, min_lon, max_lon, min_lat, max_lat, lon_name="lon", lat_name="lat")
ds_sm_l1 = ds_sm_l1.sel(time=ds_sm_l1.time.dt.month.isin([5, 6, 7, 8, 9]))
ds_sm_l1 = normalise_time(dedup_time(ds_sm_l1))

ds_sm_l2 = ds_sm_l2.sortby("lat")
ds_sm_l2 = crop_dom(ds_sm_l2, min_lon, max_lon, min_lat, max_lat, lon_name="lon", lat_name="lat")
ds_sm_l2 = ds_sm_l2.sel(time=ds_sm_l2.time.dt.month.isin([5, 6, 7, 8, 9]))
ds_sm_l2 = normalise_time(dedup_time(ds_sm_l2))

# The two layers should already share the same days (matched by file
# pair above); intersect defensively in case either file had missing
# days internally.
common_sm_times = ds_sm_l1.time.to_index().intersection(ds_sm_l2.time.to_index())
ds_sm_l1 = ds_sm_l1.sel(time=common_sm_times)
ds_sm_l2 = ds_sm_l2.sel(time=common_sm_times)

field_l1 = ds_sm_l1[SM_VARNAME_L1]
field_l2 = ds_sm_l2[SM_VARNAME_L2]
field_l2 = reconcile_grid(field_l2, field_l1, "soil moisture layer 2 (vs. layer 1 grid)")

# Depth-weighted merge -> single 0-28 cm volumetric soil moisture field.
sm_field = (L1_DEPTH_CM * field_l1 + L2_DEPTH_CM * field_l2) / TOTAL_DEPTH_CM
sm_field = sm_field.rename("sm_0_28cm")
print(f"Merged soil moisture: SM = ({L1_DEPTH_CM:.0f}*swvl1 + {L2_DEPTH_CM:.0f}*swvl2) "
      f"/ {TOTAL_DEPTH_CM:.0f}  (depth-weighted 0-{TOTAL_DEPTH_CM:.0f} cm volumetric mean)")

# ================================================================
# Load ERA5 tmax
# ================================================================
ds_tmax = xr.open_dataset(
    "/data/cmcc/ls21622/ERA5/tmax/postprocessed/ERA5_tmax_day_1975-2024_AMJJASO.nc")
ds_tmax = standard_names(ds_tmax)
ds_tmax = ds_tmax.sortby("lat")
ds_tmax = crop_dom(ds_tmax, min_lon, max_lon, min_lat, max_lat, lon_name="lon", lat_name="lat")
ds_tmax = ds_tmax.sel(time=ds_tmax.time.dt.month.isin([5, 6, 7, 8, 9]))
ds_tmax = ds_tmax.sel(time=ds_tmax.time.dt.year.isin(range(ystart, yend + 1)))
ds_tmax = normalise_time(dedup_time(ds_tmax))
tmax_field = reconcile_grid(ds_tmax.tasmax, sm_field, "ERA5 tmax")

# ================================================================
# HW footprint mask (used only for the labelled sensitivity below)
# ================================================================
ds_hw_mask = xr.open_mfdataset(
    "/work/cmcc/ls21622/HighResMIP/ERA5/ERA5_sub_heatwave_day_*_CWE.nc")
ds_hw_mask = ds_hw_mask.sortby("lat")
ds_hw_mask = crop_dom(ds_hw_mask, min_lon, max_lon, min_lat, max_lat,
                      lon_name="lon", lat_name="lat")
ds_hw_mask = normalise_time(dedup_time(ds_hw_mask))
hw_mask = reconcile_grid(ds_hw_mask.sub_hw, sm_field, "HW footprint mask")

# ================================================================
# Anomalies from a smoothed day-of-year climatology, per grid point,
# BEFORE spatial averaging (see methodological note 1).
# ================================================================
sm_anom_field = daily_anomalies(sm_field)
tmax_anom_field = daily_anomalies(tmax_field)

sm_anom = area_weighted_mean(sm_anom_field).to_series()
tmax_anom = area_weighted_mean(tmax_anom_field).to_series()

# HW-footprint-restricted means -- sensitivity only, NOT comparable to
# the whole-domain baseline (methodological note 2).
sm_anom_hwmask = area_weighted_mean(sm_anom_field.where(hw_mask == 1)).to_series()
tmax_anom_hwmask = area_weighted_mean(tmax_anom_field.where(hw_mask == 1)).to_series()

# Raw (non-anomaly) whole-domain means, kept only to quantify how much
# of the first draft's correlation was the seasonal cycle.
sm_raw = area_weighted_mean(sm_field).to_series()
tmax_raw = area_weighted_mean(tmax_field).to_series()

# ================================================================
# Regime assignment: all-days SOM and HW-days SOM
# ================================================================
ds_som_all = xr.open_mfdataset(
    f'{som_path}/som_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
ds_som_all = normalise_time(dedup_time(ds_som_all))
node_all = ds_som_all["bmu_row"].values * som_grid_cols + ds_som_all["bmu_col"].values
time_all = ds_som_all.time.values

ds_som_hw = xr.open_mfdataset(
    f'{som_path}/som_projected_HW_days_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
ds_som_hw = normalise_time(dedup_time(ds_som_hw))
node_hw = ds_som_hw["bmu_row"].values * som_grid_cols + ds_som_hw["bmu_col"].values
time_hw = ds_som_hw.time.values

all_days_dates = pd.DatetimeIndex(time_all[node_all == node_target])
hw_days_dates = pd.DatetimeIndex(time_hw[node_hw == node_target])

print(f"Regime {TARGET_REGIME}: {len(all_days_dates)} all-MJJAS days, "
      f"{len(hw_days_dates)} HW days (SOM assignment, {ystart}-{yend})")


def paired_subset(sm_series, t_series, dates, detrend=False):
    """Align an SM and a temperature series on a set of days."""
    sm = sm_series.reindex(dates).dropna()
    tt = t_series.reindex(dates).dropna()
    common = sm.index.intersection(tt.index).sort_values()
    sm, tt = sm.loc[common], tt.loc[common]
    if detrend and len(common) >= 3:
        sm, tt = detrend_series(sm), detrend_series(tt)
    return sm, tt


sm_all, tmax_all = paired_subset(sm_anom, tmax_anom, all_days_dates)
sm_hw, tmax_hw = paired_subset(sm_anom, tmax_anom, hw_days_dates)

sm_all_dt, tmax_all_dt = paired_subset(sm_anom, tmax_anom, all_days_dates, detrend=True)
sm_hw_dt, tmax_hw_dt = paired_subset(sm_anom, tmax_anom, hw_days_dates, detrend=True)

sm_hw_fp, tmax_hw_fp = paired_subset(sm_anom_hwmask, tmax_anom_hwmask, hw_days_dates)
# NEW: the analogous masked series for the all-days group. sub_hw is a
# per-pixel, per-day exceedance field (tmax > p90 at that grid cell,
# that day) computed on EVERY summer day, not only on days meeting the
# domain-wide 30% threshold that defines an official "HW day" -- so,
# unlike an earlier assumption in this script, the mask is not
# structurally restricted to HW days and CAN be applied symmetrically
# to both groups. This gives a properly comparable "locally
# hot-exceeding footprint" contrast, rather than mixing a masked group
# against a whole-domain one.
sm_all_fp, tmax_all_fp = paired_subset(sm_anom_hwmask, tmax_anom_hwmask, all_days_dates)
sm_all_raw, tmax_all_raw = paired_subset(sm_raw, tmax_raw, all_days_dates)
sm_hw_raw, tmax_hw_raw = paired_subset(sm_raw, tmax_raw, hw_days_dates)

if len(sm_all) < 10 or len(sm_hw) < 10:
    raise ValueError(
        f"Too few matched days (all: {len(sm_all)}, HW: {len(sm_hw)}). "
        f"Check that the SOM, soil-moisture and tmax time axes overlap -- "
        f"soil moisture covers {SM_PERIOD} only.")
if len(sm_all_fp) < 10 or len(sm_hw_fp) < 10:
    print(f"[warning] Masked-footprint groups are small (all: {len(sm_all_fp)}, "
          f"HW: {len(sm_hw_fp)}) -- some regime-11 days may have zero grid points "
          f"exceeding the local threshold, and are dropped rather than treated as 0.")

# ================================================================
# (A) STATIC COUPLING CONTRAST
# ================================================================
tci_all, r_all = tci(sm_all.values, tmax_all.values)
tci_hw, r_hw = tci(sm_hw.values, tmax_hw.values)
tci_all_dt, r_all_dt = tci(sm_all_dt.values, tmax_all_dt.values)
tci_hw_dt, r_hw_dt = tci(sm_hw_dt.values, tmax_hw_dt.values)
tci_hw_fp, r_hw_fp = tci(sm_hw_fp.values, tmax_hw_fp.values)
tci_all_fp, r_all_fp = tci(sm_all_fp.values, tmax_all_fp.values)
_, r_all_raw = tci(sm_all_raw.values, tmax_all_raw.values)
_, r_hw_raw = tci(sm_hw_raw.values, tmax_hw_raw.values)

# Significance test for the NOW-SYMMETRIC masked comparison, using the
# same methods as the whole-domain headline test above (Fisher z on
# the correlations; moving-block bootstrap on the TCI difference).
n_eff_all_fp = effective_n(sm_all_fp.values, tmax_all_fp.values)
n_eff_hw_fp = effective_n(sm_hw_fp.values, tmax_hw_fp.values)
z_diff_fp, p_diff_fp = fisher_z_difference(r_all_fp, n_eff_all_fp, r_hw_fp, n_eff_hw_fp)

boot_all_fp = moving_block_bootstrap_tci(sm_all_fp.values, tmax_all_fp.values)
boot_hw_fp = moving_block_bootstrap_tci(sm_hw_fp.values, tmax_hw_fp.values)
ci_all_fp = np.nanpercentile(boot_all_fp, [2.5, 97.5])
ci_hw_fp = np.nanpercentile(boot_hw_fp, [2.5, 97.5])
boot_diff_fp = boot_hw_fp - boot_all_fp
ci_diff_fp = np.nanpercentile(boot_diff_fp, [2.5, 97.5])
p_tci_diff_fp = 2 * min(np.nanmean(boot_diff_fp < 0), np.nanmean(boot_diff_fp > 0))

n_eff_all = effective_n(sm_all.values, tmax_all.values)
n_eff_hw = effective_n(sm_hw.values, tmax_hw.values)
z_diff, p_diff = fisher_z_difference(r_all, n_eff_all, r_hw, n_eff_hw)

boot_all = moving_block_bootstrap_tci(sm_all.values, tmax_all.values)
boot_hw = moving_block_bootstrap_tci(sm_hw.values, tmax_hw.values)
ci_all = np.nanpercentile(boot_all, [2.5, 97.5])
ci_hw = np.nanpercentile(boot_hw, [2.5, 97.5])
boot_diff = boot_hw - boot_all
ci_diff = np.nanpercentile(boot_diff, [2.5, 97.5])
p_tci_diff = 2 * min(np.nanmean(boot_diff < 0), np.nanmean(boot_diff > 0))

print(f"\n{'=' * 68}")
print(f"(A) Soil moisture-temperature coupling, Regime {TARGET_REGIME}")
print(f"    SM = depth-weighted 0-{TOTAL_DEPTH_CM:.0f} cm merge of swvl1+swvl2. "
      f"Anomalies from a {CLIM_SMOOTH_WINDOW}-day-smoothed day-of-year "
      f"climatology; whole-domain, cos(lat)-weighted, on BOTH sides.")
print(f"{'=' * 68}")
print(f"All MJJAS days (n={len(sm_all)}, n_eff={n_eff_all:.0f}): "
      f"r = {r_all:+.3f}   std(T') = {tmax_all.std():.3f} K   "
      f"TCI = {tci_all:+.3f}  [95% CI {ci_all[0]:+.3f}, {ci_all[1]:+.3f}]")
print(f"HW days only   (n={len(sm_hw)}, n_eff={n_eff_hw:.0f}): "
      f"r = {r_hw:+.3f}   std(T') = {tmax_hw.std():.3f} K   "
      f"TCI = {tci_hw:+.3f}  [95% CI {ci_hw[0]:+.3f}, {ci_hw[1]:+.3f}]")
print(f"\nDifference (HW - all days):")
print(f"  correlation: Fisher z = {z_diff:+.2f}, p = {p_diff:.3f} -> {stars(p_diff)}")
print(f"  TCI:         delta = {tci_hw - tci_all:+.3f} "
      f"[95% CI {ci_diff[0]:+.3f}, {ci_diff[1]:+.3f}], "
      f"p = {p_tci_diff:.3f} -> {stars(p_tci_diff)}")

print(f"\nSensitivity / robustness:")
print(f"  linearly detrended:     r_all = {r_all_dt:+.3f}, r_HW = {r_hw_dt:+.3f}, "
      f"TCI_all = {tci_all_dt:+.3f}, TCI_HW = {tci_hw_dt:+.3f}")
print(f"  RAW values (seasonal cycle NOT removed, for reference only):")
print(f"                          r_all = {r_all_raw:+.3f}, r_HW = {r_hw_raw:+.3f}")
print(f"     -> the gap between the raw and anomaly correlations is the "
      f"seasonal-cycle artefact.")

print(f"\n{'=' * 68}")
print(f"(A2) LOCALLY-MASKED coupling contrast (sub_hw==1, both groups)")
print(f"     sub_hw is a per-pixel, per-day exceedance field computed on EVERY")
print(f"     summer day (not only official HW days), so it CAN be applied")
print(f"     symmetrically to both groups -- unlike the whole-domain contrast")
print(f"     above, this compares like with like: the soil moisture-temperature")
print(f"     relationship specifically within each day's locally hot-exceeding")
print(f"     footprint, whether or not that day was part of an official HW event.")
print(f"{'=' * 68}")
print(f"All regime-11 days, masked (n={len(sm_all_fp)}, n_eff={n_eff_all_fp:.0f}): "
      f"r = {r_all_fp:+.3f}   std(T') = {tmax_all_fp.std():.3f} K   "
      f"TCI = {tci_all_fp:+.3f}  [95% CI {ci_all_fp[0]:+.3f}, {ci_all_fp[1]:+.3f}]")
print(f"HW days, masked         (n={len(sm_hw_fp)}, n_eff={n_eff_hw_fp:.0f}): "
      f"r = {r_hw_fp:+.3f}   std(T') = {tmax_hw_fp.std():.3f} K   "
      f"TCI = {tci_hw_fp:+.3f}  [95% CI {ci_hw_fp[0]:+.3f}, {ci_hw_fp[1]:+.3f}]")
print(f"\nDifference (HW - all days), masked:")
print(f"  correlation: Fisher z = {z_diff_fp:+.2f}, p = {p_diff_fp:.3f} -> {stars(p_diff_fp)}")
print(f"  TCI:         delta = {tci_hw_fp - tci_all_fp:+.3f} "
      f"[95% CI {ci_diff_fp[0]:+.3f}, {ci_diff_fp[1]:+.3f}], "
      f"p = {p_tci_diff_fp:.3f} -> {stars(p_tci_diff_fp)}")
print()
if not np.isfinite(p_tci_diff_fp) or p_tci_diff_fp >= 0.05:
    print(f"VERDICT (A2): masked coupling is NOT distinguishable between HW days and")
    print(f"  all regime-{TARGET_REGIME} days -- consistent with (A)'s whole-domain verdict.")
else:
    direction = "MORE NEGATIVE" if tci_hw_fp < tci_all_fp else "LESS negative"
    print(f"VERDICT (A2): masked TCI is significantly {direction} on HW days "
          f"(p = {p_tci_diff_fp:.3f}).")

print()
if not np.isfinite(p_tci_diff) or p_tci_diff >= 0.05:
    print(f"VERDICT (A): coupling on HW days is NOT distinguishable from coupling")
    print(f"  across all regime-{TARGET_REGIME} days (TCI difference "
          f"{tci_hw - tci_all:+.3f}, p = {p_tci_diff:.3f}).")
    print(f"  --> no support for a stronger-feedback state on the HW days of")
    print(f"      this regime; note this is a static contrast and says nothing")
    print(f"      about change over time -- see (B), which is the Reviewer's")
    print(f"      actual claim.")
elif tci_hw < tci_all:
    print(f"VERDICT (A): TCI is significantly MORE NEGATIVE on HW days "
          f"(p = {p_tci_diff:.3f}).")
    print(f"  --> consistent with enhanced land-atmosphere feedback under HW")
    print(f"      conditions within this regime.")
else:
    print(f"VERDICT (A): TCI is significantly LESS negative on HW days "
          f"(p = {p_tci_diff:.3f}).")
    print(f"  --> opposite to the enhanced-feedback alternative.")

# ================================================================
# (B) TREND CONFOUND -- the Reviewer's actual claim, with a control
# ================================================================
sm_regime_yearly = yearly_mean_with_threshold(sm_anom.reindex(all_days_dates).dropna())
sm_control_yearly = sm_anom.groupby(sm_anom.index.year).mean()  # all MJJAS days
sm_control_yearly = sm_control_yearly.loc[sm_control_yearly.index.isin(sm_regime_yearly.index)]

slope_regime, p_regime, corr_regime = mann_kendall(sm_regime_yearly.values)
slope_control, p_control, corr_control = mann_kendall(sm_control_yearly.values)

# Difference series: regime-11 SM anomaly relative to the same year's
# all-MJJAS SM anomaly. A trend here is a regime-SPECIFIC drying signal,
# i.e. what the Reviewer's confound actually requires.
sm_excess_yearly = sm_regime_yearly - sm_control_yearly
slope_excess, p_excess, _ = mann_kendall(sm_excess_yearly.values)

print(f"\n{'=' * 68}")
print(f"(B) Soil moisture TREND on Regime {TARGET_REGIME} days ({SM_PERIOD})")
print(f"{'=' * 68}")
print(f"Years with >= {MIN_DAYS_PER_YEAR} regime-{TARGET_REGIME} days: "
      f"{len(sm_regime_yearly)} of {SM_YEND - SM_YSTART + 1}")
print(f"Theil-Sen slopes on yearly-mean SM anomaly, Mann-Kendall p "
      f"(Hamed-Rao corrected for serial correlation):")
print(f"  regime-{TARGET_REGIME} days: {slope_regime:+.5f} m3 m-3 / yr, "
      f"p = {p_regime:.3f}  -> {stars(p_regime)}")
print(f"  all MJJAS days (control): {slope_control:+.5f} m3 m-3 / yr, "
      f"p = {p_control:.3f}  -> {stars(p_control)}")
print(f"  regime-specific excess:   {slope_excess:+.5f} m3 m-3 / yr, "
      f"p = {p_excess:.3f}  -> {stars(p_excess)}")

print()
regime_dries = slope_regime < 0 and p_regime < 0.05
excess_dries = slope_excess < 0 and p_excess < 0.05
if regime_dries and excess_dries:
    print(f"VERDICT (B): significant drying on regime-{TARGET_REGIME} days that "
          f"EXCEEDS the")
    print(f"  domain-wide background drying.")
    print(f"  --> supports the alternative explanation: this circulation regime is")
    print(f"      occurring under progressively drier conditions, which could")
    print(f"      account for part of the apparent 'efficiency increase'")
    print(f"      independent of the circulation-regime relationship itself.")
elif regime_dries:
    print(f"VERDICT (B): regime-{TARGET_REGIME} days do dry significantly, but the "
          f"trend is not")
    print(f"  distinguishable from the domain-wide background drying "
          f"(excess p = {p_excess:.3f}).")
    print(f"  --> the drying is not specific to this regime; it is the general")
    print(f"      MJJAS background, which affects all regimes alike and so cannot")
    print(f"      explain a regime-{TARGET_REGIME}-specific efficiency increase.")
else:
    print(f"VERDICT (B): no significant drying trend on regime-{TARGET_REGIME} days "
          f"over {SM_PERIOD}")
    print(f"  (Theil-Sen {slope_regime:+.5f} m3 m-3 / yr, p = {p_regime:.3f}).")
    print(f"  --> does not support the land-atmosphere-feedback-trend alternative;")
    print(f"      the efficiency increase is not attributable to a soil-moisture")
    print(f"      drying trend within this regime's occurrences.")

if (SM_YSTART, SM_YEND) != (ystart, yend):
    print(f"\n  CAVEAT: soil moisture covers {SM_PERIOD}, not {ystart}-{yend}. "
          f"This verdict\n  is for the shorter period only.")

# ================================================================
# Machine-readable summary
# ================================================================
summary = pd.DataFrame([
    {"quantity": "r", "subset": "regime all days (whole domain, anom)", "value": r_all,
     "n": len(sm_all), "p": np.nan},
    {"quantity": "r", "subset": "regime HW days (whole domain, anom)", "value": r_hw,
     "n": len(sm_hw), "p": np.nan},
    {"quantity": "r difference (Fisher z)", "subset": "HW - all days", "value": r_hw - r_all,
     "n": np.nan, "p": p_diff},
    {"quantity": "TCI", "subset": "regime all days (whole domain, anom)", "value": tci_all,
     "n": len(sm_all), "p": np.nan},
    {"quantity": "TCI", "subset": "regime HW days (whole domain, anom)", "value": tci_hw,
     "n": len(sm_hw), "p": np.nan},
    {"quantity": "TCI difference (bootstrap)", "subset": "HW - all days",
     "value": tci_hw - tci_all, "n": np.nan, "p": p_tci_diff},
    {"quantity": "r", "subset": "regime all days (sub_hw-masked, anom)",
     "value": r_all_fp, "n": len(sm_all_fp), "p": np.nan},
    {"quantity": "r", "subset": "regime HW days (sub_hw-masked, anom)",
     "value": r_hw_fp, "n": len(sm_hw_fp), "p": np.nan},
    {"quantity": "r difference (Fisher z), masked", "subset": "HW - all days",
     "value": r_hw_fp - r_all_fp, "n": np.nan, "p": p_diff_fp},
    {"quantity": "TCI", "subset": "regime all days (sub_hw-masked, anom)",
     "value": tci_all_fp, "n": len(sm_all_fp), "p": np.nan},
    {"quantity": "TCI", "subset": "regime HW days (sub_hw-masked, anom)",
     "value": tci_hw_fp, "n": len(sm_hw_fp), "p": np.nan},
    {"quantity": "TCI difference (bootstrap), masked", "subset": "HW - all days",
     "value": tci_hw_fp - tci_all_fp, "n": np.nan, "p": p_tci_diff_fp},
    {"quantity": "r", "subset": "regime all days (RAW, seasonal cycle included)",
     "value": r_all_raw, "n": len(sm_all_raw), "p": np.nan},
    {"quantity": "r", "subset": "regime HW days (RAW, seasonal cycle included)",
     "value": r_hw_raw, "n": len(sm_hw_raw), "p": np.nan},
    {"quantity": "SM trend (Theil-Sen, /yr)", "subset": f"regime {TARGET_REGIME} days",
     "value": slope_regime, "n": len(sm_regime_yearly), "p": p_regime},
    {"quantity": "SM trend (Theil-Sen, /yr)", "subset": "all MJJAS days (control)",
     "value": slope_control, "n": len(sm_control_yearly), "p": p_control},
    {"quantity": "SM trend (Theil-Sen, /yr)", "subset": "regime-specific excess",
     "value": slope_excess, "n": len(sm_excess_yearly), "p": p_excess},
])
summary["period"] = SM_PERIOD
summary["sm_definition"] = f"depth-weighted 0-{TOTAL_DEPTH_CM:.0f}cm (swvl1+swvl2 merge)"
summary.to_csv(f"{OUT_PREFIX}_summary.csv", index=False)
print(f"\nWrote {OUT_PREFIX}_summary.csv")

# ================================================================
# Figure
# ================================================================
fig, axs = plt.subplots(1, 4, figsize=(21.5, 5.0))
for ax in axs:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=C_GRID, linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_INK_2, labelsize=13)

# --- Panel (a): SM' vs Tmax' scatter, one mask on both sides ---
ax = axs[0]
for xs, ys, color, label in (
        (sm_all.values, tmax_all.values, C_ALLDAYS, f"All MJJAS days (n={len(sm_all)})"),
        (sm_hw.values, tmax_hw.values, C_HWDAYS, f"HW days (n={len(sm_hw)})")):
    ax.scatter(xs, ys, s=24, color=color, alpha=0.65, linewidths=0.6,
               edgecolors="#fcfcfb", label=label, zorder=3)
    fit = linregress(xs, ys)
    xl = np.linspace(xs.min(), xs.max(), 20)
    ax.plot(xl, fit.slope * xl + fit.intercept, color=color, linewidth=2, zorder=4)

ax.axhline(0, color=C_GRID, linewidth=1, zorder=1)
ax.axvline(0, color=C_GRID, linewidth=1, zorder=1)
x_abs_max = max(abs(sm_all.min()), abs(sm_all.max()), abs(sm_hw.min()), abs(sm_hw.max()))
y_abs_max = max(abs(tmax_all.min()), abs(tmax_all.max()), abs(tmax_hw.min()), abs(tmax_hw.max()))
ax.set_xlim(-x_abs_max * 1.05, x_abs_max * 1.05)
ax.set_ylim(-y_abs_max * 1.05, y_abs_max * 1.05)
ax.set_xlabel(f"Soil moisture anomaly (0-{TOTAL_DEPTH_CM:.0f}cm merged, m$^3$ m$^{{-3}}$)",
              fontsize=14, color=C_INK_2)
ax.set_ylabel("Tmax anomaly (K)", fontsize=14, color=C_INK_2)
ax.set_title(f"(a) SM'-Tmax' coupling, Regime {TARGET_REGIME}\n"
             f"all days r={r_all:+.2f}   |   HW days r={r_hw:+.2f}   "
             f"(difference {'sig.' if np.isfinite(p_diff) and p_diff < 0.05 else 'n.s.'})",
             fontsize=15, color=C_INK, loc="left")
ax.legend(fontsize=12, frameon=False, labelcolor=C_INK_2)

# --- Panel (b): TCI with bootstrap 95% CI ---
ax = axs[1]
labels = ["All MJJAS days", "HW days"]
vals = [tci_all, tci_hw]
cis = [ci_all, ci_hw]
colors = [C_ALLDAYS, C_HWDAYS]
xpos = np.arange(2)
for i, (v, ci, c) in enumerate(zip(vals, cis, colors)):
    ax.bar(xpos[i], v, width=0.5, color=c, zorder=3,
           edgecolor="#fcfcfb", linewidth=2)
    ax.plot([xpos[i], xpos[i]], ci, color=C_INK_2, linewidth=2, zorder=4)
    below = v < 0
    ax.annotate(f"{v:+.3f}", xy=(xpos[i], ci[0] if below else ci[1]),
                xytext=(0, -10 if below else 10), textcoords="offset points",
                ha="center", va="top" if below else "bottom",
                fontsize=13, color=C_INK, zorder=5)

lo = min(np.min(ci_all), np.min(ci_hw), 0.0)
hi = max(np.max(ci_all), np.max(ci_hw), 0.0)
span = (hi - lo) or 1.0
ax.set_ylim(lo - 0.22 * span, hi + 0.12 * span)
ax.axhline(0, color=C_INK_2, linewidth=1, zorder=2)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=14, color=C_INK_2)
ax.set_xlabel("whiskers: moving-block bootstrap 95% CI",
              fontsize=12, color=C_INK_2)
ax.set_ylabel("TCI  =  std(Tmax')  ×  corr(SM', Tmax')", fontsize=14, color=C_INK_2)
ax.set_title(f"(b) Terrestrial Coupling Index, whole domain\n"
             f"difference {tci_hw - tci_all:+.3f} "
             f"[{ci_diff[0]:+.3f}, {ci_diff[1]:+.3f}] "
             f"({'sig.' if p_tci_diff < 0.05 else 'n.s.'})",
             fontsize=15, color=C_INK, loc="left")

# --- Panel (c): yearly SM anomaly, regime-11 days vs all-days control ---
ax = axs[2]
for series, slope, pval, color, label in (
        (sm_regime_yearly, slope_regime, p_regime, C_ALLDAYS,
         f"Regime {TARGET_REGIME} days"),
        (sm_control_yearly, slope_control, p_control, C_CONTROL,
         "All MJJAS days (control)")):
    yrs = series.index.values.astype(float)
    ax.plot(yrs, series.values, marker="o", markersize=5, linewidth=1.2,
            color=color, alpha=0.85, markeredgecolor="#fcfcfb",
            markeredgewidth=0.6, zorder=3,
            label=f"{label}: {slope:+.5f}/yr, p={pval:.2f}")
    ts = theilslopes(series.values, yrs)
    ax.plot(yrs, ts[1] + ts[0] * yrs, color=color, linewidth=2,
            linestyle="--", zorder=4)

ax.axhline(0, color=C_GRID, linewidth=1, zorder=1)
ax.margins(y=0.18)
ax.set_xlabel("Year", fontsize=14, color=C_INK_2)
ax.set_ylabel(f"Yearly-mean SM anomaly (0-{TOTAL_DEPTH_CM:.0f}cm, m$^3$ m$^{{-3}}$)",
              fontsize=14, color=C_INK_2)
ax.set_title(f"(c) Soil moisture trend, {SM_PERIOD}\n"
             f"regime-specific excess {slope_excess:+.5f}/yr "
             f"({'sig.' if np.isfinite(p_excess) and p_excess < 0.05 else 'n.s.'})",
             fontsize=15, color=C_INK, loc="left")
ax.legend(fontsize=12, frameon=False, labelcolor=C_INK_2, loc="lower left")

# --- Panel (d): TCI with bootstrap 95% CI, sub_hw-MASKED (both groups) ---
'''ax = axs[3]
labels_fp = ["All MJJAS days\n(masked)", "HW days\n(masked)"]
vals_fp = [tci_all_fp, tci_hw_fp]
cis_fp = [ci_all_fp, ci_hw_fp]
colors_fp = [C_ALLDAYS, C_HWDAYS]
xpos_fp = np.arange(2)
for i, (v, ci, c) in enumerate(zip(vals_fp, cis_fp, colors_fp)):
    ax.bar(xpos_fp[i], v, width=0.5, color=c, zorder=3,
           edgecolor="#fcfcfb", linewidth=2)
    ax.plot([xpos_fp[i], xpos_fp[i]], ci, color=C_INK_2, linewidth=2, zorder=4)
    below = v < 0
    ax.annotate(f"{v:+.3f}", xy=(xpos_fp[i], ci[0] if below else ci[1]),
                xytext=(0, -10 if below else 10), textcoords="offset points",
                ha="center", va="top" if below else "bottom",
                fontsize=13, color=C_INK, zorder=5)

lo_fp = min(np.min(ci_all_fp), np.min(ci_hw_fp), 0.0)
hi_fp = max(np.max(ci_all_fp), np.max(ci_hw_fp), 0.0)
span_fp = (hi_fp - lo_fp) or 1.0
ax.set_ylim(lo_fp - 0.22 * span_fp, hi_fp + 0.12 * span_fp)
ax.axhline(0, color=C_INK_2, linewidth=1, zorder=2)
ax.set_xticks(xpos_fp)
ax.set_xticklabels(labels_fp, fontsize=14, color=C_INK_2)
ax.tick_params(axis="y", labelsize=13)
ax.set_xlabel("whiskers: moving-block bootstrap 95% CI",
              fontsize=12, color=C_INK_2)
ax.set_ylabel("TCI  =  std(Tmax')  ×  corr(SM', Tmax')", fontsize=14, color=C_INK_2)
ax.set_title(f"(d) TCI, sub_hw-masked (both groups)\n"
             f"difference {tci_hw_fp - tci_all_fp:+.3f} "
             f"[{ci_diff_fp[0]:+.3f}, {ci_diff_fp[1]:+.3f}] "
             f"({'sig.' if p_tci_diff_fp < 0.05 else 'n.s.'})",
             fontsize=15, color=C_INK, loc="left")'''

plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}.png", dpi=300, bbox_inches="tight",
            facecolor="#fcfcfb")
print(f"Wrote {OUT_PREFIX}.png")
if SHOW_FIGURE:
    plt.show()