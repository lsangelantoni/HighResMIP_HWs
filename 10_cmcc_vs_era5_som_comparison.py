"""
10_cmcc_vs_era5_som_comparison.py

Basic diagnostic for Reviewer 2, Sec 4.3(1)(a) ("Fairness of ERA5 phase
space"): compares CMCC-CM2-HR4's own model-trained master SOM (trained
on CMCC's own Z500 data, not projected onto ERA5's phase space) against
the ERA5-trained SOM used to define the study's phase space throughout.

UPDATE: added a small number of full (non-anomaly) Z500 field contour
lines to both the CMCC and ERA5 panels of the spatial comparison grid,
matching the approach used in 01_era5_som_composites.py (Reviewer 3,
comment 10).

IMPORTANT -- two things need your confirmation before this will run:
1. CMCC_FULL_FIELD_FILE below is a PLACEHOLDER -- I do not have the
   actual path for CMCC-CM2-HR4's full daily Z500 field. Please fill
   this in (analogous to the ERA5 path already used elsewhere).
2. This requires ds_cmcc to contain "bmu_row", "bmu_col", and "time"
   (the day-to-node assignment from CMCC's own training), not just
   "patterns". If ds_cmcc does not have these fields, the CMCC
   composite step below will fail with a clear KeyError -- if so,
   let me know and we can discuss alternatives (e.g., if CMCC's BMU
   assignments are stored in a separate file).

NOTE: this is a basic, single-model, single-direction check (no
downstream re-analysis of HW-promoting efficiency, trends, etc. using
CMCC's own SOM) -- exactly the scope requested.

Requires: 00_setup_and_functions.py (import * or run first)
"""

from importlib import import_module
setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

import pandas as pd

CMCC_SOM_FILE = (
    "/work/cmcc/ls21622/HighResMIP/CMCC-CM2-HR4/r1i1p1f1/som/claude/"
    "CMCC-CM2-HR4_zg_1975_2014_3_5_iterations_10000_learning_rate_0.1_sigma_1.0_init_RND.nc"
)

# PLACEHOLDER -- replace with the actual CMCC-CM2-HR4 full daily Z500
# field path (analogous to the ERA5 path used in
# 01_era5_som_composites.py).
CMCC_FULL_FIELD_FILE = (
    "/work/cmcc/ls21622/HighResMIP/CMCC-CM2-HR4/r1i1p1f1/zg/"
    "zg_day_CMCC-CM2-HR4_hist-1950_r1i1p1f1_gn_1951-2014_AMJJASO.nc"  # <-- CONFIRM/FIX THIS
)

# --- ERA5-trained SOM (defines the study's phase space) ---
ds_era5 = xr.open_mfdataset(
    f'{som_path}/som_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
patterns_era5 = ds_era5["patterns"].transpose("som_row", "som_col", "lat", "lon")

# --- CMCC-CM2-HR4's own model-trained SOM ---
ds_cmcc = xr.open_dataset(CMCC_SOM_FILE)
patterns_cmcc = ds_cmcc["patterns"].transpose("som_row", "som_col", "lat", "lon")

n_rows, n_cols, nlat, nlon = patterns_era5.shape
n_nodes = n_rows * n_cols

flat_era5 = patterns_era5.values.reshape(n_nodes, -1)
flat_cmcc = patterns_cmcc.values.reshape(n_nodes, -1)

# Mask out NaN grid points consistently
valid = np.all(np.isfinite(flat_era5), axis=0) & np.all(np.isfinite(flat_cmcc), axis=0)
flat_era5 = flat_era5[:, valid]
flat_cmcc = flat_cmcc[:, valid]


def spatial_corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- For each CMCC node, find its best-matching ERA5 node ---
node_labels_1idx = [i + 1 for i in range(n_nodes)]
results = []
for k in range(n_nodes):
    corrs = [spatial_corr(flat_cmcc[k], flat_era5[j]) for j in range(n_nodes)]
    best_j = int(np.argmax(corrs))
    results.append({
        "cmcc_node": node_labels_1idx[k],
        "best_match_era5_node": node_labels_1idx[best_j],
        "match_correlation": round(corrs[best_j], 4),
    })

df = pd.DataFrame(results)
print("=== CMCC-CM2-HR4 (own model-trained SOM) vs. ERA5-trained SOM ===")
print(df.to_string(index=False))
print()
print(f"Mean best-match correlation across all {n_nodes} nodes: {df['match_correlation'].mean():.4f}")
print(f"Minimum best-match correlation: {df['match_correlation'].min():.4f}  "
      f"(CMCC node {df.loc[df['match_correlation'].idxmin(), 'cmcc_node']})")
print(f"Maximum best-match correlation: {df['match_correlation'].max():.4f}")

WEAK_MATCH_THRESHOLD = 0.5
weak_matches = df[df["match_correlation"] < WEAK_MATCH_THRESHOLD]
print()
if len(weak_matches) > 0:
    print(f"[note] {len(weak_matches)} CMCC node(s) have a weak best-match correlation "
          f"(< {WEAK_MATCH_THRESHOLD}) against every ERA5 node:")
    print(weak_matches.to_string(index=False))
    print("  --> possible evidence of model-specific circulation states not well")
    print("      represented in the ERA5-defined phase space")
else:
    print(f"All CMCC nodes have a best-match correlation >= {WEAK_MATCH_THRESHOLD} against "
          f"some ERA5 node.")
    print("  --> no evidence in this basic check that CMCC develops circulation states")
    print("      absent from the ERA5-defined phase space")

df.to_csv("cmcc_vs_era5_som_comparison.csv", index=False)
print("\nSaved: cmcc_vs_era5_som_comparison.csv")

# ================================================================
# NEW: Load and preprocess the FULL (non-anomaly) daily Z500 fields
# for both CMCC and ERA5, used for the contour overlay below.
# ================================================================
def to_standard_datetime(time_array):
    """
    Robustly convert a time coordinate to standard (Gregorian)
    datetime64, handling both already-standard arrays (e.g. ERA5) and
    cftime-object arrays with non-standard calendars (e.g. CMCC's
    native 365-day "noleap" calendar, which pd.to_datetime() cannot
    convert directly). Since noleap dates never include Feb 29, every
    noleap date is also a valid Gregorian date -- reconstructing via
    (year, month, day) strings sidesteps the direct type conversion.
    """
    time_array = np.asarray(time_array)
    if len(time_array) == 0:
        return pd.to_datetime(time_array)
    if isinstance(time_array[0], np.datetime64):
        return pd.to_datetime(time_array)
    # cftime object (or similar): reconstruct via (year, month, day)
    return pd.to_datetime([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in time_array])


def load_full_field(path):
    ds_full = xr.open_dataset(path)
    if "valid_time" in ds_full.coords or "longitude" in ds_full.coords:
        ds_full = ds_full.rename({"valid_time": "time", "longitude": "lon", "latitude": "lat"})
    ds_full = ds_full.sortby("lat")
    if ds_full.lon.min() >= 0:
        ds_full = shift_lon(ds_full)
    ds_full = crop_dom(ds_full, min_lon, max_lon, min_lat, max_lat, lon_name="lon", lat_name="lat")
    ds_full = ds_full.assign_coords(time=to_standard_datetime(ds_full.time.values).normalize())
    ds_full = ds_full.sel(time=ds_full.time.dt.month.isin([5, 6, 7, 8, 9]))
    varname = "zg" if "zg" in ds_full.data_vars else list(ds_full.data_vars)[0]
    return ds_full[varname]


def composite_full_field_by_node(full_field_da, bmu_row, bmu_col, bmu_time, n_rows, n_cols):
    bmu_time = to_standard_datetime(bmu_time).normalize()
    node_idx = bmu_row * n_cols + bmu_col
    composites = np.full((n_rows * n_cols, full_field_da.lat.size, full_field_da.lon.size), np.nan)
    for k in range(n_rows * n_cols):
        dates_k = bmu_time[node_idx == k]
        if len(dates_k) == 0:
            continue
        field_k = full_field_da.sel(time=full_field_da.time.isin(dates_k))
        composites[k] = field_k.mean(dim="time", skipna=True).values
    return composites.reshape(n_rows, n_cols, full_field_da.lat.size, full_field_da.lon.size)


full_field_era5 = load_full_field(
    f"/data/cmcc/ls21622/ERA5/{VAR}/postprocessed/ERA5_{VAR}_day_1975-2024_AMJJASO.nc")
full_composite_era5 = composite_full_field_by_node(
    full_field_era5, ds_era5["bmu_row"].values, ds_era5["bmu_col"].values,
    ds_era5["time"].values, n_rows, n_cols)

# CMCC side: requires ds_cmcc to have bmu_row/bmu_col/time -- if this
# raises a KeyError, ds_cmcc does not carry the day-to-node assignment
# and the CMCC contour overlay cannot be computed this way.
full_field_cmcc = load_full_field(CMCC_FULL_FIELD_FILE)
full_composite_cmcc = composite_full_field_by_node(
    full_field_cmcc, ds_cmcc["bmu_row"].values, ds_cmcc["bmu_col"].values,
    ds_cmcc["time"].values, n_rows, n_cols)

# ================================================================
# Spatial comparison grid: CMCC's own pattern (top row) vs. its
# best-matching ERA5 pattern (bottom row), for all 15 nodes, so the
# correlation numbers can be visually verified against the actual
# patterns rather than judged as an abstract statistic alone.
# ================================================================
if VAR == 'tos':
    levels = np.linspace(-2, 2, 21)
else:
    levels = np.linspace(-150, 150, 21)
cmap = sns.color_palette("RdBu_r", len(levels) - 1, as_cmap=True)
norm = mcolors.BoundaryNorm(levels, cmap.N, clip=False, extend='both')

fig, axs = plt.subplots(n_nodes, 2, figsize=(6, 1.8 * n_nodes),
                         subplot_kw={"projection": ccrs.PlateCarree()})

for idx, row in df.iterrows():
    cmcc_node = int(row["cmcc_node"])
    era5_node = int(row["best_match_era5_node"])
    corr_val = row["match_correlation"]
    row_pos = cmcc_node - 1

    cmcc_r, cmcc_c = divmod(cmcc_node - 1, n_cols)
    era5_r, era5_c = divmod(era5_node - 1, n_cols)

    ax_left = axs[row_pos, 0]
    data_left = patterns_cmcc.isel(som_row=cmcc_r, som_col=cmcc_c)
    im = ax_left.pcolormesh(data_left.lon, data_left.lat, data_left, cmap=cmap, norm=norm,
                            shading="auto", transform=ccrs.PlateCarree(), zorder=1)
    # NEW: small number of full-field contour lines, composited over
    # CMCC's own days assigned to this node
    full_field_node_cmcc = full_composite_cmcc[cmcc_r, cmcc_c]
    if not np.all(np.isnan(full_field_node_cmcc)):
        cs_left = ax_left.contour(full_field_cmcc.lon, full_field_cmcc.lat, full_field_node_cmcc,
                                  levels=6, colors="black", linewidths=0.5, zorder=2,
                                  transform=ccrs.PlateCarree())
        ax_left.clabel(cs_left, inline=True, fontsize=5, fmt="%.0f")
    ax_left.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black",
                        linewidth=0.5, zorder=3)
    ax_left.set_title(f"CMCC {cmcc_node}", fontsize=11)

    ax_right = axs[row_pos, 1]
    data_right = patterns_era5.isel(som_row=era5_r, som_col=era5_c)
    ax_right.pcolormesh(data_right.lon, data_right.lat, data_right, cmap=cmap, norm=norm,
                        shading="auto", transform=ccrs.PlateCarree(), zorder=1)
    # NEW: small number of full-field contour lines, composited over
    # ERA5's own days assigned to this node
    full_field_node_era5 = full_composite_era5[era5_r, era5_c]
    if not np.all(np.isnan(full_field_node_era5)):
        cs_right = ax_right.contour(full_field_era5.lon, full_field_era5.lat, full_field_node_era5,
                                    levels=6, colors="black", linewidths=0.5, zorder=2,
                                    transform=ccrs.PlateCarree())
        ax_right.clabel(cs_right, inline=True, fontsize=5, fmt="%.0f")
    ax_right.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black",
                         linewidth=0.5, zorder=3)
    verdict_color = "#1a7a3c" if corr_val >= WEAK_MATCH_THRESHOLD else "#b3401f"
    ax_right.set_title(f"ERA5 {era5_node}  (r={corr_val:.2f})", fontsize=11, color=verdict_color)

axs[0, 0].text(0.5, 1.18, "CMCC (own SOM)", transform=axs[0, 0].transAxes,
              fontsize=13, fontweight="bold", ha="center", va="bottom")
axs[0, 1].text(0.5, 1.18, "Best-match ERA5 node", transform=axs[0, 1].transAxes,
              fontsize=13, fontweight="bold", ha="center", va="bottom")

cbar = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.015, pad=0.01)
cbar.set_label(f"{VAR_TIT} anomaly (m)" if VAR == 'zg' else f"{VAR_TIT} anomaly (\u00b0C)", fontsize=12)

fig.suptitle("CMCC-CM2-HR4 own model-trained SOM vs. best-matching ERA5 node"
            , fontsize=14, y=0.9)
plt.savefig("cmcc_vs_era5_som_comparison_maps.png", dpi=300, bbox_inches="tight")
plt.show()

# ================================================================
# Simple summary bar chart of match correlation per CMCC node
# ================================================================
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#1a7a3c" if v >= WEAK_MATCH_THRESHOLD else "#b3401f" for v in df["match_correlation"]]
ax.bar(df["cmcc_node"], df["match_correlation"], color=colors)
ax.axhline(WEAK_MATCH_THRESHOLD, color="black", linestyle="--", linewidth=1,
          label=f"Weak-match threshold ({WEAK_MATCH_THRESHOLD})")
ax.set_xlabel("CMCC node", fontsize=12)
ax.set_ylabel("Best-match correlation with ERA5", fontsize=12)
ax.set_xticks(df["cmcc_node"])
ax.set_title("CMCC-CM2-HR4 own SOM: best-match correlation to ERA5 phase space", fontsize=13)
ax.legend(fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig("cmcc_vs_era5_som_comparison_bars.png", dpi=300, bbox_inches="tight")
