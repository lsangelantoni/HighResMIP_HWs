"""
01_era5_som_composites.py

Corresponds to original notebook sections "ERA5 All summer season days SOM"
and "ERA5 HW composites" (Fig. 7a / 7b in the revised manuscript).

UPDATE (Reviewer 3, comment 10): added a small number of contour lines
for the FULL Z500 field (not just the anomaly already shown via
shading), composited over the same days assigned to each SOM node.
This required loading the full (non-anomaly) daily Z500 field and
compositing it per node using the SAME bmu_row/bmu_col assignments
already computed for the anomaly-based SOM.

NOTE ON CLEANUP: the original notebook contained the all-summer-day SOM
plot TWICE, back to back, with byte-for-byte identical code and an
identical output filename -- a genuine duplicate, not two different
diagnostics. Only one copy is retained here.

IMPORTANT -- please verify before running: the full-field file's
variable/coordinate names are ASSUMED to follow the same convention
already used elsewhere in this project for other ERA5 files (e.g.
09_soil_moisture_coupling_diagnostic.py), i.e. "valid_time" -> "time",
"longitude" -> "lon", "latitude" -> "lat". This has NOT been verified
against the actual file, since it isn't accessible from this
environment. If the rename step below errors or is a no-op, check the
file's actual coordinate names with `ds_full.coords` and adjust.

Requires: 00_setup_and_functions.py (import * or run first)
"""

from importlib import import_module
setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

# ================================================================
# ================================================================
# Load the anomaly SOM FIRST, so we can derive the full-field crop
# domain directly from its own lon/lat coordinates -- this guarantees
# the full field is cropped to EXACTLY the same domain as the anomaly
# patterns, regardless of whatever min_lon/max_lon/min_lat/max_lat
# might already exist in the global namespace from an unrelated
# earlier script run in the same session (e.g. the much smaller
# Western-Europe HW-detection domain used elsewhere in this project).
# ================================================================
ds = xr.open_mfdataset(
    f'{som_path}/som_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc'
)

FULL_FIELD_MIN_LON = float(ds.lon.min())
FULL_FIELD_MAX_LON = float(ds.lon.max())
FULL_FIELD_MIN_LAT = float(ds.lat.min())
FULL_FIELD_MAX_LAT = float(ds.lat.max())
print(f"Cropping full field to the anomaly SOM's own domain: "
      f"lon [{FULL_FIELD_MIN_LON}, {FULL_FIELD_MAX_LON}], "
      f"lat [{FULL_FIELD_MIN_LAT}, {FULL_FIELD_MAX_LAT}]")

# ================================================================
# Load and preprocess the FULL (non-anomaly) daily Z500 field, used
# to composite the contour overlay in both panels below.
# ================================================================
FULL_FIELD_FILE = f"/data/cmcc/ls21622/ERA5/{VAR}/postprocessed/ERA5_{VAR}_day_1975-2024_AMJJASO.nc"

ds_full = xr.open_dataset(FULL_FIELD_FILE)
if "valid_time" in ds_full.coords or "longitude" in ds_full.coords:
    ds_full = ds_full.rename({"valid_time": "time", "longitude": "lon", "latitude": "lat"})
ds_full = ds_full.sortby("lat")
if ds_full.lon.min() >= 0:
    ds_full = shift_lon(ds_full)
ds_full = crop_dom(ds_full, FULL_FIELD_MIN_LON, FULL_FIELD_MAX_LON,
                   FULL_FIELD_MIN_LAT, FULL_FIELD_MAX_LAT, lon_name="lon", lat_name="lat")
ds_full["time"] = pd.to_datetime(ds_full.time.astype("datetime64[ns]")).normalize()
ds_full = ds_full.sel(time=ds_full.time.dt.month.isin([5, 6, 7, 8, 9]))
ds_full = ds_full.sel(time=ds_full.time.dt.year.isin(range(1975, yend + 1)))

# NOTE: adjust "zg" below if the full-field file's data variable is
# named differently (e.g. "z" or "Z500").
FULL_FIELD_VARNAME = "zg" if "zg" in ds_full.data_vars else list(ds_full.data_vars)[0]
full_field_da = ds_full[FULL_FIELD_VARNAME]

# ERA5's raw SST (tos) is stored in Kelvin; convert to Celsius here so
# the contour values match the anomaly shading's units (already °C).
# Z500 (zg, metres) needs no conversion. A value check guards against
# double-converting if the file already happens to be in Celsius.
if VAR == 'tos' and float(full_field_da.mean()) > 100:
    full_field_da = full_field_da - 273.15
    full_field_da.attrs["units"] = "degC"
    print("Converted full SST field from Kelvin to Celsius.")


def composite_full_field_by_node(bmu_row, bmu_col, bmu_time, n_rows, n_cols):
    """
    For each SOM node, composite the FULL (non-anomaly) Z500 field over
    the same days assigned to that node (matched by date to
    full_field_da), returning an (n_rows, n_cols, lat, lon) array.
    """
    bmu_time = pd.to_datetime(np.asarray(bmu_time)).normalize()
    node_idx = bmu_row * n_cols + bmu_col

    composites = np.full((n_rows * n_cols, full_field_da.lat.size, full_field_da.lon.size), np.nan)
    for k in range(n_rows * n_cols):
        dates_k = bmu_time[node_idx == k]
        if len(dates_k) == 0:
            continue
        field_k = full_field_da.sel(time=full_field_da.time.isin(dates_k))
        composites[k] = field_k.mean(dim="time", skipna=True).values

    return composites.reshape(n_rows, n_cols, full_field_da.lat.size, full_field_da.lon.size)


# ================================================================
# ERA5 all-summer-day master SOM (Fig. 7a)
# (already loaded above, to derive the full-field crop domain)
# ================================================================
full_composite_all = composite_full_field_by_node(
    ds["bmu_row"].values, ds["bmu_col"].values, ds["time"].values,
    som_grid_rows, som_grid_cols
)

if VAR == 'tos':
    levels = np.linspace(-2, 2, 21)
else:
    levels = np.linspace(-150, 150, 21)
cmap = sns.color_palette("RdBu_r", len(levels) - 1, as_cmap=True)
norm = mcolors.BoundaryNorm(levels, cmap.N, clip=False, extend='both')

fig, axs = plt.subplots(
    som_grid_rows, som_grid_cols, figsize=(16, 6),
    constrained_layout=True, subplot_kw={"projection": ccrs.PlateCarree()},
)

ntime = ds.dims["time"]
for i in range(som_grid_rows):
    for j in range(som_grid_cols):
        data = ds["patterns"].isel(som_row=i, som_col=j)
        im = axs[i, j].pcolormesh(data.lon, data.lat, data, cmap=cmap, norm=norm,
                                   shading="auto", zorder=1)

        # NEW: small number of contour lines for the full Z500 field
        full_field_node = full_composite_all[i, j]
        cs = axs[i, j].contour(full_field_da.lon, full_field_da.lat, full_field_node,
                               levels=6, colors="black", linewidths=0.6, zorder=2)
        axs[i, j].clabel(cs, inline=True, fontsize=6, fmt="%.0f")

        freq_era5 = ds["hit_counts"].isel(som_row=i, som_col=j).values / ntime * 100
        n_hits_era5 = ds["hit_counts"].isel(som_row=i, som_col=j).values

        title_text = f"{freq_era5:.0f}% (n={n_hits_era5:.0f})"
        axs[i, j].text(0.01, 0.13, title_text, transform=axs[i, j].transAxes,
                       fontsize=10, va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.2))

        node_num = node_number(i, j)
        axs[i, j].set_xlabel(f'{node_num}', fontsize=12)
        axs[i, j].xaxis.set_label_coords(0.5, -0.1)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].add_feature(cfeature.COASTLINE.with_scale("50m"),
                              edgecolor="black", linewidth=0.6, zorder=3)
        if VAR == 'tos':
            axs[i, j].add_feature(cfeature.LAND, facecolor='darkgray', zorder=2)

cbar = fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.015, pad=0.02)
cbar.ax.tick_params(labelsize=14)

plt.suptitle(f"ERA5 SOM {VAR_TIT} anomaly all days (MJJAS)", fontsize=20)
if VAR == 'zg':
    cbar.set_label(f"{VAR_TIT} anomaly (m)", fontsize=14)
else:
    cbar.set_label(f"{VAR_TIT} anomaly (\u00b0C)", fontsize=14)
fig.supylabel('SOM rows', fontsize=18)
fig.supxlabel('SOM columns', fontsize=18)

plt.savefig(f'01_ERA5_SOM_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.png', dpi=300)


# ================================================================
# ERA5 HW-day composite (Fig. 7b)
# ================================================================
ds_hw = xr.open_mfdataset(
    f'{som_path}/som_projected_HW_days_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc'
)

full_composite_hw = composite_full_field_by_node(
    ds_hw["bmu_row"].values, ds_hw["bmu_col"].values, ds_hw["time"].values,
    som_grid_rows, som_grid_cols
)

if VAR == 'tos':
    levels = np.linspace(-2, 2, 21)
else:
    levels = np.linspace(-200, 200, 21)
cmap = sns.color_palette("RdBu_r", len(levels) - 1, as_cmap=True)
norm = mcolors.BoundaryNorm(levels, cmap.N, clip=False, extend='both')

fig, axs = plt.subplots(
    som_grid_rows, som_grid_cols, figsize=(16, 6),
    constrained_layout=True, subplot_kw={"projection": ccrs.PlateCarree()},
)

ntime = ds_hw.dims["time"]
for i in range(som_grid_rows):
    for j in range(som_grid_cols):
        data = ds_hw["composite"].isel(som_row=i, som_col=j)
        im = axs[i, j].pcolormesh(data.lon, data.lat, data, cmap=cmap, norm=norm,
                                   shading="auto", zorder=1)

        # NEW: small number of contour lines for the full Z500 field,
        # composited over the same HW days assigned to this node
        full_field_node = full_composite_hw[i, j]
        if not np.all(np.isnan(full_field_node)):
            cs = axs[i, j].contour(full_field_da.lon, full_field_da.lat, full_field_node,
                                   levels=6, colors="black", linewidths=0.6, zorder=2)
            axs[i, j].clabel(cs, inline=True, fontsize=6, fmt="%.0f")

        freq_era5 = ds_hw["hit_counts"].isel(som_row=i, som_col=j).values / ntime * 100
        n_hits_era5 = ds_hw["hit_counts"].isel(som_row=i, som_col=j).values

        title_text = f"{freq_era5:.0f}% (n={n_hits_era5:.0f})"
        axs[i, j].text(0.01, 0.13, title_text, transform=axs[i, j].transAxes,
                       fontsize=10, va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.2))

        node_num = node_number(i, j)
        axs[i, j].set_xlabel(f'{node_num}', fontsize=12)
        axs[i, j].xaxis.set_label_coords(0.5, -0.1)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].add_feature(cfeature.COASTLINE.with_scale("50m"),
                              edgecolor="black", linewidth=0.6, zorder=3)
        if VAR == 'tos':
            axs[i, j].add_feature(cfeature.LAND, facecolor='darkgray', zorder=2)

cbar = fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.015, pad=0.02)
cbar.ax.tick_params(labelsize=14)

plt.suptitle(f"ERA5 HW days {VAR_TIT} anomaly composite", fontsize=20)
if VAR == 'zg':
    cbar.set_label(f"{VAR_TIT} anomaly (m)", fontsize=14)
else:
    cbar.set_label(f"{VAR_TIT} anomaly (\u00b0C)", fontsize=14)
fig.supylabel('SOM rows', fontsize=18)
fig.supxlabel('SOM columns', fontsize=18)

plt.savefig(
    f'01b_ERA5_projected_SOM_HW_days_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.png', dpi=300)