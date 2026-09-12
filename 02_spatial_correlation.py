"""
02_spatial_correlation.py

Corresponds to original notebook section "SPATIAL CORRELATIONS" (Fig. 7).

NOTE ON CLEANUP: the original notebook contained, immediately after the
correct block-bootstrap Monte Carlo implementation below, TWO further
duplicate blocks that called an undefined function `mc_iteration` (the
old i.i.d.-day-sampling test, superseded when the block-bootstrap fix
was implemented and renamed to `mc_iteration_one_sided`). Those blocks
would raise a NameError if run and have been dropped entirely here.

NOTE ON SCOPE: this file reproduces the HW-day composite correlation
test (Fig. 7 in the manuscript) using the verified, correct block-
bootstrap implementation. The original notebook also contained a
similarly-structured "all summer days" composite correlation heatmap
(different diagnostic, secondary to the manuscript's main Fig. 7). That
computation lived inside one of the now-removed dead-code blocks, so it
is NOT reproduced here. If you need that figure, it should be rebuilt
using run_resolution()/mc_iteration_one_sided() below, pointed at the
som_projected_all_days_* files instead of som_projected_HW_days_*, for
methodological consistency with the Fig. 7 fix.

Requires: 00_setup_and_functions.py (import * or run first)
"""

from importlib import import_module
setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

import xarray as xr
from joblib import Parallel, delayed

# ================================================================
# Block-bootstrap helpers (Reviewer 2, Sec 2.4(1) fix)
# ================================================================
def compute_valid_block_starts(year_ids, block_length):
    ntime = len(year_ids)
    valid_starts = [
        i for i in range(ntime - block_length + 1)
        if year_ids[i] == year_ids[i + block_length - 1]
    ]
    return np.array(valid_starts, dtype=int)


def sample_block_days(n_days, block_length, valid_starts, rng):
    idx = []
    while len(idx) < n_days:
        start = valid_starts[rng.integers(len(valid_starts))]
        idx.extend(range(start, start + block_length))
    return np.array(idx[:n_days])


def mc_iteration_one_sided(X_flat, era5_flat, counts, n_nodes,
                            valid_starts, block_length):
    rng = np.random.default_rng()
    r = np.full(n_nodes, np.nan)
    for k in range(n_nodes):
        n_days = counts[k]
        if n_days < 5:
            continue
        rand_days = sample_block_days(n_days, block_length, valid_starts, rng)
        rand_comp = np.nanmean(X_flat[rand_days], axis=0)
        valid_mask = np.isfinite(rand_comp) & np.isfinite(era5_flat[k])
        if valid_mask.sum() < 10:
            continue
        r[k] = np.corrcoef(rand_comp[valid_mask], era5_flat[k][valid_mask])[0, 1]
    return r


# ================================================================
# BLOCK_LENGTH: days per contiguous block, matching the 3-7 day
# synoptic decorrelation timescale (Wilks 2006; Reviewer 2, Sec 2.4).
# ================================================================
BLOCK_LENGTH = 5
N_ITER = 2000
N_JOBS = -1

r_obs_lr = np.full((len(models), n_nodes), np.nan)
pvals_lr = np.full((len(models), n_nodes), np.nan)
r_obs_hr = np.full((len(models), n_nodes), np.nan)
pvals_hr = np.full((len(models), n_nodes), np.nan)

ds_era5 = xr.open_mfdataset(
    f'{som_path}/som_projected_HW_days_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
era5_nodes = ds_era5.composite.transpose("som_row", "som_col", "lat", "lon")
nrow, ncol, nlat, nlon = era5_nodes.shape
n_nodes = nrow * ncol
n_models = len(models)

weights = np.cos(np.deg2rad(ds_era5.lat)).values
weights_flat = np.repeat(weights[:, None], nlon, axis=1).flatten()

mask_da = None


def run_resolution(m, model, realization, grid, mod_name, resolution,
                    r_obs_arr, pvals_arr, mask_da):
    """Observed-correlation + block-bootstrap significance test for one
    model at one resolution ('lr' or 'hr')."""
    global era5_nodes, weights_flat, n_nodes, nlat, nlon

    print(f"{model} ({resolution.upper()})")

    data_all_mod = (f"/work/cmcc/ls21622/HighResMIP/{mod_name}/{realization}/{VAR}/"
                     f"{VAR}_day_{mod_name}_hist-1950_{realization}_{grid}_1951-2014_AMJJASO.nc")
    X, X_weighted, clim = preprocess_for_som(data_all_mod, VAR)
    anomaly_dt = X.transpose('time', 'lat', 'lon')

    year_ids = anomaly_dt.time.dt.year.values
    valid_starts = compute_valid_block_starts(year_ids, BLOCK_LENGTH)

    file_in = f'/work/cmcc/ls21622/HighResMIP/{mod_name}/{realization}/som'
    ds_mod = xr.open_dataset(
        f'{file_in}/som_projected_HW_days_{mod_name}_{realization}_{VAR}_{ystart}_{yend}_'
        f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
    model_comp = ds_mod.composite.transpose("som_row", "som_col", "lat", "lon")

    if m == 0 and resolution == "lr" and VAR == 'tos' and mask_da is None:
        print("Creating common land-sea mask from first model and ERA5...")
        common_mask_2d = ~(np.isnan(era5_nodes[0, 0, :, :]) | np.isnan(model_comp[0, 0, :, :]))
        mask_da = xr.DataArray(
            ~common_mask_2d, dims=['lat', 'lon'],
            coords={'lat': ds_mod.lat, 'lon': ds_mod.lon}
        )
        print(f"Mask created. Ocean pixels: {common_mask_2d.sum()}/{common_mask_2d.size}")

    anomaly_dt_masked = anomaly_dt.where(~mask_da, np.nan) if (VAR == 'tos' and mask_da is not None) else anomaly_dt
    X_flat = anomaly_dt_masked.values.reshape(anomaly_dt_masked.time.size, nlat * nlon) * weights_flat

    model_comp_masked = model_comp.where(~mask_da, np.nan) if (VAR == 'tos' and mask_da is not None) else model_comp
    model_flat = model_comp_masked.values.reshape(n_nodes, nlat * nlon) * weights_flat

    era5_nodes_masked = era5_nodes.where(~mask_da, np.nan) if (VAR == 'tos' and mask_da is not None) else era5_nodes
    era5_flat = era5_nodes_masked.values.reshape(n_nodes, nlat * nlon) * weights_flat

    for k in range(n_nodes):
        valid_mask = ~(np.isnan(model_flat[k]) | np.isnan(era5_flat[k]))
        if valid_mask.sum() < 10:
            r_obs_arr[m, k] = np.nan
            continue
        r_obs_arr[m, k] = np.corrcoef(model_flat[k][valid_mask], era5_flat[k][valid_mask])[0, 1]

    node = ds_mod.bmu_row * som_grid_cols + ds_mod.bmu_col
    counts = np.bincount(node.values, minlength=n_nodes)

    print(f"Running block-bootstrap Monte Carlo ({resolution.upper()}, block_length={BLOCK_LENGTH})...")
    r_rand = Parallel(n_jobs=N_JOBS)(
        delayed(mc_iteration_one_sided)(X_flat, era5_flat, counts, n_nodes, valid_starts, BLOCK_LENGTH)
        for _ in range(N_ITER)
    )
    r_rand = np.array(r_rand)

    for k in range(n_nodes):
        if np.isnan(r_obs_arr[m, k]):
            continue
        extreme = np.sum(r_rand[:, k] >= r_obs_arr[m, k])
        pvals_arr[m, k] = (extreme + 1) / (N_ITER + 1)

    print(f"Completed {model} ({resolution.upper()}). "
          f"Significant nodes: {np.sum(pvals_arr[m] < 0.05)}/{n_nodes}")

    return mask_da


for m, model in enumerate(models):
    realization = realizations[m]
    grid = grids[m]
    mask_da = run_resolution(m, model, realization, grid, models_lr[m], "lr",
                              r_obs_lr, pvals_lr, mask_da)
    mask_da = run_resolution(m, model, realization, grid, models_hr[m], "hr",
                              r_obs_hr, pvals_hr, mask_da)

# ================================================================
# Fig. 7: pattern correlation triangle heatmap (HW-day composites)
# ================================================================
levels = np.linspace(0.4, 1.0, 13)
cmap = sns.color_palette("YlOrRd", len(levels) - 1, as_cmap=True)
norm = mcolors.BoundaryNorm(levels, cmap.N, clip=False, extend='min')

ensemble_mean_lr = np.nanmean(r_obs_lr, axis=0)
ensemble_mean_hr = np.nanmean(r_obs_hr, axis=0)

lr_panel = np.vstack([r_obs_lr, ensemble_mean_lr])
hr_panel = np.vstack([r_obs_hr, ensemble_mean_hr])

alpha = 0.05
ensemble_sig_lr = np.mean(pvals_lr < alpha, axis=0) > 0.75
ensemble_sig_hr = np.mean(pvals_hr < alpha, axis=0) > 0.75
sig_lr = np.vstack([pvals_lr < alpha, ensemble_sig_lr])
sig_hr = np.vstack([pvals_hr < alpha, ensemble_sig_hr])

row_labels = models + ['Ensemble Mean']

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[10, 0.5], hspace=0.15)
ax_main = fig.add_subplot(gs[0, 0])
cax_corr = fig.add_subplot(gs[1, 0])

for i in range(n_models + 1):
    for j in range(n_nodes):
        val_lr = lr_panel[i, j]
        color_lr = cmap(norm(val_lr)) if not np.isnan(val_lr) else 'lightgray'
        ax_main.add_patch(create_triangle(j, i, 'lower', color_lr))

        val_hr = hr_panel[i, j]
        color_hr = cmap(norm(val_hr)) if not np.isnan(val_hr) else 'lightgray'
        ax_main.add_patch(create_triangle(j, i, 'upper', color_hr))

ax_main.set_xlim(-0.5, n_nodes - 0.5)
ax_main.set_ylim(-0.5, n_models + 0.5)
ax_main.invert_yaxis()
for x in np.arange(-0.5, n_nodes, 1):
    ax_main.axvline(x, color='black', linewidth=0.8)
for y in np.arange(-0.5, n_models + 1, 1):
    ax_main.axhline(y, color='black', linewidth=0.8)
ax_main.axhline(y=n_models - 0.5, color='black', linewidth=2)

ax_main.set_xticks(np.arange(n_nodes))
ax_main.set_xticklabels(range(1, n_nodes + 1), ha='center', fontsize=16)
ax_main.set_yticks(np.arange(n_models + 1))
ax_main.set_yticklabels(row_labels, fontsize=16)
ax_main.set_xlabel("Regimes (SOM Nodes)", fontsize=16)

for i in range(n_models + 1):
    for j in range(n_nodes):
        val_lr = lr_panel[i, j]
        if not np.isnan(val_lr):
            marker = '*' if sig_lr[i, j] else ''
            ax_main.text(j - 0.2, i - 0.15, marker, ha='center', va='center', fontsize=12, color='black')
        val_hr = hr_panel[i, j]
        if not np.isnan(val_hr):
            marker = '*' if sig_hr[i, j] else ''
            ax_main.text(j + 0.2, i + 0.15, marker, ha='center', va='center', fontsize=12, color='black')

cbar_corr = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=cax_corr, orientation='horizontal')
cbar_corr.set_label("Pattern Correlation (r)", fontsize=16)
cbar_corr.ax.tick_params(labelsize=16)
if VAR == 'zg' : 
    plt.suptitle("Z500 Pattern correlation between ERA5 and models' HW day composites \n"
            "Low-res (\u25be) High-res (\u25b4)\n", fontsize=22, y=0.97)
else : 
    plt.suptitle("SST Pattern correlation between ERA5 and models' HW day composites \n"
            "Low-res (\u25be) High-res (\u25b4)\n", fontsize=22, y=0.97)
plt.tight_layout()
plt.savefig(f'02_HW_days_composites_correlation_{VAR}_{ystart}_{yend}.png',
            dpi=300, bbox_inches='tight')
