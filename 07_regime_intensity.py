"""
07_regime_intensity.py

Corresponds to original notebook section "Regime-conditioned HW
intensity" -> "Associate HW magnitude to each regime" (Fig. 9).

NOTE: n_min (minimum HW-day sample size to compute a regime-conditioned
mean) is set via N_MIN below (currently 10 HW days), applied
consistently to both ERA5 and models, and used both to gate the mean
computation and to determine which regimes are plotted.

Requires: 00_setup_and_functions.py (import * or run first)
"""

from importlib import import_module
setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines
import scipy.stats

N_MIN = 10  # minimum number of HW days required to compute a node's mean anomaly


def process_file_era5_here(file_path, var_name, min_lon, max_lon, min_lat, max_lat,
                            ystart, yend, mask=False):
    ds = xr.open_dataset(file_path).rename({'valid_time': 'time', 'longitude': 'lon', 'latitude': 'lat'})
    ds = ds.sortby('lat')
    ds = crop_dom(ds, min_lon, max_lon, min_lat, max_lat, lon_name='lon', lat_name='lat')
    ds = ds.sel(time=ds.time.dt.month.isin([5, 6, 7, 8, 9]))
    ds = ds.sel(time=ds.time.dt.year.isin(range(ystart, yend + 1)))

    if 'sst' in ds:
        ds = ds.rename({'sst': 'tos'})
    if 'z' in ds:
        ds = ds.rename({'z': 'zg'}).squeeze()
        ds['zg'] = ds['zg'] / 9.8

    if mask:
        lsm = xr.open_dataset(
            '/data/cmcc/ls21622/ERA5/tos/postprocessed/ERA5_tos_day_1975-2024_AMJJASO.nc'
        ).rename({'longitude': 'lon', 'latitude': 'lat', 'valid_time': 'time'})
        lsm = lsm.sortby('lat')
        if lsm.lon.min() >= 0:
            lsm = shift_lon(lsm)
        lsm = crop_dom(lsm, min_lon, max_lon, min_lat, max_lat, lon_name='lon', lat_name='lat')
        lsm = lsm.isel(time=0)
        mask_arr = np.isnan(lsm.tos)
        ds = ds.where(mask_arr)
        return ds, mask_arr
    return ds


def process_file_models_here(file_path, var_name, min_lon, max_lon, min_lat, max_lat,
                              ystart, yend, mask=False):
    ds = xr.open_dataset(file_path)
    if 'tas' in ds:
        ds = ds.rename({'tas': 'tasmax'})
    if ds.lon.min() >= 0:
        ds = shift_lon(ds)
    ds = crop_dom(ds, min_lon, max_lon, min_lat, max_lat, lon_name='lon', lat_name='lat')
    ds = ds.sel(time=ds.time.dt.month.isin([5, 6, 7, 8, 9]))
    ds = ds.sel(time=ds.time.dt.year.isin(range(ystart, yend + 1)))

    if mask:
        lsm = xr.open_dataset(
            '/work/cmcc/ls21622/HighResMIP/ECMWF-IFS-HR/r1i1p1f1/tos/'
            'tos_day_ECMWF-IFS-HR_hist-1950_r1i1p1f1_gr_1951-2014_AMJJASO.nc')
        if lsm.lon.min() >= 0:
            lsm = shift_lon(lsm)
        lsm = crop_dom(lsm, min_lon, max_lon, min_lat, max_lat, lon_name='lon', lat_name='lat')
        lsm = lsm.isel(time=0)
        lsm = regrid(lsm, ds, method)
        mask_arr = np.isnan(lsm.tos)
        ds = ds.where(mask_arr)
        return ds, mask_arr
    return ds


# ================================================================
# ERA5 regime-conditioned HW intensity
# ================================================================
min_lon, max_lon = -10, 15
min_lat, max_lat = 42, 58

file_tasmax = "/data/cmcc/ls21622/ERA5/tmax/postprocessed/ERA5_tmax_day_1975-2024_AMJJASO.nc"
file_p90 = "/data/cmcc/ls21622/ERA5/tmax/p90/tasmax_p90_day_15_ERA5_1975-1994.nc"
hw_mask_path = '/work/cmcc/ls21622/HighResMIP/ERA5/'

ds_tasmax, mask = process_file_era5_here(file_tasmax, 'tasmax', min_lon, max_lon, min_lat, max_lat,
                                         ystart, yend, mask=True)
da_tasmax = ds_tasmax.tasmax
ds_p90, mask = process_file_era5_here(file_p90, 'tasmax', min_lon, max_lon, min_lat, max_lat,
                                      ystart, yend, mask=True)

ds_hw_mask = xr.open_mfdataset(f'{hw_mask_path}/ERA5_sub_heatwave_day_*_CWE.nc')
ds_hw_mask = ds_hw_mask.sel(time=ds_hw_mask.time.dt.month.isin([5, 6, 7, 8, 9]))

ds_som_hw = xr.open_mfdataset(
    f'{som_path}/som_projected_HW_days_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
ds_som_hw = ds_som_hw.sel(time=ds_som_hw.time.dt.month.isin([5, 6, 7, 8, 9]))

hw_dates = ds_som_hw.time.values
tasmax_hw = da_tasmax.sel(time=hw_dates)
tasmax_hw_mask = ds_hw_mask.sel(time=hw_dates)

p90_lookup = (
    ds_p90.assign_coords(dayofyear=ds_p90.time.dt.dayofyear)
    .swap_dims({"time": "dayofyear"})
    .drop_vars("time")
)
hw_anomaly = tasmax_hw.groupby("time.dayofyear") - p90_lookup.tasmax
hw_anomaly = hw_anomaly.where(tasmax_hw_mask.sub_hw == 1)

df, node_id_da = build_hw_intensity_df(hw_anomaly, ds_som_hw)

n_rows = ds_som_hw.som_row.size
n_cols = ds_som_hw.som_col.size
total_nodes = n_rows * n_cols

mean_anomaly_array = np.full((n_rows, n_cols), np.nan)
freq_array = np.full((n_rows, n_cols), np.nan)
number_array = np.full((n_rows, n_cols), np.nan)

for node_id in range(total_nodes):
    row = node_id // n_cols
    col = node_id % n_cols
    node_data = df[df['node_id'] == node_id]
    n_events = len(node_data)
    freq = (n_events / len(df)) * 100 if n_events > 1 else 0
    node_mean = node_data['tmax_anom'].mean() if n_events >= N_MIN else np.nan

    mean_anomaly_array[row, col] = node_mean
    freq_array[row, col] = freq
    number_array[row, col] = n_events

era5_mag_flat = mean_anomaly_array.ravel()
era5_freq_flat = freq_array.ravel()

# ================================================================
# Models: regime-conditioned HW intensity, LR and HR
# ================================================================
n_models = len(models)
n_nodes = som_grid_rows * som_grid_cols

mag_matrix_hr = np.full((n_models, n_nodes), np.nan)
mag_matrix_lr = np.full((n_models, n_nodes), np.nan)
freq_matrix_hr = np.zeros((n_models, n_nodes))
freq_matrix_lr = np.zeros((n_models, n_nodes))

for m_index in range(n_models):
    print(f"Processing {models[m_index]}")
    model = models[m_index]
    realization = realizations[m_index]
    mod_lr = models_lr[m_index]
    mod_hr = models_hr[m_index]
    grid = grids[m_index]

    for resolution, mod_name, mag_matrix, freq_matrix in [
        ("LR", mod_lr, mag_matrix_lr, freq_matrix_lr),
        ("HR", mod_hr, mag_matrix_hr, freq_matrix_hr),
    ]:
        if model[:5] == 'ECMWF':
            file_tasmax = (f"/work/cmcc/ls21622/HighResMIP/{mod_name}/{realization}/tasmax/"
                          f"tasmax_day_{mod_name}_historical_{realization}_{grid}_1975-2014_AMJJASO.nc")
        else:
            file_tasmax = (f"/work/cmcc/ls21622/HighResMIP/{mod_name}/{realization}/tasmax/"
                          f"tasmax_day_{mod_name}_historical_{realization}_{grid}_1975-2024_AMJJASO.nc")

        file_p90 = (f"/work/cmcc/ls21622/HighResMIP/{mod_name}/{realization}/tasmax/p90/"
                   f"tasmax_p90_day_15_{mod_name}_{realization}_HISTORICAL_day_1975-1994.nc")

        ds_tasmax, _ = process_file_models_here(file_tasmax, 'tasmax', min_lon, max_lon, min_lat, max_lat,
                                                ystart, yend, mask=True)
        ds_tasmax["time"] = pd.to_datetime(ds_tasmax.time.astype("datetime64[ns]")).normalize()
        da_tasmax = ds_tasmax.tasmax

        ds_p90, _ = process_file_models_here(file_p90, 'tasmax', min_lon, max_lon, min_lat, max_lat,
                                             ystart, yend, mask=True)

        ds_hw_mask = xr.open_mfdataset(
            f'/work/cmcc/ls21622/HighResMIP/{mod_name}/{realization}/hw_metrics/*sub_heatwave_day_*_CWE.nc')
        ds_hw_mask["time"] = pd.to_datetime(ds_hw_mask.time.astype("datetime64[ns]")).normalize()
        ds_hw_mask = ds_hw_mask.sel(time=ds_hw_mask.time.dt.month.isin([5, 6, 7, 8, 9]))

        ds_som_hw = xr.open_mfdataset(
            f'/work/cmcc/ls21622/HighResMIP/{mod_name}/{realization}/som/'
            f'som_projected_HW_days_{mod_name}_{realization}_{VAR}_{ystart}_{yend}_'
            f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
        ds_som_hw["time"] = pd.to_datetime(ds_som_hw.time.astype("datetime64[ns]")).normalize()
        ds_som_hw = ds_som_hw.sel(time=ds_som_hw.time.dt.month.isin([5, 6, 7, 8, 9]))

        hw_dates = ds_som_hw.time.values
        tasmax_hw = da_tasmax.sel(time=hw_dates)
        tasmax_hw_mask = ds_hw_mask.sel(time=hw_dates)

        p90_lookup = (
            ds_p90.assign_coords(dayofyear=ds_p90.time.dt.dayofyear)
            .swap_dims({"time": "dayofyear"})
            .drop_vars("time")
        )
        hw_anomaly = tasmax_hw.groupby("time.dayofyear") - p90_lookup.tasmax
        hw_anomaly = hw_anomaly.where(tasmax_hw_mask.sub_hw == 1)

        df, _ = build_hw_intensity_df(hw_anomaly, ds_som_hw)

        for node_id in range(total_nodes):
            node_data = df[df['node_id'] == node_id]
            n_events = len(node_data)
            freq = (n_events / len(df)) * 100 if n_events > 0 else 0
            freq_matrix[m_index, node_id] = freq
            mag_matrix[m_index, node_id] = node_data['tmax_anom'].mean() if n_events >= N_MIN else np.nan

# ================================================================
# Fig. 9: triangle heatmap
# ================================================================
max_bias = 1.0
levels_bias = np.linspace(-max_bias, max_bias, 11)
cmap_bias = sns.color_palette("RdBu_r", len(levels_bias) - 1, as_cmap=True)
norm_bias = BoundaryNorm(levels_bias, cmap_bias.N, clip=False, extend='both')

levels_era5 = np.linspace(1, 4, 13)
cmap_era5 = sns.color_palette("YlOrRd", len(levels_era5) - 1, as_cmap=True)
norm_era5 = BoundaryNorm(levels_era5, cmap_era5.N, clip=False, extend='both')

ensemble_mean_lr_phys = np.nanmean(mag_matrix_lr, axis=0)
ensemble_mean_hr_phys = np.nanmean(mag_matrix_hr, axis=0)

bias_matrix_lr = mag_matrix_lr - era5_mag_flat
bias_matrix_hr = mag_matrix_hr - era5_mag_flat
bias_ensemble_lr = ensemble_mean_lr_phys - era5_mag_flat
bias_ensemble_hr = ensemble_mean_hr_phys - era5_mag_flat

bias_lr_panel = np.vstack([bias_matrix_lr, bias_ensemble_lr])
bias_hr_panel = np.vstack([bias_matrix_hr, bias_ensemble_hr])
era5_row = era5_mag_flat
row_labels = models + ['Ensemble Mean', 'ERA5']
n_rows_total = n_models + 2

fig = plt.figure(figsize=(19, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[10, 0.5, 0.5], hspace=0.2)
ax_main = fig.add_subplot(gs[0, 0])
cax_bias = fig.add_subplot(gs[1, 0])
cax_era5 = fig.add_subplot(gs[2, 0])

for i in range(n_models + 1):
    for j in range(n_nodes):
        val_lr = bias_lr_panel[i, j]
        color_lr = cmap_bias(norm_bias(val_lr)) if not np.isnan(val_lr) else 'lightgray'
        ax_main.add_patch(create_triangle(j, i, 'lower', color_lr))
        val_hr = bias_hr_panel[i, j]
        color_hr = cmap_bias(norm_bias(val_hr)) if not np.isnan(val_hr) else 'lightgray'
        ax_main.add_patch(create_triangle(j, i, 'upper', color_hr))

i_era5 = n_models + 1
for j in range(n_nodes):
    val_era5 = era5_row[j]
    color_era5 = cmap_era5(norm_era5(val_era5)) if not np.isnan(val_era5) else 'lightgray'
    ax_main.add_patch(plt.Rectangle((j - 0.5, i_era5 - 0.5), 1, 1,
                                     facecolor=color_era5, edgecolor='black', linewidth=0.5))

ax_main.set_xlim(-0.5, n_nodes - 0.5)
ax_main.set_ylim(-0.5, n_rows_total - 0.5)
ax_main.invert_yaxis()
for x in np.arange(-0.5, n_nodes, 1):
    ax_main.axvline(x, color='black', linewidth=0.8)
for y in np.arange(-0.5, n_rows_total, 1):
    ax_main.axhline(y, color='black', linewidth=0.8)
ax_main.axhline(y=n_models - 0.5, color='black', linewidth=2)
ax_main.axhline(y=n_models + 0.5, color='black', linewidth=2)

ax_main.set_xticks(np.arange(n_nodes))
ax_main.set_xticklabels(range(1, n_nodes + 1), ha='center', fontsize=16)
ax_main.set_yticks(np.arange(n_rows_total))
ax_main.set_yticklabels(row_labels, fontsize=16)
ax_main.set_xlabel("Regimes (SOM Nodes)", fontsize=16)

for i in range(n_rows_total):
    for j in range(n_nodes):
        if i == n_models + 1:
            val = era5_row[j]
            if not np.isnan(val):
                ax_main.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=12, color='black')
        elif i == n_models:
            val_lr = bias_lr_panel[i, j]
            if not np.isnan(val_lr):
                ax_main.text(j - 0.2, i - 0.15, f"{val_lr:+.1f}", ha='center', va='center',
                            fontsize=12, color='black')
            val_hr = bias_hr_panel[i, j]
            if not np.isnan(val_hr):
                ax_main.text(j + 0.2, i + 0.15, f"{val_hr:+.1f}", ha='center', va='center',
                            fontsize=12, color='black')

cbar_bias = fig.colorbar(plt.cm.ScalarMappable(norm=norm_bias, cmap=cmap_bias),
                         cax=cax_bias, orientation='horizontal')
cbar_bias.set_label("Regime-conditioned HW Intensity Bias (\u00b0C)", fontsize=16)
cbar_bias.ax.tick_params(labelsize=16)
cbar_bias.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

cbar_era5 = fig.colorbar(plt.cm.ScalarMappable(norm=norm_era5, cmap=cmap_era5),
                         cax=cax_era5, orientation='horizontal')
cbar_era5.set_label("ERA5 HW Intensity (deviation from 90p in \u00b0C)", fontsize=16)
cbar_era5.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
cbar_era5.ax.tick_params(labelsize=16)

plt.suptitle("Regime-conditioned HW intensity bias \n Low-res (\u25be) vs High-res (\u25b4)", fontsize=20, y=0.95)
plt.tight_layout()
plt.savefig(f'07_HW_magnitude_bias_{VAR}_{ystart}_{yend}.png', dpi=300, bbox_inches='tight')

# ================================================================
# Whisker-style redesign (ensemble spread vs. ERA5), with sample-size
# pre-filtering so no "insufficient sample" placeholder panels appear
# ================================================================
def plot_magnitude_spread(mag_lr, mag_hr, era5_mag, regimes_1idx, fname,
                          min_valid_models=3, n_cols=4):
    kept_regimes, dropped_regimes = [], []
    for regime_num in regimes_1idx:
        node = regime_num - 1
        n_valid_lr = np.sum(~np.isnan(mag_lr[:, node]))
        n_valid_hr = np.sum(~np.isnan(mag_hr[:, node]))
        era5_valid = not np.isnan(era5_mag[node])  # ERA5 itself has >= N_MIN HW days at this node
        if n_valid_lr >= min_valid_models and n_valid_hr >= min_valid_models and era5_valid:
            kept_regimes.append(regime_num)
        else:
            dropped_regimes.append((regime_num, n_valid_lr, n_valid_hr, era5_valid))

    if dropped_regimes:
        print(f"Excluded regimes (insufficient sample: <{min_valid_models} valid models in "
              f"LR and/or HR, or ERA5 itself has fewer than N_MIN HW days at this node):")
        for regime_num, n_lr, n_hr, era5_ok in dropped_regimes:
            print(f"  Regime {regime_num}: LR n={n_lr}, HR n={n_hr}, ERA5 sufficient={era5_ok}")
    regimes_1idx = kept_regimes

    n_panels = len(regimes_1idx)
    n_rows_ = int(np.ceil(n_panels / n_cols))
    fig, axs = plt.subplots(n_rows_, n_cols, figsize=(4.4 * n_cols, 4.6 * n_rows_), sharey=True)
    axs = np.atleast_2d(axs)

    X_LR, X_HR = -0.25, 0.25
    HALF_WIDTH = 0.07

    def plot_group(ax, x_center, vals, color):
        vals = vals[~np.isnan(vals)]
        if len(vals) < min_valid_models:
            ax.text(x_center, 0, "insufficient\nsample", ha="center", va="center",
                    fontsize=9, color="gray", style="italic")
            return np.nan, np.nan, vals
        ens_mean = np.nanmean(vals)
        v_min, v_max = np.nanmin(vals), np.nanmax(vals)
        v_range = v_max - v_min
        x_jitter = x_center + np.linspace(-0.04, 0.04, len(vals))
        ax.plot([x_center, x_center], [v_min, v_max], color="gray", linewidth=2.2,
                solid_capstyle="butt", zorder=1)
        ax.plot([x_center - HALF_WIDTH, x_center + HALF_WIDTH], [v_min, v_min], color="gray", linewidth=2.2, zorder=1)
        ax.plot([x_center - HALF_WIDTH, x_center + HALF_WIDTH], [v_max, v_max], color="gray", linewidth=2.2, zorder=1)
        ax.plot([x_center - HALF_WIDTH * 1.6, x_center + HALF_WIDTH * 1.6], [ens_mean, ens_mean],
                color=color, linewidth=2.2, zorder=3)
        ax.scatter(x_jitter, vals, color=color, s=42, zorder=4, edgecolor="white", linewidth=0.5)
        return ens_mean, v_range, vals

    all_bias_lr, all_bias_hr = [], []
    all_range_lr, all_range_hr = [], []

    for idx, regime_num in enumerate(regimes_1idx):
        row, col = divmod(idx, n_cols)
        ax = axs[row, col]
        node = regime_num - 1
        era5_val = era5_mag[node]

        mean_lr, range_lr, vals_lr = plot_group(ax, X_LR, mag_lr[:, node], "#4C72B0")
        mean_hr, range_hr, vals_hr = plot_group(ax, X_HR, mag_hr[:, node], "#DD8452")

        if not np.isnan(mean_lr):
            all_bias_lr.append(mean_lr - era5_val)
            all_range_lr.append(range_lr)
        if not np.isnan(mean_hr):
            all_bias_hr.append(mean_hr - era5_val)
            all_range_hr.append(range_hr)

        ax.scatter([0], [era5_val], color="tab:red", marker="D", s=65, zorder=5,
                  edgecolor="black", linewidth=0.4)

        if not np.isnan(mean_lr):
            ax.plot([X_LR + HALF_WIDTH * 1.6, 0], [mean_lr, era5_val], color="#4C72B0",
                    linewidth=0.9, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)
        if not np.isnan(mean_hr):
            ax.plot([X_HR - HALF_WIDTH * 1.6, 0], [mean_hr, era5_val], color="#DD8452",
                    linewidth=0.9, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)

        ax.set_xticks([X_LR, X_HR])
        ax.set_xticklabels(["LR", "HR"], fontsize=14)
        ax.set_xlim(-0.42, 0.42)

        fw = "normal"
        ax.set_title(f"Regime {regime_num}", fontsize=18, fontweight=fw)

        for label, rng, gap_val, y_off, color in [
            ("LR", range_lr, (mean_lr - era5_val) if not np.isnan(mean_lr) else np.nan, -0.08, "#4C72B0"),
            ("HR", range_hr, (mean_hr - era5_val) if not np.isnan(mean_hr) else np.nan, -0.17, "#DD8452"),
        ]:
            if not np.isnan(rng):
                verdict_color = "#1a7a3c" if rng >= abs(gap_val) else "#b3401f"
                ax.text(0.5, y_off, f"{label}: spread={rng:.2f}  |bias|={abs(gap_val):.2f}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=12, color=verdict_color)

    for idx in range(n_panels, n_rows_ * n_cols):
        row, col = divmod(idx, n_cols)
        axs[row, col].axis("off")
    for row in range(n_rows_):
        axs[row, 0].set_ylabel("HW intensity (\u00b0C, deviation from 90p)", fontsize=15)

    legend_handles = [
        mlines.Line2D([], [], color="#4C72B0", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="LR models"),
        mlines.Line2D([], [], color="#DD8452", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="HR models"),
        mlines.Line2D([], [], color="gray", linewidth=2.5, label="Ensemble spread (min\u2013max range)"),
        mlines.Line2D([], [], color="tab:red", marker="D", linestyle="None", markersize=8,
                      markeredgecolor="black", label="ERA5 reference"),
        mlines.Line2D([], [], color="gray", linewidth=1.2, linestyle=(0, (2, 1)),
                      label="Ensemble-mean\u2013ERA5 bias"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=16)

    mean_abs_bias_lr = np.mean(np.abs(all_bias_lr)) if all_bias_lr else np.nan
    mean_abs_bias_hr = np.mean(np.abs(all_bias_hr)) if all_bias_hr else np.nan
    mean_range_lr = np.mean(all_range_lr) if all_range_lr else np.nan
    mean_range_hr = np.mean(all_range_hr) if all_range_hr else np.nan
    summary_text = (
        f"Ensemble mean |bias| vs. ERA5:  LR = {mean_abs_bias_lr:.2f}   HR = {mean_abs_bias_hr:.2f}"
        f"     |     Ensemble spread (mean min\u2013max range):  LR = {mean_range_lr:.2f}   HR = {mean_range_hr:.2f}"
    )
    fig.text(0.5, 1.025, summary_text, ha="center", va="center", fontsize=16,
             fontweight="bold", color="#333333")
    fig.suptitle("Regime-conditioned HW intensity", fontsize=22, y=1.1)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")


# Regimes to plot: determined dynamically from ERA5's own HW-day count
# per node (number_array, computed in the ERA5 section above), rather
# than a hardcoded regime list -- Z500 and SST use different SOMs with
# different regime numbering, so a fixed list would not transfer between
# the two variables. Threshold matches N_MIN (currently >10 HW days).
era5_hw_count = number_array.ravel()
regimes_to_plot = [node + 1 for node in range(n_nodes) if era5_hw_count[node] >= N_MIN]
print(f"Regimes plotted (ERA5 HW-day count >= {N_MIN}): {regimes_to_plot}")

plot_magnitude_spread(mag_matrix_lr, mag_matrix_hr, era5_mag_flat,
                      regimes_1idx=regimes_to_plot,
                      fname=f"07b_HW_magnitude_ensemble_whisker_{VAR}.png")
