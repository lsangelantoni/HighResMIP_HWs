"""
04_regime_persistence.py

Corresponds to original notebook section "Mean regime persistence bias".

NOTE ON CLEANUP: the original notebook contained an "#### old version"
persistence calculation that counted consecutive ARRAY INDICES as
consecutive days, without checking actual calendar dates. This is
incorrect for HW-only datasets (which have temporal gaps between
events) and is superseded by the "#### Adjusted" version below. Only
the corrected version is kept here.

Requires: 00_setup_and_functions.py (import * or run first)
"""

from importlib import import_module
setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

import matplotlib.lines as mlines

# Set all_days=True for the all-summer-day persistence diagnostic,
# or False for the HW-day-only diagnostic (both are used in the paper)
all_days = False


def compute_node_persistence(ds, som_grid_cols, n_nodes):
    """
    Mean persistence of each SOM node.

    Persistence is the mean length of consecutive calendar-day streaks
    assigned to the same SOM node.

    This works for both:
    - all-summer-day datasets
    - HW-only datasets with temporal gaps
    """
    node_series = (
        ds["bmu_row"].values.astype(int) * som_grid_cols
        + ds["bmu_col"].values.astype(int)
    )
    time = ds.time.values.astype("datetime64[D]")

    persistence = np.full(n_nodes, np.nan)

    for node in range(n_nodes):
        streaks = []
        current_streak = 0
        for i, node_i in enumerate(node_series):
            if i == 0:
                consecutive_day = True
            else:
                consecutive_day = (time[i] - time[i - 1]).astype(int) == 1

            if node_i == node and (current_streak == 0 or consecutive_day):
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = 0
                if node_i == node:
                    current_streak = 1

        if current_streak > 0:
            streaks.append(current_streak)
        if len(streaks) > 0:
            persistence[node] = np.mean(streaks)

    return persistence


# ================================================================
# Heatmap color parameters
# ================================================================
levels_bias = np.linspace(-40, 40, 21)
if VAR == "tos":
    levels_bias = np.linspace(-100, 100, 21)
cmap_bias = sns.color_palette("RdBu_r", len(levels_bias) - 1, as_cmap=True)
norm_bias = mcolors.BoundaryNorm(levels_bias, cmap_bias.N, clip=False, extend="both")

levels_era5 = np.linspace(1, 4, 7)
if VAR == "tos":
    levels_era5 = np.linspace(1, 6, 11)
if VAR == "tos" and all_days:
    levels_era5 = np.linspace(1, 20, 20)
cmap_era5 = sns.color_palette("YlOrRd", len(levels_era5) - 1, as_cmap=True)
norm_era5 = BoundaryNorm(levels_era5, cmap_era5.N, clip=False, extend="both")

n_models = len(models)
n_nodes = som_grid_rows * som_grid_cols

persistence_matrix_lr = np.full((n_models, n_nodes), np.nan)
persistence_matrix_hr = np.full((n_models, n_nodes), np.nan)
bias_matrix_lr = np.full((n_models, n_nodes), np.nan)
bias_matrix_hr = np.full((n_models, n_nodes), np.nan)

# ================================================================
# ERA5 persistence
# ================================================================
if all_days:
    era5_file = (f"{som_path}/som_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_"
                 f"iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc")
else:
    era5_file = (f"{som_path}/som_projected_HW_days_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_"
                 f"iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc")

ds_era5 = xr.open_mfdataset(era5_file)
ds_era5 = ds_era5.sel(time=ds_era5.time.dt.month.isin([5, 6, 7, 8, 9]))
persistence_era5 = compute_node_persistence(ds_era5, som_grid_cols=som_grid_cols, n_nodes=n_nodes)

# ERA5's own HW-day count per node -- nodes with zero HW days are left
# blank throughout (heatmap and whisker plot alike).
ds_era5_hw = xr.open_mfdataset(
    f"{som_path}/som_projected_HW_days_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_"
    f"iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc")
node_era5_hw = (ds_era5_hw["bmu_row"].values.astype(int) * som_grid_cols
                + ds_era5_hw["bmu_col"].values.astype(int))
era5_hw_count = np.array([np.sum(node_era5_hw == k) for k in range(n_nodes)])

# A regime with zero ERA5 HW days has genuinely no data to compute HW-day
# persistence from, so it is blanked out in that mode. In the all-days
# mode, the same regime still occurs on ordinary summer days and has a
# well-defined all-day persistence value -- whether it happens to never
# host a heatwave is unrelated to that, so nothing is blanked there.
no_hw_in_era5 = (era5_hw_count == 0) if not all_days else np.zeros(n_nodes, dtype=bool)

# ================================================================
# Loop over models
# ================================================================
for m_index in range(n_models):
    model = models[m_index]
    realization = realizations[m_index]
    mod_lr = models_lr[m_index]
    mod_hr = models_hr[m_index]

    if all_days:
        if yend == 2024 and model[:5] == "ECMWF":
            file_lr = (f"som_projected_all_days_{mod_lr}_{realization}_{VAR}_{ystart}_2014_"
                       f"{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}")
        else:
            file_lr = (f"som_projected_all_days_{mod_lr}_{realization}_{VAR}_{ystart}_{yend}_"
                       f"{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}")
    else:
        if yend == 2024 and model[:5] == "ECMWF":
            file_lr = (f"som_projected_HW_days_{mod_lr}_{realization}_{VAR}_{ystart}_2014_"
                       f"{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}")
        else:
            file_lr = (f"som_projected_HW_days_{mod_lr}_{realization}_{VAR}_{ystart}_{yend}_"
                       f"{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}")

    ds_lr = xr.open_dataset(f"/work/cmcc/ls21622/HighResMIP/{mod_lr}/{realization}/som/{file_lr}.nc")
    ds_lr = ds_lr.sel(time=ds_lr.time.dt.month.isin([5, 6, 7, 8, 9]))
    persistence_matrix_lr[m_index, :] = compute_node_persistence(ds_lr, som_grid_cols=som_grid_cols, n_nodes=n_nodes)

    if all_days:
        if yend == 2024 and model[:5] == "ECMWF":
            file_hr = (f"som_projected_all_days_{mod_hr}_{realization}_{VAR}_{ystart}_2014_"
                       f"{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}")
        else:
            file_hr = (f"som_projected_all_days_{mod_hr}_{realization}_{VAR}_{ystart}_{yend}_"
                       f"{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}")
    else:
        if yend == 2024 and model[:5] == "ECMWF":
            file_hr = (f"som_projected_HW_days_{mod_hr}_{realization}_{VAR}_{ystart}_2014_"
                       f"{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}")
        else:
            file_hr = (f"som_projected_HW_days_{mod_hr}_{realization}_{VAR}_{ystart}_{yend}_"
                       f"{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}")

    ds_hr = xr.open_dataset(f"/work/cmcc/ls21622/HighResMIP/{mod_hr}/{realization}/som/{file_hr}.nc")
    ds_hr = ds_hr.sel(time=ds_hr.time.dt.month.isin([5, 6, 7, 8, 9]))
    persistence_matrix_hr[m_index, :] = compute_node_persistence(ds_hr, som_grid_cols=som_grid_cols, n_nodes=n_nodes)

# ================================================================
# Percentage persistence bias (model - ERA5, normalized by ERA5)
# ================================================================
for m_index in range(n_models):
    valid_lr = (np.isfinite(persistence_matrix_lr[m_index, :]) & np.isfinite(persistence_era5)
                & (persistence_era5 != 0))
    valid_hr = (np.isfinite(persistence_matrix_hr[m_index, :]) & np.isfinite(persistence_era5)
                & (persistence_era5 != 0))
    bias_matrix_lr[m_index, valid_lr] = (
        (persistence_matrix_lr[m_index, valid_lr] - persistence_era5[valid_lr])
        / persistence_era5[valid_lr] * 100)
    bias_matrix_hr[m_index, valid_hr] = (
        (persistence_matrix_hr[m_index, valid_hr] - persistence_era5[valid_hr])
        / persistence_era5[valid_hr] * 100)

ensemble_mean_lr = np.nanmean(bias_matrix_lr, axis=0)
ensemble_mean_hr = np.nanmean(bias_matrix_hr, axis=0)

bias_lr_panel = np.vstack([bias_matrix_lr, ensemble_mean_lr])
bias_hr_panel = np.vstack([bias_matrix_hr, ensemble_mean_hr])
era5_row = persistence_era5
row_labels = models + ["Ensemble Mean", "ERA5"]
n_rows_total = n_models + 2

# ================================================================
# Triangle heatmap (regimes with no ERA5 HW days left blank)
# ================================================================
fig = plt.figure(figsize=(19, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[10, 0.5, 0.5], hspace=0.2)
ax_main = fig.add_subplot(gs[0, 0])
cax_bias = fig.add_subplot(gs[1, 0])
cax_era5 = fig.add_subplot(gs[2, 0])

for i in range(n_models + 1):
    for j in range(n_nodes):
        val_lr = bias_lr_panel[i, j]
        color_lr = 'lightgray' if (no_hw_in_era5[j] or np.isnan(val_lr)) else cmap_bias(norm_bias(val_lr))
        ax_main.add_patch(create_triangle(j, i, 'lower', color_lr))
        val_hr = bias_hr_panel[i, j]
        color_hr = 'lightgray' if (no_hw_in_era5[j] or np.isnan(val_hr)) else cmap_bias(norm_bias(val_hr))
        ax_main.add_patch(create_triangle(j, i, 'upper', color_hr))

i_era5 = n_models + 1
for j in range(n_nodes):
    val_era5 = era5_row[j]
    color_era5 = 'lightgray' if (no_hw_in_era5[j] or np.isnan(val_era5)) else cmap_era5(norm_era5(val_era5))
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
        if no_hw_in_era5[j]:
            continue
        if i == n_models + 1:
            val = era5_row[j]
            if not np.isnan(val):
                ax_main.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=12, color='black')
        elif i == n_models:
            val_lr = bias_lr_panel[i, j]
            if not np.isnan(val_lr):
                ax_main.text(j - 0.2, i - 0.15, f"{val_lr:+.0f}", ha='center', va='center',
                            fontsize=12, color='black')
            val_hr = bias_hr_panel[i, j]
            if not np.isnan(val_hr):
                ax_main.text(j + 0.2, i + 0.15, f"{val_hr:+.0f}", ha='center', va='center',
                            fontsize=12, color='black')

cbar_bias = fig.colorbar(plt.cm.ScalarMappable(norm=norm_bias, cmap=cmap_bias),
                         cax=cax_bias, orientation='horizontal')
cbar_bias.set_label("Persistence bias (%)", fontsize=16)
cbar_bias.ax.tick_params(labelsize=16)
cbar_era5 = fig.colorbar(plt.cm.ScalarMappable(norm=norm_era5, cmap=cmap_era5),
                         cax=cax_era5, orientation='horizontal')
cbar_era5.set_label("ERA5 mean persistence (days)", fontsize=16)
cbar_era5.ax.tick_params(labelsize=16)

if all_days:
    plt.suptitle("Mean regime persistence bias all days (MJJAS) \nLow-res (\u25be) vs High-res (\u25b4)",
                fontsize=20, y=0.95)
    plt.tight_layout()
    plt.savefig(f'regime_persistence_bias_all_days_{VAR}_{ystart}_{yend}.png', dpi=300, bbox_inches='tight')
else:
    plt.suptitle("Mean regime persistence bias HW-day only \nLow-res (\u25be) vs High-res (\u25b4)",
                fontsize=20, y=0.95)
    plt.tight_layout()
    plt.savefig(f'regime_persistence_bias_HW_days_{VAR}_{ystart}_{yend}.png', dpi=300, bbox_inches='tight')

# ================================================================
# Whisker-style redesign: same "no HW days in ERA5 -> blank" rule
# ================================================================
def plot_persistence_spread(persist_lr, persist_hr, era5_persist, no_hw_in_era5, n_nodes, fname,
                            n_cols=5, min_valid_models=3):
    regimes_1idx = list(range(1, n_nodes + 1))
    n_rows_ = int(np.ceil(n_nodes / n_cols))
    fig, axs = plt.subplots(n_rows_, n_cols, figsize=(4.2 * n_cols, 4.4 * n_rows_), sharey=True)
    axs = np.atleast_2d(axs)

    X_LR, X_HR = -0.25, 0.25
    HALF_WIDTH = 0.07

    def plot_group(ax, x_center, vals, color):
        vals = vals[~np.isnan(vals)]
        if len(vals) < min_valid_models:
            ax.text(x_center, 0, "insufficient\nsample", ha="center", va="center",
                    fontsize=8, color="gray", style="italic")
            return np.nan, np.nan
        ens_mean = np.nanmean(vals)
        v_min, v_max = np.nanmin(vals), np.nanmax(vals)
        v_range = v_max - v_min
        x_jitter = x_center + np.linspace(-0.04, 0.04, len(vals))
        ax.plot([x_center, x_center], [v_min, v_max], color="gray", linewidth=2.0,
                solid_capstyle="butt", zorder=1)
        ax.plot([x_center - HALF_WIDTH, x_center + HALF_WIDTH], [v_min, v_min], color="gray", linewidth=2.0, zorder=1)
        ax.plot([x_center - HALF_WIDTH, x_center + HALF_WIDTH], [v_max, v_max], color="gray", linewidth=2.0, zorder=1)
        ax.plot([x_center - HALF_WIDTH * 1.6, x_center + HALF_WIDTH * 1.6], [ens_mean, ens_mean],
                color=color, linewidth=2.0, zorder=3)
        ax.scatter(x_jitter, vals, color=color, s=36, zorder=4, edgecolor="white", linewidth=0.5)
        return ens_mean, v_range

    all_bias_lr, all_bias_hr = [], []
    all_range_lr, all_range_hr = [], []

    for idx, regime_num in enumerate(regimes_1idx):
        row, col = divmod(idx, n_cols)
        ax = axs[row, col]
        node = regime_num - 1

        ax.set_xticks([X_LR, X_HR])
        ax.set_xticklabels(["LR", "HR"], fontsize=14)
        ax.set_xlim(-0.42, 0.42)
        ax.set_title(f"Regime {regime_num}", fontsize=18)

        if no_hw_in_era5[node]:
            ax.set_facecolor("#f2f2f2")
            continue

        era5_val = era5_persist[node]
        mean_lr, range_lr = plot_group(ax, X_LR, persist_lr[:, node], "#4C72B0")
        mean_hr, range_hr = plot_group(ax, X_HR, persist_hr[:, node], "#DD8452")

        if not np.isnan(mean_lr):
            all_bias_lr.append(mean_lr - era5_val)
            all_range_lr.append(range_lr)
        if not np.isnan(mean_hr):
            all_bias_hr.append(mean_hr - era5_val)
            all_range_hr.append(range_hr)

        ax.scatter([0], [era5_val], color="tab:red", marker="D", s=60, zorder=5,
                  edgecolor="black", linewidth=0.4)

        if not np.isnan(mean_lr):
            ax.plot([X_LR + HALF_WIDTH * 1.6, 0], [mean_lr, era5_val], color="#4C72B0",
                    linewidth=0.8, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)
        if not np.isnan(mean_hr):
            ax.plot([X_HR - HALF_WIDTH * 1.6, 0], [mean_hr, era5_val], color="#DD8452",
                    linewidth=0.8, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)

        for label, rng, gap_val, y_off, color in [
            ("LR", range_lr, (mean_lr - era5_val) if not np.isnan(mean_lr) else np.nan, -0.08, "#4C72B0"),
            ("HR", range_hr, (mean_hr - era5_val) if not np.isnan(mean_hr) else np.nan, -0.17, "#DD8452"),
        ]:
            if not np.isnan(rng):
                verdict_color = "#1a7a3c" if rng >= abs(gap_val) else "#b3401f"
                ax.text(0.5, y_off, f"{label}: spread={rng:.2f}  |bias|={abs(gap_val):.2f}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=12, color=verdict_color)

    for row in range(n_rows_):
        axs[row, 0].set_ylabel("Regime persistence (days)", fontsize=15)

    legend_handles = [
        mlines.Line2D([], [], color="#4C72B0", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="LR models"),
        mlines.Line2D([], [], color="#DD8452", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="HR models"),
        mlines.Line2D([], [], color="gray", linewidth=2.5, label="Ensemble spread (min\u2013max)"),
        mlines.Line2D([], [], color="tab:red", marker="D", linestyle="None", markersize=8,
                      markeredgecolor="black", label="ERA5 reference"),
        mlines.Line2D([], [], color="gray", linewidth=1.0, linestyle=(0, (2, 1)),
                      label="Ensemble-mean\u2013ERA5 gap"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=14)

    mean_abs_bias_lr = np.mean(np.abs(all_bias_lr)) if all_bias_lr else np.nan
    mean_abs_bias_hr = np.mean(np.abs(all_bias_hr)) if all_bias_hr else np.nan
    mean_range_lr = np.mean(all_range_lr) if all_range_lr else np.nan
    mean_range_hr = np.mean(all_range_hr) if all_range_hr else np.nan
    summary_text = (
        f"Ensemble mean |bias|:  LR = {mean_abs_bias_lr:.2f}   HR = {mean_abs_bias_hr:.2f}"
        f"     |     Ensemble spread:  LR = {mean_range_lr:.2f}   HR = {mean_range_hr:.2f}"
    )
    fig.text(0.5, 1.09, summary_text, ha="center", va="center", fontsize=16,
             fontweight="bold", color="#333333")
    fig.suptitle("Regime persistence: ensemble spread vs. ERA5", fontsize=22, y=1.16)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")


day_tag = "all_days" if all_days else "HW_days"
plot_persistence_spread(
    persistence_matrix_lr, persistence_matrix_hr, persistence_era5, no_hw_in_era5,
    n_nodes=som_grid_rows * som_grid_cols,
    fname=f"Figure_persistence_ensemble_spread_{day_tag}.png"
)
