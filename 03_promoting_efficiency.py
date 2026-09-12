"""
03_promoting_efficiency.py

Corresponds to original notebook sections "REGIMES BIAS" (all-day
frequency bias) and "HW-promoting efficiency bias" (Fig. 8).

NOTE ON CLEANUP: this section contained THREE separate instances of
duplicated computation:
  1. The all-day regime frequency bias loop (hits_matrix_lr/hr) was
     executed twice, back to back, with identical logic. Only the
     final occurrence (the one immediately followed by the triangle
     plot) is kept here.
  2. The ERA5-level HW-promoting efficiency computation originally
     appeared WITHOUT the suppression test (dead/superseded), followed
     by a "#### Claude" block that redid the same computation WITH the
     added suppression test (Reviewer 2, Sec 2.4(2)). Only the latter
     is kept.
  3. The per-model LR/HR promoting-efficiency loop had the same
     old-then-patched duplication. Only the patched ("#### Claude")
     version, which also populates signif_suppress_matrix_lr/hr, is
     kept.

Requires: 00_setup_and_functions.py (import * or run first)
"""

from importlib import import_module
setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines

# ================================================================
# All-day regime frequency bias
# ================================================================
levels_bias = np.linspace(-100, 100, 21)
cmap_bias = sns.color_palette("RdBu_r", len(levels_bias) - 1, as_cmap=True)
norm_bias = mcolors.BoundaryNorm(levels_bias, cmap_bias.N, clip=False, extend='both')

levels_era5 = np.linspace(0, 600, 21)
cmap_era5 = sns.color_palette("YlOrRd", len(levels_era5) - 1, as_cmap=True)
norm_era5 = BoundaryNorm(levels_era5, cmap_era5.N, clip=False, extend='max')

n_models = len(models)
n_nodes = som_grid_rows * som_grid_cols

hits_matrix_hr = np.zeros((n_models, n_nodes))
hits_matrix_lr = np.zeros((n_models, n_nodes))
bias_matrix_hr = np.zeros((n_models, n_nodes))
bias_matrix_lr = np.zeros((n_models, n_nodes))

for m_index in range(n_models):
    model = models[m_index]
    realization = realizations[m_index]
    mod_lr = models_lr[m_index]
    mod_hr = models_hr[m_index]

    file_in = (f'som_projected_all_days_{mod_lr}_{realization}_{VAR}_{ystart}_{yend}_'
               f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
    ds = xr.open_dataset(f'/work/cmcc/ls21622/HighResMIP/{mod_lr}/{realization}/som/{file_in}.nc')
    hits_matrix_lr[m_index, :] = ds["hit_counts"].values.ravel()

    file_in = (f'som_projected_all_days_{mod_hr}_{realization}_{VAR}_{ystart}_{yend}_'
               f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
    ds = xr.open_dataset(f'/work/cmcc/ls21622/HighResMIP/{mod_hr}/{realization}/som/{file_in}.nc')
    hits_matrix_hr[m_index, :] = ds["hit_counts"].values.ravel()

ds_era5 = xr.open_mfdataset(
    f'{som_path}/som_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
hits_era5 = ds_era5['hit_counts'].values.ravel()

for m_index in range(n_models):
    with np.errstate(divide='ignore', invalid='ignore'):
        bias_matrix_lr[m_index, :] = (hits_matrix_lr[m_index, :] - hits_era5) / hits_era5 * 100
        bias_matrix_hr[m_index, :] = (hits_matrix_hr[m_index, :] - hits_era5) / hits_era5 * 100
    bias_matrix_lr[m_index, :] = np.nan_to_num(bias_matrix_lr[m_index, :], nan=0, posinf=0, neginf=0)
    bias_matrix_hr[m_index, :] = np.nan_to_num(bias_matrix_hr[m_index, :], nan=0, posinf=0, neginf=0)

ensemble_mean_lr_freq = np.mean(bias_matrix_lr, axis=0)
ensemble_mean_hr_freq = np.mean(bias_matrix_hr, axis=0)

bias_lr_panel_freq = np.vstack([bias_matrix_lr, ensemble_mean_lr_freq])
bias_hr_panel_freq = np.vstack([bias_matrix_hr, ensemble_mean_hr_freq])
era5_row_freq = hits_era5
row_labels_freq = models + ['Ensemble Mean', 'ERA5']
n_rows_total_freq = n_models + 2

fig = plt.figure(figsize=(19, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[10, 0.5, 0.5], hspace=0.2)
ax_main = fig.add_subplot(gs[0, 0])
cax_bias = fig.add_subplot(gs[1, 0])
cax_era5 = fig.add_subplot(gs[2, 0])

for i in range(n_models + 1):
    for j in range(n_nodes):
        val_lr = bias_lr_panel_freq[i, j]
        ax_main.add_patch(create_triangle(j, i, 'lower', cmap_bias(norm_bias(val_lr))))
        val_hr = bias_hr_panel_freq[i, j]
        ax_main.add_patch(create_triangle(j, i, 'upper', cmap_bias(norm_bias(val_hr))))

i_era5 = n_models + 1
for j in range(n_nodes):
    color_era5 = cmap_era5(norm_era5(era5_row_freq[j]))
    ax_main.add_patch(plt.Rectangle((j - 0.5, i_era5 - 0.5), 1, 1,
                                     facecolor=color_era5, edgecolor='black', linewidth=0.5))

ax_main.set_xlim(-0.5, n_nodes - 0.5)
ax_main.set_ylim(-0.5, n_rows_total_freq - 0.5)
ax_main.invert_yaxis()
for x in np.arange(-0.5, n_nodes, 1):
    ax_main.axvline(x, color='black', linewidth=0.8)
for y in np.arange(-0.5, n_rows_total_freq, 1):
    ax_main.axhline(y, color='black', linewidth=0.8)
ax_main.axhline(y=n_models - 0.5, color='black', linewidth=2)
ax_main.axhline(y=n_models + 0.5, color='black', linewidth=2)

ax_main.set_xticks(np.arange(n_nodes))
ax_main.set_xticklabels(range(1, n_nodes + 1), ha='center', fontsize=16)
ax_main.set_yticks(np.arange(n_rows_total_freq))
ax_main.set_yticklabels(row_labels_freq, fontsize=16)
ax_main.set_xlabel("Regimes (SOM Nodes)", fontsize=16)

for i in range(n_rows_total_freq):
    for j in range(n_nodes):
        if i == n_models + 1:
            ax_main.text(j, i, f"{int(era5_row_freq[j])}", ha='center', va='center',
                        fontsize=10, color='black', fontweight='bold')
        else:
            fw = 'bold' if i == n_models else 'normal'
            ax_main.text(j - 0.2, i - 0.15, f"{bias_lr_panel_freq[i, j]:+.0f}",
                        ha='center', va='center', fontsize=9, color='black', fontweight=fw)
            ax_main.text(j + 0.2, i + 0.15, f"{bias_hr_panel_freq[i, j]:+.0f}",
                        ha='center', va='center', fontsize=9, color='black', fontweight=fw)

cbar_bias = fig.colorbar(plt.cm.ScalarMappable(norm=norm_bias, cmap=cmap_bias),
                         cax=cax_bias, orientation='horizontal')
cbar_bias.set_label("Bias (%)", fontsize=16)
cbar_bias.ax.tick_params(labelsize=16)
cbar_era5 = fig.colorbar(plt.cm.ScalarMappable(norm=norm_era5, cmap=cmap_era5),
                         cax=cax_era5, orientation='horizontal')
cbar_era5.set_label("ERA5 regime frequency (number of days)", fontsize=16)
cbar_era5.ax.tick_params(labelsize=16)
plt.suptitle("Regimes frequency bias \n Low-res (\u25be) vs High-res (\u25b4)", fontsize=20, y=0.95)
plt.tight_layout()
#plt.savefig(f'03_regime_frequency_bias_{VAR}_{ystart}_{yend}.png', dpi=300, bbox_inches='tight')

# ================================================================
# HW-promoting / HW-suppressing efficiency: ERA5 level
# (patched version with both one-sided tests -- Reviewer 2, Sec 2.4(2))
# ================================================================
ds = xr.open_mfdataset(
    f'{som_path}/som_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
ds_hw = xr.open_mfdataset(
    f'{som_path}/som_projected_HW_days_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
    f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')

row_all = ds["bmu_row"].values
col_all = ds["bmu_col"].values
n_rows = ds.dims["som_row"]
n_cols = ds.dims["som_col"]
node_all = row_all * n_cols + col_all

row_hw = ds_hw["bmu_row"].values
col_hw = ds_hw["bmu_col"].values
node_hw = row_hw * n_cols + col_hw

n_nodes = n_rows * n_cols
obs_counts = np.array([np.sum(node_hw == k) for k in range(n_nodes)])

N_all = len(node_all)
N_hw = len(node_hw)
mc_counts = np.zeros((n_iter, n_nodes))
for i in range(n_iter):
    random_days = np.random.choice(N_all, size=N_hw, replace=False)
    random_nodes = node_all[random_days]
    for k in range(n_nodes):
        mc_counts[i, k] = np.sum(random_nodes == k)

alpha = 0.05

# --- upper-tail: HW-promoting ---
pvals = np.zeros(n_nodes)
for k in range(n_nodes):
    extreme = np.sum(mc_counts[:, k] >= obs_counts[k])
    pvals[k] = (extreme + 1) / (n_iter + 1)
promoting = pvals < alpha
promoting_grid = promoting.reshape(n_rows, n_cols)
pvals_grid = pvals.reshape(n_rows, n_cols)
obs_grid = obs_counts.reshape(n_rows, n_cols)

# --- lower-tail: HW-suppressing (Reviewer 2, Sec 2.4(2)) ---
pvals_suppress = np.zeros(n_nodes)
for k in range(n_nodes):
    extreme_suppress = np.sum(mc_counts[:, k] <= obs_counts[k])
    pvals_suppress[k] = (extreme_suppress + 1) / (n_iter + 1)
suppressing = pvals_suppress < alpha
suppressing_grid = suppressing.reshape(n_rows, n_cols)
pvals_suppress_grid = pvals_suppress.reshape(n_rows, n_cols)

baseline_freq = np.array([np.sum(node_all == k) / len(node_all) for k in range(n_nodes)])
expected = baseline_freq * len(node_hw)
enrichment = obs_counts / expected
enrichment_grid = enrichment.reshape(n_rows, n_cols)

# ================================================================
# HW-promoting / HW-suppressing efficiency: per-model LR/HR loop
# (patched version -- Reviewer 2, Sec 2.4(2))
# ================================================================
n_models = len(models)
n_nodes = som_grid_rows * som_grid_cols

enrichment_matrix_hr = np.zeros((n_models, n_nodes))
enrichment_matrix_lr = np.zeros((n_models, n_nodes))
signif_matrix_hr = np.zeros((n_models, n_nodes))
signif_matrix_lr = np.zeros((n_models, n_nodes))
signif_suppress_matrix_hr = np.zeros((n_models, n_nodes))
signif_suppress_matrix_lr = np.zeros((n_models, n_nodes))

for m_index in range(n_models):
    model = models[m_index]
    realization = realizations[m_index]
    mod_lr = models_lr[m_index]
    mod_hr = models_hr[m_index]

    # ----------------------
    # LOW RESOLUTION
    # ----------------------
    if yend == 2024 and model[:5] == 'ECMWF':
        file_all = (f'som_projected_all_days_{mod_lr}_{realization}_{VAR}_{ystart}_2014_'
                    f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
        file_hw = (f'som_projected_HW_days_{mod_lr}_{realization}_{VAR}_{ystart}_2014_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
    else:
        file_all = (f'som_projected_all_days_{mod_lr}_{realization}_{VAR}_{ystart}_{yend}_'
                    f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
        file_hw = (f'som_projected_HW_days_{mod_lr}_{realization}_{VAR}_{ystart}_{yend}_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')

    ds = xr.open_dataset(f'/work/cmcc/ls21622/HighResMIP/{mod_lr}/{realization}/som/{file_all}.nc')
    ds_hw = xr.open_dataset(f'/work/cmcc/ls21622/HighResMIP/{mod_lr}/{realization}/som/{file_hw}.nc')

    row_all = ds["bmu_row"].values
    col_all = ds["bmu_col"].values
    n_cols = ds.dims["som_col"]
    n_rows = ds.dims["som_row"]
    node_all = row_all * n_cols + col_all

    row_hw = ds_hw["bmu_row"].values
    col_hw = ds_hw["bmu_col"].values
    node_hw = row_hw * n_cols + col_hw

    n_nodes = n_rows * n_cols
    obs_counts = np.array([np.sum(node_hw == k) for k in range(n_nodes)])
    baseline_freq = np.array([np.sum(node_all == k) / len(node_all) for k in range(n_nodes)])
    expected = baseline_freq * len(node_hw)
    enrichment = obs_counts / expected
    enrichment_matrix_lr[m_index, :] = enrichment

    N_all = len(node_all)
    N_hw = len(node_hw)
    mc_counts = np.zeros((n_iter, n_nodes))
    for i in range(n_iter):
        random_days = np.random.choice(N_all, size=N_hw, replace=False)
        random_nodes = node_all[random_days]
        for k in range(n_nodes):
            mc_counts[i, k] = np.sum(random_nodes == k)

    pvals = np.zeros(n_nodes)
    for k in range(n_nodes):
        extreme = np.sum(mc_counts[:, k] >= obs_counts[k])
        pvals[k] = (extreme + 1) / (n_iter + 1)
    signif_matrix_lr[m_index, :] = pvals < 0.05

    pvals_suppress = np.zeros(n_nodes)
    for k in range(n_nodes):
        extreme_suppress = np.sum(mc_counts[:, k] <= obs_counts[k])
        pvals_suppress[k] = (extreme_suppress + 1) / (n_iter + 1)
    signif_suppress_matrix_lr[m_index, :] = pvals_suppress < 0.05

    # ----------------------
    # HIGH RESOLUTION
    # ----------------------
    if yend == 2024 and model[:5] == 'ECMWF':
        file_all = (f'som_projected_all_days_{mod_hr}_{realization}_{VAR}_{ystart}_2014_'
                    f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
        file_hw = (f'som_projected_HW_days_{mod_hr}_{realization}_{VAR}_{ystart}_2014_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
    else:
        file_all = (f'som_projected_all_days_{mod_hr}_{realization}_{VAR}_{ystart}_{yend}_'
                    f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
        file_hw = (f'som_projected_HW_days_{mod_hr}_{realization}_{VAR}_{ystart}_{yend}_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')

    ds = xr.open_dataset(f'/work/cmcc/ls21622/HighResMIP/{mod_hr}/{realization}/som/{file_all}.nc')
    ds_hw = xr.open_dataset(f'/work/cmcc/ls21622/HighResMIP/{mod_hr}/{realization}/som/{file_hw}.nc')

    row_all = ds["bmu_row"].values
    col_all = ds["bmu_col"].values
    n_cols = ds.dims["som_col"]
    n_rows = ds.dims["som_row"]
    node_all = row_all * n_cols + col_all

    row_hw = ds_hw["bmu_row"].values
    col_hw = ds_hw["bmu_col"].values
    node_hw = row_hw * n_cols + col_hw

    n_nodes = n_rows * n_cols
    obs_counts = np.array([np.sum(node_hw == k) for k in range(n_nodes)])
    baseline_freq = np.array([np.sum(node_all == k) / len(node_all) for k in range(n_nodes)])
    expected = baseline_freq * len(node_hw)
    enrichment = obs_counts / expected
    enrichment_matrix_hr[m_index, :] = enrichment

    N_all = len(node_all)
    N_hw = len(node_hw)
    mc_counts = np.zeros((n_iter, n_nodes))
    for i in range(n_iter):
        random_days = np.random.choice(N_all, size=N_hw, replace=False)
        random_nodes = node_all[random_days]
        for k in range(n_nodes):
            mc_counts[i, k] = np.sum(random_nodes == k)

    pvals = np.zeros(n_nodes)
    for k in range(n_nodes):
        extreme = np.sum(mc_counts[:, k] >= obs_counts[k])
        pvals[k] = (extreme + 1) / (n_iter + 1)
    signif_matrix_hr[m_index, :] = pvals < 0.05

    pvals_suppress = np.zeros(n_nodes)
    for k in range(n_nodes):
        extreme_suppress = np.sum(mc_counts[:, k] <= obs_counts[k])
        pvals_suppress[k] = (extreme_suppress + 1) / (n_iter + 1)
    signif_suppress_matrix_hr[m_index, :] = pvals_suppress < 0.05

# ================================================================
# Ensemble means, biases, combined matrices
# ================================================================
ensemble_mean_hr = np.mean(enrichment_matrix_hr, axis=0)
ensemble_mean_lr = np.mean(enrichment_matrix_lr, axis=0)

era5_values = enrichment_grid.ravel()

bias_matrix_hr = enrichment_matrix_hr - era5_values
bias_matrix_lr = enrichment_matrix_lr - era5_values
bias_ensemble_hr = ensemble_mean_hr - era5_values
bias_ensemble_lr = ensemble_mean_lr - era5_values

era5_signif_flat = promoting_grid.ravel()
era5_suppress_signif_flat = suppressing_grid.ravel()

signif_lr_panel = np.vstack([signif_matrix_lr, np.zeros(n_nodes, dtype=bool)])
signif_hr_panel = np.vstack([signif_matrix_hr, np.zeros(n_nodes, dtype=bool)])
signif_suppress_lr_panel = np.vstack([signif_suppress_matrix_lr, np.zeros(n_nodes, dtype=bool)])
signif_suppress_hr_panel = np.vstack([signif_suppress_matrix_hr, np.zeros(n_nodes, dtype=bool)])

bias_lr_panel = np.vstack([bias_matrix_lr, bias_ensemble_lr])
bias_hr_panel = np.vstack([bias_matrix_hr, bias_ensemble_hr])

hr_panel_enrich = np.vstack([enrichment_matrix_hr, ensemble_mean_hr, era5_values])
lr_panel_enrich = np.vstack([enrichment_matrix_lr, ensemble_mean_lr, era5_values])

lr_labels = models + ['Ensemble Mean', 'ERA5']

era5_row = era5_values
era5_signif_row = era5_signif_flat
era5_suppress_signif_row = era5_suppress_signif_flat
row_labels = models + ['Ensemble Mean', 'ERA5']
n_rows_total = n_models + 2

# ================================================================
# Optional diagnostic: check whether specific low-ERA5-efficiency
# nodes show spurious model efficiency. Update the node list as
# needed for the current 3x5 SOM configuration.
# ================================================================
CHECK_NODES_1IDX = []  # e.g. [6, 11] if reproducing the earlier 20-node check
if CHECK_NODES_1IDX:
    rows = []
    for node_1idx in CHECK_NODES_1IDX:
        node_0idx = node_1idx - 1
        era5_val = era5_values[node_0idx]
        lr_vals = enrichment_matrix_lr[:, node_0idx]
        hr_vals = enrichment_matrix_hr[:, node_0idx]
        for m_idx, model_label in enumerate(models):
            rows.append({"node": node_1idx, "era5_efficiency": round(era5_val, 4),
                        "model": model_label, "resolution": "LR",
                        "model_efficiency": round(lr_vals[m_idx], 4),
                        "bias": round(lr_vals[m_idx] - era5_val, 4)})
            rows.append({"node": node_1idx, "era5_efficiency": round(era5_val, 4),
                        "model": model_label, "resolution": "HR",
                        "model_efficiency": round(hr_vals[m_idx], 4),
                        "bias": round(hr_vals[m_idx] - era5_val, 4)})
    pd.DataFrame(rows).to_csv("node_efficiency_check.csv", index=False)
    print("Saved: node_efficiency_check.csv")

# ================================================================
# Fig. 8: triangle heatmap, bold = significant promotion,
# italic = significant suppression
# ================================================================
if VAR == 'tos':
    max_bias = 2
else:
    max_bias = 1
levels_bias = np.linspace(-max_bias, max_bias, 101)
cmap_bias = sns.color_palette("RdBu_r", len(levels_bias) - 1, as_cmap=True)
norm_bias = mcolors.BoundaryNorm(levels_bias, cmap_bias.N, clip=False, extend='both')

levels_era5 = np.linspace(0, 5, 21)
cmap_era5 = sns.color_palette("YlOrRd", len(levels_era5) - 1, as_cmap=True)
norm_era5 = BoundaryNorm(levels_era5, cmap_era5.N, clip=False, extend='max')

fig = plt.figure(figsize=(19, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[10, 0.5, 0.5], hspace=0.2)
ax_main = fig.add_subplot(gs[0, 0])
cax_bias = fig.add_subplot(gs[1, 0])
cax_era5 = fig.add_subplot(gs[2, 0])

for i in range(n_models + 1):
    for j in range(n_nodes):
        val_lr = bias_lr_panel[i, j]
        ax_main.add_patch(create_triangle(j, i, 'lower', cmap_bias(norm_bias(val_lr))))
        val_hr = bias_hr_panel[i, j]
        ax_main.add_patch(create_triangle(j, i, 'upper', cmap_bias(norm_bias(val_hr))))

i_era5 = n_models + 1
for j in range(n_nodes):
    color_era5 = cmap_era5(norm_era5(era5_row[j]))
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
            fw = 'bold' if era5_signif_row[j] else 'normal'
            fs = 'italic' if era5_suppress_signif_row[j] else 'normal'
            ax_main.text(j, i, f"{era5_row[j]:.1f}", ha='center', va='center', fontsize=12,
                        color='black', fontweight=fw, fontstyle=fs)
        if i == n_models:
            val_lr = lr_panel_enrich[i, j]
            fw_lr = 'bold' if np.sum(signif_matrix_lr[:, j]) > len(models) / 2 else 'normal'
            fs_lr = 'italic' if np.sum(signif_suppress_matrix_lr[:, j]) > len(models) / 2 else 'normal'
            ax_main.text(j - 0.2, i - 0.15, f"{val_lr:.1f}", ha='center', va='center', fontsize=12,
                        color='black', fontweight=fw_lr, fontstyle=fs_lr)

            val_hr = hr_panel_enrich[i, j]
            fw_hr = 'bold' if np.sum(signif_matrix_hr[:, j]) > len(models) / 2 else 'normal'
            fs_hr = 'italic' if np.sum(signif_suppress_matrix_hr[:, j]) > len(models) / 2 else 'normal'
            ax_main.text(j + 0.2, i + 0.15, f"{val_hr:.1f}", ha='center', va='center', fontsize=12,
                        color='black', fontweight=fw_hr, fontstyle=fs_hr)

cbar_bias = fig.colorbar(plt.cm.ScalarMappable(norm=norm_bias, cmap=cmap_bias),
                         cax=cax_bias, orientation='horizontal')
cbar_bias.set_label("Bias", fontsize=16)
cbar_bias.ax.tick_params(labelsize=16)
cbar_era5 = fig.colorbar(plt.cm.ScalarMappable(norm=norm_era5, cmap=cmap_era5),
                         cax=cax_era5, orientation='horizontal')
cbar_era5.set_label("ERA5 regime HW-promoting efficiency (dimensionless ratio; 1 = climatological frequency)",
                    fontsize=16)
cbar_era5.ax.tick_params(labelsize=16)

plt.suptitle(
    "Bias of regime HW-promoting efficiency \n Low-res (\u25be) vs High-res (\u25b4)\n"
    "Bold = significantly promoting; Italic = significantly suppressing (p < 0.05)",
    fontsize=18, y=0.97)
plt.tight_layout()
plt.savefig(f'03_hw_promoting_bias_{VAR}_{ystart}_{yend}.png', dpi=300, bbox_inches='tight')

# ================================================================
# Whisker-style redesign of Fig. 8 (ensemble spread vs. ERA5)
# ================================================================
def plot_enrichment_spread(enrich_lr, enrich_hr, era5_vals, era5_promote_sig, era5_suppress_sig,
                            n_nodes, fname, n_cols=5, min_valid_models=3):
    """Grid figure, one panel per regime: full-ensemble LR/HR spread in
    HW-promoting efficiency against ERA5, with a neutral (efficiency=1)
    reference line and promotion/suppression significance symbols."""
    regimes_1idx = list(range(1, n_nodes + 1))
    n_rows_ = int(np.ceil(n_nodes / n_cols))
    fig, axs = plt.subplots(n_rows_, n_cols, figsize=(4.2 * n_cols, 4.4 * n_rows_), sharey=True)
    axs = np.atleast_2d(axs)

    X_LR, X_HR = -0.25, 0.25
    HALF_WIDTH = 0.07

    def plot_group(ax, x_center, vals, color):
        vals = vals[~np.isnan(vals)]
        if len(vals) < min_valid_models:
            ax.text(x_center, 1, "insufficient\nsample", ha="center", va="center",
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
        era5_val = era5_vals[node]

        mean_lr, range_lr = plot_group(ax, X_LR, enrich_lr[:, node], "#4C72B0")
        mean_hr, range_hr = plot_group(ax, X_HR, enrich_hr[:, node], "#DD8452")

        if not np.isnan(mean_lr):
            all_bias_lr.append(mean_lr - era5_val)
            all_range_lr.append(range_lr)
        if not np.isnan(mean_hr):
            all_bias_hr.append(mean_hr - era5_val)
            all_range_hr.append(range_hr)

        sig_symbol = ""
        if era5_promote_sig[node]:
            sig_symbol = " \u2605"
        elif era5_suppress_sig[node]:
            sig_symbol = " \u2020"

        ax.scatter([0], [era5_val], color="tab:red", marker="D", s=60, zorder=5,
                  edgecolor="black", linewidth=0.4)
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5, zorder=0)

        if not np.isnan(mean_lr):
            ax.plot([X_LR + HALF_WIDTH * 1.6, 0], [mean_lr, era5_val], color="#4C72B0",
                    linewidth=0.8, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)
        if not np.isnan(mean_hr):
            ax.plot([X_HR - HALF_WIDTH * 1.6, 0], [mean_hr, era5_val], color="#DD8452",
                    linewidth=0.8, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)

        ax.set_xticks([X_LR, X_HR])
        ax.set_xticklabels(["LR", "HR"], fontsize=14)
        ax.set_xlim(-0.42, 0.42)
        ax.set_title(f"Regime {regime_num}{sig_symbol}", fontsize=18)

        for label, rng, gap_val, y_off, color in [
            ("LR", range_lr, (mean_lr - era5_val) if not np.isnan(mean_lr) else np.nan, -0.08, "#4C72B0"),
            ("HR", range_hr, (mean_hr - era5_val) if not np.isnan(mean_hr) else np.nan, -0.17, "#DD8452"),
        ]:
            if not np.isnan(rng):
                verdict_color = "#1a7a3c" if rng >= abs(gap_val) else "#b3401f"
                ax.text(0.5, y_off, f"{label}: spread={rng:.2f}  |gap|={abs(gap_val):.2f}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=12, color=verdict_color)

    for idx in range(n_nodes, n_rows_ * n_cols):
        row, col = divmod(idx, n_cols)
        axs[row, col].axis("off")
    for row in range(n_rows_):
        axs[row, 0].set_ylabel("HW-promoting efficiency\n(1 = climatological frequency)", fontsize=15)

    legend_handles = [
        mlines.Line2D([], [], color="#4C72B0", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="LR models"),
        mlines.Line2D([], [], color="#DD8452", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="HR models"),
        mlines.Line2D([], [], color="gray", linewidth=2.5, label="Ensemble spread (min\u2013max)"),
        mlines.Line2D([], [], color="tab:red", marker="D", linestyle="None", markersize=8,
                      markeredgecolor="black", label="ERA5 (\u2605 promoting, \u2020 suppressing)"),
        mlines.Line2D([], [], color="gray", linewidth=1.0, linestyle=(0, (2, 1)),
                      label="Ensemble-mean\u2013ERA5 gap"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=16)

    mean_abs_bias_lr = np.mean(np.abs(all_bias_lr)) if all_bias_lr else np.nan
    mean_abs_bias_hr = np.mean(np.abs(all_bias_hr)) if all_bias_hr else np.nan
    mean_range_lr = np.mean(all_range_lr) if all_range_lr else np.nan
    mean_range_hr = np.mean(all_range_hr) if all_range_hr else np.nan
    summary_text = (
        f"Ensemble mean |bias|:  LR = {mean_abs_bias_lr:.2f}   HR = {mean_abs_bias_hr:.2f}"
        f"     |     Ensemble spread:  LR = {mean_range_lr:.2f}   HR = {mean_range_hr:.2f}"
    )
    fig.text(0.5, 1.025, summary_text, ha="center", va="center", fontsize=16,
             fontweight="bold", color="#333333")

    fig.suptitle("HW-promoting efficiency", fontsize=22, y=1.1)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")


plot_enrichment_spread(
    enrichment_matrix_lr, enrichment_matrix_hr, era5_values,
    era5_promote_sig=promoting_grid.ravel(),
    era5_suppress_sig=suppressing_grid.ravel(),
    n_nodes=som_grid_rows * som_grid_cols,
    fname=f"03b_hw_promoting_bias_{VAR}_{ystart}_{yend}_whisker.png"
)
