"""
06_regime_trends.py

Corresponds to original notebook section "REGIME TRENDS" (Fig. 10, plus
the internal-variability and full-ensemble-spread companion figures
developed in response to Reviewer #1).

This section did NOT contain the old/dead-code duplication pattern seen
elsewhere in the notebook -- the trend computation here is a single,
clean implementation.

Requires: 00_setup_and_functions.py (import * or run first)
"""

from importlib import import_module
setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

import matplotlib.lines as mlines
from scipy.stats import linregress
import pymannkendall as mk

# ================================================================
# Settings
# ================================================================
all_days = True        # False = HW-day regime trends, True = all-summer-day regime trends
alpha = 0.05
extend_era5 = False
test = 'OLS'             # 'OLS' or 'MK' -- see Reviewer 2, Sec 2.4/2.5 discussion
                          # re: trend-estimator sensitivity for intermittent series
n_year_threshold = 10

n_models = len(models)
n_nodes = som_grid_rows * som_grid_cols

trend_matrix_hr = np.zeros((n_models, n_nodes))
trend_matrix_lr = np.zeros((n_models, n_nodes))
signif_matrix_hr = np.zeros((n_models, n_nodes), dtype=bool)
signif_matrix_lr = np.zeros((n_models, n_nodes), dtype=bool)

# ================================================================
# ERA5 trends
# ================================================================
if all_days == 0:
    ds_era5 = xr.open_mfdataset(
        f'{som_path}/som_projected_HW_days_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
        f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
    if extend_era5:
        ds_era5 = xr.open_mfdataset(
            f'{som_path}/som_projected_HW_days_ERA5_{VAR}_1975_2024_{som_grid_rows}_{som_grid_cols}_'
            f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
else:
    ds_era5 = xr.open_mfdataset(
        f'{som_path}/som_ERA5_{VAR}_1975_{yend}_{som_grid_rows}_{som_grid_cols}_'
        f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')
    if extend_era5:
        ds_era5 = xr.open_mfdataset(
            f'{som_path}/som_ERA5_{VAR}_1975_2024_{som_grid_rows}_{som_grid_cols}_'
            f'iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}.nc')

node = ds_era5["bmu_row"] * som_grid_cols + ds_era5["bmu_col"]
node = node.values
years = ds_era5["time"].dt.year.values
unique_years = np.unique(years)
n_years = len(unique_years)

freq_year_era5 = np.zeros((n_years, n_nodes))
for i, yr in enumerate(unique_years):
    mask = years == yr
    node_year = node[mask]
    for k in range(n_nodes):
        freq_year_era5[i, k] = np.sum(node_year == k)

trend_era5 = np.zeros(n_nodes)
pval_era5 = np.zeros(n_nodes)
for k in range(n_nodes):
    if np.sum(freq_year_era5[:, k] != 0) < n_year_threshold:
        trend_era5[k] = np.nan
        pval_era5[k] = np.nan
    else:
        if test == 'MK':
            mk_result = mk.original_test(freq_year_era5[:, k])
            trend_era5[k] = mk_result.slope
            pval_era5[k] = mk_result.p
        else:
            slope, _, _, p, _ = linregress(unique_years, freq_year_era5[:, k])
            trend_era5[k] = slope
            pval_era5[k] = p

era5_signif = pval_era5 < alpha

# ================================================================
# Per-model LR/HR trends
# ================================================================
for m_index in range(n_models):
    model = models[m_index]
    realization = realizations[m_index]
    mod_lr = models_lr[m_index]
    mod_hr = models_hr[m_index]

    # ----------------------
    # LOW RESOLUTION
    # ----------------------
    if all_days == 0:
        file_in = (f'som_projected_HW_days_{mod_lr}_{realization}_{VAR}_{ystart}_{yend}_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
    else:
        file_in = (f'som_projected_all_days_{mod_lr}_{realization}_{VAR}_{ystart}_{yend}_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
    ds = xr.open_dataset(f'/work/cmcc/ls21622/HighResMIP/{mod_lr}/{realization}/som/{file_in}.nc')

    node = ds["bmu_row"] * som_grid_cols + ds["bmu_col"]
    node = node.values
    years = ds["time"].dt.year.values
    unique_years = np.unique(years)
    n_years = len(unique_years)

    freq_year = np.zeros((n_years, n_nodes))
    for i, yr in enumerate(unique_years):
        mask = years == yr
        node_year = node[mask]
        for k in range(n_nodes):
            freq_year[i, k] = np.sum(node_year == k)

    for k in range(n_nodes):
        if np.sum(freq_year[:, k] != 0) < n_year_threshold:
            trend_matrix_lr[m_index, k] = np.nan
            signif_matrix_lr[m_index, k] = False
        else:
            if test == 'MK':
                mk_result = mk.original_test(freq_year[:, k])
                trend_matrix_lr[m_index, k] = mk_result.slope
                signif_matrix_lr[m_index, k] = mk_result.p < alpha
            else:
                slope, _, _, p, _ = linregress(unique_years, freq_year[:, k])
                trend_matrix_lr[m_index, k] = slope
                signif_matrix_lr[m_index, k] = p < alpha

    # ----------------------
    # HIGH RESOLUTION
    # ----------------------
    if all_days == 0:
        file_in = (f'som_projected_HW_days_{mod_hr}_{realization}_{VAR}_{ystart}_{yend}_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
    else:
        file_in = (f'som_projected_all_days_{mod_hr}_{realization}_{VAR}_{ystart}_{yend}_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
    ds = xr.open_dataset(f'/work/cmcc/ls21622/HighResMIP/{mod_hr}/{realization}/som/{file_in}.nc')

    node = ds["bmu_row"] * som_grid_cols + ds["bmu_col"]
    node = node.values
    years = ds["time"].dt.year.values
    unique_years = np.unique(years)
    n_years = len(unique_years)

    freq_year = np.zeros((n_years, n_nodes))
    for i, yr in enumerate(unique_years):
        mask = years == yr
        node_year = node[mask]
        for k in range(n_nodes):
            freq_year[i, k] = np.sum(node_year == k)

    for k in range(n_nodes):
        if np.sum(freq_year[:, k] != 0) < n_year_threshold:
            trend_matrix_hr[m_index, k] = np.nan
            signif_matrix_hr[m_index, k] = False
        else:
            if test == 'MK':
                mk_result = mk.original_test(freq_year[:, k])
                trend_matrix_hr[m_index, k] = mk_result.slope
                signif_matrix_hr[m_index, k] = mk_result.p < alpha
            else:
                slope, _, _, p, _ = linregress(unique_years, freq_year[:, k])
                trend_matrix_hr[m_index, k] = slope
                signif_matrix_hr[m_index, k] = p < alpha

ensemble_mean_lr = np.nanmean(trend_matrix_lr, axis=0)
ensemble_mean_hr = np.nanmean(trend_matrix_hr, axis=0)

lr_panel = np.vstack([trend_matrix_lr, ensemble_mean_lr, trend_era5])
hr_panel = np.vstack([trend_matrix_hr, ensemble_mean_hr, trend_era5])

signif_combined_lr = np.vstack([signif_matrix_lr, np.zeros(n_nodes, dtype=bool), era5_signif])
signif_combined_hr = np.vstack([signif_matrix_hr, np.zeros(n_nodes, dtype=bool), era5_signif])

# ================================================================
# Internal-variability spread (multi-realization subset: EC-Earth3P,
# ECMWF-IFS) -- addresses Reviewer #1, major comment 2
# ================================================================
def member_spread_stats(trend_matrix, member_rows, regimes_1idx, era5_trend):
    rows = []
    for regime_num in regimes_1idx:
        node_i = regime_num - 1
        member_trends = trend_matrix[member_rows, node_i]
        member_trends = member_trends[~np.isnan(member_trends)]
        if len(member_trends) == 0:
            continue
        ens_mean = np.nanmean(member_trends)
        era5_val = era5_trend[node_i]
        gap = ens_mean - era5_val
        spread_range = np.nanmax(member_trends) - np.nanmin(member_trends)
        rows.append({"regime": regime_num, "member_trends": member_trends.tolist(),
                    "ensemble_mean": ens_mean, "era5_trend": era5_val,
                    "bias": gap, "spread_range": spread_range,
                    "spread_exceeds_bias": bool(spread_range >= abs(gap))})
    return pd.DataFrame(rows)


def lr_hr_spread_comparison(trend_matrix_lr, trend_matrix_hr, member_rows, regimes_1idx, model_label):
    rows = []
    for regime_num in regimes_1idx:
        node_i = regime_num - 1
        lr_vals = trend_matrix_lr[member_rows, node_i]
        hr_vals = trend_matrix_hr[member_rows, node_i]
        lr_vals = lr_vals[~np.isnan(lr_vals)]
        hr_vals = hr_vals[~np.isnan(hr_vals)]
        lr_range = np.nanmax(lr_vals) - np.nanmin(lr_vals) if len(lr_vals) else np.nan
        hr_range = np.nanmax(hr_vals) - np.nanmin(hr_vals) if len(hr_vals) else np.nan
        rows.append({"model": model_label, "regime": regime_num,
                    "LR_range": lr_range, "HR_range": hr_range})
    return pd.DataFrame(rows)


def inter_model_spread(trend_matrix, regimes_1idx, model_groups):
    """Inter-model (not inter-member) spread across the full ensemble."""
    rows = []
    for regime_num in regimes_1idx:
        node_i = regime_num - 1
        vals = trend_matrix[:, node_i]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        rows.append({"regime": regime_num, "range": np.nanmax(vals) - np.nanmin(vals),
                    "std": np.nanstd(vals, ddof=1) if len(vals) > 1 else np.nan})
    return pd.DataFrame(rows)


def regimes_with_full_member_coverage(panel_specs, n_nodes):
    """
    Determine which regimes have a valid (non-NaN) trend value for
    EVERY member of EVERY model in panel_specs, in both LR and HR --
    i.e., the regimes where the inter-member spread can be computed
    on the full available sample (all 3 realizations) for every model
    shown, rather than a degraded subset.
    """
    valid_regimes = []
    for node in range(n_nodes):
        ok = True
        for row_label, trend_lr_, trend_hr_, member_rows in panel_specs:
            vals_lr = trend_lr_[member_rows, node]
            vals_hr = trend_hr_[member_rows, node]
            if np.any(np.isnan(vals_lr)) or np.any(np.isnan(vals_hr)):
                ok = False
                break
        if ok:
            valid_regimes.append(node + 1)
    return valid_regimes


# --------------------------------------------------------------
# Single, unified regime selection -- computed ONCE here, and reused
# for both the diagnostic print statements below AND the final
# plot_member_spread_grid call, so the two can never disagree with
# each other (replaces the previously hardcoded, Z500-specific
# regimes_of_interest = [1, 11, 14]).
# --------------------------------------------------------------
panel_specs = [
    ("EC-Earth3P", trend_matrix_lr, trend_matrix_hr, ec_earth_idx),
    ("ECMWF-IFS", trend_matrix_lr, trend_matrix_hr, ecmwf_idx),
]
regimes_for_spread = regimes_with_full_member_coverage(panel_specs, n_nodes)


# ================================================================
# Figure 11: internal-variability spread, LR/HR side-by-side within
# each panel, one row per multi-realization model. Always standalone
# (this is a dedicated SM figure, never combined with anything else).
# ================================================================
def plot_member_spread_grid(panel_specs, regimes_1idx, era5_trend, era5_signif, fname):
    if len(regimes_1idx) == 0:
        print(f"[plot_member_spread_grid] No regimes to plot -- skipping '{fname}'.")
        return None

    n_rows_ = len(panel_specs)
    n_cols_ = len(regimes_1idx)
    fig, axs = plt.subplots(n_rows_, n_cols_, figsize=(5.2 * n_cols_, 4.8 * n_rows_), sharey="row",
                             gridspec_kw={"hspace": 0.6, "wspace": 0.3})
    if n_rows_ == 1:
        axs = axs.reshape(1, -1)

    X_LR, X_HR = -0.25, 0.25
    HALF_WIDTH = 0.07

    def plot_group(ax, x_center, vals, color, label):
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            ax.text(x_center, 0, "insufficient\nsample", ha="center", va="center",
                    fontsize=10, color="gray", style="italic")
            return np.nan, np.nan
        ens_mean = np.nanmean(vals)
        v_min, v_max = np.nanmin(vals), np.nanmax(vals)
        v_range = v_max - v_min
        x_jitter = (x_center + np.linspace(-0.04, 0.04, len(vals)) if len(vals) > 1 else [x_center])
        ax.plot([x_center, x_center], [v_min, v_max], color="gray", linewidth=2.2,
                solid_capstyle="butt", zorder=1)
        ax.plot([x_center - HALF_WIDTH, x_center + HALF_WIDTH], [v_min, v_min], color="gray", linewidth=2.2, zorder=1)
        ax.plot([x_center - HALF_WIDTH, x_center + HALF_WIDTH], [v_max, v_max], color="gray", linewidth=2.2, zorder=1)
        ax.plot([x_center - HALF_WIDTH * 1.6, x_center + HALF_WIDTH * 1.6], [ens_mean, ens_mean],
                color=color, linewidth=2.2, zorder=3)
        ax.scatter(x_jitter, vals, color=color, s=38, zorder=4, edgecolor="white", linewidth=0.5)
        return ens_mean, v_range

    for i, (row_label, trend_lr_, trend_hr_, member_rows) in enumerate(panel_specs):
        for j, regime_num in enumerate(regimes_1idx):
            ax = axs[i, j]
            node_i = regime_num - 1
            era5_val = era5_trend[node_i]

            vals_lr = trend_lr_[member_rows, node_i]
            vals_hr = trend_hr_[member_rows, node_i]

            mean_lr, range_lr = plot_group(ax, X_LR, vals_lr, "#4C72B0", "LR")
            mean_hr, range_hr = plot_group(ax, X_HR, vals_hr, "#DD8452", "HR")

            gap_lr = mean_lr - era5_val
            gap_hr = mean_hr - era5_val

            ax.scatter([0], [era5_val], color="tab:red", marker="D", s=60, zorder=5,
                      edgecolor="black", linewidth=0.4)

            ax.plot([X_LR + HALF_WIDTH * 1.6, 0], [mean_lr, era5_val], color="#4C72B0",
                    linewidth=0.9, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)
            ax.plot([X_HR - HALF_WIDTH * 1.6, 0], [mean_hr, era5_val], color="#DD8452",
                    linewidth=0.9, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)

            ax.axhline(0, color="black", linewidth=0.4, zorder=0)
            ax.set_xticks([X_LR, X_HR])
            ax.set_xticklabels(["LR", "HR"], fontsize=14)
            ax.set_xlim(-0.42, 0.42)

            if i == 0:
                sig_symbol = " *" if era5_signif[node_i] else ""
                ax.set_title(f"Regime {regime_num}{sig_symbol}", fontsize=16)
            if j == 0:
                ax.set_ylabel(f"{row_label}\nFrequency change / yr", fontsize=16)

            for k, (label, rng, gap, y_off, color) in enumerate([
                ("LR", range_lr, gap_lr, -0.1, "#4C72B0"),
                ("HR", range_hr, gap_hr, -0.19, "#DD8452"),
            ]):
                if np.isnan(rng):
                    continue  # plot_group already printed "insufficient sample" on the panel
                verdict_color = "#1a7a3c" if rng >= abs(gap) else "#b3401f"
                ax.text(0.5, y_off, f"{label}: spread={rng:.3f}  |bias|={abs(gap):.3f}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=14, color=verdict_color)

    legend_handles = [
        mlines.Line2D([], [], color="#4C72B0", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="LR realizations"),
        mlines.Line2D([], [], color="#DD8452", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="HR realizations"),
        mlines.Line2D([], [], color="gray", linewidth=2.5, label="Inter-member spread (min\u2013max range)"),
        mlines.Line2D([], [], color="tab:red", marker="D", linestyle="None", markersize=8,
                      markeredgecolor="black", label="ERA5 reference trend"),
        mlines.Line2D([], [], color="gray", linewidth=1.2, linestyle=(0, (2, 1)),
                      label="Ensemble-mean\u2013ERA5 gap"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, .98), frameon=False, fontsize=16)
    fig.suptitle("Inter-member trend spread across models and resolutions", fontsize=22, y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    return axs


# Standalone Supplementary Materials figure: internal-variability spread
# (EC-Earth3P, ECMWF-IFS only), addressing Reviewer #1's specific
# question about distinguishing the observed trend from sampling noise.
plot_member_spread_grid(panel_specs, regimes_for_spread, trend_era5, era5_signif,
                        f"SM_internal_variability_combined_{test}.png")


# ================================================================
# Full-ensemble spread (all 8 models). Always standalone (this is a
# dedicated companion figure to the heatmap, never combined into one
# shared figure with it).
# ================================================================
def plot_trend_spread(trend_lr, trend_hr, era5_trend, era5_signif, n_nodes, fname,
                       n_cols=5, min_valid_models=3,
                       suptitle_text="Regime-specific HW-day frequency trend"):
    n_models_total = trend_lr.shape[0]
    all_regimes_1idx = list(range(1, n_nodes + 1))

    kept_regimes, dropped_regimes = [], []
    for regime_num in all_regimes_1idx:
        node_i = regime_num - 1
        n_valid_lr = np.sum(~np.isnan(trend_lr[:, node_i]))
        n_valid_hr = np.sum(~np.isnan(trend_hr[:, node_i]))
        all_models_valid = (n_valid_lr == n_models_total) and (n_valid_hr == n_models_total)
        era5_valid = not np.isnan(era5_trend[node_i])  # ERA5 itself has enough
                                                        # non-zero years (n_year_threshold)
        if all_models_valid and era5_valid:
            kept_regimes.append(regime_num)
        else:
            dropped_regimes.append((regime_num, n_valid_lr, n_valid_hr, era5_valid))

    if dropped_regimes:
        print(f"Excluded regimes (not all {n_models_total} models valid in LR and/or HR, "
              f"or ERA5 itself lacks a sufficient number of years/events):")
        for regime_num, n_lr, n_hr, era5_ok in dropped_regimes:
            print(f"  Regime {regime_num}: LR n={n_lr}/{n_models_total}, "
                  f"HR n={n_hr}/{n_models_total}, ERA5 sufficient={era5_ok}")
    regimes_1idx = kept_regimes

    if len(regimes_1idx) == 0:
        print(f"[plot_trend_spread] No regimes to plot -- skipping '{fname}'.")
        return None

    n_panels = len(regimes_1idx)
    n_rows_ = int(np.ceil(n_panels / n_cols))
    fig, axs = plt.subplots(n_rows_, n_cols, figsize=(5.2 * n_cols, 6 * n_rows_), sharey=True)
    axs = np.atleast_2d(axs)

    X_LR, X_HR = -0.25, 0.25
    HALF_WIDTH = 0.07

    def plot_group(ax, x_center, vals, color):
        vals = vals[~np.isnan(vals)]
        if len(vals) < min_valid_models:
            ax.text(x_center, 0, "insufficient\nsample", ha="center", va="center",
                    fontsize=12, color="gray", style="italic")
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
        node_i = regime_num - 1
        era5_val = era5_trend[node_i]

        mean_lr, range_lr = plot_group(ax, X_LR, trend_lr[:, node_i], "#4C72B0")
        mean_hr, range_hr = plot_group(ax, X_HR, trend_hr[:, node_i], "#DD8452")

        if not np.isnan(mean_lr):
            all_bias_lr.append(mean_lr - era5_val)
            all_range_lr.append(range_lr)
        if not np.isnan(mean_hr):
            all_bias_hr.append(mean_hr - era5_val)
            all_range_hr.append(range_hr)

        ax.scatter([0], [era5_val], color="tab:red", marker="D", s=60, zorder=5,
                  edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5, zorder=0)

        if not np.isnan(mean_lr):
            ax.plot([X_LR + HALF_WIDTH * 1.6, 0], [mean_lr, era5_val], color="#4C72B0",
                    linewidth=0.8, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)
        if not np.isnan(mean_hr):
            ax.plot([X_HR - HALF_WIDTH * 1.6, 0], [mean_hr, era5_val], color="#DD8452",
                    linewidth=0.8, linestyle=(0, (2, 1)), alpha=0.6, zorder=2)

        ax.set_xticks([X_LR, X_HR])
        ax.set_xticklabels(["LR", "HR"], fontsize=14)
        ax.set_xlim(-0.42, 0.42)

        is_significant = era5_signif[node_i]
        title_str = f"Regime {regime_num}" + (" *" if is_significant else "")
        ax.set_title(title_str, fontsize=18)

        for label, rng, gap_val, y_off, color in [
            ("LR", range_lr, (mean_lr - era5_val) if not np.isnan(mean_lr) else np.nan, -0.08, "#4C72B0"),
            ("HR", range_hr, (mean_hr - era5_val) if not np.isnan(mean_hr) else np.nan, -0.17, "#DD8452"),
        ]:
            if not np.isnan(rng):
                verdict_color = "#1a7a3c" if rng >= abs(gap_val) else "#b3401f"
                ax.text(0.5, y_off, f"{label}: spread={rng:.3f}  |bias|={abs(gap_val):.3f}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=16, color=verdict_color)

    for idx in range(n_panels, n_rows_ * n_cols):
        row, col = divmod(idx, n_cols)
        axs[row, col].axis("off")
    for row in range(n_rows_):
        axs[row, 0].set_ylabel("HW-day frequency trend\n(days / year)", fontsize=16)

    legend_handles = [
        mlines.Line2D([], [], color="#4C72B0", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="LR models"),
        mlines.Line2D([], [], color="#DD8452", marker="o", linestyle="None", markersize=8,
                      markeredgecolor="white", label="HR models"),
        mlines.Line2D([], [], color="gray", linewidth=2.5, label="Ensemble spread (min\u2013max)"),
        mlines.Line2D([], [], color="tab:red", marker="D", linestyle="None", markersize=8,
                      markeredgecolor="black", label="ERA5 reference"),
        mlines.Line2D([], [], color="gray", linewidth=1.0, linestyle=(0, (2, 1)),
                      label="Ensemble-mean\u2013ERA5 bias"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=16)

    mean_abs_bias_lr = np.mean(np.abs(all_bias_lr)) if all_bias_lr else np.nan
    mean_abs_bias_hr = np.mean(np.abs(all_bias_hr)) if all_bias_hr else np.nan
    mean_range_lr = np.mean(all_range_lr) if all_range_lr else np.nan
    mean_range_hr = np.mean(all_range_hr) if all_range_hr else np.nan
    summary_text = (
        f"Ensemble mean |bias| vs. ERA5:  LR = {mean_abs_bias_lr:.3f}   HR = {mean_abs_bias_hr:.3f}"
        f"     |     Ensemble spread:  LR = {mean_range_lr:.3f}   HR = {mean_range_hr:.3f}"
    )
    fig.text(0.5, 1.06, summary_text, ha="center", va="center", fontsize=16,
             fontweight="bold", color="#333333")
    fig.suptitle(suptitle_text, fontsize=22, y=1.16)

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    return axs


# ================================================================
# Trend triangle heatmap -- Fig. 10a
# ================================================================
def build_trend_heatmap(fig):
    """Build the trend triangle heatmap into the given Figure,
    returning the main axis."""
    levels = np.linspace(-0.2, 0.2, 11)
    cmap = sns.color_palette("RdBu_r", len(levels) - 1, as_cmap=True)
    norm = mcolors.BoundaryNorm(levels, cmap.N, clip=False, extend='both')

    trend_lr_panel = np.vstack([trend_matrix_lr, ensemble_mean_lr])
    trend_hr_panel = np.vstack([trend_matrix_hr, ensemble_mean_hr])
    signif_lr_panel = np.vstack([signif_matrix_lr, np.zeros(n_nodes, dtype=bool)])
    signif_hr_panel = np.vstack([signif_matrix_hr, np.zeros(n_nodes, dtype=bool)])
    era5_row = trend_era5
    era5_signif_row = era5_signif
    row_labels = models + ['Ensemble Mean', 'ERA5']
    n_rows_total = n_models + 2

    gs = fig.add_gridspec(2, 1, height_ratios=[10, 0.5], hspace=0.15)
    ax_main = fig.add_subplot(gs[0, 0])
    cax_trend = fig.add_subplot(gs[1, 0])

    for i in range(n_models + 1):
        for j in range(n_nodes):
            val_lr = trend_lr_panel[i, j]
            color_lr = cmap(norm(val_lr)) if not np.isnan(val_lr) else 'lightgray'
            ax_main.add_patch(create_triangle(j, i, 'lower', color_lr))
            val_hr = trend_hr_panel[i, j]
            color_hr = cmap(norm(val_hr)) if not np.isnan(val_hr) else 'lightgray'
            ax_main.add_patch(create_triangle(j, i, 'upper', color_hr))

    i_era5 = n_models + 1
    for j in range(n_nodes):
        val_era5 = era5_row[j]
        color_era5 = cmap(norm(val_era5)) if not np.isnan(val_era5) else 'lightgray'
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
                    fw = 'bold' if era5_signif_row[j] else 'normal'
                    ax_main.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=14,
                                color='black', fontweight=fw)
            else:
                val_lr = trend_lr_panel[i, j]
                if not np.isnan(val_lr) and i == n_rows_total - 2:
                    fw_lr = ('bold' if (i == n_models and np.sum(signif_matrix_lr[:, j]) > len(models) / 2)
                            else ('bold' if signif_lr_panel[i, j] else 'normal'))
                    ax_main.text(j - 0.2, i - 0.15, f"{val_lr:.2f}", ha='center', va='center',
                                fontsize=14, color='black', fontweight=fw_lr)
                val_hr = trend_hr_panel[i, j]
                if not np.isnan(val_hr) and i == n_rows_total - 2:
                    fw_hr = ('bold' if (i == n_models and np.sum(signif_matrix_hr[:, j]) > len(models) / 2)
                            else ('bold' if signif_hr_panel[i, j] else 'normal'))
                    ax_main.text(j + 0.2, i + 0.15, f"{val_hr:.2f}", ha='center', va='center',
                                fontsize=14, color='black', fontweight=fw_hr)

    cbar_trend = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                              cax=cax_trend, orientation='horizontal')
    cbar_trend.set_label("HW days yr\u207b\u00b9 (trend)", fontsize=16)
    cbar_trend.ax.tick_params(labelsize=16)

    return ax_main


# ================================================================
# Fig. 10a: trend triangle heatmap (standalone figure)
# ================================================================
fig = plt.figure(figsize=(18, 12))
ax_main = build_trend_heatmap(fig)

if all_days == 0:
    fig.suptitle("Trend in mean HW days frequency per regime (1975-2014)\n"
                "Low-res (\u25be) vs High-res (\u25b4)", fontsize=22, y=0.95)
    fname_heatmap = f'trend_frequency_HW_days_{VAR}_{ystart}_{yend}.png'
else:
    fig.suptitle("Trend regimes frequency, all days (1975-2014)\n"
                "Low-res (\u25be) vs High-res (\u25b4)", fontsize=22, y=0.95)
    fname_heatmap = f'trend_frequency_all_days_regimes_{VAR}_{ystart}_{yend}_{test}.png'

plt.tight_layout()
plt.savefig(fname_heatmap, dpi=300, bbox_inches='tight')


# ================================================================
# Fig. 10b: full-ensemble spread, standalone figure (all 8 models,
# same set shown in the heatmap above)
# ================================================================
if all_days == 0:
    fname_spread = f'trend_frequency_HW_days_full_ensemble_spread_{VAR}_{ystart}_{yend}.png'
    spread_title = "Ensemble spread vs. ERA5 (1975-2014)"
else:
    fname_spread = f'trend_frequency_all_days_full_ensemble_spread_{VAR}_{ystart}_{yend}_{test}.png'
    spread_title = "Ensemble spread vs. ERA5 (1975-2014)"

plot_trend_spread(trend_matrix_lr, trend_matrix_hr, trend_era5, era5_signif,
                  n_nodes=som_grid_rows * som_grid_cols, fname=fname_spread,
                  suptitle_text=spread_title)
