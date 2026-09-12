"""
05_model_composites.py

Corresponds to original notebook section "MODEL COMPOSITES" -> "Make an
example of HW composite for one model".

NOTE ON CLEANUP: the original notebook contained a "#### old maps"
section directly after this one, re-implementing the same two plots
with the same variable names and nearly identical logic. That section
is dead/superseded and has been dropped entirely.

Requires: 00_setup_and_functions.py (import * or run first)
"""

from importlib import import_module
setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

m_index = 0          # CMCC-CM2, by default -- change to inspect a different model
all_days = False      # True = all-summer-day composite, False = HW-day composite

if VAR == 'tos':
    levels = np.linspace(-2, 2, 21)
else:
    levels = np.linspace(-200, 200, 21)
cmap = sns.color_palette("RdBu_r", len(levels) - 1, as_cmap=True)
norm = mcolors.BoundaryNorm(levels, cmap.N, clip=False, extend='both')

model = models[m_index]
realization = realizations[m_index]


def plot_example_composite(mod_name, resolution_label):
    if all_days:
        file_in = (f'som_projected_all_days_{mod_name}_{realization}_{VAR}_{ystart}_{yend}_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')
    else:
        file_in = (f'som_projected_HW_days_{mod_name}_{realization}_{VAR}_{ystart}_{yend}_'
                   f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}')

    ds_hw = xr.open_dataset(f'/work/cmcc/ls21622/HighResMIP/{mod_name}/{realization}/som/{file_in}.nc')

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
            freq = ds_hw["hit_counts"].isel(som_row=i, som_col=j).values / ntime * 100
            n_hits = ds_hw["hit_counts"].isel(som_row=i, som_col=j).values

            axs[i, j].text(0.01, 0.13, f"{freq:.0f}% (n={n_hits})",
                          transform=axs[i, j].transAxes, fontsize=10, va='top', ha='left',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

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

    day_type = "all days" if all_days else "HW days"
    plt.suptitle(f"{mod_name} {day_type} {VAR_TIT} anomaly composite", fontsize=20)
    if VAR == 'zg':
        cbar.set_label(f"{VAR_TIT} anomaly (m)", fontsize=14)
    else:
        cbar.set_label(f"{VAR_TIT} anomaly (\u00b0C)", fontsize=14)
    fig.supylabel('SOM rows', fontsize=18)
    fig.supxlabel('SOM columns', fontsize=18)

    day_tag = "all_days" if all_days else "HW_days"
    plt.savefig(
        f'05_som_projected_{day_tag}_{mod_name}_{VAR}_{ystart}_{yend}_'
        f'{som_grid_rows}_{som_grid_cols}_iterations_{iterations}_learning_rate_{learning_rate}_sigma_{sigma}_{VAR}.png',
        dpi=300)


plot_example_composite(models_lr[m_index], "LR")
plot_example_composite(models_hr[m_index], "HR")
