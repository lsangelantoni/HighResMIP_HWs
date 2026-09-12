"""
08_cmcc_evap_fraction_diagnostic.py

Preliminary diagnostic for Reviewer 2, Sec 3.3(2): does CMCC's
evaporative-fraction dry bias intensify at high resolution as an
AMPLIFICATION of the same spatial pattern (consistent with a fixed,
untuned CLM4.5 land-surface parameterization; Scoccimarro et al. 2022),
or as a qualitatively different pattern?

No daily soil moisture is available for CMCC (only monthly means, not
usable for a HW-day-specific diagnostic), so this uses evaporative
fraction (EF = hfls / (hfls + hfss)) during HW days instead, which IS
available daily.

Requires: 00_setup_and_functions.py (import * or run first)
"""

from importlib import import_module
setup = import_module("00_setup_and_functions")
globals().update(vars(setup))

CMCC_INDEX = models.index("CMCC-CM2_r1")
mod_lr = models_lr[CMCC_INDEX]
mod_hr = models_hr[CMCC_INDEX]
realization = realizations[CMCC_INDEX]
grid = grids[CMCC_INDEX]

min_lon, max_lon = -10, 15
min_lat, max_lat = 42, 58


def load_ef(hfls_file, hfss_file, hw_mask_file):
    """Load hfls/hfss, restrict to HW days, return mean evaporative
    fraction field over the domain."""
    ds_hfls = xr.open_dataset(hfls_file).sortby('lat')
    ds_hfss = xr.open_dataset(hfss_file).sortby('lat')
    # Normalize the model's own time coordinate to midnight, matching
    # ds_hw_mask below -- without this, .where(sub_hw == 1) silently
    # returns all-NaN if the model's native timestamps carry a
    # non-midnight time-of-day component (e.g. 12:00:00), since the
    # time coordinates then never align.
    ds_hfls["time"] = pd.to_datetime(ds_hfls.time.astype("datetime64[ns]")).normalize()
    ds_hfss["time"] = pd.to_datetime(ds_hfss.time.astype("datetime64[ns]")).normalize()
    if ds_hfls.lon.min() >= 0:
        ds_hfls = shift_lon(ds_hfls)
        ds_hfss = shift_lon(ds_hfss)
    ds_hfls = crop_dom(ds_hfls, min_lon, max_lon, min_lat, max_lat, lon_name='lon', lat_name='lat')
    ds_hfss = crop_dom(ds_hfss, min_lon, max_lon, min_lat, max_lat, lon_name='lon', lat_name='lat')
    ds_hfls = ds_hfls.sel(time=ds_hfls.time.dt.month.isin([5, 6, 7, 8, 9]))
    ds_hfss = ds_hfss.sel(time=ds_hfss.time.dt.month.isin([5, 6, 7, 8, 9]))

    ds_hw_mask = xr.open_mfdataset(hw_mask_file)
    ds_hw_mask["time"] = pd.to_datetime(ds_hw_mask.time.astype("datetime64[ns]")).normalize()
    ds_hw_mask = ds_hw_mask.sel(time=ds_hw_mask.time.dt.month.isin([5, 6, 7, 8, 9]))

    hfls_hw = ds_hfls.hfls.where(ds_hw_mask.sub_hw == 1)
    hfss_hw = ds_hfss.hfss.where(ds_hw_mask.sub_hw == 1)
    ef = hfls_hw / (hfls_hw + hfss_hw)
    return ef.reduce(np.nanmean, dim="time")


# --- ERA5 ---
ef_era5 = load_ef(
    "/data/cmcc/ls21622/ERA5/hfls/postprocessed/ERA5_hfls_day_1975-2024_AMJJASO.nc",
    "/data/cmcc/ls21622/ERA5/hfss/postprocessed/ERA5_hfss_day_1975-2024_AMJJASO.nc",
    "/work/cmcc/ls21622/HighResMIP/ERA5/ERA5_sub_heatwave_day_*_CWE.nc",
)

# --- CMCC LR and HR ---
ef_cmcc = {}
for res, mod_name in [("lr", mod_lr), ("hr", mod_hr)]:
    hfls_file = (f"/data/cmcc/am35323/HighResMIP/{mod_name}/{realization}/hfls/1975_2024/"
                 f"hfls_day_{mod_name}_historical_{realization}_{grid}_1975-2024_AMJJASO.nc")
    hfss_file = (f"/data/cmcc/am35323/HighResMIP/{mod_name}/{realization}/hfss/1975_2024/"
                 f"hfss_day_{mod_name}_historical_{realization}_{grid}_1975-2024_AMJJASO.nc")
    hw_mask_file = f"/work/cmcc/ls21622/HighResMIP/{mod_name}/{realization}/hw_metrics/*sub_heatwave_day_*_CWE.nc"
    ef_cmcc[res] = load_ef(hfls_file, hfss_file, hw_mask_file)

# --- Regrid CMCC to ERA5 grid, compute bias ---
ef_lr_regrid = regrid(ef_cmcc["lr"].to_dataset(name="ef"), ef_era5.to_dataset(name="ef"), method).ef
ef_hr_regrid = regrid(ef_cmcc["hr"].to_dataset(name="ef"), ef_era5.to_dataset(name="ef"), method).ef

bias_lr = ef_lr_regrid.values - ef_era5.values
bias_hr = ef_hr_regrid.values - ef_era5.values

valid = np.isfinite(bias_lr) & np.isfinite(bias_hr)
bias_lr_flat = bias_lr[valid]
bias_hr_flat = bias_hr[valid]

# --- Diagnostics ---
spatial_corr = np.corrcoef(bias_lr_flat, bias_hr_flat)[0, 1]
mean_bias_lr = np.nanmean(bias_lr)
mean_bias_hr = np.nanmean(bias_hr)

DRY_THRESHOLD = -0.1
frac_dry_lr = np.mean(bias_lr_flat < DRY_THRESHOLD) * 100
frac_dry_hr = np.mean(bias_hr_flat < DRY_THRESHOLD) * 100

max_dry_lr = np.nanmin(bias_lr)
max_dry_hr = np.nanmin(bias_hr)

print("=== CMCC evaporative fraction bias: LR vs HR spatial pattern ===")
print(f"Spatial correlation (LR bias vs HR bias pattern): r = {spatial_corr:.3f}")
print(f"Domain-mean bias:            LR = {mean_bias_lr:+.3f}   HR = {mean_bias_hr:+.3f}")
print(f"Area with bias < {DRY_THRESHOLD} (%): LR = {frac_dry_lr:.1f}%   HR = {frac_dry_hr:.1f}%")
print(f"Most extreme (driest) bias:  LR = {max_dry_lr:+.3f}   HR = {max_dry_hr:+.3f}")
print()
print("High spatial correlation + larger |mean bias|/area/extreme at HR")
print("  --> supports amplification of the same pattern (fixed, untuned CLM4.5)")
print("Low spatial correlation --> suggests a qualitatively different bias at HR")

# ================================================================
# Physical evaporative fraction maps: LR and HR, native grids (not
# regridded), so any resolution-related spatial detail is preserved
# rather than smoothed away onto ERA5's coarser grid.
# ================================================================
levels_phys = np.linspace(0, 1, 11)
cmap_phys = sns.color_palette("YlOrBr", len(levels_phys) - 1, as_cmap=True)
norm_phys = mcolors.BoundaryNorm(levels_phys, cmap_phys.N, clip=False, extend='neither')

fig, axs = plt.subplots(1, 2, figsize=(14, 6),
                         subplot_kw={"projection": ccrs.PlateCarree()})

for ax, ef_da, title in [(axs[0], ef_cmcc["lr"], "LR"), (axs[1], ef_cmcc["hr"], "HR")]:
    im_phys = ax.pcolormesh(ef_da.lon, ef_da.lat, ef_da, cmap=cmap_phys, norm=norm_phys,
                            shading="auto", transform=ccrs.PlateCarree(), zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black",
                    linewidth=0.6, zorder=2)
    ax.set_title(f"CMCC {title} (native grid)", fontsize=14)

cbar_phys = fig.colorbar(im_phys, ax=axs, orientation="horizontal", fraction=0.06, pad=0.08)
cbar_phys.set_label("Evaporative fraction", fontsize=13)

fig.suptitle("CMCC evaporative fraction during HW days: LR vs. HR (physical values)",
             fontsize=16, y=1.02)
plt.savefig("cmcc_evap_fraction_physical_maps.png", dpi=300, bbox_inches="tight")
plt.show()

# ================================================================
# Bias maps: LR and HR evaporative fraction bias vs. ERA5, side by
# side, same color scale so the two are directly comparable.
# ================================================================
bias_lr_da = ef_lr_regrid - ef_era5
bias_hr_da = ef_hr_regrid - ef_era5

levels = np.linspace(-0.3, 0.3, 13)
cmap = sns.color_palette("RdBu_r", len(levels) - 1, as_cmap=True)
norm = mcolors.BoundaryNorm(levels, cmap.N, clip=False, extend='both')

fig, axs = plt.subplots(1, 2, figsize=(14, 6),
                         subplot_kw={"projection": ccrs.PlateCarree()})

for ax, bias_da, title in [(axs[0], bias_lr_da, "LR"), (axs[1], bias_hr_da, "HR")]:
    im = ax.pcolormesh(bias_da.lon, bias_da.lat, bias_da, cmap=cmap, norm=norm,
                        shading="auto", transform=ccrs.PlateCarree(), zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black",
                    linewidth=0.6, zorder=2)
    ax.set_title(f"CMCC {title} \u2212 ERA5", fontsize=14)

cbar = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.06, pad=0.08)
cbar.set_label("Evaporative fraction bias (CMCC \u2212 ERA5)", fontsize=13)

fig.suptitle("CMCC evaporative fraction bias during HW days: LR vs. HR", fontsize=16, y=1.02)
plt.savefig(f"cmcc_evap_fraction_bias_maps.png", dpi=300, bbox_inches="tight")
plt.show()
