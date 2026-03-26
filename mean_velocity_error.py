import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from pyproj import Transformer
from sklearn.metrics import mean_squared_error, mean_absolute_error


Y_OFFSET = 2000000
STAKE_CRS = "EPSG:27572"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    data[data == src.nodata] = np.nan if src.nodata is not None else data
    return data, profile, transform, crs


def _build_glacier_mask(shape_path, ref_tif):
    gdf = gpd.read_file(shape_path)
    with rasterio.open(ref_tif) as src:
        gdf_proj = gdf.to_crs(src.crs)
        mask = geometry_mask(
            [g for g in gdf_proj.geometry],
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width)
        )
    return mask


def _reproject_trend_to_match(trend_path, ref_tif_path):
    """Reproject and resample trend TIF to match the reference TIF grid."""
    from rasterio.warp import reproject, Resampling
    with rasterio.open(ref_tif_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    with rasterio.open(trend_path) as src:
        trend_data = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=trend_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )

    trend_data = trend_data.astype(float)
    trend_data[trend_data == 0] = np.nan
    return trend_data


# ---------------------------------------------------------------------------
# Glacier-wide error vs reference velocity field
# ---------------------------------------------------------------------------

def compute_glacier_error(mean_velocity_tif, trend_tif, shape_path):
    mean_vel, profile, transform, crs = _read_tif(mean_velocity_tif)
    glacier_mask = _build_glacier_mask(shape_path, mean_velocity_tif)

    trend = _reproject_trend_to_match(trend_tif, mean_velocity_tif)

    valid = glacier_mask & ~np.isnan(mean_vel) & ~np.isnan(trend)
    if valid.sum() == 0:
        raise RuntimeError("No valid overlapping pixels between velocity map and trend.")

    v_pred = mean_vel[valid]
    v_ref  = trend[valid]

    rmse = np.sqrt(mean_squared_error(v_ref, v_pred))
    mae  = mean_absolute_error(v_ref, v_pred)
    bias = np.mean(v_pred - v_ref)

    print(f"=== Glacier-wide error ===")
    print(f"  Valid pixels : {valid.sum()}")
    print(f"  RMSE         : {rmse:.2f} m/yr")
    print(f"  MAE          : {mae:.2f} m/yr")
    print(f"  Bias         : {bias:.2f} m/yr")

    diff = np.full_like(mean_vel, np.nan)
    diff[valid] = v_pred - v_ref

    return rmse, mae, bias, diff


def plot_glacier_error(diff, ref_tif, shape_path, out_path=None):
    glacier_mask = _build_glacier_mask(shape_path, ref_tif)
    diff[~glacier_mask] = np.nan

    vmax = np.nanpercentile(np.abs(diff), 95)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(diff, cmap="RdBu_r", origin="upper", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Velocity difference (m/yr)")
    plt.title("Mean velocity − Reference trend")
    plt.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()


def save_error_tif(diff, ref_tif, out_path):
    with rasterio.open(ref_tif) as src:
        profile = src.profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(diff.astype(np.float32)[np.newaxis, :, :])
    print(f"Error TIF saved: {out_path}")


# ---------------------------------------------------------------------------
# Stake-level validation
# ---------------------------------------------------------------------------

def sample_velocity_at_stakes(mean_velocity_tif, csv_path,
                               window_size=3, y_offset=Y_OFFSET):
    df = pd.read_csv(csv_path)
    df = df[df["profile_name"] == "Tacul"].copy()

    vel, profile, transform, crs = _read_tif(mean_velocity_tif)

    transformer = Transformer.from_crs(STAKE_CRS, crs, always_xy=True)
    x_proj, y_proj = transformer.transform(
        df["x_lambert2e_start"].values,
        df["y_lambert2e_start"].values + y_offset
    )

    rows, cols = rasterio.transform.rowcol(transform, x_proj, y_proj)
    rows = np.array(rows)
    cols = np.array(cols)

    H, W = vel.shape
    half = window_size // 2

    sampled = []
    for r, c in zip(rows, cols):
        r_min = max(0, r - half)
        r_max = min(H, r + half + 1)
        c_min = max(0, c - half)
        c_max = min(W, c + half + 1)
        window = vel[r_min:r_max, c_min:c_max]
        sampled.append(np.nanmean(window))

    df["velocity_sampled"] = sampled
    return df


def compute_stake_error(mean_velocity_tif, csv_path,
                        window_size=3, y_offset=Y_OFFSET):
    df = sample_velocity_at_stakes(mean_velocity_tif, csv_path,
                                   window_size, y_offset)

    valid = ~np.isnan(df["velocity_sampled"]) & ~np.isnan(df["annual_speed"])
    df_valid = df[valid].copy()

    if len(df_valid) == 0:
        raise RuntimeError("No valid stake comparisons — check coordinates or velocity map coverage.")

    rmse = np.sqrt(mean_squared_error(df_valid["annual_speed"],
                                      df_valid["velocity_sampled"]))
    mae  = mean_absolute_error(df_valid["annual_speed"],
                               df_valid["velocity_sampled"])
    bias = np.mean(df_valid["velocity_sampled"] - df_valid["annual_speed"])

    print(f"\n=== Stake-level error ===")
    print(f"  Valid stakes : {len(df_valid)} / {len(df)}")
    print(f"  RMSE         : {rmse:.2f} m/yr")
    print(f"  MAE          : {mae:.2f} m/yr")
    print(f"  Bias         : {bias:.2f} m/yr")
    print()
    print(df_valid[["stake_number", "annual_speed", "velocity_sampled"]].to_string(index=False))

    return df_valid, rmse, mae, bias


def plot_stake_error(df_valid, year, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: sampled vs measured
    vmax = max(df_valid["annual_speed"].max(), df_valid["velocity_sampled"].max()) * 1.1
    axes[0].scatter(df_valid["annual_speed"], df_valid["velocity_sampled"],
                    s=60, color="steelblue", zorder=5)
    axes[0].plot([0, vmax], [0, vmax], "r--", linewidth=1, label="1:1 line")
    for _, row in df_valid.iterrows():
        axes[0].annotate(str(int(row["stake_number"])),
                         (row["annual_speed"], row["velocity_sampled"]),
                         fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[0].set_xlabel("Field measurement (m/yr)")
    axes[0].set_ylabel("Sampled velocity (m/yr)")
    axes[0].set_title(f"Sampled vs measured velocity {year}")
    axes[0].legend()

    # Bar: error per stake
    errors = df_valid["velocity_sampled"] - df_valid["annual_speed"]
    colors = ["red" if e > 0 else "steelblue" for e in errors]
    axes[1].bar(df_valid["stake_number"].astype(str), errors, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Stake number")
    axes[1].set_ylabel("Error (m/yr)")
    axes[1].set_title(f"Per-stake error (sampled − measured) {year}")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":


    # YEAR = [2020, 2019, 2018, 2017]
    YEAR = [2020]
    for y in YEAR:

        MEAN_VEL_TIF  = f"Non_sub_Sentinel_2_{y}-01-01_{y}-12-31_32TLR_Cloud_cover80_tif_visualisation/velocity_tifs/mean_velocity.tif"
        TREND_TIF     = f"dataverse_files/v{y}_{y+1}_ALPES_ALL_ANNUALv2016-2021.tiff"
        SHAPE         = "RGI2000-v7.0-G-11_central_europe/RGI2000-v7.0-G-11_central_europe.shp"
        STAKE_CSV     = f"GlacioClim_csv/MDG_Tacul_Langue_speed_{y}.csv"
        OUT_DIR       = f"Non_sub_validation_{y}"

        os.makedirs(OUT_DIR, exist_ok=True)

        # --- Glacier-wide error vs reference trend ---
        rmse, mae, bias, diff = compute_glacier_error(MEAN_VEL_TIF, TREND_TIF, SHAPE)
        plot_glacier_error(diff, MEAN_VEL_TIF, SHAPE,
                        out_path=os.path.join(OUT_DIR, "glacier_error_map.png"))
        save_error_tif(diff, MEAN_VEL_TIF,
                    out_path=os.path.join(OUT_DIR, "glacier_error.tif"))

        # --- Stake-level validation ---
        df_valid, rmse_s, mae_s, bias_s = compute_stake_error(
            MEAN_VEL_TIF, STAKE_CSV, window_size=3, y_offset=Y_OFFSET
        )
        plot_stake_error(df_valid, year=y,
                        out_path=os.path.join(OUT_DIR, f"stake_error_{y}.png"))