import os
import re
import glob
import subprocess
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import generic_filter
import geopandas as gpd

from utils import convert_jp2_folder_to_tif, process_tif_folder, copy_georef, filter_cloudy_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_from_tif(tif_path):
    match = re.search(r"(\d{8})T\d{6}", os.path.basename(tif_path))
    if not match:
        raise ValueError(f"Cannot parse date from: {tif_path}")
    return datetime.strptime(match.group(1), "%Y%m%d")


def _load_tif_stack(tif_paths, crop=None):
    arrays = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            data = src.read(1).astype(np.float32)
            if crop:
                x_min, y_min, x_size, y_size = crop
                data = data[y_min:y_min+y_size, x_min:x_min+x_size]
        arrays.append(data)
    return np.stack(arrays, axis=0)


def _build_mask_array(shape_path, ref_tif):
    gdf = gpd.read_file(shape_path)
    with rasterio.open(ref_tif) as src:
        gdf_proj = gdf.to_crs(src.crs)
        mask_arr = geometry_mask(
            [g for g in gdf_proj.geometry],
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width)
        )
    return mask_arr

def _nan_median_filter_nooverlap(arr, size):
    H, W = arr.shape
    out = np.full_like(arr, np.nan)
    for r in range(0, H - size + 1, size):
        for c in range(0, W - size + 1, size):
            window = arr[r:r+size, c:c+size]
            val = np.nanmedian(window)
            out[r:r+size, c:c+size] = val
    return out


# ---------------------------------------------------------------------------
# Mask extraction
# ---------------------------------------------------------------------------

def extract_glacier_mask_tif(shape_path, ref_tif, out_path):
    gdf = gpd.read_file(shape_path)
    with rasterio.open(ref_tif) as src:
        gdf_proj = gdf.to_crs(src.crs)
        mask_arr = geometry_mask(
            [g for g in gdf_proj.geometry],
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width)
        ).astype(np.uint8)
        profile = src.profile.copy()

    mask_arr = mask_arr * 255  # MicMac expects 0/255 not 0/1

    profile.update(dtype=rasterio.uint8, count=1, nodata=None)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask_arr[np.newaxis, :, :])

    print(f"Glacier mask saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_temporal_distribution(tif_paths, out_path=None):
    dates = sorted([_date_from_tif(p) for p in tif_paths])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.vlines(dates, 0, 1, color="steelblue", linewidth=1.5, alpha=0.7)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_title(f"Temporal distribution — {len(dates)} images")
    ax.set_xlim(dates[0], dates[-1])
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()


def plot_tsne(tif_paths, crop=None, perplexity=30, out_path=None):
    stack = _load_tif_stack(tif_paths, crop=crop)
    stack[stack == 0] = np.nan

    n = stack.shape[0]
    flat = stack.reshape(n, -1)

    col_mask = ~np.any(np.isnan(flat), axis=0)
    flat = flat[:, col_mask]

    flat = StandardScaler().fit_transform(flat)

    perplexity = min(perplexity, n - 1)
    embedding = TSNE(n_components=2, perplexity=perplexity,
                     random_state=42).fit_transform(flat)

    dates = [_date_from_tif(p) for p in tif_paths]
    timestamps = np.array([d.timestamp() for d in dates])

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=timestamps, cmap="plasma", s=40, alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Date")
    cbar.set_ticks(np.linspace(timestamps.min(), timestamps.max(), 5))
    cbar.set_ticklabels([datetime.fromtimestamp(t).strftime("%Y-%m")
                         for t in np.linspace(timestamps.min(), timestamps.max(), 5)])

    for i, d in enumerate(dates):
        ax.annotate(d.strftime("%m/%y"), (embedding[i, 0], embedding[i, 1]),
                    fontsize=6, alpha=0.6)

    ax.set_title(f"t-SNE of image stack (n={n})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Multi-pair MicMac correlation
# ---------------------------------------------------------------------------

def build_pairs(tif_paths, low_cycle=[2, 100], high_cycle=[400, 500]):
    sorted_paths = sorted(tif_paths, key=_date_from_tif)
    min_date = _date_from_tif(sorted_paths[0])
    max_date = _date_from_tif(sorted_paths[-1])
    pairs = []
    for i, j in itertools.combinations(range(len(sorted_paths)), 2):
        d1 = _date_from_tif(sorted_paths[i])
        d2 = _date_from_tif(sorted_paths[j])
        days = abs((d2 - d1).days)
        if low_cycle[0] <= days <= low_cycle[1] or high_cycle[0] <= days <= high_cycle[1]:
            pairs.append((sorted_paths[i], sorted_paths[j]))
    print(f"Found {len(pairs)} pairs between {min_date} and {max_date}")
    return pairs


def run_micmac_correlation(img1, img2, out_dir, sz_w, reg, inc, gamma_cor,
                           cor_min, zoom_init, mask_path=None):
    
    work_dir    = os.path.dirname(os.path.abspath(img1))
    abs_out_dir = os.path.abspath(out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)

    cmd = ["mm3d", "MM2DPosSism",
           os.path.basename(img1),
           os.path.basename(img2),
           f"SzW={sz_w}", f"Reg={reg}", f"Inc={inc}",
           f"GamaCor={gamma_cor}", f"CorMin={cor_min}", f"ZoomInit={zoom_init}"]

    if mask_path is not None:
        mask_dst = os.path.join(work_dir, os.path.basename(mask_path))
        if os.path.abspath(mask_path) != os.path.abspath(mask_dst):
            shutil.copy2(mask_path, mask_dst)
        cmd.append(f"Masq={os.path.basename(mask_dst)}")

    subprocess.run(cmd, check=True, cwd=work_dir)

    mec_dir = os.path.join(work_dir, "MEC")
    for f in glob.glob(os.path.join(mec_dir, "*")):
        shutil.move(f, abs_out_dir)
    shutil.rmtree(mec_dir, ignore_errors=True)

    for tif in glob.glob(os.path.join(abs_out_dir, "*.tif")):
        copy_georef(img1, tif)

    return out_dir


def run_all_pairs(pairs, corr_base_dir, sz_w, reg, gamma_cor, cor_min, zoom_init,
                  mask_path=None):
    pair_results = []
    for img1, img2 in pairs:

        dt1 = _date_from_tif(img1)
        dt2 = _date_from_tif(img2)
        delta_time = abs((dt2 - dt1).days)

        d1 = _date_from_tif(img1).strftime("%Y%m%d")
        d2 = _date_from_tif(img2).strftime("%Y%m%d")
        out_dir = os.path.join(corr_base_dir, f"{d1}_{d2}")

        if delta_time < 30:
            inc = 1
        elif 30 <= delta_time < 60:
            inc = 2
        else:
            inc = 3
        try:
            run_micmac_correlation(img1, img2, out_dir,
                                   sz_w=sz_w, reg=reg, inc=inc,
                                   gamma_cor=gamma_cor, cor_min=cor_min, zoom_init=zoom_init,
                                   mask_path=mask_path)
            pair_results.append({"img1": img1, "img2": img2, "corr_dir": out_dir})
            print(f"  Correlated: {d1} → {d2}")
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {d1} → {d2}: {e}")
    return pair_results


def find_displacement_files(corr_dir, pxl_precision):
    def _find(pattern):
        matches = glob.glob(os.path.join(corr_dir, pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern} in {corr_dir}")
        return matches[0]

    dx = _find(f"*Px1_Num{pxl_precision}*.tif")
    dy = _find(f"*Px2_Num{pxl_precision}*.tif")
    return dx, dy


def find_correlation_file(corr_dir):
    matches = glob.glob(os.path.join(corr_dir, "Correl_LeChantier_Num_4.tif"))
    if not matches:
        raise FileNotFoundError(f"No correlation file found in {corr_dir}")
    return matches[0]

def _sigmoid_corr_weight(corr, center=0.5, steepness=10.0):
    return 1.0 / (1.0 + np.exp(-steepness * (corr - center)))


def compute_mean_velocity(corr_base_dir, pxl_precision, shape_path,
                          correl_threshold=0.3, min_correlated_fraction=0.0,
                          sigmoid_center=0.5, sigmoid_steepness=10.0,
                          pixel_size_m=10,
                          start_date=None, end_date=None):
    dx_stack     = []
    dy_stack     = []
    weights      = []
    profile      = None
    glacier_mask = None
    good_folders = []

    pair_dirs = sorted(glob.glob(os.path.join(corr_base_dir, "*_*")))
    if not pair_dirs:
        raise FileNotFoundError(f"No pair directories found in {corr_base_dir}")
    

    nb_pairs_in_folder = len(pair_dirs)
    nb_pairs      = 0
    skipped_pairs = 0

    for pair_dir in pair_dirs:
        folder_name = os.path.basename(pair_dir)
        match = re.match(r"(\d{8})_(\d{8})", folder_name)
        if not match:
            continue

        d1 = datetime.strptime(match.group(1), "%Y%m%d")
        d2 = datetime.strptime(match.group(2), "%Y%m%d")

        if start_date is not None and d1 < datetime.strptime(start_date, "%Y%m%d"):
            continue
        if end_date is not None and d1 > datetime.strptime(end_date, "%Y%m%d"):
            continue
        
        nb_pairs += 1

        days = abs((d2 - d1).days)
        scale_factor = pixel_size_m * 365.25 / days

        try:
            dx_path, dy_path = find_displacement_files(pair_dir, pxl_precision)
        except FileNotFoundError:
            print(f"  Skipping {folder_name} — displacement files not found")
            skipped_pairs += 1
            continue

        try:
            corr_path = find_correlation_file(pair_dir)
            with rasterio.open(corr_path) as src:
                corr = src.read(1).astype(float)
            corr_min_val = corr.min()
            corr = (corr - corr_min_val) / (255 - corr_min_val)
        except FileNotFoundError:
            print(f"  WARNING: no correlation map for {folder_name}, skipping")
            skipped_pairs += 1
            continue

        with rasterio.open(dx_path) as src:
            dx = src.read(1).astype(float)
            profile = src.profile.copy()
        with rasterio.open(dy_path) as src:
            dy = src.read(1).astype(float)

        if glacier_mask is None:
            glacier_mask = _build_mask_array(shape_path, dx_path)

        # Optional: skip pairs where too few glacier pixels are well-correlated
        if min_correlated_fraction > 0.0:
            glacier_pixels        = glacier_mask.sum()
            correlated_on_glacier = (glacier_mask & (corr >= correl_threshold)).sum()
            corr_fraction = correlated_on_glacier / glacier_pixels if glacier_pixels > 0 else 0.0
            print(f"  {folder_name}: {corr_fraction*100:.1f}% glacier pixels above threshold")
            if corr_fraction < min_correlated_fraction:
                print(f"  Skipping — below threshold ({min_correlated_fraction*100:.0f}%)")
                skipped_pairs += 1
                continue

        # Mask low-correlation pixels
        hard_mask = corr < correl_threshold

        dx[hard_mask] = np.nan
        dy[hard_mask] = np.nan
        # dx[dx == 0]  = np.nan
        # dy[dy == 0]  = np.nan

        dx = dx * scale_factor
        dy = dy * scale_factor

        e = 0.1 * pixel_size_m * 365.25 / days
        w_time = 1.0 / e**2

        w_corr = _sigmoid_corr_weight(corr, center=sigmoid_center, steepness=sigmoid_steepness)

        w = w_time*w_corr

        dx_stack.append(dx)
        dy_stack.append(dy)
        weights.append(w)
        good_folders.append(pair_dir)

    if not dx_stack:
        raise RuntimeError("No valid displacement files found.")

    print(f"Nb pairs in folder: {nb_pairs_in_folder}")
    print(f"\nUsed {len(dx_stack)} / {nb_pairs} pairs "
          f"({skipped_pairs} skipped)")

    dx_stack = np.stack(dx_stack, axis=0)
    dy_stack = np.stack(dy_stack, axis=0)
    weights  = np.stack(weights, axis=0)

    dx_median = np.nanmedian(dx_stack, axis=0)
    dy_median = np.nanmedian(dy_stack, axis=0)
    dx_std    = np.nanstd(dx_stack, axis=0)
    dy_std    = np.nanstd(dy_stack, axis=0)

    dx_stack[np.abs(dx_stack - dx_median) > 1.0 * dx_std] = np.nan
    dy_stack[np.abs(dy_stack - dy_median) > 1.0 * dy_std] = np.nan

    w_masked_dx = np.where(~np.isnan(dx_stack), weights, 0.0)
    w_masked_dy = np.where(~np.isnan(dy_stack), weights, 0.0)

    w_sum_dx = w_masked_dx.sum(axis=0)
    w_sum_dy = w_masked_dy.sum(axis=0)

    dx_w_mean = np.where(w_sum_dx > 0,
                         (w_masked_dx * np.nan_to_num(dx_stack)).sum(axis=0) / w_sum_dx,
                         np.nan)
    dy_w_mean = np.where(w_sum_dy > 0,
                         (w_masked_dy * np.nan_to_num(dy_stack)).sum(axis=0) / w_sum_dy,
                         np.nan)

    dx_wstd = np.sqrt(np.where(w_sum_dx > 0,
                               (w_masked_dx * np.nan_to_num(
                                   (dx_stack - dx_w_mean)**2)).sum(axis=0) / w_sum_dx,
                               np.nan))
    dy_wstd = np.sqrt(np.where(w_sum_dy > 0,
                               (w_masked_dy * np.nan_to_num(
                                   (dy_stack - dy_w_mean)**2)).sum(axis=0) / w_sum_dy,
                               np.nan))

    return dx_w_mean, dy_w_mean, dx_wstd, dy_wstd, profile, good_folders


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_velocity(mean_dx, mean_dy, std_w_dx, std_w_dy, shape_path, ref_tif,
                  step=10, scale=100,
                  median_filter_size=None,
                  out_path=None, out_tif_dir=None):

    valid_mask = _build_mask_array(shape_path, ref_tif)
    mean_dx[~valid_mask]  = np.nan
    mean_dy[~valid_mask]  = np.nan
    std_w_dx[~valid_mask] = np.nan
    std_w_dy[~valid_mask] = np.nan

    if median_filter_size is not None and median_filter_size > 1:
        mean_dx = _nan_median_filter_nooverlap(mean_dx, median_filter_size)
        mean_dy = _nan_median_filter_nooverlap(mean_dy, median_filter_size)

    mean_magnitude = np.sqrt(mean_dx**2 + mean_dy**2)
    std_magnitude  = np.sqrt(std_w_dx**2 + std_w_dy**2)

    H, W = mean_magnitude.shape
    if H != std_magnitude.shape[0] or W != std_magnitude.shape[1]:
        raise ValueError("Shape of mean and std arrays differ")

    if out_tif_dir:
        os.makedirs(out_tif_dir, exist_ok=True)
        with rasterio.open(ref_tif) as src:
            profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

        for name, arr in [("mean_velocity.tif",  mean_magnitude),
                          ("std_velocity.tif",   std_magnitude),
                          ("mean_dx.tif",        mean_dx),
                          ("mean_dy.tif",        mean_dy)]:
            with rasterio.open(os.path.join(out_tif_dir, name), "w", **profile) as dst:
                dst.write(arr.astype(np.float32)[np.newaxis, :, :])

        print(f"Saved velocity TIFs to {out_tif_dir}")

    y, x = np.mgrid[0:H:step, 0:W:step]
    u_down = mean_dx[::step, ::step]
    v_down = -mean_dy[::step, ::step]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im = axes[0].imshow(mean_magnitude, cmap="inferno", origin="upper")
    axes[0].quiver(x, y, u_down, v_down, color="white", scale=scale)
    axes[0].set_title("Weighted mean velocity (m/yr)")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    plt.colorbar(im, ax=axes[0], label="m/yr")

    im2 = axes[1].imshow(std_magnitude, cmap="viridis", origin="upper")
    axes[1].set_title("Weighted mean standard deviation (m/yr)")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=axes[1], label="m/yr")

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

        JP2_FOLDER    = f"Sentinel_2_{y}-01-01_{y}-12-31_32TLR_Cloud_cover80"
        TIF_FOLDER    = f"Sentinel_2_{y}-01-01_{y}-12-31_32TLR_Cloud_cover80_tif"
        SHAPE         = "RGI2000-v7.0-G-11_central_europe/RGI2000-v7.0-G-11_central_europe.shp"
        CLEAN_IMG_TIF = f"{TIF_FOLDER}_Cleaned"
        OUT_DIR       = f"{TIF_FOLDER}_img_processed"
        CORR_DIR      = f"Non_sub_{TIF_FOLDER}_pairs"
        VIZ_DIR       = f"Non_sub_{TIF_FOLDER}_visualisation"
        MASK_TIF      = "glacier_mask.tif"

        sz_w          = 3
        reg           = 0.5
        gamma_cor     = 1
        zoom_init     = 4
        cor_min       = 0.6
        pxl_precision = 5
        correl_threshold = 0.70

        os.makedirs(VIZ_DIR, exist_ok=True)
        os.makedirs(CLEAN_IMG_TIF, exist_ok=True)

        # --- 1. Convert JP2 to TIF ---
        # convert_jp2_folder_to_tif(JP2_FOLDER, TIF_FOLDER)

        # # --- 2. Filter cloudy images ---
        # tif_paths = sorted(glob.glob(os.path.join(TIF_FOLDER, "*.tif")))
        # clean_paths, _ = filter_cloudy_images(
        #     tif_paths, scl_folder=os.path.join(JP2_FOLDER, "SCL"),
        #     x_min=3772, y_min=1272, x_size=512, y_size=512,
        #     max_cloud_fraction=0.05
        # )
        # plot_temporal_distribution(clean_paths,
        #     out_path=os.path.join(VIZ_DIR, "temporal_distribution.png"))
        # for p in clean_paths:
        #     shutil.copy2(p, os.path.join(CLEAN_IMG_TIF, os.path.basename(p)))

        # # --- 3. Process TIFs (mask, crop, Wallis) ---
        # results = process_tif_folder(
        #     tif_folder=CLEAN_IMG_TIF, out_dir=OUT_DIR, shape_path=SHAPE,
        #     x_min=3772, y_min=1272, x_size=512, y_size=512,
        #     do_wallis=True, do_census=False, do_wallis_census=False, do_masked=False,
        #     keep_intermediate=False
        # )

        wallis_paths = sorted(glob.glob(os.path.join(OUT_DIR, "wallis", "*.tif")))

        # --- 4. Extract glacier mask TIF (once, from first Wallis image) ---
        # extract_glacier_mask_tif(SHAPE, wallis_paths[0], MASK_TIF)

        # --- 5. t-SNE visualisation ---
        # plot_tsne(wallis_paths, perplexity=30,
        #           out_path=os.path.join(VIZ_DIR, "tsne_wallis.png"))

        # --- 6. Build pairs and run correlations ---
        # pairs = build_pairs(wallis_paths)
        # run_all_pairs(pairs, CORR_DIR, sz_w=sz_w, reg=reg, zoom_init=zoom_init,
        #               gamma_cor=gamma_cor, cor_min=cor_min, mask_path=MASK_TIF)

        # # --- 7. Compute mean velocity ---
        dx_mean, dy_mean, dx_wstd, dy_wstd, profile, good_folders = compute_mean_velocity(
            CORR_DIR, pxl_precision, shape_path=SHAPE,
            correl_threshold=correl_threshold,
            min_correlated_fraction=0.0,
            sigmoid_center=0.7, sigmoid_steepness=10.0,
            pixel_size_m=10,
            start_date="20200101", end_date="20200630"
        )

        # # --- 8. Plot and save ---
        plot_velocity(
            dx_mean, dy_mean, dx_wstd, dy_wstd, SHAPE,
            ref_tif=wallis_paths[0],
            step=30, scale=800,
            median_filter_size=None,
            out_path=os.path.join(VIZ_DIR, f"mean_velocity_Num{pxl_precision}.png"),
            out_tif_dir=os.path.join(VIZ_DIR, "velocity_tifs")
        )