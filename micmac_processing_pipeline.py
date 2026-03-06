import os
import glob
import subprocess
import shutil
import numpy as np
import rasterio
import re
from rasterio.mask import mask
from rasterio.features import geometry_mask
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal
from datetime import datetime


# ---------------------------------------------------------------------------
# SAFE folder, looks for specified band at given resolution, converts jp2 to tif, mask the image then crop it to given region + Wallis filter or Census transform
# ---------------------------------------------------------------------------

def date_from_safe(safe_path):
    match = re.search(r"_(\d{8})T\d{6}_", os.path.basename(safe_path))
    if not match:
        raise ValueError(f"Cannot parse date from SAFE filename: {safe_path}")
    return datetime.strptime(match.group(1), "%Y%m%d")


def find_band_in_safe(safe_dir, band, resolution="R10m"):
    pattern = os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA", resolution, f"*{band}*.jp2")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Band {band} not found in {safe_dir} at {resolution}")
    return matches[0]


def jp2_to_tif(jp2_path, tif_path):
    gdal.Translate(tif_path, jp2_path, format="GTiff", outputType=gdal.GDT_UInt16)


def apply_mask(tif_path, shape_path, out_path):
    gdf = gpd.read_file(shape_path)
    with rasterio.open(tif_path) as src:
        gdf_proj = gdf.to_crs(src.crs)
        geoms = [geom for geom in gdf_proj.geometry]
        out_image, out_transform = mask(src, geoms, crop=False, nodata=0)
        out_meta = src.meta.copy()
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_image)


def crop_tif(tif_path, out_path, x_min, y_min, x_size, y_size):
    gdal.Translate(out_path, tif_path, srcWin=[x_min, y_min, x_size, y_size])


def wallis_filter(tif_path, out_path):
    subprocess.run(
        ["mm3d", "Nikrup",
         f"* 256 erfcc / - (=I {tif_path}) (=M ( moy @I 20 4)) "
         "(sqrt max 0.001 (- (moy square @I 20 4) (square @M)))",
         out_path],
        check=True
    )


def census_transform(tif_path, out_path, radius=3):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    from scipy.ndimage import uniform_filter
    census = np.zeros_like(data, dtype=np.uint16)
    bit = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(data, dy, axis=0), dx, axis=1)
            census |= (data > shifted).astype(np.uint16) << (bit % 16)
            bit += 1

    profile.update(dtype=rasterio.uint16)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(census[np.newaxis, :, :])


def copy_georef(ref_path, target_path):
    ref = gdal.Open(ref_path)
    target = gdal.Open(target_path, gdal.GA_Update)
    target.SetGeoTransform(ref.GetGeoTransform())
    target.SetProjection(ref.GetProjection())
    ref = None
    target = None


def process_safe_pair(safe1, safe2, shape_path, out_dir,
                      band="B02", resolution="R10m",
                      x_min=3772, y_min=1272, x_size=512, y_size=512):
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for safe in (safe1, safe2):
        jp2 = find_band_in_safe(safe, band, resolution)
        base = os.path.splitext(os.path.basename(jp2))[0]

        tif_raw     = os.path.join(out_dir, f"{base}.tif")
        tif_masked  = os.path.join(out_dir, f"{base}_masked.tif")
        tif_cropped = os.path.join(out_dir, f"{base}_cropped.tif")
        tif_wallis  = os.path.join(out_dir, f"{base}_cropped_lisse.tif")
        tif_census  = os.path.join(out_dir, f"{base}_cropped_census.tif")
        tif_wallis_census  = os.path.join(out_dir, f"{base}_cropped_lisse_census.tif")


        jp2_to_tif(jp2, tif_raw)
        apply_mask(tif_raw, shape_path, tif_masked)
        crop_tif(tif_masked, tif_cropped, x_min, y_min, x_size, y_size)
        wallis_filter(tif_cropped, tif_wallis)
        copy_georef(tif_cropped, tif_wallis)
        census_transform(tif_cropped, tif_census)
        copy_georef(tif_cropped, tif_census)
        census_transform(tif_wallis, tif_wallis_census)
        copy_georef(tif_cropped, tif_wallis_census)


        results[safe] = {"cropped": tif_cropped, "wallis": tif_wallis, "census": tif_census, "wallis_census": tif_wallis_census}
        print(f"Processed: {base}")

    return results


# ---------------------------------------------------------------------------
# MicMac correlation
# ---------------------------------------------------------------------------

def run_micmac_correlation(img1, img2, out_dir, sz_w, reg, inc, gamma_cor, cor_min):
    work_dir = os.path.dirname(os.path.abspath(img1))
    abs_out_dir = os.path.abspath(out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)

    subprocess.run(
        ["mm3d", "MM2DPosSism",
         os.path.basename(img1),
         os.path.basename(img2),
         f"SzW={sz_w}", f"Reg={reg}", f"Inc={inc}", f"GamaCor={gamma_cor}", f"CorMin={cor_min}"],
        check=True,
        cwd=work_dir
    )

    mec_dir = os.path.join(work_dir, "MEC")
    for f in glob.glob(os.path.join(mec_dir, "*")):
        shutil.move(f, abs_out_dir)
    shutil.rmtree(mec_dir, ignore_errors=True)

    for tif in glob.glob(os.path.join(abs_out_dir, "*.tif")):
        copy_georef(img1, tif)

    return out_dir


def find_displacement_files(corr_dir, pxl_precision):
    def _find(pattern):
        matches = glob.glob(os.path.join(corr_dir, pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern} in {corr_dir}")
        return matches[0]

    dx = _find(f"*Px1_Num{pxl_precision}*")
    dy = _find(f"*Px2_Num{pxl_precision}*")
    return dx, dy


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _read_band(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data


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


def plot_displacement(dx_path, dy_path, shape_path, safe1, safe2,
                      pixel_size_m=10, step=10, scale=100, title_suffix="",
                      out_path=None):
    dx = _read_band(dx_path)
    dy = _read_band(dy_path)

    valid_mask = _build_mask_array(shape_path, dx_path)
    dx[~valid_mask] = np.nan
    dy[~valid_mask] = np.nan

    date1 = date_from_safe(safe1)
    date2 = date_from_safe(safe2)
    days = abs((date2 - date1).days)
    scale_factor = pixel_size_m * 365.25 / days
    magnitude_myr = np.sqrt(dx**2 + dy**2) * scale_factor

    H, W = magnitude_myr.shape
    y, x = np.mgrid[0:H:step, 0:W:step]
    u_down = dx[::step, ::step]
    v_down = -dy[::step, ::step]

    title = f"Displacement  |  {date1.date()} - {date2.date()}  ({days} days)"
    if title_suffix:
        title += f"  [{title_suffix}]"

    plt.figure(figsize=(10, 8))
    im = plt.imshow(magnitude_myr, cmap="inferno", origin="upper")
    plt.quiver(x, y, u_down, v_down, color="white", scale=scale)
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.colorbar(im, label="Displacement (m/yr)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAFE1    = "Images/S2A_MSIL2A_20250907T103031_N0511_R108_T32TLR_20250907T142815.SAFE"
    SAFE2    = "Images/S2B_MSIL2A_20250920T102619_N0511_R108_T32TLR_20250920T142941.SAFE"
    SHAPE    = "RGI2000-v7.0-G-11_central_europe/RGI2000-v7.0-G-11_central_europe.shp"
    OUT_DIR  = "Images_out"
    CORR_WALLIS = "Displ_MicMac_wallis"
    CORR_CENSUS = "Displ_MicMac_census"
    CORR_WALLIS_CENSUS = "Displ_MicMac_wallis_Census"

    sz_w = 10  #size of the correlation window (note that it is not sz_w*sz_w but (2*sz_w+1)*(2*sz_w+1))
    reg = 0.5  #Regularisation param
    inc = 1    #Initial uncertainty of the algo, higher value means larger exploration (can take a long time)
    gamma_cor = 1   #weight given to the high correlation matches
    cor_min = 0.3   #minimal correlation to be considered a match
    pxl_precision = 3  # between 1 and 5, MicMac returns 5 different displacement map with different resolutions. The lower resolution seems to yield better results in term of expected displacement values but we loose spatial resolution.

    processed = process_safe_pair(
        SAFE1, SAFE2, SHAPE, OUT_DIR,
        band="B08", resolution="R10m",
        x_min=3772, y_min=1272, x_size=512, y_size=512
    )

    wallis_files = [v["wallis"] for v in processed.values()]
    census_files = [v["census"] for v in processed.values()]
    wallis_census_files = [v["wallis_census"] for v in processed.values()]

    run_micmac_correlation(
        wallis_files[0], wallis_files[1],
        out_dir=CORR_WALLIS,
        sz_w=sz_w, reg=reg, inc=inc, gamma_cor=gamma_cor, cor_min=cor_min
    )

    run_micmac_correlation(
        census_files[0], census_files[1],
        out_dir=CORR_CENSUS,
        sz_w=sz_w, reg=reg, inc=inc, gamma_cor=gamma_cor, cor_min=cor_min
    )
    
    run_micmac_correlation(
        wallis_census_files[0], wallis_census_files[1],
        out_dir=CORR_WALLIS_CENSUS,
        sz_w=sz_w, reg=reg, inc=inc, gamma_cor=gamma_cor, cor_min=cor_min
    )

    dx_w, dy_w = find_displacement_files(CORR_WALLIS, pxl_precision=pxl_precision)
    plot_displacement(dx_w, dy_w, SHAPE, SAFE1, SAFE2,
                      step=20, scale=200, title_suffix="Wallis",
                      out_path=os.path.join(OUT_DIR, "displacement_wallis.png"))

    dx_c, dy_c = find_displacement_files(CORR_CENSUS, pxl_precision=pxl_precision)
    plot_displacement(dx_c, dy_c, SHAPE, SAFE1, SAFE2,
                      step=20, scale=200, title_suffix="Census",
                      out_path=os.path.join(OUT_DIR, "displacement_census.png"))
    
    dx_w_c, dy_w_c = find_displacement_files(CORR_WALLIS_CENSUS, pxl_precision=pxl_precision)
    plot_displacement(dx_w_c, dy_w_c, SHAPE, SAFE1, SAFE2,
                      step=20, scale=200, title_suffix="Wallis + Census",
                      out_path=os.path.join(OUT_DIR, "displacement_wallis_census.png"))