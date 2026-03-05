import os
import glob
import subprocess
import shutil
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.windows import Window
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from osgeo import gdal


# ---------------------------------------------------------------------------
# SAFE folder, looks for specified band at given resolution,  converts jp2 to tif, mask the image then crop it to given region + Wallis filter
# --------------------------------------------------------------------------

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
    cmd = (
        f'mm3d Nikrup " * 256 erfcc / - (=I {tif_path}) (=M ( moy @I 20 4)) '
        f'(sqrt max 0.001 (- (moy square @I 20 4) (square @M))) " {out_path}'
    )
    subprocess.run(
        ["mm3d", "Nikrup",
         f"* 256 erfcc / - (=I {tif_path}) (=M ( moy @I 20 4)) "
         "(sqrt max 0.001 (- (moy square @I 20 4) (square @M)))",
         out_path],
        check=True
    )


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

        jp2_to_tif(jp2, tif_raw)
        apply_mask(tif_raw, shape_path, tif_masked)
        crop_tif(tif_masked, tif_cropped, x_min, y_min, x_size, y_size)
        wallis_filter(tif_cropped, tif_wallis)
        copy_georef(tif_cropped, tif_wallis)

        results[safe] = {"cropped": tif_cropped, "wallis": tif_wallis}
        print(f"Processed: {base}")

    return results


# ---------------------------------------------------------------------------
# MicMac correlation
# ---------------------------------------------------------------------------

def run_micmac_correlation(img1_wallis, img2_wallis, out_dir,
                           sz_w=4, reg=0.1, inc=8):
    work_dir = os.path.dirname(os.path.abspath(img1_wallis))
    abs_out_dir = os.path.abspath(out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)

    subprocess.run(
        ["mm3d", "MM2DPosSism",
         os.path.basename(img1_wallis),
         os.path.basename(img2_wallis),
         f"SzW={sz_w}", f"Reg={reg}", f"Inc={inc}"],
        check=True,
        cwd=work_dir
    )

    mec_dir = os.path.join(work_dir, "MEC")
    for f in glob.glob(os.path.join(mec_dir, "*")):
        shutil.move(f, abs_out_dir)
    shutil.rmtree(mec_dir, ignore_errors=True)

    for tif in glob.glob(os.path.join(abs_out_dir, "*.tif")):
        copy_georef(img1_wallis, tif)

    return out_dir


def find_displacement_files(corr_dir):
    def _find(pattern):
        matches = glob.glob(os.path.join(corr_dir, pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern} in {corr_dir}")
        return matches[0]

    dx = _find("*Px1_Num5*")
    dy = _find("*Px2_Num5*")
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


def plot_displacement(dx_path, dy_path, shape_path, step=10, scale=100, out_path=None):
    dx = _read_band(dx_path)
    dy = _read_band(dy_path)

    valid_mask = _build_mask_array(shape_path, dx_path)
    dx[~valid_mask] = np.nan
    dy[~valid_mask] = np.nan

    magnitude = np.sqrt(dx**2 + dy**2)

    print("Min:", np.nanmin(magnitude))
    print("Max:", np.nanmax(magnitude))
    print("Count > 1:", np.sum(magnitude > 1))
    print("Total valid:", np.sum(~np.isnan(magnitude)))

    plt.hist(magnitude, bins=10)
    plt.show()

    H, W = magnitude.shape
    y, x = np.mgrid[0:H:step, 0:W:step]

    u_down = dx[::step, ::step]
    v_down = dy[::step, ::step]
    v_down_plot = -v_down
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        magnitude,
        cmap='inferno',
        origin='upper'
        )    
    plt.quiver(x, y, u_down, v_down_plot, color='white', scale=scale)

    plt.title("Scale and direction of displacement")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.colorbar(im, label="Displacement magnitude (pixels)",)
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAFE1      = "Images/S2A_MSIL2A_20250629T102701_N0511_R108_T32TLR_20250629T143707.SAFE"
    SAFE2      = "Images/S2B_MSIL2A_20250920T102619_N0511_R108_T32TLR_20250920T142941.SAFE"
    SHAPE      = "RGI2000-v7.0-G-11_central_europe/RGI2000-v7.0-G-11_central_europe.shp"
    TREND      = "v2020_2021_ALPES_ALL_ANNUALv2016-2021.tiff"
    OUT_DIR    = "Images_out"
    CORR_DIR   = "Displ_MicMac_cor"

    processed = process_safe_pair(
        SAFE1, SAFE2, SHAPE, OUT_DIR,
        band="B02", resolution="R10m",
        x_min=3772, y_min=1272, x_size=512, y_size=512
    )

    wallis_files = [v["wallis"] for v in processed.values()]
    run_micmac_correlation(
        wallis_files[0], wallis_files[1],
        out_dir=CORR_DIR,
        sz_w=4, reg=0.5, inc=20
    )

    dx_path, dy_path = find_displacement_files(CORR_DIR)

    plot_displacement(dx_path, dy_path, SHAPE, step=20, scale=200,
                      out_path=os.path.join(OUT_DIR, "displacement_map.png"))
