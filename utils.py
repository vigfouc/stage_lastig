import os
import glob
import subprocess
import numpy as np
import rasterio
import re
from rasterio.mask import mask
from rasterio.windows import Window
import geopandas as gpd
from osgeo import gdal
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer


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


def convert_jp2_folder_to_tif(jp2_folder_path, tif_folder_path):
    if not os.path.isdir(jp2_folder_path):
        raise NotADirectoryError(f"Invalid JP2 folder: {jp2_folder_path}")
    os.makedirs(tif_folder_path, exist_ok=True)
    for file in os.listdir(jp2_folder_path):
        if not file.lower().endswith(".jp2"):
            continue
        jp2_path = os.path.join(jp2_folder_path, file)
        if not os.path.isfile(jp2_path):
            continue
        base = os.path.splitext(file)[0]
        tif_img = os.path.join(tif_folder_path, f"{base}.tif")
        jp2_to_tif(jp2_path, tif_img)


def apply_mask(tif_path, shape_path, out_path=None):
    gdf = gpd.read_file(shape_path)
    with rasterio.open(tif_path) as src:
        gdf_proj = gdf.to_crs(src.crs)
        geoms = [geom for geom in gdf_proj.geometry]
        out_image, out_transform = mask(src, geoms, crop=False, nodata=0)
        out_meta = src.meta.copy()
    if out_path:
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)
    return out_image, out_meta


def crop_tif(tif_path, out_path=None, x_min=0, y_min=0, x_size=512, y_size=512):
    with rasterio.open(tif_path) as src:
        window = Window(x_min, y_min, x_size, y_size)
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(width=x_size, height=y_size, transform=transform)
    if out_path:
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
    return data, profile


def wallis_filter(tif_path, out_path):
    subprocess.run(
        ["mm3d", "Nikrup",
         f"* 256 erfcc / - (=I {tif_path}) (=M ( moy @I 20 4)) "
         "(sqrt max 0.001 (- (moy square @I 20 4) (square @M)))",
         out_path],
        check=True
    )


def census_transform(tif_path, out_path=None, radius=3):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

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
    if out_path:
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(census[np.newaxis, :, :])
    return census, profile


def copy_georef(ref_path, target_path):
    ref = gdal.Open(ref_path)
    target = gdal.Open(target_path, gdal.GA_Update)
    target.SetGeoTransform(ref.GetGeoTransform())
    target.SetProjection(ref.GetProjection())
    ref = None
    target = None


def process_tif_img(tif_img, out_dir, shape_path,
                    x_min=3772, y_min=1272, x_size=512, y_size=512,
                    do_wallis=True, do_census=True, do_wallis_census=True, do_masked=True,
                    keep_intermediate=True):

    masked_dir       = os.path.join(out_dir, "masked")
    cropped_dir      = os.path.join(out_dir, "cropped")
    wallis_dir       = os.path.join(out_dir, "wallis")
    census_dir       = os.path.join(out_dir, "census")
    wallis_census_dir = os.path.join(out_dir, "wallis_census")

    for d in (masked_dir, cropped_dir):
        os.makedirs(d, exist_ok=True)

    base = os.path.splitext(os.path.basename(tif_img))[0]

    masked_path       = os.path.join(masked_dir,       f"{base}_masked.tif")
    cropped_path      = os.path.join(cropped_dir,      f"{base}_cropped.tif")
    wallis_path       = os.path.join(wallis_dir,       f"{base}_cropped_lisse.tif")
    census_path       = os.path.join(census_dir,       f"{base}_cropped_census.tif")
    wallis_census_path = os.path.join(wallis_census_dir, f"{base}_cropped_lisse_census.tif")

    if do_masked:
        apply_mask(tif_img, shape_path, masked_path)
        crop_tif(masked_path, cropped_path, x_min, y_min, x_size, y_size)
    else:
        crop_tif(tif_img, cropped_path, x_min, y_min, x_size, y_size)

    results = {"cropped": cropped_path}

    if do_wallis:
        os.makedirs(wallis_dir, exist_ok=True)
        wallis_filter(cropped_path, wallis_path)
        copy_georef(cropped_path, wallis_path)
        results["wallis"] = wallis_path

    if do_census:
        os.makedirs(census_dir, exist_ok=True)
        census_transform(cropped_path, census_path)
        copy_georef(cropped_path, census_path)
        results["census"] = census_path

    if do_wallis_census and do_wallis:
        os.makedirs(wallis_census_dir, exist_ok=True)
        census_transform(wallis_path, wallis_census_path)
        copy_georef(cropped_path, wallis_census_path)
        results["wallis_census"] = wallis_census_path

    if not keep_intermediate:
        if os.path.exists(masked_path):
            os.remove(masked_path)

    print(f"Processed: {base}")
    return results


def process_tif_folder(tif_folder, out_dir, shape_path,
                       x_min=3772, y_min=1272, x_size=512, y_size=512,
                       do_wallis=True, do_census=True, do_wallis_census=True, do_masked=True,
                       keep_intermediate=True):
    tif_files = sorted(glob.glob(os.path.join(tif_folder, "*.tif")))
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {tif_folder}")

    print(f"Found {len(tif_files)} TIF files in {tif_folder}")
    all_results = {}

    for tif_path in tif_files:
        results = process_tif_img(
            tif_path, out_dir, shape_path,
            x_min=x_min, y_min=y_min, x_size=x_size, y_size=y_size,
            do_wallis=do_wallis, do_census=do_census,
            do_wallis_census=do_wallis_census, do_masked=do_masked,
            keep_intermediate=keep_intermediate
        )
        all_results[tif_path] = results

    return all_results


def save_stakes_on_tif(tif_path, csv_path, out_tif_path, y_offset=2000000):
    df = pd.read_csv(csv_path)

    with rasterio.open(tif_path) as src:
        img_crs   = src.crs
        transform = src.transform
        data      = src.read(1).astype(float)
        profile   = src.profile.copy()

    transformer = Transformer.from_crs("EPSG:27572", img_crs, always_xy=True)

    x_start, y_start = transformer.transform(
        df["x_lambert3_start"].values,
        df["y_lambert3_start"].values + y_offset
    )
    x_end, y_end = transformer.transform(
        df["x_lambert3_end"].values,
        df["y_lambert3_end"].values + y_offset
    )

    H, W = data.shape

    def to_pixels(x_proj, y_proj):
        rows, cols = rasterio.transform.rowcol(transform, x_proj, y_proj)
        rows = np.array(rows)
        cols = np.array(cols)
        in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        return rows, cols, in_bounds

    rows_s, cols_s, in_s = to_pixels(x_start, y_start)
    rows_e, cols_e, in_e = to_pixels(x_end, y_end)

    norm = data / np.nanmax(data)
    norm = np.nan_to_num(norm)
    rgb = np.stack([norm, norm, norm], axis=0)

    def paint(rgb, rows, cols, in_bounds, r, g, b):
        for row, col in zip(rows[in_bounds], cols[in_bounds]):
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        rgb[0, rr, cc] = r
                        rgb[1, rr, cc] = g
                        rgb[2, rr, cc] = b
        return rgb

    rgb = paint(rgb, rows_s, cols_s, in_s, 1.0, 0.0, 0.0)  # red = start
    rgb = paint(rgb, rows_e, cols_e, in_e, 0.0, 0.0, 1.0)  # blue = end

    profile.update(count=3, dtype=rasterio.float32)
    with rasterio.open(out_tif_path, "w", **profile) as dst:
        dst.write(rgb.astype(np.float32))

    print(f"Saved: {out_tif_path}")


# SCL cloud classes
# 3  = cloud shadow
# 8  = medium probability cloud
# 9  = high probability cloud
# 10 = thin cirrus
CLOUD_CLASSES = {3, 8, 9, 10}
 
 
def get_scl_path(tif_path, scl_folder):
    """Find the SCL file corresponding to a B08 TIF by matching the date."""
    match = re.search(r"(\d{8})T\d{6}", os.path.basename(tif_path))
    if not match:
        raise ValueError(f"Cannot parse date from: {tif_path}")
    date_str = match.group(1)
    candidates = glob.glob(os.path.join(scl_folder, f"*{date_str}*.jp2"))
    if not candidates:
        raise FileNotFoundError(f"No SCL file found for date {date_str} in {scl_folder}")
    return candidates[0]
 
 
def cloud_fraction_in_crop(scl_path, x_min, y_min, x_size, y_size,
                            scl_scale=2):
    """
    Compute cloud fraction within the crop region using the SCL band.
    scl_scale=2 because SCL is at 20m while B08 is at 10m.
    """
    scl_x_min  = x_min  // scl_scale
    scl_y_min  = y_min  // scl_scale
    scl_x_size = x_size // scl_scale
    scl_y_size = y_size // scl_scale
 
    with rasterio.open(scl_path) as src:
        window = Window(scl_x_min, scl_y_min, scl_x_size, scl_y_size)
        scl = src.read(1, window=window).astype(np.uint8)
 
    cloud_pixels = np.isin(scl, list(CLOUD_CLASSES)).sum()
    total_pixels = scl.size
    return cloud_pixels / total_pixels
 
 
def filter_cloudy_images(tif_paths, scl_folder, x_min, y_min, x_size, y_size,
                         max_cloud_fraction=0.05):
    clean  = []
    cloudy = []

    nb_tif_files = len(tif_paths)
 
    for tif_path in tif_paths:
        try:
            scl_path = get_scl_path(tif_path, scl_folder)
            fraction = cloud_fraction_in_crop(scl_path, x_min, y_min,
                                              x_size, y_size)
            if fraction <= max_cloud_fraction:
                clean.append(tif_path)
            else:
                cloudy.append((tif_path, fraction))
                print(f"  Cloudy ({fraction*100:.1f}%): {os.path.basename(tif_path)}")
        except FileNotFoundError as e:
            print(f"  SCL not found, keeping image: {e}")
            clean.append(tif_path)

        print(f"Processed: {len(clean)+len(cloudy)}/{nb_tif_files}")
 
    print(f"\n{len(clean)} clean / {len(cloudy)} cloudy out of {len(tif_paths)} images")
    return clean, cloudy


if __name__ == "__main__":

    csv_path = "GlacioClim_csv/Saint_Sorlin_speed_2020.csv"
    img_path = "Sentinel_2_2020-01-01_2020-12-31_32TLR_Cloud_cover80_tif/T32TLR_20200519T102559_B08_10m.tif"
    out_path = "stakes_on_img.tif"

    save_stakes_on_tif(img_path, csv_path, out_path)