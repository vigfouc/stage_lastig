import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
from rasterio.crs import CRS


def export_velocity_tifs(nc_path, out_dir, epsg=32632):
    os.makedirs(out_dir, exist_ok=True)
    ds = xr.open_dataset(nc_path)

    x = ds.coords["x"].values
    y = ds.coords["y"].values
    times = ds.coords["time"].values

    pixel_size_x = (x.max() - x.min()) / (len(x) - 1)
    pixel_size_y = (y.max() - y.min()) / (len(y) - 1)

    transform = rasterio.transform.from_origin(
        west=x.min() - pixel_size_x / 2,
        north=y.max() + pixel_size_y / 2,
        xsize=pixel_size_x,
        ysize=pixel_size_y
    )
    crs = CRS.from_epsg(epsg)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": len(x),
        "height": len(y),
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": np.nan
    }

    for i, t in enumerate(times):
        date_str = str(t).replace(" ", "_").replace("-", "")

        for var in ["v"]:
            data = ds[var].isel(time=i).values

            if ds[var].dims[0] == "y":
                arr = data
            else:
                arr = data.T

            out_path = os.path.join(out_dir, f"{var}_{date_str}.tif")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr[np.newaxis, :, :])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"Exported {i + 1}/{len(times)}: {date_str}")

    print(f"\nDone. {len(times)} TIFs written to {out_dir}")


def compute_median_velocity_tifs(nc_path, epsg=32632):
    ds = xr.open_dataset(nc_path)

    x = ds.coords["x"].values
    y = ds.coords["y"].values

    pixel_size_x = (x.max() - x.min()) / (len(x) - 1)
    pixel_size_y = (y.max() - y.min()) / (len(y) - 1)

    transform = rasterio.transform.from_origin(
        west=x.min() - pixel_size_x / 2,
        north=y.max() + pixel_size_y / 2,
        xsize=pixel_size_x,
        ysize=pixel_size_y
    )
    crs = CRS.from_epsg(epsg)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": len(x),
        "height": len(y),
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": np.nan
    }

    median_velocity_field = np.nanmedian(ds["v"].values, axis=2)
    with rasterio.open("median_velocity_field.tif", "w", **profile) as dst: 
        dst.write(median_velocity_field[np.newaxis, :, :])


def plot_velocity_tiff(velocity_tif_path):
    with rasterio.open(velocity_tif_path) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr == src.nodata] = np.nan

    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr, cmap="inferno")
    plt.colorbar(im, label="m/yr")
    plt.title(os.path.basename(velocity_tif_path))
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    NC_PATH = "stack_median_pleiades_alllayers_2012-2022.nc"
    OUT_DIR = "Pleiades_velocity_tifs"

    # compute_median_velocity_tifs(NC_PATH, epsg=32632)
    plot_velocity_tiff("median_velocity_field.tif")
