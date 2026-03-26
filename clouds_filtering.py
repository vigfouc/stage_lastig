import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import matplotlib.dates as mdates
from micmac_velocity_field import _date_from_tif

def plot_img_histogram(img_path):
    with rasterio.open(img_path) as src:
        img = src.read(1).astype(float)

    img[img == 0] = np.nan

    if np.all(np.isnan(img)):
        print(f"Skipping {os.path.basename(img_path)} — all NaN")
        return

    vmin = np.nanpercentile(img, 2)
    vmax = np.nanpercentile(img, 98)
    if vmax == vmin:
        print(f"Skipping {os.path.basename(img_path)} — no variance")
        return

    img_8bit = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(img_8bit, cmap="gray")
    axes[0].set_title(os.path.basename(img_path))
    axes[0].axis("off")

    valid = img_8bit[img_8bit > 0].ravel()
    axes[1].hist(valid, bins=255, range=(0, 255), color="steelblue", edgecolor="none")
    axes[1].set_xlabel("Pixel value (0-255)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Histogram (non-zero pixels)")

    plt.tight_layout()
    plt.show()

def compute_fourier_anomaly_score(tif_paths):
    spectra = []
    valid_paths = []

    for p in tif_paths:
        with rasterio.open(p) as src:
            img = src.read(1).astype(float)
        img[img == 0] = np.nan
        if np.all(np.isnan(img)):
            continue

        img = np.nan_to_num(img)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        spectra.append(magnitude.ravel())
        valid_paths.append(p)

    spectra = np.stack(spectra, axis=0)
    median_spectrum = np.median(spectra, axis=0)
    scores = np.linalg.norm(spectra - median_spectrum, axis=1)

    return valid_paths, scores


def plot_fourier_anomaly(tif_paths, threshold_sigma=2.0, out_path=None):
    valid_paths, scores = compute_fourier_anomaly_score(tif_paths)

    dates = [_date_from_tif(p) for p in valid_paths]
    median = np.median(scores)
    std    = np.std(scores)
    threshold = median + threshold_sigma * std

    flagged = [p for p, s in zip(valid_paths, scores) if s < threshold]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(dates, scores, s=20, color="steelblue", label="Images")
    ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold_sigma}σ)")
    ax.scatter([d for d, s in zip(dates, scores) if s < threshold],
               [s for s in scores if s < threshold],
               color="red", s=40, zorder=5, label="Flagged")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Anomaly score")
    ax.set_title("Fourier anomaly score per image")
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()

    print(f"\nFlagged {len(flagged)} images:")
    for p in flagged:
        print(f"  {os.path.basename(p)}")

    return flagged

if __name__ == "__main__":

    tif_images_path = "MicMac_correl_wallis_cleaned/wallis/"
    tif_paths = sorted(glob.glob(os.path.join(tif_images_path, "*.tif")))
    # flagged = plot_fourier_anomaly(tif_paths, threshold_sigma=0.3, out_path=None)

    for flagged_path in tif_paths:
        with rasterio.open(flagged_path) as src:
            img = src.read(1).astype(float)
            plt.imshow(img, cmap="gray")
            plt.title(f"Image flagged: {os.path.basename(flagged_path)}")
            plt.show()

    corel_path = ""
