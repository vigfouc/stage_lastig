import os
import time
import requests
from datetime import date
from dateutil.relativedelta import relativedelta
from pystac_client import Client
from tqdm import tqdm


TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_URL   = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/stac"


def get_token(username, password):
    response = requests.post(TOKEN_URL, data={
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public"
    })
    if response.status_code != 200:
        raise RuntimeError(f"Authentication failed: {response.text}")
    return response.json()["access_token"], time.time()


def monthly_search(catalog, collection, tile_id, start_date, end_date,
                   cloud_cover_percentage=None, orbit=108):
    all_items = []
    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    while current < end:
        month_end = min(current + relativedelta(months=1), end)
        print(f"Searching {current} → {month_end} ...")
        for attempt in range(5):
            try:
                query = {
                    "grid:code": {"eq": f"MGRS-{tile_id}"},
                    "sat:relative_orbit": {"eq": orbit}
                }
                if cloud_cover_percentage is not None:
                    query["eo:cloud_cover"] = {"lte": cloud_cover_percentage}

                search = catalog.search(
                    collections=[collection],
                    datetime=f"{current}/{month_end}",
                    query=query,
                )
                items = list(search.items())
                print(f"  → {len(items)} image(s) found")
                all_items.extend(items)
                break
            except Exception as e:
                if "429" in str(e):
                    wait = 5 * (2 ** attempt)
                    print(f"  Rate limited — retrying in {wait}s (attempt {attempt+1}/5)...")
                    time.sleep(wait)
                else:
                    print(f"  Error: {e} — skipping month")
                    break
        current = month_end

    return all_items


def get_product_id_from_name(product_name):
    url = f"{ODATA_URL}/Products?$filter=Name eq '{product_name}'&$select=Id,Name"
    response = requests.get(url)
    results = response.json().get("value", [])
    if not results:
        raise ValueError(f"Product not found in OData: {product_name}")
    return results[0]["Id"]


def download_band_odata(session, product_id, product_name, band_filename,
                        out_path, resolution="R10m"):
    safe_node    = f"{ODATA_URL}/Products({product_id})/Nodes({product_name})"
    granule_resp = session.get(f"{safe_node}/Nodes(GRANULE)/Nodes")
    granule_name = granule_resp.json()["result"][0]["Id"]

    imgdata_node = (f"{safe_node}/Nodes(GRANULE)/Nodes({granule_name})"
                    f"/Nodes(IMG_DATA)/Nodes({resolution})/Nodes({band_filename})/$value")

    response = session.get(imgdata_node, allow_redirects=False, stream=True)
    while response.status_code in (301, 302, 303, 307):
        response = session.get(response.headers["Location"],
                               allow_redirects=False, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Download failed ({response.status_code}): {response.text[:200]}")

    total = int(response.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(
        desc=band_filename, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_items(items, output_folder, username, password,
                   band="B08_10m", download_scl=True):
    os.makedirs(output_folder, exist_ok=True)
    scl_folder = os.path.join(output_folder, "SCL")
    if download_scl:
        os.makedirs(scl_folder, exist_ok=True)

    access_token, token_time = get_token(username, password)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {access_token}"})

    failed = []

    for i, item in enumerate(items):
        if time.time() - token_time > 480:
            print("Refreshing access token...")
            access_token, token_time = get_token(username, password)
            session.headers.update({"Authorization": f"Bearer {access_token}"})

        product_name = item.id if item.id.endswith(".SAFE") else item.id + ".SAFE"

        if band not in item.assets:
            print(f"  WARNING: asset '{band}' not found in {item.id}, "
                  f"available: {list(item.assets.keys())}")
            failed.append(item.id)
            continue

        band_filename = item.assets[band].href.split("/")[-1]
        out_path = os.path.join(output_folder, band_filename)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
            print(f"  Skipping {band_filename} (already downloaded)")
        else:
            print(f"[{i+1}/{len(items)}] Downloading {band_filename}")
            try:
                product_id = get_product_id_from_name(product_name)
                download_band_odata(session, product_id, product_name,
                                    band_filename, out_path, resolution="R10m")
                if os.path.getsize(out_path) < 1_000_000:
                    print(f"  WARNING: suspiciously small ({os.path.getsize(out_path)} bytes)")
                    failed.append(item.id)
                    continue
            except Exception as e:
                print(f"  ERROR: {e}")
                failed.append(item.id)
                continue

        # Download SCL (Scene Classification Layer) at 20m resolution
        if download_scl:
            scl_asset_key = "SCL_20m"
            if scl_asset_key not in item.assets:
                # fallback: try common alternative key names
                for key in item.assets:
                    if "SCL" in key.upper():
                        scl_asset_key = key
                        break
                else:
                    print(f"  WARNING: SCL asset not found for {item.id}")
                    continue

            scl_filename = item.assets[scl_asset_key].href.split("/")[-1]
            scl_path = os.path.join(scl_folder, scl_filename)

            if os.path.exists(scl_path) and os.path.getsize(scl_path) > 100_000:
                print(f"  Skipping SCL {scl_filename} (already downloaded)")
            else:
                print(f"  Downloading SCL: {scl_filename}")
                try:
                    product_id = get_product_id_from_name(product_name)
                    download_band_odata(session, product_id, product_name,
                                        scl_filename, scl_path, resolution="R20m")
                except Exception as e:
                    print(f"  SCL ERROR: {e}")

    if failed:
        print(f"\n{len(failed)} failed: {failed}")
    else:
        print("\nAll downloads completed successfully.")


if __name__ == "__main__":
    START_DATE            = "2017-01-01"
    END_DATE              = "2017-12-31"
    TILE_ID               = "32TLR"
    BAND                  = "B08_10m"
    CLOUD_COVER_MAX       = 80
    OUTPUT_FOLDER         = f"Sentinel_2_{START_DATE}_{END_DATE}_{TILE_ID}_Cloud_cover{CLOUD_COVER_MAX}"

    username = input("Copernicus username (email): ")
    password = input("Copernicus password: ")

    catalog = Client.open(CATALOG_URL)
    print(f"Searching for tile {TILE_ID} from {START_DATE} to {END_DATE}...\n")
    items = monthly_search(catalog, "sentinel-2-l2a", TILE_ID,
                           START_DATE, END_DATE,
                           cloud_cover_percentage=CLOUD_COVER_MAX)
    print(f"\nTotal images found: {len(items)}")

    if not items:
        print("No images found, exiting.")
        exit()

    print(f"Sample asset keys: {list(items[0].assets.keys())}")
    max_nb = int(input(f"Number of images to download (max {len(items)}): "))
    items = items[:max_nb]

    download_items(items, OUTPUT_FOLDER, username, password,
                   band=BAND, download_scl=True)