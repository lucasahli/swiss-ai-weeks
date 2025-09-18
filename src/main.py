import re
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import show

from PIL import Image
import numpy as np
from typing import List, Tuple

def bounds_from_filename(url: str) -> Tuple[int, int, int, int]:
    """
    Extract bounding box from a swissimage-dop10 filename.
    Returns (left, bottom, right, top) in EPSG:2056.
    """
    # Find pattern like _2594-1138_
    m = re.search(r"_(\d+)-(\d+)_", url)
    if not m:
        raise ValueError(f"Could not parse tile coords from {url}")

    tile_x, tile_y = map(int, m.groups())
    left = tile_x * 1000
    right = left + 1000
    bottom = tile_y * 1000
    top = bottom + 1000

    # print(url)
    # print("bounds: left: " + str(left) + " right: " + str(right) + " bottom: " + str(bottom) + " top: ", str(top))

    return (left, bottom, right, top)

def find_covering_files(coord: Tuple[float, float], files: List[str]) -> List[str]:
    """
    Find GeoTIFF files that contain the given coordinate.

    Args:
        coord: (x, y, canton) tuple in EPSG:2056
        files: list of GeoTIFF file paths

    Returns:
        List of file paths that contain the coordinate.
    """
    matching_files = []
    x, y, canton = coord

    for f in files:
        try:
            left, bottom, right, top = bounds_from_filename(f)
            if left <= x <= right and bottom <= y <= top:
                matching_files.append(f)
        except Exception as e:
            print(f"⚠️ Could not open {f}: {e}")

    return matching_files

electronic_power_producer_path = '../data/electronic_power_producers/ElectricityProductionPlant.csv'

import csv

coords = []
image_urls = []
with open("../data/imageURLs.csv", "r") as f:
    for line in f:
        url = line.strip()
        if url:  # skip empty lines
            image_urls.append(url)


# Open the CSV file
with open(electronic_power_producer_path, newline='', encoding="utf-8") as csvfile:
    # reader = csv.reader(csvfile)
    reader = csv.DictReader(csvfile)

    for i, row in enumerate(reader):
        try:
            coords.append((int(row["_x"]), int(row["_y"]), row['Canton']))
        except ValueError:
            print("Invalid coord: " + row["_x"] + " " + row["_y"])



# Check which files cover which coords
for i, c in enumerate(coords):
    if i >= 10:   # stop after 5 rows
        break
    hits = find_covering_files(c, image_urls)
    if hits:
        print(f"✅ {c} is inside: {hits}")
    else:
        print(f"❌ {c} not found in any file")

image_name = 'https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/swissimage-dop10_2021_2594-1138/swissimage-dop10_2021_2594-1138_0.1_2056.tif'

with rasterio.open(image_name) as dataset:
    # Read all bands (e.g., RGB)
    data = dataset.read()

    # Rasterio returns (bands, height, width) → rearrange to (height, width, bands)
    img = np.transpose(data, (1, 2, 0))

    # Convert to uint8 (JPEG needs this, and many GeoTIFFs are uint16 or float)
    if img.dtype != np.uint8:
        img = (255 * (img.astype(np.float32) / img.max())).astype(np.uint8)

    # Create and save with PIL
    im = Image.fromarray(img)
    im.save("output.jpg", "JPEG")