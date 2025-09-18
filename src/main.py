import os
import multiprocessing
from functools import partial
import re
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import show

from PIL import Image
import numpy as np
from typing import List, Tuple

def highlight_area(img: np.ndarray, row: int, col: int, size: int = 50):
    """
    Set all pixels around (row, col) to white (255) in a square of ±size.
    Works on RGB or grayscale images.
    """
    rows, cols = img.shape[:2]

    # Clamp the window to image bounds
    r0 = max(0, row - size)
    r1 = min(rows, row + size + 1)
    c0 = max(0, col - size)
    c1 = min(cols, col + size + 1)

    img[r0:r1, c0:c1, :] = [255, 0, 0]
    return img

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
            if row['Canton'] == 'BE':
                coords.append((int(row["_x"]), int(row["_y"]), row['Canton']))
        except ValueError:
            continue
            # print("Invalid coord: " + row["_x"] + " " + row["_y"])

import pandas as pd
houses_df = pd.read_parquet('../data/houses.parquet', engine='fastparquet')
# import pandas as pd
# houses_df = pd.read_parquet('../data/houses.parquet', engine='pyarrow')

# Function to check if point (x, y) is inside a bbox
def is_point_in_bbox(row, x, y):
    min_x, min_y, max_x, max_y = row['bbox']
    return min_x <= x <= max_x and min_y <= y <= max_y

# Check which files cover which coords
# for i, coord in enumerate(coords):
#     if i >= 20:   # stop after 5 rows
#         break
#     hits = find_covering_files(coord, image_urls)
#     if hits:
#         # print(f"✅ {coord} is inside: {hits}")
#         image_name = hits[0]
#         with rasterio.open(image_name) as dataset:
#             # Read all bands (e.g., RGB)
#             data = dataset.read()
#
#             # Rasterio returns (bands, height, width) → rearrange to (height, width, bands)
#             img = np.transpose(data, (1, 2, 0))
#
#             # Convert to uint8 (JPEG needs this, and many GeoTIFFs are uint16 or float)
#             if img.dtype != np.uint8:
#                 img = (255 * (img.astype(np.float32) / img.max())).astype(np.uint8)
#
#             x, y, canton = coord
#             row, col = dataset.index(x, y)
#
#             # img[row, col,:] = 255
#             # img = highlight_area(img, row=row, col=col, size=20)
#
#             # Apply the function to filter rows
#             # matches = houses_df[houses_df.apply(is_point_in_bbox, axis=1, args=(x, y))]
#             print(houses_df.columns)
#             # Filter rows where the point is inside the bounding box
#             matches = houses_df[
#                 (houses_df['bbox.min_x'] <= x) & (x <= houses_df['bbox.max_x']) &
#                 (houses_df['bbox.min_y'] <= y) & (y <= houses_df['bbox.max_y']) & (houses_df['art'] == 0)
#                 ]
#
#             # Get the 'name' column (or any other column) for matching rows
#             result = matches['bbartt_bez_de']
#             print(result)
#
#             # Step 2: Crop the image for each matching row
#             if matches.empty:
#                 print("No bounding box contains the point.")
#             else:
#                 for index, row in matches.iterrows():
#                     # Extract scalar values from the row
#                     min_x = int(row['bbox.min_x'])
#                     max_x = int(row['bbox.max_x'])
#                     min_y = int(row['bbox.min_y'])
#                     max_y = int(row['bbox.max_y'])
#
#                     row_max, col_min = dataset.index(min_x, min_y)
#                     row_min, col_max = dataset.index(max_x, max_y)
#
#                     extension_amount = 20
#                     row_min = row_min - extension_amount
#                     row_max = row_max + extension_amount
#                     col_min = col_min - extension_amount
#                     col_max = col_max + extension_amount
#
#                     # Ensure indices are within image bounds
#                     if (row_min >= 0 and row_max <= img.shape[0] and col_min >= 0 and col_max <= img.shape[1]):
#                         # Crop the image
#                         crop_img = img[row_min:row_max, col_min:col_max]
#                         print(f"Cropped image for row {index}")
#                         cim = Image.fromarray(crop_img)
#                         cim.save(f'crop{index}.jpg', "JPEG")
#                     else:
#                         print(f"Bounding box for row {index} is out of image bounds.")
#
#             # crop_img = img[int(houses_df['bbox.min_x']):int(houses_df['bbox.max_x']), int(houses_df['bbox.min_y']):int(houses_df['bbox.max_y'])]
#             # cim = Image.fromarray(crop_img)
#             # cim.save("crop.jpg", "JPEG")
#
#             # Create and save with PIL
#             im = Image.fromarray(img)
#             im.save("output.jpg", "JPEG")
#     else:
#         print(f"❌ {coord} not found in any file")


def process_coord(coord, houses_df, image_urls, output_dir="output"):
    """Process a single coordinate and return results."""
    try:
        hits = find_covering_files(coord, image_urls)
        if not hits:
            return f"❌ {coord} not found in any file"

        image_name = hits[0]
        with rasterio.open(image_name) as dataset:
            # Read all bands
            data = dataset.read()
            # Rearrange to (height, width, bands)
            img = np.transpose(data, (1, 2, 0))

            # Convert to uint8
            if img.dtype != np.uint8:
                img = (255 * (img.astype(np.float32) / img.max())).astype(np.uint8)

            x, y, canton = coord
            row, col = dataset.index(x, y)

            # Filter houses_df
            matches = houses_df[
                (houses_df['bbox.min_x'] <= x) & (x <= houses_df['bbox.max_x']) &
                (houses_df['bbox.min_y'] <= y) & (y <= houses_df['bbox.max_y']) &
                (houses_df['art'] == 0)
                ]

            result = matches['bbartt_bez_de'].tolist()
            output = [f"✅ {coord} is inside: {hits}", f"Matches: {result}"]

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Crop images for each matching row
            if matches.empty:
                output.append("No bounding box contains the point.")
            else:
                for index, row in matches.iterrows():
                    min_x, max_x = int(row['bbox.min_x']), int(row['bbox.max_x'])
                    min_y, max_y = int(row['bbox.min_y']), int(row['bbox.max_y'])

                    row_max, col_min = dataset.index(min_x, min_y)
                    row_min, col_max = dataset.index(max_x, max_y)

                    extension_amount = 20
                    row_min -= extension_amount
                    row_max += extension_amount
                    col_min -= extension_amount
                    col_max += extension_amount

                    # Check bounds
                    if (row_min >= 0 and row_max <= img.shape[0] and
                            col_min >= 0 and col_max <= img.shape[1]):
                        crop_img = img[row_min:row_max, col_min:col_max]
                        crop_img_path = os.path.join(output_dir, f"crop_{x}_{y}_{index}.jpg")
                        cim = Image.fromarray(crop_img)
                        cim.save(crop_img_path, "JPEG")
                        output.append(f"Cropped image for row {index}: {crop_img_path}")
                    else:
                        output.append(f"Bounding box for row {index} is out of image bounds.")

        return "\n".join(output)
    except Exception as e:
        return f"Error processing {coord}: {str(e)}"

def parallel_process_coords(coords, houses_df, image_urls, max_iterations=20, num_processes=None):
    """Parallelize coordinate processing with multiprocessing."""
    # Limit to max_iterations
    coords = coords[:max_iterations]

    # Use number of CPU cores if not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Create a partial function with shared data
    process_func = partial(process_coord, houses_df=houses_df, image_urls=image_urls)

    # Use Pool to parallelize
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, coords)

    # Print results
    for result in results:
        print(result)

# Example usage
if __name__ == "__main__":
    # Assuming coords, houses_df, image_urls, and find_covering_files are defined
    parallel_process_coords(coords, houses_df, image_urls, max_iterations=10000)
