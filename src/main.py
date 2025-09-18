import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import show

import rasterio
from PIL import Image
import numpy as np

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