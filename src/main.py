import rasterio
import rasterio.features
import rasterio.warp

with rasterio.open('https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/swissimage-dop10_2021_2594-1138/swissimage-dop10_2021_2594-1138_0.1_2056.tif') as dataset:

    # Read the dataset's valid data mask as a ndarray.
    mask = dataset.dataset_mask()

    # Extract feature shapes and values from the array.
    for geom, val in rasterio.features.shapes(
            mask, transform=dataset.transform):

        # Transform shapes from the dataset's own coordinate
        # reference system to CRS84 (EPSG:4326).
        geom = rasterio.warp.transform_geom(
            dataset.crs, 'EPSG:4326', geom, precision=6)

        # Print GeoJSON shapes to stdout.
        print(geom)