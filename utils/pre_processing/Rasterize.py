import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

def lines_to_raster(
    shapefile_path: str,
    output_dir: str,
    manual_type: int,
    buffer_base: float = 200,
    resolution: float = 10,
    target_type: int = 2,
) -> str:
    # Build output path
    output_name = os.path.splitext(os.path.basename(shapefile_path))[0] + ".tif"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # Read data
    gdf = gpd.read_file(shapefile_path)
    if "Size" not in gdf.columns:
        raise ValueError("Shapefile must contain the 'Size' column")

    # Compute buffer
    gdf["Buffer"] = gdf["Size"] * buffer_base

    # Filter levees
    gdf = gdf[(gdf["Type"] == target_type) & (gdf["geometry"].notnull())]

    # Apply buffer
    gdf["geometry"] = gdf.apply(lambda row: row.geometry.buffer(row["Buffer"]), axis=1)

    # Define metadata
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Rasterize
    shapes = ((geom, manual_type) for geom in gdf["geometry"])
    rasterized_levee = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int32",
    )

    # Save
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterized_levee.dtype,
        crs=gdf.crs,
        transform=transform,
    ) as dst:
        dst.write(rasterized_levee, 1)

    return output_path