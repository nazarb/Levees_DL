import os
from typing import List, Tuple, Union

import numpy as np
import rasterio
import tifffile as tiff
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, calculate_default_transform


def get_filename_without_extension(file_path: str) -> str:
    """
    Extract the filename without its extension from a file path.

    Args:
        file_path: Path to the file (can be absolute or relative).

    Returns:
        The filename without its extension.

    Example:
        >>> get_filename_without_extension("/path/to/image.tif")
        'image'
        >>> get_filename_without_extension("data/raster.geotiff")
        'raster'
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def create_tfw_file(transform: Affine, tfw_path: str) -> None:
    """
    Create a TFW (world file) from an Affine transform.

    A TFW file contains six lines with georeference information:
    - Line 1: pixel size in the x-direction (scale)
    - Line 2: rotation term (0 for north-up images)
    - Line 3: rotation term (0 for north-up images)
    - Line 4: pixel size in the y-direction (usually negative)
    - Line 5: x-coordinate of the upper-left corner
    - Line 6: y-coordinate of the upper-left corner

    Args:
        transform: An Affine transform object containing georeference info.
        tfw_path: Path where the TFW file will be saved.

    Example:
        >>> from affine import Affine
        >>> transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
        >>> create_tfw_file(transform, "output/image.tfw")
    """
    print(f"Creating .tfw file at {tfw_path}")
    with open(tfw_path, 'w') as f:
        f.write(f"{transform.a}\n")  # pixel size in the x-direction
        f.write(f"{transform.b}\n")  # rotation term (always 0 for north-up images)
        f.write(f"{transform.d}\n")  # rotation term (always 0 for north-up images)
        f.write(f"{transform.e}\n")  # pixel size in the y-direction (usually negative)
        f.write(f"{transform.c}\n")  # x-coordinate of the upper-left corner
        f.write(f"{transform.f}\n")  # y-coordinate of the upper-left corner


def read_tfw_coordinates(tif_path: str) -> Affine:
    """
    Read georeference coordinates from a TFW (world file) associated with a TIF.

    The function expects a .tfw file with the same name as the input TIF file.

    Args:
        tif_path: Path to the TIF file (the function looks for a matching .tfw file).

    Returns:
        An Affine transform object containing the georeference information.

    Raises:
        FileNotFoundError: If the corresponding .tfw file does not exist.
        ValueError: If the .tfw file does not have exactly 6 lines.

    Example:
        >>> transform = read_tfw_coordinates("data/image.tif")
        >>> print(transform.c, transform.f)  # Upper-left corner coordinates
    """
    # Normalize the path to handle duplicates and inconsistent separators
    tif_path = os.path.normpath(os.path.abspath(tif_path))
    tfw_path = os.path.splitext(tif_path)[0] + ".tfw"
    
    if not os.path.exists(tfw_path):
        raise FileNotFoundError(f"World file not found: {tfw_path}")

    with open(tfw_path, "r") as f:
        lines = f.readlines()
    if len(lines) != 6:
        raise ValueError(f"Invalid world file format: {tfw_path}")

    # Parse six parameters
    A = float(lines[0])  # pixel size in X
    D = float(lines[1])  # rotation term
    B = float(lines[2])  # rotation term
    E = float(lines[3])  # pixel size in Y
    C = float(lines[4])  # X coordinate of center of upper-left pixel
    F = float(lines[5])  # Y coordinate of center of upper-left pixel

    transform = Affine(A, B, C, D, E, F)
    print("\n[TFW] Affine transform read from file:")

    return transform


def load_large_image(image_path: str) -> Tuple[np.ndarray, Affine]:
    """
    Load a large raster image using rasterio.

    Reads all bands from a raster file and returns the image data along
    with its affine transformation matrix for georeferencing.

    Args:
        image_path: Path to the raster image file (e.g., GeoTIFF).

    Returns:
        A tuple containing:
        - image: numpy array with shape (bands, height, width)
        - transform: Affine transformation matrix for georeferencing

    Example:
        >>> image, transform = load_large_image("data/satellite_image.tif")
        >>> print(f"Image shape: {image.shape}")
        >>> print(f"Upper-left corner: ({transform.c}, {transform.f})")
    """
    print(f"Loading large image from {image_path}")
    with rasterio.open(image_path) as src:
        image = src.read()
        transform = src.transform  # Capture the affine transformation matrix
    return image, transform

def load_large_binary_image(image_path: str) -> Tuple[np.ndarray, Affine]:
    """
    Load a large raster image using rasterio.

    Reads all bands from a raster file and returns the image data along
    with its affine transformation matrix for georeferencing.

    Args:
        image_path: Path to the raster image file (e.g., GeoTIFF).

    Returns:
        A tuple containing:
        - image: numpy array with shape (bands, height, width)
        - transform: Affine transformation matrix for georeferencing

    Example:
        >>> image, transform = load_large_image("data/satellite_image.tif")
        >>> print(f"Image shape: {image.shape}")
        >>> print(f"Upper-left corner: ({transform.c}, {transform.f})")
    """
    print(f"Loading large image from {image_path}")
    with rasterio.open(image_path) as src:
        image = src.read()
        image = src.read(1).copy()  # ← Add .copy() here
        transform = src.transform  # Capture the affine transformation matrix
    return image, transform

def convert_tiles_to_npy(tile_paths: List[str], npy_dir: str) -> List[str]:
    """
    Convert TIFF tiles to NumPy (.npy) format.

    Reads each TIFF tile and saves it as a NumPy array file in the specified
    directory. Useful for preprocessing tiles before model inference.

    Args:
        tile_paths: List of paths to TIFF tile files.
        npy_dir: Directory where the .npy files will be saved.

    Returns:
        List of paths to the saved .npy files.

    Example:
        >>> tile_paths = ["tiles/tile_0_0.tif", "tiles/tile_0_96.tif"]
        >>> npy_paths = convert_tiles_to_npy(tile_paths, "tiles_npy/")
        >>> print(npy_paths)
        ['tiles_npy/tile_0_0.npy', 'tiles_npy/tile_0_96.npy']
    """
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    npy_paths = []
    for tile_path in tile_paths:
        image = tiff.imread(tile_path)
        npy_path = os.path.join(npy_dir, os.path.basename(tile_path).replace('.tif', '.npy'))
        print(f"Converting {tile_path} to {npy_path}")
        np.save(npy_path, image)
        npy_paths.append(npy_path)
    return npy_paths


def merge_tiles(tiles: List[str], image_shape: Tuple[int, int, int], tile_size: int) -> np.ndarray:
    """
    Merge prediction tiles back into a full raster image.

    Reconstructs the full image from individual tile predictions by parsing
    tile positions from filenames (expected format: tile_{row}_{col}.tif).

    Args:
        tiles: List of paths to tile TIFF files.
        image_shape: Shape of the original image as (num_channels, height, width).
        tile_size: Size of each square tile in pixels.

    Returns:
        A 2D numpy array (height, width) containing the merged single-channel image.

    Note:
        Tiles must follow the naming convention 'tile_{i}_{j}.tif' where i and j
        are the row and column pixel offsets from the top-left corner.

    Example:
        >>> tiles = ["pred/tile_0_0.tif", "pred/tile_0_96.tif", "pred/tile_96_0.tif"]
        >>> image_shape = (3, 192, 192)
        >>> merged = merge_tiles(tiles, image_shape, tile_size=96)
        >>> print(merged.shape)
        (192, 192)
    """
    print(f"Merging tiles back into full image of shape {image_shape}")
    _, height, width = image_shape  # Assume image_shape is in the format (num_channels, height, width)
    full_image = np.zeros((height, width), dtype=np.uint8)  # Only one channel for the full image

    for tile_path in tiles:
        tile = tiff.imread(tile_path)
        # Adjust the parsing to match the filename pattern used in split_to_tiles
        filename = os.path.basename(tile_path)
        parts = filename.replace('tile_', '').replace('.tif', '').split('_')
        if len(parts) == 2:  # Ensure the filename format is correct
            i, j = map(int, parts)
            full_image[i:i+tile_size, j:j+tile_size] = tile  # Single channel tile assignment
        else:
            print(f"Unexpected filename format: {filename}")

    return full_image


def save_full_raster(
    predictions: List[str],
    image_shape: Tuple[int, int, int],
    tile_size: int,
    save_path: str
) -> None:
    """
    Merge prediction tiles and save as a full GeoTIFF raster.

    This function combines `merge_tiles` with saving the result to a GeoTIFF file.
    It merges all prediction tiles back into the original image dimensions and
    writes the result as a single-band raster.

    Args:
        predictions: List of paths to prediction tile TIFF files.
        image_shape: Shape of the original image as (num_channels, height, width).
        tile_size: Size of each square tile in pixels.
        save_path: Path where the merged GeoTIFF will be saved.

    Example:
        >>> predictions = ["pred/tile_0_0.tif", "pred/tile_0_96.tif"]
        >>> image_shape = (3, 192, 192)
        >>> save_full_raster(predictions, image_shape, tile_size=96, save_path="output/full_prediction.tif")
    """
    print(f"Saving full raster to {save_path}")
    full_raster = merge_tiles(predictions, image_shape, tile_size)
    with rasterio.open(save_path, 'w', driver='GTiff', height=full_raster.shape[0],
                       width=full_raster.shape[1], count=1, dtype=full_raster.dtype) as dst:
        dst.write(full_raster, 1)


def split_to_tiles(image: np.ndarray, tile_size: int, save_dir: str) -> List[str]:
    """
    Split a large image into square tiles and save them as TIFF files.

    Divides the input image into non-overlapping tiles of the specified size.
    Only complete tiles (matching tile_size exactly) are saved. Tiles are named
    using their top-left pixel coordinates: tile_{row}_{col}.tif

    Args:
        image: Input image array with shape (channels, height, width).
        tile_size: Size of each square tile in pixels.
        save_dir: Directory where tile TIFF files will be saved.

    Returns:
        List of paths to the saved tile TIFF files.

    Note:
        Partial tiles at the edges (smaller than tile_size) are discarded.

    Example:
        >>> image = np.random.rand(3, 384, 384)  # 3 channels, 384x384 pixels
        >>> tiles = split_to_tiles(image, tile_size=96, save_dir="tiles/")
        >>> print(len(tiles))  # 16 tiles (4x4 grid)
        16
    """
    print(f"Splitting image into tiles of size {tile_size}x{tile_size}")
    tiles = []
    num_channels, height, width = image.shape
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tile = image[:, i:i+tile_size, j:j+tile_size]
            if tile.shape[1] == tile_size and tile.shape[2] == tile_size:
                tile_path = os.path.join(save_dir, f'tile_{i}_{j}.tif')
                print(f"Saving tile to {tile_path}")
                tiff.imwrite(tile_path, tile)
                tiles.append(tile_path)
    
    print(f"Total number of tiles: {len(tiles)}")
    return tiles


def align_reference_with_crs(
    reference_tif: str,
    predicted_tif: str,
    predicted_transform: Affine,
    dst_crs: Union[CRS, str],
    output_path: str,
    resampling_method: Resampling = Resampling.nearest,
    num_threads: int = 2
) -> str:
    """
    Align a reference raster to match the spatial extent and CRS of a predicted raster.

    Reprojects the reference raster to match the dimensions, transform, and CRS
    of the predicted raster. Useful for comparing predictions with ground truth
    that may be in different coordinate systems or resolutions.

    Args:
        reference_tif: Path to the reference raster file (e.g., ground truth).
        predicted_tif: Path to the predicted raster file (used for dimensions).
        predicted_transform: Affine transform for the output raster.
        dst_crs: Destination coordinate reference system (CRS object or EPSG string).
        output_path: Path where the aligned raster will be saved.
        resampling_method: Resampling method to use (default: nearest neighbor).
        num_threads: Number of threads for reprojection (default: 2).

    Returns:
        Path to the saved aligned raster file.

    Raises:
        ValueError: If the reference raster has no CRS defined.

    Example:
        >>> from rasterio.crs import CRS
        >>> aligned_path = align_reference_with_crs(
        ...     reference_tif="ground_truth.tif",
        ...     predicted_tif="prediction.tif",
        ...     predicted_transform=transform,
        ...     dst_crs=CRS.from_epsg(32638),
        ...     output_path="aligned_ground_truth.tif"
        ... )
    """
    with rasterio.open(predicted_tif) as pred:
        dst_width = pred.width
        dst_height = pred.height
        dst_profile = pred.profile.copy()

    with rasterio.open(reference_tif) as ref:
        if ref.crs is None:
            raise ValueError("Reference raster has no CRS — cannot reproject.")
        src_dtype = ref.dtypes[0]
        dest = np.zeros((dst_height, dst_width), dtype=src_dtype)

        reproject(
            source=rasterio.band(ref, 1),
            destination=dest,
            src_transform=ref.transform,
            src_crs=ref.crs,
            dst_transform=predicted_transform,
            dst_crs=dst_crs,
            resampling=resampling_method,
            num_threads=num_threads
        )

    dst_profile.update(
        dtype=src_dtype,
        count=1,
        compress="lzw",
        driver="GTiff",
        transform=predicted_transform,
        crs=dst_crs,
        width=dst_width,
        height=dst_height
    )

    with rasterio.open(output_path, "w", **dst_profile) as dst:
        dst.write(dest, 1)

    return output_path


def align_reference(
    reference_tif: str,
    predicted_tif: str,
    predicted_transform: Affine,
    output_path: str,
    resampling_method: Resampling = Resampling.nearest,
    num_threads: int = 2
) -> str:
    """
    Align a reference raster to match a predicted raster, using the predicted raster's CRS.

    Similar to `align_reference_with_crs`, but automatically uses the CRS from the
    predicted raster file instead of requiring it as a parameter.

    Args:
        reference_tif: Path to the reference raster file (e.g., ground truth).
        predicted_tif: Path to the predicted raster file (used for dimensions and CRS).
        predicted_transform: Affine transform for the output raster.
        output_path: Path where the aligned raster will be saved.
        resampling_method: Resampling method to use (default: nearest neighbor).
        num_threads: Number of threads for reprojection (default: 2).

    Returns:
        Path to the saved aligned raster file.

    Raises:
        ValueError: If the reference raster has no CRS defined.

    Example:
        >>> aligned_path = align_reference(
        ...     reference_tif="ground_truth.tif",
        ...     predicted_tif="prediction.tif",
        ...     predicted_transform=transform,
        ...     output_path="aligned_ground_truth.tif"
        ... )
    """
    with rasterio.open(predicted_tif) as pred:
        dst_crs = pred.crs
        dst_width = pred.width
        dst_height = pred.height
        dst_profile = pred.profile.copy()

    with rasterio.open(reference_tif) as ref:
        if ref.crs is None:
            raise ValueError("Reference raster has no CRS — cannot reproject.")
        src_dtype = ref.dtypes[0]
        dest = np.zeros((dst_height, dst_width), dtype=src_dtype)

        reproject(
            source=rasterio.band(ref, 1),
            destination=dest,
            src_transform=ref.transform,
            src_crs=ref.crs,
            dst_transform=predicted_transform,
            dst_crs=dst_crs,
            resampling=resampling_method,
            num_threads=num_threads
        )

    dst_profile.update(
        dtype=src_dtype,
        count=1,
        compress="lzw",
        driver="GTiff",
        transform=predicted_transform,
        crs=dst_crs,
        width=dst_width,
        height=dst_height
    )

    with rasterio.open(output_path, "w", **dst_profile) as dst:
        dst.write(dest, 1)

    print(f"\nAligned reference raster saved to:\n{output_path}")
    return output_path


def reproject_raster(
    input_tif: str,
    output_path: str,
    dst_crs: Union[CRS, str, int] = 3857,
    src_crs: Union[CRS, str, int, None] = None,
    use_tfw: bool = False,
    resampling_method: Resampling = Resampling.nearest
) -> str:
    """
    Reproject a raster to a different coordinate reference system.

    Reads a raster file (optionally with its TFW world file for transform info)
    and reprojects it to the specified destination CRS.

    Args:
        input_tif: Path to the input raster file.
        output_path: Path where the reprojected raster will be saved.
        dst_crs: Destination CRS. Can be a CRS object, EPSG string (e.g., "EPSG:3857"),
            or EPSG code integer (e.g., 3857). Default is 3857 (Web Mercator).
        src_crs: Source CRS. If None, uses the CRS from the input file. Required if
            the input file has no embedded CRS.
        use_tfw: If True, reads the transform from the associated .tfw file using
            read_tfw_coordinates(). Default is False.
        resampling_method: Resampling method to use (default: nearest neighbor).

    Returns:
        Path to the saved reprojected raster file.

    Raises:
        ValueError: If no source CRS is available (neither in file nor provided).

    Example:
        >>> # Reproject to EPSG:3857 (Web Mercator)
        >>> output = reproject_raster("prediction.tif", "prediction_3857.tif", dst_crs=3857)
        
        >>> # Reproject with explicit source CRS
        >>> output = reproject_raster(
        ...     "prediction.tif",
        ...     "prediction_3857.tif",
        ...     dst_crs=3857,
        ...     src_crs=32638  # UTM zone 38N
        ... )
        
        >>> # Use TFW file for transform
        >>> output = reproject_raster(
        ...     "prediction.tif",
        ...     "prediction_3857.tif",
        ...     dst_crs="EPSG:3857",
        ...     use_tfw=True
        ... )
    """
    # Normalize input path to handle duplicates and inconsistent separators
    input_tif = os.path.normpath(os.path.abspath(input_tif))
    output_path = os.path.normpath(os.path.abspath(output_path))
    
    # Convert dst_crs to CRS object if needed
    if isinstance(dst_crs, int):
        dst_crs = CRS.from_epsg(dst_crs)
    elif isinstance(dst_crs, str):
        dst_crs = CRS.from_string(dst_crs)

    # Read TFW transform if requested
    tfw_transform = None
    if use_tfw:
        tfw_transform = read_tfw_coordinates(input_tif)

    with rasterio.open(input_tif) as src:
        # Determine source CRS
        if src_crs is not None:
            if isinstance(src_crs, int):
                source_crs = CRS.from_epsg(src_crs)
            elif isinstance(src_crs, str):
                source_crs = CRS.from_string(src_crs)
            else:
                source_crs = src_crs
        elif src.crs is not None:
            source_crs = src.crs
        else:
            raise ValueError("No source CRS available. Please provide src_crs parameter.")

        # Use TFW transform or source transform
        source_transform = tfw_transform if tfw_transform is not None else src.transform

        # Calculate the transform and dimensions for the destination
        dst_transform, dst_width, dst_height = calculate_default_transform(
            source_crs,
            dst_crs,
            src.width,
            src.height,
            *rasterio.transform.array_bounds(src.height, src.width, source_transform)
        )

        # Update profile for the output
        dst_profile = src.profile.copy()
        dst_profile.update(
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height
        )

        # Create output and reproject each band
        with rasterio.open(output_path, "w", **dst_profile) as dst:
            for band in range(1, src.count + 1):
                # Read source data as array
                src_array = src.read(band)
                dest_array = np.zeros((dst_height, dst_width), dtype=src.dtypes[band - 1])
                
                reproject(
                    source=src_array,
                    destination=dest_array,
                    src_transform=source_transform,
                    src_crs=source_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method
                )
                dst.write(dest_array, band)

    print(f"Reprojected raster saved to: {output_path}")
    return output_path
