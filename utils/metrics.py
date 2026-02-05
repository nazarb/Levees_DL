from typing import Dict, Optional

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion

from typing import Dict, Optional, Union
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString


def compare_vectors(
    reference_path: str,
    predicted_path: str,
    buffer_distance: float,
    output_path: Optional[str] = None,
    crs: Optional[Union[str, int]] = None
) -> Dict[str, float]:
    """
    Compare predicted vector with reference vector and compute evaluation metrics.
    
    Performs a buffered comparison between reference and predicted geometries
    using geometric buffer operations. Computes precision, recall, F1 score, 
    and IoU metrics based on area overlap.
    
    The comparison output contains polygons with the following classification:
    - 1: True Positive (TP) - predicted area within buffered reference
    - -1: False Positive (FP) - predicted area outside buffered reference
    - 2: False Negative (FN) - buffered reference area not covered by prediction
    
    Args:
        reference_path: Path to the reference (ground truth) vector file.
        predicted_path: Path to the predicted vector file.
        buffer_distance: Buffer distance in CRS units (meters for projected CRS).
            Applied to reference geometry for tolerance matching.
        output_path: Optional path for comparison vector output. If None, generates
            path based on predicted_path with "_comparison_buf{buffer_distance}" suffix.
        crs: Optional CRS to reproject both datasets to before comparison.
            Recommended to use a projected CRS (e.g., EPSG:3857) for accurate
            area calculations. If None, uses reference CRS.
    
    Returns:
        Dictionary containing evaluation metrics:
        - precision: TP_area / (TP_area + FP_area)
        - recall: TP_area / (TP_area + FN_area)
        - f1_score: 2 * (precision * recall) / (precision + recall)
        - iou: TP_area / (TP_area + FP_area + FN_area)
        - tp_area: Total true positive area
        - fp_area: Total false positive area
        - fn_area: Total false negative area
    
    Raises:
        ValueError: If either input file is empty or has no valid geometry.
    
    Example:
        >>> metrics = compare_vectors(
        ...     reference_path="ground_truth.gpkg",
        ...     predicted_path="prediction.gpkg",
        ...     buffer_distance=10,  # 10 meter tolerance
        ...     crs=3857
        ... )
        >>> print(f"IoU: {metrics['iou']:.4f}")
    """
    # Load vector files
    ref = gpd.read_file(reference_path)
    pred = gpd.read_file(predicted_path)
    
    if len(ref) == 0 or ref.geometry.is_empty.all():
        raise ValueError("Reference vector file is empty or has no valid geometry.")
    if len(pred) == 0 or pred.geometry.is_empty.all():
        raise ValueError("Predicted vector file is empty or has no valid geometry.")
    
    # Reproject to common CRS
    target_crs = crs if crs else ref.crs
    ref = ref.to_crs(target_crs)
    pred = pred.to_crs(target_crs)
    
    # Dissolve to single geometries
    ref_union = unary_union(ref.geometry)
    pred_union = unary_union(pred.geometry)
    
    # Buffer reference (equivalent to dilation in raster)
    buffered_ref = ref_union.buffer(buffer_distance)
    
    # Compute TP, FP, FN geometries
    tp_geom = pred_union.intersection(buffered_ref)
    fp_geom = pred_union.difference(buffered_ref)
    fn_geom = buffered_ref.difference(pred_union)
    
    # Calculate areas
    tp_area = tp_geom.area if not tp_geom.is_empty else 0
    fp_area = fp_geom.area if not fp_geom.is_empty else 0
    fn_area = fn_geom.area if not fn_geom.is_empty else 0
    
    # Compute metrics
    precision = tp_area / (tp_area + fp_area) if (tp_area + fp_area) > 0 else 0
    recall = tp_area / (tp_area + fn_area) if (tp_area + fn_area) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp_area / (tp_area + fp_area + fn_area) if (tp_area + fp_area + fn_area) > 0 else 0
    
    print(f"\n--- Evaluation Metrics (Vector) ---")
    print(f"Buffer distance: {buffer_distance} units")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1_score:.4f}")
    print(f"IoU:        {iou:.4f}")
    print(f"\nAreas:")
    print(f"  TP: {tp_area:,.2f} sq units")
    print(f"  FP: {fp_area:,.2f} sq units")
    print(f"  FN: {fn_area:,.2f} sq units")
    
    # Build comparison GeoDataFrame
    comparison_records = []
    
    if not tp_geom.is_empty:
        comparison_records.append({
            "geometry": tp_geom,
            "class": 1,
            "label": "TP",
            "area": tp_area
        })
    if not fp_geom.is_empty:
        comparison_records.append({
            "geometry": fp_geom,
            "class": -1,
            "label": "FP",
            "area": fp_area
        })
    if not fn_geom.is_empty:
        comparison_records.append({
            "geometry": fn_geom,
            "class": 2,
            "label": "FN",
            "area": fn_area
        })
    
    comparison_gdf = gpd.GeoDataFrame(comparison_records, crs=target_crs)
    
    # Determine output path
    if output_path is None:
        pred_path = Path(predicted_path)
        output_path = str(pred_path.parent / f"{pred_path.stem}_comparison_buf{buffer_distance}.gpkg")
    
    # Handle fid column conflict for GPKG
    if 'fid' in comparison_gdf.columns:
        comparison_gdf = comparison_gdf.rename(columns={'fid': 'orig_fid'})
    
    comparison_gdf.to_file(output_path, driver="GPKG")
    print(f"\nComparison vector saved to:\n{output_path}")
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "iou": float(iou),
        "tp_area": float(tp_area),
        "fp_area": float(fp_area),
        "fn_area": float(fn_area)
    }
    

def compare_rasters(
    reference_tif: str,
    predicted_tif: str,
    IoU_buf: int,
    output_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Compare predicted raster with reference raster and compute evaluation metrics.

    Performs a buffered comparison between reference and predicted binary rasters
    using morphological operations (dilation on reference, erosion on prediction).
    Computes precision, recall, F1 score, and IoU metrics.

    The comparison raster output uses the following encoding:
    - 1: True Positive (TP)
    - -1: False Positive (FP)
    - 2: False Negative (FN)
    - 0: True Negative (background)

    Args:
        reference_tif: Path to the reference (ground truth) raster file.
        predicted_tif: Path to the predicted raster file.
        IoU_buf: Buffer size for morphological operations (dilation/erosion kernel size).
        output_path: Optional path for comparison raster output. If None, generates
            path based on predicted_tif with "_comparison_buf{IoU_buf}.tif" suffix.

    Returns:
        Dictionary containing evaluation metrics:
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1_score: 2 * (precision * recall) / (precision + recall)
        - iou: TP / (TP + FP + FN)

    Raises:
        ValueError: If rasters have different dimensions.

    Example:
        >>> metrics = compare_rasters(
        ...     reference_tif="ground_truth.tif",
        ...     predicted_tif="prediction.tif",
        ...     IoU_buf=4
        ... )
        >>> print(f"IoU: {metrics['iou']:.4f}")
    """
    with rasterio.open(reference_tif) as ref_src:
        ref_data = ref_src.read(1)
        profile = ref_src.profile

    with rasterio.open(predicted_tif) as pred_src:
        pred_data = pred_src.read(1)

    if ref_data.shape != pred_data.shape:
        raise ValueError("Rasters must have the same dimensions and alignment.")

    struct_element = np.ones((IoU_buf, IoU_buf))
    buffered_ref = binary_dilation(ref_data, structure=struct_element).astype(ref_data.dtype)

    TP = (buffered_ref == 1) & (pred_data == 1)
    FP = (buffered_ref == 0) & (pred_data == 1)
    FN = (buffered_ref == 1) & (pred_data == 0)

    output_data = np.zeros_like(pred_data, dtype=np.int8)
    output_data[TP] = 1
    output_data[FP] = -1
    output_data[FN] = 2

    tp_count = np.sum(TP)
    fp_count = np.sum(FP)
    fn_count = np.sum(FN)

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp_count / (tp_count + fp_count + fn_count) if (tp_count + fp_count + fn_count) > 0 else 0

    print(f"\n--- Evaluation Metrics ---")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1_score:.4f}")
    print(f"IoU:        {iou:.4f}")

    # Determine output path
    if output_path is None:
        output_path = predicted_tif.replace(".tif", f"_comparison_buf{IoU_buf}.tif")
    
    profile.update(dtype=rasterio.int8, count=1)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_data, 1)

    print(f"Comparison raster saved to:\n{output_path}")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "iou": float(iou)
    }


def segmentize_line(geom, segment_length):
    """
    Split a line into segments of approximate length.
    
    Parameters
    ----------
    geom : shapely.geometry.LineString or MultiLineString
        The geometry to segmentize
    segment_length : float
        Target length for each segment in CRS units
    
    Returns
    -------
    list
        List of LineString segments
    """
    if geom.is_empty:
        return []
    
    if geom.geom_type == 'MultiLineString':
        segments = []
        for line in geom.geoms:
            segments.extend(segmentize_line(line, segment_length))
        return segments
    
    if geom.geom_type != 'LineString':
        return []
    
    total_length = geom.length
    if total_length <= segment_length:
        return [geom]
    
    num_segments = int(np.ceil(total_length / segment_length))
    segments = []
    
    for i in range(num_segments):
        start_frac = i / num_segments
        end_frac = (i + 1) / num_segments
        
        start_pt = geom.interpolate(start_frac, normalized=True)
        end_pt = geom.interpolate(end_frac, normalized=True)
        
        # Extract segment between two points
        start_dist = start_frac * total_length
        end_dist = end_frac * total_length
        
        coords = list(geom.coords)
        segment_coords = [start_pt.coords[0]]
        
        cumulative = 0
        for j in range(len(coords) - 1):
            p1, p2 = coords[j], coords[j + 1]
            seg_len = LineString([p1, p2]).length
            
            if cumulative + seg_len > start_dist and cumulative < end_dist:
                if cumulative >= start_dist:
                    segment_coords.append(p1)
                if cumulative + seg_len <= end_dist:
                    segment_coords.append(p2)
            
            cumulative += seg_len
        
        segment_coords.append(end_pt.coords[0])
        
        if len(segment_coords) >= 2:
            seg = LineString(segment_coords)
            if seg.length > 0:
                segments.append(seg)
    
    return segments


def compare_line_segments(
    reference_path: Union[str, gpd.GeoDataFrame],
    predicted_path: Union[str, gpd.GeoDataFrame],
    segment_length: float,
    buffer_distance: float,
    output_path: Optional[str] = None,
    crs: Optional[Union[str, int]] = None,
    reference_layer: Optional[str] = None,
    predicted_layer: Optional[str] = None
) -> Dict[str, float]:
    """
    Compare predicted line segments with reference line segments using length-based metrics.
    
    This function segments both reference and predicted lines into smaller segments,
    then classifies each segment as True Positive (TP), False Positive (FP), or 
    False Negative (FN) based on buffer distance matching. Metrics are computed
    based on segment lengths rather than areas, making it suitable for linear 
    features like levees, channels, or road networks.
    
    The comparison output contains line segments with the following classification:
    - 1: True Positive (TP) - predicted segment within buffered reference
    - -1: False Positive (FP) - predicted segment outside buffered reference
    - 2: False Negative (FN) - reference segment not detected by prediction
    
    Parameters
    ----------
    reference_path : str or GeoDataFrame
        Path to the reference (ground truth) vector file, or a GeoDataFrame.
        If a file path, can be any format supported by geopandas (GPKG, GeoJSON, etc.).
    predicted_path : str or GeoDataFrame
        Path to the predicted vector file, or a GeoDataFrame.
        If a file path and GPKG format, specify layer name if needed.
    segment_length : float
        Target length for segmenting lines, in CRS units (meters for projected CRS).
        Lines are split into segments of approximately this length for comparison.
    buffer_distance : float
        Buffer distance in CRS units (meters for projected CRS).
        Used to determine if predicted segments match reference segments.
    output_path : str, optional
        Path for comparison vector output. If None, generates path based on 
        predicted_path with "_comp_segm_{segment_length}_dist_{buffer_distance}" suffix.
    crs : str or int, optional
        CRS to reproject both datasets to before comparison.
        Recommended to use a projected CRS (e.g., EPSG:3857) for accurate
        length calculations. If None, uses reference CRS.
    reference_layer : str, optional
        Layer name if reference_path is a GPKG file with multiple layers.
        Defaults to first layer.
    predicted_layer : str, optional
        Layer name if predicted_path is a GPKG file with multiple layers.
        Defaults to 'edges' layer if it exists, otherwise first layer.
    
    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - precision: TP_length / (TP_length + FP_length)
        - recall: TP_length / (TP_length + FN_length)
        - f1_score: 2 * (precision * recall) / (precision + recall)
        - iou: TP_length / (TP_length + FP_length + FN_length)
        - tp_length: Total true positive length
        - fp_length: Total false positive length
        - fn_length: Total false negative length
        - tp_count: Number of TP segments
        - fp_count: Number of FP segments
        - fn_count: Number of FN segments
    
    Raises
    ------
    ValueError
        If either input file is empty or has no valid geometry.
    
    Example
    -------
    >>> from utils import compare_line_segments
    >>> metrics = compare_line_segments(
    ...     reference_path="ground_truth.gpkg",
    ...     predicted_path="prediction.gpkg",
    ...     segment_length=250,  # 250 meter segments
    ...     buffer_distance=250,  # 250 meter tolerance
    ...     crs=3857
    ... )
    >>> print(f"F1 Score: {metrics['f1_score']:.4f}")
    >>> print(f"Total TP length: {metrics['tp_length']/1000:.2f} km")
    """
    # Load vector files
    if isinstance(reference_path, gpd.GeoDataFrame):
        ref = reference_path.copy()
    else:
        ref = gpd.read_file(reference_path, layer=reference_layer)
    
    if isinstance(predicted_path, gpd.GeoDataFrame):
        pred = predicted_path.copy()
    else:
        # Try 'edges' layer first (common for network outputs), fallback to default
        try:
            pred = gpd.read_file(predicted_path, layer=predicted_layer or 'edges')
        except (ValueError, KeyError):
            pred = gpd.read_file(predicted_path, layer=predicted_layer)
    
    if len(ref) == 0 or ref.geometry.is_empty.all():
        raise ValueError("Reference vector file is empty or has no valid geometry.")
    if len(pred) == 0 or pred.geometry.is_empty.all():
        raise ValueError("Predicted vector file is empty or has no valid geometry.")
    
    # Reproject to common CRS
    target_crs = crs if crs else ref.crs
    ref = ref.to_crs(epsg=target_crs if isinstance(target_crs, int) else target_crs)
    pred = pred.to_crs(epsg=target_crs if isinstance(target_crs, int) else target_crs)
    
    # Segment reference lines
    print(f"Segmenting reference lines into {segment_length}m sections...")
    ref_segments = []
    for idx, row in ref.iterrows():
        segs = segmentize_line(row.geometry, segment_length)
        for seg in segs:
            ref_segments.append({'geometry': seg, 'orig_idx': idx})
    
    ref_segmented = gpd.GeoDataFrame(ref_segments, crs=ref.crs)
    print(f"Reference: {len(ref)} lines → {len(ref_segmented)} segments")
    
    # Segment predicted lines
    print(f"Segmenting predicted lines into {segment_length}m sections...")
    pred_segments = []
    for idx, row in pred.iterrows():
        segs = segmentize_line(row.geometry, segment_length)
        for seg in segs:
            pred_segments.append({'geometry': seg, 'orig_idx': idx})
    
    pred_segmented = gpd.GeoDataFrame(pred_segments, crs=pred.crs)
    print(f"Predicted: {len(pred)} lines → {len(pred_segmented)} segments")
    
    # Create buffered geometries for matching
    ref_union = unary_union(ref_segmented.geometry)
    ref_buffered = ref_union.buffer(buffer_distance)
    
    pred_union = unary_union(pred_segmented.geometry)
    pred_buffered = pred_union.buffer(buffer_distance)
    
    # Classify predicted segments: TP or FP
    pred_segmented['near_ref'] = pred_segmented.geometry.intersects(ref_buffered)
    pred_segmented['class'] = pred_segmented['near_ref'].map({True: 1, False: -1})
    pred_segmented['label'] = pred_segmented['near_ref'].map({True: 'TP', False: 'FP'})
    
    # Classify reference segments: TP or FN
    ref_segmented['detected'] = ref_segmented.geometry.intersects(pred_buffered)
    ref_segmented['class'] = ref_segmented['detected'].map({True: 1, False: 2})
    ref_segmented['label'] = ref_segmented['detected'].map({True: 'TP', False: 'FN'})
    
    # Get only FN reference segments
    ref_fn = ref_segmented[ref_segmented['class'] == 2].copy()
    
    # Combine
    comparison_lines = gpd.GeoDataFrame(
        pd.concat([
            pred_segmented[['geometry', 'class', 'label']],
            ref_fn[['geometry', 'class', 'label']]
        ], ignore_index=True),
        crs=pred.crs
    )
    
    comparison_lines['length'] = comparison_lines.geometry.length
    
    # Determine output path
    if output_path is None:
        if isinstance(predicted_path, str):
            pred_path = Path(predicted_path)
            output_path = str(pred_path.parent / f"{pred_path.stem}_comp_segm_{segment_length}_dist_{buffer_distance}.gpkg")
        else:
            output_path = f"comparison_segm_{segment_length}_dist_{buffer_distance}.gpkg"
    
    # Handle fid column conflict for GPKG
    if 'fid' in comparison_lines.columns:
        comparison_lines = comparison_lines.rename(columns={'fid': 'orig_fid'})
    
    comparison_lines.to_file(output_path, driver="GPKG")
    
    # Calculate metrics
    tp_length = comparison_lines[comparison_lines['class'] == 1]['length'].sum()
    fp_length = comparison_lines[comparison_lines['class'] == -1]['length'].sum()
    fn_length = comparison_lines[comparison_lines['class'] == 2]['length'].sum()
    
    tp_count = len(comparison_lines[comparison_lines['class'] == 1])
    fp_count = len(comparison_lines[comparison_lines['class'] == -1])
    fn_count = len(comparison_lines[comparison_lines['class'] == 2])
    
    precision = tp_length / (tp_length + fp_length) if (tp_length + fp_length) > 0 else 0
    recall = tp_length / (tp_length + fn_length) if (tp_length + fn_length) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp_length / (tp_length + fp_length + fn_length) if (tp_length + fp_length + fn_length) > 0 else 0
    
    print(f"\n--- Line Comparison Summary ({segment_length}m segments) ---")
    for cls, label in [(1, 'TP'), (-1, 'FP'), (2, 'FN')]:
        subset = comparison_lines[comparison_lines['class'] == cls]
        total_length = subset['length'].sum()
        print(f"{label}: {len(subset)} segments, {total_length/1000:,.2f} km")
    
    print(f"\n--- Evaluation Metrics (Length-based) ---")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1_score:.4f}")
    print(f"IoU:        {iou:.4f}")
    
    print(f"\nSaved to: {output_path}")
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "iou": float(iou),
        "tp_length": float(tp_length),
        "fp_length": float(fp_length),
        "fn_length": float(fn_length),
        "tp_count": int(tp_count),
        "fp_count": int(fp_count),
        "fn_count": int(fn_count)
    }
