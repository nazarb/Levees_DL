"""
Raster to Network Utilities

Skeletonise a binary channel raster (multi-pixel thickness -> 1-pixel centreline)
and convert it to a network graph with gap bridging, snapping, and analysis features.
"""

import os
import math
import numpy as np
import rasterio
import networkx as nx

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

try:
    from scipy.ndimage import label
except ImportError:
    label = None

try:
    import sknw
except ImportError:
    sknw = None

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.strtree import STRtree
    from shapely.ops import substring, nearest_points
except ImportError:
    gpd = None
    Point = None
    LineString = None
    STRtree = None
    substring = None
    nearest_points = None


# ============================================================================
# SKELETONIZATION
# ============================================================================

def skeletonize_raster(in_raster, out_raster, binarize=None):
    """
    Skeletonise a binary channel raster (multi-pixel thickness -> 1-pixel centreline).
    
    Output is a georeferenced uint8 raster: 1 = skeleton, 0 = background 
    (and preserves nodata if present).
    
    Parameters
    ----------
    in_raster : str
        Path to input binary raster
    out_raster : str
        Path to output skeleton raster
    binarize : callable, optional
        Function to binarize the input array. Default: lambda arr: (arr > 0)
        Examples:
        - if raster is 0/1 -> (arr == 1)
        - if raster is probability 0..1 -> (arr >= 0.5)
        - if raster is 0/255 -> (arr > 0)
    
    Returns
    -------
    str
        Path to output skeleton raster
    """
    if skeletonize is None:
        raise ImportError(
            "This function requires scikit-image. "
            "Install with: pip install scikit-image (or conda install scikit-image)"
        )
    
    if binarize is None:
        binarize = lambda arr: (arr > 0)
    
    with rasterio.open(in_raster) as src:
        prof = src.profile.copy()
        nodata = src.nodata
        arr = src.read(1, masked=True)  # masked array if nodata exists
    
    # Build boolean mask for channels, respecting nodata
    binary = np.zeros(arr.shape, dtype=bool)
    valid = ~arr.mask if np.ma.isMaskedArray(arr) else np.ones(arr.shape, dtype=bool)
    binary[valid] = binarize(np.asarray(arr)[valid])
    
    # Skeletonise: 1-pixel-wide centreline
    skel = skeletonize(binary).astype(np.uint8)  # 0/1
    
    # Optionally restore nodata (so nodata isn't forced to 0)
    if nodata is not None and np.any(~valid):
        skel = skel.astype(np.uint8)
        # Keep as uint8 for GIS friendliness; nodata will be set in metadata below
        skel[~valid] = 0  # keep nodata pixels as 0 in data; nodata flag handles masking in GIS
    
    # Write output
    prof.update(dtype=rasterio.uint8, count=1, compress="deflate", nodata=0 if nodata is None else nodata)
    with rasterio.open(out_raster, "w", **prof) as dst:
        dst.write(skel, 1)
    
    return out_raster


# ============================================================================
# CONNECTED COMPONENT FILTERING
# ============================================================================

def filter_connected_components(in_raster, out_raster, min_pixels=50, connectivity=8, binarize=None):
    """
    Connected-pixel filtering of a binary channel raster (remove small components).
    
    - Reads a binary raster (0/1 or 0/255 etc.)
    - Labels connected components
    - Removes components with fewer than `min_pixels` pixels
    - Writes a filtered binary raster with the same georeferencing
    
    Parameters
    ----------
    in_raster : str
        Path to input binary raster
    out_raster : str
        Path to output filtered raster
    min_pixels : int, default=50
        Remove components smaller than this (in pixels)
    connectivity : int, default=8
        4 or 8 (8 is usually better for channel networks)
    binarize : callable, optional
        Function to binarize the input array. Default: lambda arr: (arr > 0)
    
    Returns
    -------
    str
        Path to output filtered raster
    """
    if label is None:
        raise ImportError(
            "This function requires scipy. "
            "Install it with: pip install scipy (or conda install scipy)"
        )
    
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    
    if binarize is None:
        binarize = lambda arr: (arr > 0)
    
    # Structuring element defines pixel connectivity
    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        if connectivity == 4
        else np.ones((3, 3), dtype=np.uint8)
    )
    
    with rasterio.open(in_raster) as src:
        prof = src.profile.copy()
        nodata = src.nodata
        
        band = src.read(1, masked=True)  # masked array if nodata exists
        # Build boolean binary mask for channels, respecting nodata
        binary = np.zeros(band.shape, dtype=bool)
        valid = ~band.mask if np.ma.isMaskedArray(band) else np.ones(band.shape, dtype=bool)
        binary[valid] = binarize(np.asarray(band)[valid])
        
        # Label connected components (only on True pixels)
        labels, nlab = label(binary, structure=structure)
        
        if nlab == 0:
            # Nothing to filter; just write an empty binary raster (or original)
            filtered = binary.astype(np.uint8)
        else:
            # Component sizes (labels start at 1; label 0 is background)
            sizes = np.bincount(labels.ravel())
            keep = sizes >= min_pixels
            keep[0] = False  # never keep background
            
            filtered = keep[labels].astype(np.uint8)
        
        # If you want to preserve nodata (rather than forcing nodata to 0), apply it back:
        if nodata is not None and np.any(~valid):
            filtered = filtered.astype(np.uint8)
            # set nodata pixels to nodata value (commonly 0, but not always)
            filtered = (
                filtered.astype(np.float32)
                if prof["dtype"] in ("float32", "float64")
                else filtered
            )
            filtered[~valid] = nodata
        
        # Write output: keep it binary uint8 unless you explicitly need another dtype
        prof.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0 if nodata is None else nodata,
            compress="deflate",
        )
        
        with rasterio.open(out_raster, "w", **prof) as dst:
            dst.write(filtered.astype(np.uint8), 1)
    
    return out_raster


# ============================================================================
# NUMERIC HELPERS
# ============================================================================

def angle_deg(v1, v2):
    """Calculate angle between two vectors in degrees."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def angle_diff_180(a_deg, b_deg):
    """Smallest angular difference modulo 180 degrees (for undirected parallel comparison)."""
    d = abs((a_deg - b_deg) % 180.0)
    return min(d, 180.0 - d)


def bearing_cw_from_north(p0, p1):
    """Bearing degrees clockwise from North for (x,y) with +y = North."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    ang = np.degrees(np.arctan2(dx, dy))  # atan2(x,y) => cw from north
    return float((ang + 360.0) % 360.0)


def convert_slope_dir_to_cw_from_north(val_deg, convention):
    """Convert raster direction to cw-from-north degrees."""
    if not np.isfinite(val_deg):
        return np.nan
    v = float(val_deg)
    if convention == "cw_from_north":
        return v % 360.0
    elif convention == "ccw_from_east":
        # 0=east ccw -> 90=north; convert to cw-from-north: cw = (90 - ccw_east) mod 360
        return float((90.0 - v) % 360.0)
    else:
        raise ValueError("slope_dir_convention must be 'cw_from_north' or 'ccw_from_east'")


def pixel_size_from_transform(transform):
    """Calculate average pixel size from transform."""
    px = np.hypot(transform.a, transform.b)
    py = np.hypot(transform.d, transform.e)
    return float((px + py) / 2.0)


# ============================================================================
# RASTER/GEOMETRY HELPERS
# ============================================================================

def pix_to_xy(transform, rc):
    """Convert pixel coordinates (row, col) to geographic coordinates (x, y)."""
    rows = rc[:, 0].astype(int)
    cols = rc[:, 1].astype(int)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.column_stack([np.asarray(xs), np.asarray(ys)])


def node_xy(G, n, transform):
    """Get geographic coordinates of a node from NetworkX graph."""
    rc = np.array(G.nodes[n]["o"], dtype=float).reshape(1, 2)
    return pix_to_xy(transform, rc)[0]


def snap_edge_endpoints_to_nodes(G, u, v, k, transform):
    """
    Edge coords oriented u->v, and endpoints snapped EXACTLY to node coords.
    """
    pts_rc = G.edges[u, v, k]["pts"]
    xy = pix_to_xy(transform, pts_rc)
    
    xu = node_xy(G, u, transform)
    xv = node_xy(G, v, transform)
    
    # orient so start closer to u
    if np.linalg.norm(xy[0] - xu) <= np.linalg.norm(xy[-1] - xu):
        xy_oriented = xy.copy()
    else:
        xy_oriented = xy[::-1].copy()
    
    xy_oriented[0] = xu
    xy_oriented[-1] = xv
    return xy_oriented


def chaikin_smooth(coords, n_iter=2):
    """Chaikin smoothing, keeping endpoints."""
    if n_iter <= 0 or len(coords) < 3:
        return coords
    out = coords
    for _ in range(n_iter):
        P = out
        Q = 0.75 * P[:-1] + 0.25 * P[1:]
        R = 0.25 * P[:-1] + 0.75 * P[1:]
        out2 = np.vstack([P[0], np.column_stack([Q, R]).reshape(-1, 2), P[-1]])
        _, idx = np.unique(out2, axis=0, return_index=True)
        out = out2[np.sort(idx)]
        if len(out) < 3:
            break
    return out


def simplify_coords(coords, tol):
    """Shapely simplify; preserve endpoints."""
    if tol is None or tol <= 0 or len(coords) < 3:
        return coords
    ls = LineString(coords)
    ls2 = ls.simplify(float(tol), preserve_topology=True)
    c = np.asarray(ls2.coords)
    c[0] = coords[0]
    c[-1] = coords[-1]
    return c


def split_linestring_at_point(line, pt, eps=1e-9):
    """
    Split LineString into two at projection of pt onto line.
    Returns (seg1, seg2, at_endpoint_bool).
    """
    if line.is_empty or line.length == 0:
        return line, None, True
    d = float(line.project(pt))
    L = float(line.length)
    if d <= eps or d >= L - eps:
        return line, None, True
    seg1 = substring(line, 0.0, d)
    seg2 = substring(line, d, L)
    return seg1, seg2, False


def endpoint_direction_from_linestring(endpoint_xy, line_coords, step=3):
    """
    line_coords is oriented so that coords[0] is endpoint (or very close).
    returns vector from endpoint into the line interior.
    """
    if len(line_coords) < 2:
        return np.array([0.0, 0.0])
    i1 = min(int(step), len(line_coords) - 1)
    return np.asarray(line_coords[i1], float) - np.asarray(line_coords[0], float)


def raster_value_at_xy(src, xy):
    """Get raster value at geographic coordinates."""
    try:
        val = next(src.sample([(float(xy[0]), float(xy[1]))]))[0]
    except Exception:
        return np.nan
    if val is None:
        return np.nan
    try:
        if np.ma.isMaskedArray(val) and val.mask:
            return np.nan
    except Exception:
        pass
    return float(val)


def cost_sum_along_line(line, cost_arr, transform, nodata, step, nodata_to_nan=True, unique_cells=True):
    """
    Approximate sum of raster cost along a line.
    - If unique_cells=True: sample points, map to cells, sum unique traversed cells.
    - Else: sum values at sample points (can double count cells).
    """
    if line.is_empty or line.length == 0:
        return 0.0
    
    L = float(line.length)
    n = max(2, int(np.ceil(L / float(step))) + 1)
    ds = np.linspace(0.0, L, n)  # keep deterministic
    
    pts = [line.interpolate(float(d)) for d in ds]
    xs = np.array([p.x for p in pts], dtype=float)
    ys = np.array([p.y for p in pts], dtype=float)
    
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rc = np.column_stack([rows, cols]).astype(int)
    
    h, w = cost_arr.shape
    ok = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
    rc = rc[ok]
    if rc.size == 0:
        return 0.0
    
    if unique_cells:
        rc = np.unique(rc, axis=0)
    
    vals = cost_arr[rc[:, 0], rc[:, 1]].astype(float)
    
    if nodata is not None:
        m = vals == nodata
        if m.any():
            if nodata_to_nan:
                vals[m] = np.nan
            else:
                vals[m] = 0.0
    
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.sum(vals))


# ============================================================================
# NETWORK BUILDING
# ============================================================================

def build_nodes_edges_gdfs(G, transform, crs, simplify_tol, smooth_iters, smooth_then_simplify=True):
    """Build GeoDataFrames of nodes and edges from NetworkX graph."""
    # nodes
    node_rows = []
    node_xy_cache = {}
    for n in G.nodes:
        xy = node_xy(G, n, transform)
        node_xy_cache[n] = xy
        node_rows.append(
            {
                "node": int(n),
                "degree": int(G.degree[n]),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "geometry": Point(xy),
            }
        )
    nodes_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=crs)
    
    # edges
    edge_rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        xy = snap_edge_endpoints_to_nodes(G, u, v, k, transform)
        
        xy = simplify_coords(xy, simplify_tol)
        if smooth_iters > 0:
            xy = chaikin_smooth(xy, n_iter=smooth_iters)
            if smooth_then_simplify:
                xy = simplify_coords(xy, simplify_tol)
        
        geom = LineString(xy)
        edge_rows.append(
            {
                "u": int(u),
                "v": int(v),
                "k": int(k),
                "n_vert": int(len(xy)),
                "length": float(geom.length),
                "is_bridge": bool(data.get("is_bridge", False)),
                "bridge_type": str(data.get("bridge_type", ""))
                if data.get("is_bridge", False)
                else "",
                "geometry": geom,
            }
        )
    edges_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=crs)
    return nodes_gdf, edges_gdf


def contract_degree2_graph(G, nodes_gdf, edges_gdf):
    """
    Contract degree-2 nodes by merging chains into single edges between 'kept' nodes.
    """
    node_geom = {int(r.node): r.geometry for r in nodes_gdf.itertuples(index=False)}
    node_xy_ = {
        int(r.node): np.array([float(r.x), float(r.y)])
        for r in nodes_gdf.itertuples(index=False)
    }
    deg = dict(G.degree)
    
    keep = {int(n) for n in G.nodes if deg[n] != 2}
    
    # pure cycles: keep one node per all-degree-2 component
    for comp in nx.connected_components(nx.Graph(G)):
        comp = {int(n) for n in comp}
        if comp and all(deg[n] == 2 for n in comp):
            keep.add(min(comp))
    
    coords_map = {}
    for r in edges_gdf.itertuples(index=False):
        coords_map[(int(r.u), int(r.v), int(r.k))] = np.asarray(
            r.geometry.coords, dtype=float
        )
    
    def norm_eid(a, b, k):
        a = int(a)
        b = int(b)
        k = int(k)
        return (a, b, k) if a <= b else (b, a, k)
    
    def get_seg_coords(a, b, k):
        a = int(a)
        b = int(b)
        k = int(k)
        if (a, b, k) in coords_map:
            return coords_map[(a, b, k)]
        if (b, a, k) in coords_map:
            return coords_map[(b, a, k)][::-1]
        return np.vstack([node_xy_[a], node_xy_[b]])
    
    visited = set()
    out_edges = []
    
    for s in sorted(keep):
        for nbr, keydict in G[s].items():
            for k in keydict.keys():
                eid = norm_eid(s, nbr, k)
                if eid in visited:
                    continue
                
                prev = int(s)
                curr = int(nbr)
                kk = int(k)
                
                chain_coords = list(get_seg_coords(prev, curr, kk))
                visited.add(eid)
                chain_n_segs = 1
                
                while curr not in keep and deg[curr] == 2:
                    found = False
                    for nbr2, keydict2 in G[curr].items():
                        nbr2 = int(nbr2)
                        if nbr2 == prev:
                            continue
                        for k2 in keydict2.keys():
                            k2 = int(k2)
                            eid2 = norm_eid(curr, nbr2, k2)
                            if eid2 in visited:
                                continue
                            seg = list(get_seg_coords(curr, nbr2, k2))
                            chain_coords.extend(seg[1:])
                            visited.add(eid2)
                            chain_n_segs += 1
                            prev, curr = curr, nbr2
                            found = True
                            break
                        if found:
                            break
                    if not found:
                        break
                
                t = int(curr)
                if len(chain_coords) >= 2:
                    geom = LineString(chain_coords)
                    out_edges.append(
                        {
                            "u": int(s),
                            "v": int(t),
                            "n_segs": int(chain_n_segs),
                            "length": float(geom.length),
                            "is_bridge": False,
                            "bridge_type": "",
                            "geometry": geom,
                        }
                    )
    
    edges2_gdf = gpd.GeoDataFrame(out_edges, geometry="geometry", crs=edges_gdf.crs)
    
    # degrees in contracted graph
    H = nx.Graph()
    for r in edges2_gdf.itertuples(index=False):
        H.add_edge(int(r.u), int(r.v))
    
    node_rows = []
    for n in sorted(keep):
        if n not in node_geom:
            continue
        xy = node_xy_[n]
        node_rows.append(
            {
                "node": int(n),
                "degree": int(H.degree[n]) if n in H else 0,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "geometry": node_geom[n],
            }
        )
    nodes2_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=nodes_gdf.crs)
    return nodes2_gdf, edges2_gdf


# ============================================================================
# MUTABLE NETWORK UTILITIES
# ============================================================================

def recompute_node_degrees(nodes, edges):
    """Recompute node degrees from edge list."""
    deg = {int(n): 0 for n in nodes.keys()}
    for e in edges:
        deg[int(e["u"])] = deg.get(int(e["u"]), 0) + 1
        deg[int(e["v"])] = deg.get(int(e["v"]), 0) + 1
    return deg


def build_node_dict(nodes_gdf):
    """Build node dictionary from GeoDataFrame."""
    nodes = {}
    for r in nodes_gdf.itertuples(index=False):
        nodes[int(r.node)] = np.array([float(r.x), float(r.y)], dtype=float)
    return nodes


def build_rounding_index(nodes, nd=6):
    """Build index mapping rounded coordinates to node IDs."""
    idx = {}
    for nid, xy in nodes.items():
        idx[(round(float(xy[0]), nd), round(float(xy[1]), nd))] = int(nid)
    return idx


def ensure_node(nodes, idx, xy, next_id):
    """Ensure a node exists at xy, creating it if needed."""
    key = (round(float(xy[0]), 6), round(float(xy[1]), 6))
    if key in idx:
        return idx[key], next_id
    nid = int(next_id)
    next_id += 1
    nodes[nid] = np.array([float(xy[0]), float(xy[1])], dtype=float)
    idx[key] = nid
    return nid, next_id


def edge_coords_oriented_from_node(edge_geom, node_xy):
    """Get edge coordinates oriented from given node."""
    coords = np.asarray(edge_geom.coords, dtype=float)
    if np.linalg.norm(coords[0] - node_xy) <= np.linalg.norm(coords[-1] - node_xy):
        return coords
    return coords[::-1].copy()


def edge_bearing_mod180(edge_geom):
    """Get edge bearing modulo 180 degrees."""
    coords = np.asarray(edge_geom.coords, dtype=float)
    b = bearing_cw_from_north(coords[0], coords[-1])
    return float(b % 180.0)


def split_edge_at_point(
    nodes, idx, edges, edge_i, pt, next_node_id, cost_ctx=None, parent_cost=None, eps=1e-9
):
    """
    Split edges[edge_i] at pt (projected), inserting a node if interior.
    Returns: (node_id_at_split, next_node_id, did_split_bool)
    """
    e = edges[edge_i]
    geom = e["geometry"]
    seg1, seg2, at_endpoint = split_linestring_at_point(geom, pt, eps=eps)
    
    if at_endpoint or seg2 is None:
        # snap to closest endpoint node id (existing)
        xy = np.array([pt.x, pt.y], float)
        uxy = nodes[int(e["u"])]
        vxy = nodes[int(e["v"])]
        if np.linalg.norm(uxy - xy) <= np.linalg.norm(vxy - xy):
            return int(e["u"]), next_node_id, False
        return int(e["v"]), next_node_id, False
    
    # interior split: create/reuse node at split point
    split_xy = np.array([pt.x, pt.y], dtype=float)
    nid, next_node_id = ensure_node(nodes, idx, split_xy, next_node_id)
    
    # replace edge with seg1 and append seg2
    u = int(e["u"])
    v = int(e["v"])
    attrs = {
        k: e[k]
        for k in e.keys()
        if k not in ("geometry", "u", "v", "length", "cost_sum")
    }
    # first piece
    e1 = {
        **attrs,
        "u": u,
        "v": nid,
        "geometry": seg1,
        "length": float(seg1.length),
    }
    # second piece
    e2 = {
        **attrs,
        "u": nid,
        "v": v,
        "geometry": seg2,
        "length": float(seg2.length),
    }
    
    if cost_ctx is not None:
        e1["cost_sum"] = cost_sum_along_line(
            seg1,
            cost_ctx["arr"],
            cost_ctx["transform"],
            cost_ctx["nodata"],
            cost_ctx["step"],
            nodata_to_nan=cost_ctx["nodata_to_nan"],
            unique_cells=cost_ctx["unique_cells"],
        )
        e2["cost_sum"] = cost_sum_along_line(
            seg2,
            cost_ctx["arr"],
            cost_ctx["transform"],
            cost_ctx["nodata"],
            cost_ctx["step"],
            nodata_to_nan=cost_ctx["nodata_to_nan"],
            unique_cells=cost_ctx["unique_cells"],
        )
    elif "cost_sum" in e:
        # if you had a cost but no ctx, carry proportionally (fallback)
        pc = float(e.get("cost_sum", 0.0))
        L = float(geom.length) if geom.length else 1.0
        e1["cost_sum"] = pc * (float(seg1.length) / L)
        e2["cost_sum"] = pc * (float(seg2.length) / L)
    
    edges[edge_i] = e1
    edges.append(e2)
    return int(nid), next_node_id, True


# ============================================================================
# ENDPOINT BRIDGING AND SNAPPING
# ============================================================================

def build_endpoint_directions(nodes, edges, direction_step=3):
    """
    Compute for degree-1 nodes:
      - outward direction vector (pointing OUT of the endpoint) = - (into-line vector)
    Returns: endpoints list, dir_map[endpoint_id] = outward_vector (not normalized)
    """
    deg = {nid: 0 for nid in nodes.keys()}
    for e in edges:
        deg[int(e["u"])] = deg.get(int(e["u"]), 0) + 1
        deg[int(e["v"])] = deg.get(int(e["v"]), 0) + 1
    
    endpoints = [int(n) for n, d in deg.items() if d == 1]
    dir_map = {}
    
    # build adjacency: endpoint -> (edge index)
    incident = {int(n): [] for n in nodes.keys()}
    for i, e in enumerate(edges):
        incident[int(e["u"])].append(i)
        incident[int(e["v"])].append(i)
    
    for n in endpoints:
        if len(incident[n]) != 1:
            continue
        ei = incident[n][0]
        e = edges[ei]
        nxy = nodes[n]
        coords = edge_coords_oriented_from_node(e["geometry"], nxy)
        d_in = endpoint_direction_from_linestring(nxy, coords, step=direction_step)
        d_out = -d_in
        dir_map[n] = d_out
    
    return endpoints, dir_map


def bridge_endpoints_round(
    nodes,
    edges,
    max_gap_dist,
    max_angle_deg,
    direction_step,
    max_bridges_per_endpoint,
    bridge_type="endpoint_endpoint",
):
    """Bridge endpoints that face each other within angle and distance constraints."""
    endpoints, out_dir = build_endpoint_directions(nodes, edges, direction_step=direction_step)
    used = {n: 0 for n in endpoints}
    added = 0
    
    # naive O(E^2) over endpoints; ok for moderate endpoint counts
    for a in endpoints:
        if used[a] >= max_bridges_per_endpoint:
            continue
        if a not in out_dir:
            continue
        xa = nodes[a]
        da = out_dir[a]
        
        best_b = None
        best_score = None
        
        for b in endpoints:
            if b == a:
                continue
            if used[b] >= max_bridges_per_endpoint:
                continue
            if b not in out_dir:
                continue
            
            xb = nodes[b]
            ab = xb - xa
            dist = float(np.linalg.norm(ab))
            if dist == 0 or dist > float(max_gap_dist):
                continue
            
            ang_a = angle_deg(da, ab)
            db = out_dir[b]
            ang_b = angle_deg(db, -ab)
            
            if ang_a <= float(max_angle_deg) and ang_b <= float(max_angle_deg):
                score = dist + 0.2 * (ang_a + ang_b)
                if best_score is None or score < best_score:
                    best_score = score
                    best_b = b
        
        if best_b is not None:
            b = best_b
            geom = LineString([tuple(nodes[a]), tuple(nodes[b])])
            edges.append(
                {
                    "u": int(a),
                    "v": int(b),
                    "length": float(geom.length),
                    "n_segs": 1,
                    "is_bridge": True,
                    "bridge_type": str(bridge_type),
                    "geometry": geom,
                }
            )
            used[a] += 1
            used[b] += 1
            added += 1
    
    return added


def build_vertex_index_for_edges(edges, stride=1):
    """
    Build STRtree over interior vertices of edges.
    Returns: (tree, meta, sig_to_i)
      meta[i] = {"edge_i": int, "vidx": int, "xy": np.array, "tangent": np.array}
      sig_to_i maps (round(x,6), round(y,6)) -> meta index (for Shapely 1.x geometry-return cases)
    """
    pts = []
    meta = []
    stride = max(1, int(stride))
    
    for ei, e in enumerate(edges):
        geom = e["geometry"]
        coords = np.asarray(geom.coords, dtype=float)
        if len(coords) < 3:
            continue
        for j in range(1, len(coords) - 1, stride):
            p = coords[j]
            tan = coords[j + 1] - coords[j - 1]
            g = Point(float(p[0]), float(p[1]))
            pts.append(g)
            meta.append(
                {
                    "edge_i": int(ei),
                    "vidx": int(j),
                    "xy": p.copy(),
                    "tangent": tan.copy(),
                }
            )
    
    if not pts:
        return None, [], {}
    
    tree = STRtree(pts)
    sig_to_i = {(round(p.x, 6), round(p.y, 6)): i for i, p in enumerate(pts)}
    return tree, meta, sig_to_i


def snap_endpoints_to_vertices_round(
    nodes,
    edges,
    next_node_id,
    cost_ctx,
    max_snap_dist,
    max_snap_angle_deg,
    max_target_angle_deg,
    direction_step,
    max_snaps_per_endpoint,
    vertex_stride,
):
    """
    Endpoint -> interior vertex snapping with Shapely 1/2 STRtree compatibility.
    """
    endpoints, out_dir = build_endpoint_directions(nodes, edges, direction_step=direction_step)
    
    tree, meta, sig_to_i = build_vertex_index_for_edges(edges, stride=vertex_stride)
    if tree is None:
        return 0, next_node_id
    
    idx = build_rounding_index(nodes, nd=6)
    used = {n: 0 for n in endpoints}
    added = 0
    
    for a in endpoints:
        if used[a] >= int(max_snaps_per_endpoint):
            continue
        if a not in out_dir:
            continue
        
        xa = nodes[a]
        da = out_dir[a]
        
        # IMPORTANT: initialize for each endpoint
        best = None
        best_score = None
        
        search_geom = Point(float(xa[0]), float(xa[1])).buffer(float(max_snap_dist))
        hits = tree.query(search_geom)
        
        for h in hits:
            # Shapely 2 often returns integer indices; Shapely 1 returns geometries
            if isinstance(h, (int, np.integer)):
                mi = int(h)
            else:
                mi = sig_to_i.get((round(h.x, 6), round(h.y, 6)), None)
            
            if mi is None or mi < 0 or mi >= len(meta):
                continue
            
            m = meta[mi]
            ei = int(m["edge_i"])
            vxy = np.asarray(m["xy"], dtype=float)
            
            ab = vxy - xa
            dist = float(np.linalg.norm(ab))
            if dist == 0 or dist > float(max_snap_dist):
                continue
            
            ang_ext = angle_deg(da, ab)
            if ang_ext > float(max_snap_angle_deg):
                continue
            
            tan = np.asarray(m["tangent"], dtype=float)
            ang_target = min(angle_deg(tan, -ab), angle_deg(-tan, -ab))
            if ang_target > float(max_target_angle_deg):
                continue
            
            score = dist + 0.2 * ang_ext + 0.1 * ang_target
            if best_score is None or score < best_score:
                best_score = score
                best = (ei, Point(float(vxy[0]), float(vxy[1])))
        
        if best is None:
            continue
        
        target_ei, vpt = best
        
        parent_cost = edges[target_ei].get("cost_sum", None)
        nid, next_node_id, did_split = split_edge_at_point(
            nodes, idx, edges, target_ei, vpt, next_node_id, cost_ctx=cost_ctx, parent_cost=parent_cost
        )
        
        geom = LineString([tuple(nodes[a]), tuple(nodes[nid])])
        conn = {
            "u": int(a),
            "v": int(nid),
            "length": float(geom.length),
            "n_segs": 1,
            "is_bridge": True,
            "bridge_type": "endpoint_vertex",
            "geometry": geom,
        }
        if cost_ctx is not None:
            conn["cost_sum"] = cost_sum_along_line(
                geom,
                cost_ctx["arr"],
                cost_ctx["transform"],
                cost_ctx["nodata"],
                cost_ctx["step"],
                nodata_to_nan=cost_ctx["nodata_to_nan"],
                unique_cells=cost_ctx["unique_cells"],
            )
        
        edges.append(conn)
        used[a] += 1
        added += 1
    
    return added, next_node_id


# ============================================================================
# DIRECTIONALITY
# ============================================================================

def add_directionality(edges, slope_raster_path, convention, is_downhill):
    """
    Adds fields:
      - slope_dir_deg (converted to cw_from_north)
      - from, to (node ids)
    """
    if not slope_raster_path:
        return
    
    with rasterio.open(slope_raster_path) as src:
        for e in edges:
            geom = e["geometry"]
            mid = geom.interpolate(0.5, normalized=True)
            sraw = raster_value_at_xy(src, (mid.x, mid.y))
            sdir = convert_slope_dir_to_cw_from_north(sraw, convention)
            if np.isfinite(sdir) and not bool(is_downhill):
                sdir = (sdir + 180.0) % 360.0  # raster indicates uphill; convert to downhill
            
            e["slope_dir_deg"] = float(sdir) if np.isfinite(sdir) else np.nan
            
            # choose orientation closest to downhill direction
            coords = np.asarray(geom.coords, float)
            b_uv = bearing_cw_from_north(coords[0], coords[-1])
            b_vu = (b_uv + 180.0) % 360.0
            
            if not np.isfinite(sdir):
                e["from"] = int(e["u"])
                e["to"] = int(e["v"])
                continue
            
            d_uv = abs(((b_uv - sdir + 180.0) % 360.0) - 180.0)
            d_vu = abs(((b_vu - sdir + 180.0) % 360.0) - 180.0)
            if d_uv <= d_vu:
                e["from"] = int(e["u"])
                e["to"] = int(e["v"])
            else:
                e["from"] = int(e["v"])
                e["to"] = int(e["u"])
