"""
Network Metrics Utilities

Compute connectivity metrics for network graphs from GeoPackage files.
Adds various centrality, clustering, and connectivity measures to nodes and edges.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import networkx as nx


def _sanitize_path(p: str) -> Path:
    """Sanitize and resolve file path."""
    p = str(p).strip()
    p = re.sub(r"^[cC]/home/", "/home/", p)  # fix c/home typo
    outp = Path(p).expanduser()
    if not outp.is_absolute():
        outp = (Path.cwd() / outp).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    return outp


def _safe_unlink(p: Path):
    """Safely remove a file."""
    try:
        if p.exists():
            p.unlink()
    except Exception as e:
        raise OSError(
            f"Cannot remove existing file: {p}\nClose it in QGIS/other apps and retry.\n{e}"
        ) from e


def compute_network_metrics(
    gpkg_in,
    gpkg_out,
    nodes_lyr="nodes",
    edges_lyr="edges",
    overwrite_in_place=False,
    directed=False,
    use_weights=True,
    length_col="length",
    compute_current_flow=False,
    compute_local_edge_connectivity=False,
):
    """
    Add connectivity metrics directly into the existing nodes/edges tables:
     1) Read nodes+edges from an existing GPKG
     2) Compute metrics with NetworkX
     3) Write new GeoPackage file with metrics
    
    Parameters
    ----------
    gpkg_in : str
        Path to input GeoPackage file
    gpkg_out : str
        Path to output GeoPackage file
    nodes_lyr : str, default="nodes"
        Name of nodes layer in GPKG
    edges_lyr : str, default="edges"
        Name of edges layer in GPKG
    overwrite_in_place : bool, default=False
        If True, replaces gpkg_in with gpkg_out at the end
    directed : bool, default=False
        Whether the graph is directed
    use_weights : bool, default=True
        Use edge weights (length) in calculations
    length_col : str, default="length"
        Column name for edge length (created from geometry if missing)
    compute_current_flow : bool, default=False
        Compute current-flow betweenness (slow, can be fragile)
    compute_local_edge_connectivity : bool, default=False
        Compute local edge connectivity (can be expensive)
    
    Returns
    -------
    dict
        Dictionary with keys: 'nodes_gdf', 'edges_gdf', 'stats'
    """
    # ---------- READ ----------
    gpkg_in_p = _sanitize_path(gpkg_in)
    gpkg_out_p = _sanitize_path(gpkg_out)
    
    nodes = gpd.read_file(gpkg_in_p.as_posix(), layer=nodes_lyr)
    edges = gpd.read_file(gpkg_in_p.as_posix(), layer=edges_lyr)
    
    if "node" not in nodes.columns:
        raise ValueError(f"'{nodes_lyr}' layer must contain a 'node' column.")
    for c in ("u", "v"):
        if c not in edges.columns:
            raise ValueError(f"'{edges_lyr}' layer must contain '{c}' columns.")
    if "geometry" not in edges.columns:
        raise ValueError(f"'{edges_lyr}' layer must contain geometries.")
    
    edges_out = edges.copy()
    if length_col not in edges_out.columns:
        edges_out[length_col] = edges_out.geometry.length.astype(float)
    
    has_k = "k" in edges_out.columns
    
    # ---------- BUILD GRAPH ----------
    if directed:
        G = nx.MultiDiGraph() if has_k else nx.DiGraph()
        G_simple = nx.DiGraph()
    else:
        G = nx.MultiGraph() if has_k else nx.Graph()
        G_simple = nx.Graph()
    
    G.add_nodes_from(nodes["node"].astype(int).tolist())
    
    for r in edges_out.itertuples(index=False):
        u = int(getattr(r, "u"))
        v = int(getattr(r, "v"))
        w = float(getattr(r, length_col))
        if has_k:
            k = int(getattr(r, "k"))
            G.add_edge(u, v, key=k, **{length_col: w})
        else:
            G.add_edge(u, v, **{length_col: w})
        
        # Simple view: keep minimum weight between u-v
        if G_simple.has_edge(u, v):
            if w < G_simple[u][v].get(length_col, np.inf):
                G_simple[u][v][length_col] = w
        else:
            G_simple.add_edge(u, v, **{length_col: w})
    
    node_ids = nodes["node"].astype(int)
    
    # ---------- NODE METRICS ----------
    degree = dict(G.degree())
    strength = (
        dict(G.degree(weight=length_col)) if use_weights else {n: np.nan for n in G.nodes()}
    )
    
    betweenness = nx.betweenness_centrality(
        G, weight=(length_col if use_weights else None), normalized=True
    )
    closeness = nx.closeness_centrality(
        G, distance=(length_col if use_weights else None)
    )
    
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight=None)
    except Exception:
        eigenvector = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-8, weight=None)
    
    harmonic = nx.harmonic_centrality(
        G, distance=(length_col if use_weights else None)
    )
    
    try:
        pagerank = nx.pagerank(G, weight=(length_col if use_weights else None))
    except Exception:
        pagerank = {n: np.nan for n in G.nodes()}
    
    # clustering/core best on undirected view
    try:
        clustering = nx.clustering(G_simple if not directed else nx.Graph(G_simple))
    except Exception:
        clustering = {n: np.nan for n in G.nodes()}
    
    try:
        core = nx.core_number(G_simple if not directed else nx.Graph(G_simple))
    except Exception:
        core = {n: np.nan for n in G.nodes()}
    
    # components
    if directed:
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    comp_id, comp_size = {}, {}
    for i, comp in enumerate(comps):
        s = len(comp)
        for n in comp:
            comp_id[n] = i
            comp_size[n] = s
    
    nodes2 = nodes.copy()
    nodes2["degree"] = node_ids.map(degree).astype(float)
    nodes2["strength"] = node_ids.map(strength).astype(float)
    nodes2["betweenness"] = node_ids.map(betweenness).astype(float)
    nodes2["closeness"] = node_ids.map(closeness).astype(float)
    nodes2["eigenvector"] = node_ids.map(eigenvector).astype(float)
    
    nodes2["harmonic"] = node_ids.map(harmonic).astype(float)
    nodes2["pagerank"] = node_ids.map(pagerank).astype(float)
    nodes2["clustering"] = node_ids.map(clustering).astype(float)
    nodes2["core"] = node_ids.map(core).astype(float)
    nodes2["component"] = node_ids.map(comp_id).astype(int)
    nodes2["comp_size"] = node_ids.map(comp_size).astype(int)
    
    if directed:
        indeg = dict(G.in_degree())
        outdeg = dict(G.out_degree())
        nodes2["in_degree"] = node_ids.map(indeg).astype(float)
        nodes2["out_degree"] = node_ids.map(outdeg).astype(float)
    
    # ---------- EDGE METRICS (emphasis) ----------
    deg_map = nodes2.set_index("node")["degree"].to_dict()
    bet_map = nodes2.set_index("node")["betweenness"].to_dict()
    clo_map = nodes2.set_index("node")["closeness"].to_dict()
    eig_map = nodes2.set_index("node")["eigenvector"].to_dict()
    
    edges2 = edges_out.copy()
    
    # Endpoint metrics
    edges2["u_degree"] = edges2["u"].astype(int).map(deg_map).astype(float)
    edges2["v_degree"] = edges2["v"].astype(int).map(deg_map).astype(float)
    edges2["u_betweenness"] = edges2["u"].astype(int).map(bet_map).astype(float)
    edges2["v_betweenness"] = edges2["v"].astype(int).map(bet_map).astype(float)
    edges2["u_closeness"] = edges2["u"].astype(int).map(clo_map).astype(float)
    edges2["v_closeness"] = edges2["v"].astype(int).map(clo_map).astype(float)
    edges2["u_eigenvector"] = edges2["u"].astype(int).map(eig_map).astype(float)
    edges2["v_eigenvector"] = edges2["v"].astype(int).map(eig_map).astype(float)
    
    edges2["deg_sum"] = edges2["u_degree"] + edges2["v_degree"]
    edges2["deg_diff"] = (edges2["u_degree"] - edges2["v_degree"]).abs()
    
    # Edge betweenness (kept)
    edge_betw = nx.edge_betweenness_centrality(
        G, weight=(length_col if use_weights else None), normalized=True
    )
    
    def _edge_betw_lookup(u, v, k=None):
        if has_k:
            if (u, v, k) in edge_betw:
                return edge_betw[(u, v, k)]
            if (v, u, k) in edge_betw and not directed:
                return edge_betw[(v, u, k)]
            return np.nan
        else:
            if (u, v) in edge_betw:
                return edge_betw[(u, v)]
            if (v, u) in edge_betw and not directed:
                return edge_betw[(v, u)]
            return np.nan
    
    if has_k:
        edges2["edge_betweenness"] = [
            float(_edge_betw_lookup(int(u), int(v), int(k)))
            for u, v, k in zip(edges2["u"], edges2["v"], edges2["k"])
        ]
    else:
        edges2["edge_betweenness"] = [
            float(_edge_betw_lookup(int(u), int(v)))
            for u, v in zip(edges2["u"], edges2["v"])
        ]
    
    # Bridges (cut edges) on undirected simple graph
    if not directed:
        try:
            bridge_pairs = set(tuple(sorted(e)) for e in nx.bridges(G_simple))
        except Exception:
            bridge_pairs = set()
        
        if has_k:
            # If parallel edges exist between u-v, it cannot be a bridge
            multiplicity = {}
            for u, v, k in G.edges(keys=True):
                a, b = (u, v) if u <= v else (v, u)
                multiplicity[(a, b)] = multiplicity.get((a, b), 0) + 1
            
            edges2["is_bridge"] = [
                bool(
                    (tuple(sorted((int(u), int(v)))) in bridge_pairs)
                    and (multiplicity.get(tuple(sorted((int(u), int(v)))), 0) == 1)
                )
                for u, v in zip(edges2["u"], edges2["v"])
            ]
        else:
            edges2["is_bridge"] = [
                bool(tuple(sorted((int(u), int(v)))) in bridge_pairs)
                for u, v in zip(edges2["u"], edges2["v"])
            ]
    else:
        edges2["is_bridge"] = False
    
    # Geometry-based edge measures
    def _straight_dist(geom):
        if geom is None or geom.is_empty:
            return np.nan
        coords = np.asarray(geom.coords)
        if coords.shape[0] < 2:
            return np.nan
        return float(np.linalg.norm(coords[-1] - coords[0]))
    
    edges2["straight_dist"] = edges2.geometry.apply(_straight_dist).astype(float)
    edges2["sinuosity"] = (edges2[length_col] / edges2["straight_dist"]).replace(
        [np.inf, -np.inf], np.nan
    )
    edges2["straightness"] = (edges2["straight_dist"] / edges2[length_col]).replace(
        [np.inf, -np.inf], np.nan
    )
    
    # Triangle-based edge measures (undirected)
    def _common_neighbors(u, v):
        try:
            return len(list(nx.common_neighbors(G_simple, int(u), int(v))))
        except Exception:
            return 0
    
    if not directed:
        tris = [_common_neighbors(u, v) for u, v in zip(edges2["u"], edges2["v"])]
        edges2["edge_triangles"] = np.asarray(tris, dtype=float)
        
        ecc = []
        for u, v, tri in zip(edges2["u"], edges2["v"], edges2["edge_triangles"]):
            du = G_simple.degree(int(u))
            dv = G_simple.degree(int(v))
            denom = min(du - 1, dv - 1)
            ecc.append(float(tri / denom) if denom > 0 else 0.0)
        edges2["edge_clustering"] = np.asarray(ecc, dtype=float)
    else:
        edges2["edge_triangles"] = np.nan
        edges2["edge_clustering"] = np.nan
    
    # Local edge connectivity (optional)
    def _local_ec(u, v):
        if not compute_local_edge_connectivity:
            return np.nan
        try:
            return float(nx.local_edge_connectivity(G_simple, int(u), int(v)))
        except Exception:
            return np.nan
    
    edges2["local_edge_connectivity"] = [
        _local_ec(u, v) for u, v in zip(edges2["u"], edges2["v"])
    ]
    
    # Current-flow edge betweenness (optional; often fragile/slow)
    edges2["edge_current_flow_betweenness"] = np.nan
    if (not directed) and compute_current_flow:
        edge_cf = {}
        for comp in nx.connected_components(G_simple):
            H = G_simple.subgraph(comp).copy()
            if H.number_of_nodes() < 3 or H.number_of_edges() < 2:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    part = nx.edge_current_flow_betweenness_centrality(
                        H, weight=(length_col if use_weights else None), normalized=True
                    )
                    for e, val in part.items():
                        edge_cf[e] = float(val) if np.isfinite(val) else np.nan
                except Exception:
                    continue
        
        def _cf_lookup(u, v):
            if (u, v) in edge_cf:
                return edge_cf[(u, v)]
            if (v, u) in edge_cf:
                return edge_cf[(v, u)]
            return np.nan
        
        edges2["edge_current_flow_betweenness"] = [
            float(_cf_lookup(int(u), int(v))) for u, v in zip(edges2["u"], edges2["v"])
        ]
    
    # ---------- WRITE ----------
    _safe_unlink(gpkg_out_p)
    
    # Use pyogrio if available; otherwise use fiona
    engine = "pyogrio"
    try:
        import pyogrio  # noqa: F401
    except Exception:
        engine = "fiona"
    
    print(f"Writing with engine: {engine}")
    nodes2.to_file(gpkg_out_p.as_posix(), layer=nodes_lyr, driver="GPKG", engine=engine)
    edges2.to_file(gpkg_out_p.as_posix(), layer=edges_lyr, driver="GPKG", engine=engine)
    
    print(f"Written: {gpkg_out_p}")
    print(f"Layers overwritten in output: {nodes_lyr}, {edges_lyr}")
    
    # Optional: replace input file (only if everything succeeded)
    if overwrite_in_place:
        # Ensure input is not open elsewhere
        tmp_backup = gpkg_in_p.with_suffix(".backup.gpkg")
        try:
            if tmp_backup.exists():
                tmp_backup.unlink()
            gpkg_in_p.rename(tmp_backup)
            gpkg_out_p.rename(gpkg_in_p)
            tmp_backup.unlink(missing_ok=True)
            print(f"Replaced original GPKG in-place: {gpkg_in_p}")
        except Exception as e:
            raise OSError(
                "Failed to replace original file. The output GPKG is still available.\n"
                f"Output: {gpkg_out_p}\nBackup (if created): {tmp_backup}\n{e}"
            ) from e
    
    stats = {
        "num_nodes": len(nodes2),
        "num_edges": len(edges2),
        "num_components": len(comps),
        "directed": directed,
        "use_weights": use_weights,
    }
    
    return {
        "nodes_gdf": nodes2,
        "edges_gdf": edges2,
        "stats": stats,
    }
