"""
GPS localisation using SuperPoint + LightGlue.

This is a drop-in replacement for the SIFT-based localize.py in gpsless_mapping.
The two-phase structure is identical; only the feature extractor and matcher change.

┌──────────────────────────────────────────────────────────────────────────────┐
│  SIFT version              │  This version (SuperPoint + LightGlue)          │
│  ─────────────────────── │  ───────────────────────────────────────────────│
│  cv2.SIFT_create()         │  SuperPoint (neural net, GPU)                   │
│  FLANN knnMatch + ratio    │  FLANN vote  (Phase 1, same idea)               │
│  BFMatcher + ratio test    │  LightGlue   (neural matcher, replaces ratio+BF)│
│  cv2.findHomography RANSAC │  cv2.findHomography RANSAC (kept — gives GPS)   │
└──────────────────────────────────────────────────────────────────────────────┘

Algorithm
---------
Phase 1 — coarse (FLANN vote):
  Extract SuperPoint from query → match against the pooled tile descriptor pool
  via FLANN (L2, same as SIFT version) → vote to rank candidate tiles.

Phase 2 — fine (SuperPoint + LightGlue):
  Stitch the winning tile + 8 neighbours into a 3×3 composite;
  run SuperPoint on both query and composite;
  run LightGlue to get verified match pairs (no manual ratio test needed);
  RANSAC homography → project query centre to GPS.

Usage:
    python localize.py photo.jpg
    python localize.py photo.jpg --zoom 19 --show
"""

import argparse
import bisect
import pickle
from collections import defaultdict
from pathlib import Path

import os as _os
# Jetson fix: set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 in the SHELL
# before starting Python (env var is read at CUDA driver init, not here).
# Use run_test.sh which sets this automatically.

import cv2
import numpy as np
import torch

from lightglue import SuperPoint, LightGlue
from lightglue.utils import rbd

from config import (
    ZOOM_LEVEL, INDEX_DIR, TILE_DIR,
    MATCH_RATIO, MIN_INLIERS, TOP_CANDIDATES, TILE_SIZE,
    SP_QUERY_MAX_KEYPOINTS, SP_COMPOSITE_MAX_KEYPOINTS, SP_KEYPOINT_THRESHOLD,
)
from tile_utils import pixel_to_latlon, tile_bounds

device = torch.device(
    'cpu' if _os.environ.get('FORCE_CPU') else
    ('cuda' if torch.cuda.is_available() else 'cpu')
)


# ── model singletons (loaded once) ───────────────────────────────────────────

_extractor = None
_matcher   = None


def _get_models():
    global _extractor, _matcher
    if _extractor is None:
        print(f'Loading SuperPoint + LightGlue on {device} …')
        _extractor = SuperPoint(
            max_num_keypoints=SP_QUERY_MAX_KEYPOINTS,
            detection_threshold=SP_KEYPOINT_THRESHOLD,
        ).eval().to(device)
        # Jetson NvMap fix: ImagePreprocessor inside extract() targets resize_max
        # as a resize TARGET (not a cap), upscaling 640px input → 1024px → crash.
        # Capping at 640 keeps the tensor at the size _img_to_tensor already set.
        # Jetson NvMap fix: ImagePreprocessor inside extract() uses preprocess_conf
        # {'resize': 1024} to UPSCALE images to max-side 1024, causing NvMap crash.
        # Override as instance attribute (shadows class-level dict) to cap at 640.
        _extractor.preprocess_conf = {'resize': 640}
        _matcher = LightGlue(features='superpoint').eval().to(device)
    return _extractor, _matcher


# ── helpers ───────────────────────────────────────────────────────────────────

def _img_to_tensor(img_bgr: np.ndarray, max_side: int = 640) -> torch.Tensor:
    """
    BGR uint8 ndarray → [1, 1, H, W] float32 tensor in [0,1] on device.
    Resizes so the longer edge is at most max_side (default 640).
    Jetson NvMap can't reliably allocate the dual ~225 MB conv buffers
    needed for 1280x720 in SuperPoint's encoder; 640x360 avoids this.
    """
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    t = torch.from_numpy(gray).float() / 255.0   # (H, W)
    return t.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)


def _extract(img_bgr: np.ndarray, max_kp: int) -> dict:
    """Run SuperPoint on a BGR image; return feature dict (no batch dim)."""
    extractor, _ = _get_models()
    # Temporarily override max keypoints for this call
    orig = extractor.conf.max_num_keypoints
    extractor.conf.max_num_keypoints = max_kp
    tensor = _img_to_tensor(img_bgr)
    with torch.no_grad():
        feats = extractor.extract(tensor)
    extractor.conf.max_num_keypoints = orig
    return rbd(feats)  # remove batch dim


# ── index helpers ─────────────────────────────────────────────────────────────

def load_index(index_dir, zoom):
    path = Path(index_dir) / f'sp_index_z{zoom}.pkl'
    if not path.exists():
        raise FileNotFoundError(
            f'SuperPoint index not found: {path}\n'
            f'Run build_index.py first.')
    print(f'Loading index from {path} …')
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Phase 1: nearest-neighbour vote ──────────────────────────────────────────

def flann_vote(descs_q: np.ndarray, index: list) -> list:
    """
    Vote for the best-matching tile using top-1 nearest-neighbour search.

    Why no ratio test here:
      SuperPoint descriptors are L2-normalised 256-dim vectors. In high
      dimensions, pairwise L2 distances concentrate (all ~√2), so Lowe's
      ratio test almost never fires — it was designed for SIFT's unnormalised
      128-dim histograms.  For deep features the right strategy is plain
      nearest-neighbour voting: for every query keypoint, cast one vote for
      whichever tile contains its closest match.

    Returns a ranked list of (tile_idx, vote_count) sorted by vote count.
    """
    tile_offsets = [0]
    for entry in index:
        tile_offsets.append(tile_offsets[-1] + len(entry['descs']))

    all_descs = np.vstack([e['descs'] for e in index]).astype(np.float32)
    print(f'Total train kp : {len(all_descs):,}  — building FLANN …')

    # Use inner-product space (cosine similarity) via FLANN LSH — correct for
    # L2-normalised vectors.  We negate distances later; here we just get the
    # top-1 nearest neighbour per query descriptor.
    index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann.add([all_descs])
    flann.train()

    # k=1: only top-1 match, no ratio test
    raw = flann.knnMatch(descs_q.astype(np.float32), k=1)
    votes = defaultdict(int)
    for (m,) in raw:
        tile_idx = bisect.bisect_right(tile_offsets, m.trainIdx) - 1
        votes[tile_idx] += 1

    print(f'Query kp voted : {len(raw)}  top tile votes: '
          f'{max(votes.values()) if votes else 0}')

    return sorted(votes.items(), key=lambda kv: kv[1], reverse=True)


# ── Phase 2: SuperPoint + LightGlue on stitched tiles ────────────────────────

def stitch_tiles(cx: int, cy: int, zoom: int, tile_dir: Path,
                 radius: int = 1) -> tuple:
    """
    Load a (2r+1)×(2r+1) grid of tiles centred on (cx, cy) and stitch into
    one BGR image.  Returns (composite_bgr, x_origin, y_origin).
    """
    r    = radius
    size = 2 * r + 1
    canvas = np.zeros((size * TILE_SIZE, size * TILE_SIZE, 3), dtype=np.uint8)
    for dx in range(size):
        for dy in range(size):
            tx, ty = cx - r + dx, cy - r + dy
            path   = tile_dir / str(zoom) / str(tx) / f'{ty}.png'
            tile   = cv2.imread(str(path))
            if tile is not None:
                canvas[dy * TILE_SIZE:(dy + 1) * TILE_SIZE,
                       dx * TILE_SIZE:(dx + 1) * TILE_SIZE] = tile
    return canvas, cx - r, cy - r


def lightglue_match(query_bgr: np.ndarray, cx: int, cy: int,
                    zoom: int, tile_dir: Path,
                    radius: int = 1,
                    min_inliers: int = MIN_INLIERS) -> tuple | None:
    """
    Stitch a 3×3 tile grid, run SuperPoint + LightGlue between
    the query and the composite, then RANSAC homography → GPS.

    Returns (lat, lon, n_inliers, H, composite_bgr) or None.

    Key difference from SIFT version:
      - LightGlue replaces BFMatcher + ratio test.  It returns only
        high-confidence match pairs — no manual filtering needed.
      - RANSAC is still used to compute the homography robustly and
        to project the query centre into GPS coordinates.
    """
    _, matcher = _get_models()

    composite, x_orig, y_orig = stitch_tiles(cx, cy, zoom, tile_dir, radius)

    # ── extract features ────────────────────────────────────────────────────
    feats_q = _extract(query_bgr,    max_kp=SP_QUERY_MAX_KEYPOINTS)
    feats_c = _extract(composite,    max_kp=SP_COMPOSITE_MAX_KEYPOINTS)

    if (feats_q['keypoints'].shape[0] < min_inliers or
            feats_c['keypoints'].shape[0] < min_inliers):
        return None

    # LightGlue needs batch dimension restored for the matcher
    def _add_batch(d):
        return {k: v.unsqueeze(0) if v.dim() == 2 else
                   v.unsqueeze(0) if v.dim() == 1 else v
                for k, v in d.items()}

    with torch.no_grad():
        matches01 = matcher({
            'image0': _add_batch(feats_q),
            'image1': _add_batch(feats_c),
        })

    # ── extract matched coordinates ─────────────────────────────────────────
    matches01 = rbd(matches01)
    m_pairs   = matches01['matches'].cpu().numpy()   # (K, 2) index pairs
    n_matches = len(m_pairs)

    if n_matches < min_inliers:
        return None

    kpts_q = feats_q['keypoints'].cpu().numpy()  # (N, 2)
    kpts_c = feats_c['keypoints'].cpu().numpy()  # (M, 2)

    src_pts = kpts_q[m_pairs[:, 0]].reshape(-1, 1, 2).astype(np.float32)
    dst_pts = kpts_c[m_pairs[:, 1]].reshape(-1, 1, 2).astype(np.float32)

    # ── RANSAC homography ────────────────────────────────────────────────────
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    n_inliers = int(mask.ravel().sum())
    if n_inliers < min_inliers:
        return None

    # Inlier ratio guard: if fewer than 30% of LightGlue matches survive
    # RANSAC, the geometry is likely wrong (false positive tile).
    inlier_ratio = n_inliers / n_matches
    if inlier_ratio < 0.30:
        return None

    # ── resized dimensions (H is defined in these coordinate spaces) ─────────
    # _img_to_tensor resizes both query and composite to max-side 640 before
    # SuperPoint runs, so all keypoints and therefore H live in these spaces.
    # Using the original (pre-resize) pixel coords when projecting through H
    # would project the wrong point entirely.
    _max_side = 640  # must match _img_to_tensor default
    h_orig, w_orig = query_bgr.shape[:2]
    if max(h_orig, w_orig) > _max_side:
        _sq = _max_side / max(h_orig, w_orig)
        qw, qh = int(w_orig * _sq), int(h_orig * _sq)
    else:
        qw, qh = w_orig, h_orig

    comp_orig = (2 * radius + 1) * TILE_SIZE   # e.g. 768 for radius=1
    _sc = _max_side / comp_orig if comp_orig > _max_side else 1.0

    # Sanity-check: projected query corners should form a convex quad with
    # a reasonable area (not degenerate or flipped).
    corners  = cv2.perspectiveTransform(
        np.float32([[[0,0],[qw,0],[qw,qh],[0,qh]]]), H)[0]
    area = cv2.contourArea(corners)
    if area < (qw * qh * 0.05) or area > (qw * qh * 50):
        return None
    if not cv2.isContourConvex(np.int32(corners)):
        return None

    # ── project query centre → composite pixel (in resized space) ────────────
    centre = cv2.perspectiveTransform(
        np.float32([[[qw / 2, qh / 2]]]), H)[0][0]
    cpx, cpy = float(centre[0]), float(centre[1])

    # Convert from resized composite space back to original tile-pixel space
    # so that TILE_SIZE subdivision is correct.
    cpx_t = cpx / _sc
    cpy_t = cpy / _sc

    margin = TILE_SIZE
    if not (-margin <= cpx_t <= comp_orig + margin and
            -margin <= cpy_t <= comp_orig + margin):
        return None

    tile_dx  = int(cpx_t / TILE_SIZE)
    tile_dy  = int(cpy_t / TILE_SIZE)
    local_px = cpx_t - tile_dx * TILE_SIZE
    local_py = cpy_t - tile_dy * TILE_SIZE
    lat, lon = pixel_to_latlon(local_px, local_py,
                               x_orig + tile_dx,
                               y_orig + tile_dy,
                               zoom)
    return lat, lon, n_inliers, H, composite


# ── main ─────────────────────────────────────────────────────────────────────

def localize(image_path, zoom=None, index_dir=None, tile_dir=None, show=False):
    zoom      = zoom      or ZOOM_LEVEL
    index_dir = index_dir or INDEX_DIR
    tile_dir  = Path(tile_dir or TILE_DIR)

    # ── load query ───────────────────────────────────────────────────────────
    query = cv2.imread(str(image_path))
    if query is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    print(f'Query image    : {image_path}  ({query.shape[1]}×{query.shape[0]} px)')

    # ── extract query features (SuperPoint) ──────────────────────────────────
    feats_q = _extract(query, max_kp=SP_QUERY_MAX_KEYPOINTS)
    descs_q = feats_q['descriptors'].cpu().numpy()   # (N, 256)
    n_kp    = len(descs_q)
    print(f'Query features : {n_kp} keypoints  (SuperPoint, device={device})')

    if n_kp < 5:
        print('Too few features in query image.')
        return None

    # ── Phase 1: FLANN coarse vote ───────────────────────────────────────────
    index  = load_index(index_dir, zoom)
    print(f'Tile index     : {len(index)} tiles')
    ranked = flann_vote(descs_q, index)

    print('\nTop candidate tiles:')
    for idx, votes in ranked[:TOP_CANDIDATES]:
        e = index[idx]
        print(f'  tile ({e["tile_x"]},{e["tile_y"]})  votes={votes}')

    # ── Phase 2: SuperPoint + LightGlue on stitched composite ───────────────
    result = None
    for tile_idx, _ms in ranked[:TOP_CANDIDATES]:
        entry = index[tile_idx]
        cx, cy = entry['tile_x'], entry['tile_y']
        print(f'\nLightGlue match on 3×3 grid centred ({cx},{cy}) …')

        out = lightglue_match(query, cx, cy, zoom, tile_dir,
                              radius=1, min_inliers=MIN_INLIERS)
        if out is not None:
            lat, lon, n_inliers, H, composite = out
            result = (lat, lon, n_inliers, cx, cy, H, composite)
            break

    # ── report ────────────────────────────────────────────────────────────────
    print()
    if result is None:
        print('Could not determine location.')
        print('Suggestions:')
        print('  • Check that the photo is within the downloaded tile area')
        print('  • Try a nadir (straight-down) photo with clear ground detail')
        print('  • Lower SP_KEYPOINT_THRESHOLD in config.py to detect more kps')
        print('  • Re-run build_index.py with a higher SP_MAX_KEYPOINTS')
        return None

    lat, lon, n_inliers, cx, cy, H, composite = result
    print('=' * 55)
    print(f'  GPS Location : {lat:.7f},  {lon:.7f}')
    print(f'  RANSAC inliers: {n_inliers}  |  centre tile ({cx},{cy}) z{zoom}')
    print(f'  Google Maps  : https://maps.google.com/?q={lat:.7f},{lon:.7f}')
    print('=' * 55)

    if show:
        _save_result(query, composite, H, lat, lon)

    return lat, lon


# ── visualisation ─────────────────────────────────────────────────────────────

def _save_result(query_img, composite, H, lat, lon):
    # H maps from resized-query space to resized-composite space.
    # Scale projected points back to original composite (768×768) for drawing.
    h_orig, w_orig = query_img.shape[:2]
    _max_side = 640
    if max(h_orig, w_orig) > _max_side:
        _sq = _max_side / max(h_orig, w_orig)
        qw, qh = int(w_orig * _sq), int(h_orig * _sq)
    else:
        qw, qh = w_orig, h_orig

    comp_h, comp_w = composite.shape[:2]  # original composite dims (e.g. 768×768)
    _sc = _max_side / max(comp_h, comp_w) if max(comp_h, comp_w) > _max_side else 1.0

    corners   = np.float32([[0,0],[qw,0],[qw,qh],[0,qh]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H) / _sc  # scale to composite space
    comp_vis  = cv2.polylines(composite.copy(), [np.int32(projected)],
                              True, (0, 255, 0), 2)

    centre = cv2.perspectiveTransform(
        np.float32([[[qw / 2, qh / 2]]]), H)[0][0] / _sc
    cv2.drawMarker(comp_vis, (int(centre[0]), int(centre[1])),
                   (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.imwrite('result_tile.png', comp_vis)
    cv2.imwrite('result_query.png', query_img)
    print('Match saved → result_tile.png  (3×3 composite with footprint)')
    print('Query saved → result_query.png')


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GPS localise a drone photo using SuperPoint + LightGlue')
    parser.add_argument('image',        help='Path to the query drone photo')
    parser.add_argument('--zoom',       type=int, default=None)
    parser.add_argument('--index-dir',  default=None)
    parser.add_argument('--tile-dir',   default=None)
    parser.add_argument('--show',       action='store_true',
                        help='Save match visualisation to result_tile.png')
    args = parser.parse_args()

    localize(args.image, zoom=args.zoom,
             index_dir=args.index_dir,
             tile_dir=args.tile_dir,
             show=args.show)
