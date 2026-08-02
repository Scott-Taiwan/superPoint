"""
Extract SuperPoint features from every downloaded tile and save a searchable index.

Why SuperPoint instead of SIFT?
  - SuperPoint is a neural network trained specifically for matching across
    different viewpoints and lighting conditions (satellite tile vs drone photo).
  - Its 256-dim descriptors are L2-normalised — matching is more discriminative
    than SIFT's 128-dim float histogram.
  - Runs on GPU (Jetson Orin), making batch indexing fast.

Index format (list of dicts, one per tile):
    {
        'tile_x': int,
        'tile_y': int,
        'zoom':   int,
        'kps':    float32 array (N, 2)   — keypoint x,y pixel coords
        'descs':  float32 array (N, 256) — L2-normalised SuperPoint descriptors
        'scores': float32 array (N,)     — detector confidence scores
    }

Usage:
    python build_index.py                       # uses ZOOM_LEVEL from config
    python build_index.py --zoom 17
    python build_index.py --tile-dir /path/to/tiles --index-dir ./index
"""

import argparse
import pickle
from pathlib import Path

import os as _os
if not _os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
    _os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

import cv2
import numpy as np
import torch
from tqdm import tqdm

from lightglue import SuperPoint
from lightglue.utils import rbd

from config import (TILE_DIR, ZOOM_LEVEL, INDEX_DIR,
                    SP_MAX_KEYPOINTS, SP_KEYPOINT_THRESHOLD,
                    TILE_SOURCES, choose_tile_source)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tile_to_tensor(path: Path) -> torch.Tensor | None:
    """
    Load a tile PNG as a [1, H, W] float32 tensor in [0, 1].
    SuperPoint works on grayscale — we convert from BGR here.
    Returns None if the file cannot be read.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).float() / 255.0   # (H, W)
    return tensor.unsqueeze(0)                         # (1, H, W)


def build_index(tile_dir=None, zoom=None, index_dir=None, index_name=None):
    tile_dir  = Path(tile_dir  or TILE_DIR)
    zoom      = zoom           or ZOOM_LEVEL
    index_dir = Path(index_dir or INDEX_DIR)
    index_name = index_name    or f'sp_index_z{zoom}.pkl'
    index_dir.mkdir(exist_ok=True)

    zoom_dir = tile_dir / str(zoom)
    if not zoom_dir.exists():
        raise FileNotFoundError(
            f'No tiles found at {zoom_dir}.\n'
            f'Run download_tiles.py (from gpsless_mapping) first, or point '
            f'--tile-dir at an existing tiles folder.')

    tile_paths = sorted(zoom_dir.rglob('*.png'))
    if not tile_paths:
        raise FileNotFoundError(f'No PNG tiles found under {zoom_dir}')

    print(f'Device        : {device}')
    print(f'Tiles found   : {len(tile_paths)} (zoom {zoom})')
    print(f'Max kp/tile   : {SP_MAX_KEYPOINTS}')

    # ── load SuperPoint ───────────────────────────────────────────────────────
    # SuperPoint downloads a small (~5 MB) pretrained model on first run.
    extractor = SuperPoint(
        max_num_keypoints=SP_MAX_KEYPOINTS,
        detection_threshold=SP_KEYPOINT_THRESHOLD,
    ).eval().to(device)
    # Jetson fix: default preprocess_conf {'resize': 1024} upscales tiles to
    # 1024x1024, triggering NvMap crash and mismatching query scale (640px).
    # Override to 640 so index and query features are extracted at the same scale.
    extractor.preprocess_conf = {'resize': 640}

    index   = []
    skipped = 0

    for path in tqdm(tile_paths, unit='tile'):
        # Tile path structure: tiles/{zoom}/{x}/{y}.png
        tile_x = int(path.parent.name)
        tile_y = int(path.stem)

        tensor = tile_to_tensor(path)
        if tensor is None:
            skipped += 1
            continue

        # Add batch dimension: (1, 1, H, W)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            feats = extractor.extract(tensor)

        # rbd = remove batch dimension; convert to numpy
        feats = rbd(feats)
        kps    = feats['keypoints'].cpu().numpy()       # (N, 2) float32
        descs  = feats['descriptors'].cpu().numpy()     # (N, 256) float32
        scores = feats['keypoint_scores'].cpu().numpy() # (N,)  float32

        if len(kps) < 5:
            skipped += 1
            continue

        index.append({
            'tile_x': tile_x,
            'tile_y': tile_y,
            'zoom':   zoom,
            'kps':    kps,
            'descs':  descs,
            'scores': scores,
        })

    # ── save ─────────────────────────────────────────────────────────────────
    out_path = index_dir / index_name
    with open(out_path, 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_kps = sum(len(e['kps']) for e in index)
    size_mb   = out_path.stat().st_size / 1e6

    print(f'\nIndex saved → {out_path}')
    print(f'  Tiles indexed  : {len(index):,}  (skipped {skipped})')
    print(f'  Total keypoints: {total_kps:,}')
    print(f'  Index file size: {size_mb:.1f} MB')
    print()
    print('Memory note: pooling all descriptors for Phase-1 vote requires')
    print(f'  ≈ {total_kps * 256 * 4 / 1e6:.0f} MB RAM.  '
          'Reduce SP_MAX_KEYPOINTS in config.py if this is too large.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build SuperPoint feature index from satellite tiles')
    parser.add_argument('--source',    choices=list(TILE_SOURCES.keys()), default=None,
                        help='Tile source (esri / tgos). Prompts if omitted.')
    parser.add_argument('--zoom',      type=int, default=None,
                        help='Zoom level (default: from config.py)')
    parser.add_argument('--tile-dir',  default=None,
                        help='Path to tiles folder (overrides --source default)')
    parser.add_argument('--index-dir', default=None,
                        help='Where to save the index (default: from config.py)')
    args = parser.parse_args()

    # ── source selection ──────────────────────────────────────────────────────
    if args.source:
        selected = TILE_SOURCES[args.source]
    else:
        selected = choose_tile_source()

    tile_dir  = args.tile_dir  or selected['dir']
    index_dir = args.index_dir or INDEX_DIR

    print(f'\n來源：{selected["name"]}')
    print(f'Tile 目錄：{tile_dir}')
    print(f'Index 儲存：{index_dir}/{Path(selected["index"]).name}')

    build_index(tile_dir=tile_dir,
                zoom=args.zoom,
                index_dir=index_dir,
                index_name=Path(selected['index']).name)
