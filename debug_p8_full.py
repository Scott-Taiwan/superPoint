"""
Photo #8 full pipeline diagnostic:
- Apply large-canvas correction (best angle from debug_p8_force.py)
- Run Phase 1 (FLANN) and report top-20 votes with distance to true tile
- Run Phase 2 on ALL top candidates AND on the true tile
"""

import sys, os, pickle
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
sys.path.insert(0, str(Path(__file__).parent))

from localize import lightglue_match, _get_models, _extract, load_index, flann_vote
from correct_tilt import correction_homography
from config import (ZOOM_LEVEL, INDEX_DIR, TILE_DIR, MATCH_RATIO,
                    TOP_CANDIDATES, MIN_INLIERS, SP_QUERY_MAX_KEYPOINTS)

PHOTO    = Path(__file__).parent / 'photo_20260705/original/25.0431990_121.4743276-NOFIX_NOFIX__NAm__59.9m.png'
TRUE_TX, TRUE_TY = 439053, 224452
ZOOM     = 19
TILE_DIR_P = Path(__file__).parent.parent / 'gpsless_mapping/tiles'

print('Loading models + index …')
_get_models()
index = load_index(INDEX_DIR, ZOOM)
print(f'Index: {len(index)} tiles.\n')


def large_canvas_correct(img, pitch_deg, roll_deg, yaw_deg, fov_h_deg=80.0):
    h, w = img.shape[:2]
    H = correction_homography(img.shape, pitch_deg, roll_deg, yaw_deg, fov_h_deg)
    corners = np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]], dtype=np.float64).T
    mapped  = H @ corners
    mapped /= mapped[2:3, :]
    min_x, max_x = mapped[0].min(), mapped[0].max()
    min_y, max_y = mapped[1].min(), mapped[1].max()
    T  = np.array([[1,0,-min_x],[0,1,-min_y],[0,0,1]], dtype=np.float64)
    corrected = cv2.warpPerspective(img, T @ H,
                                    (int(np.ceil(max_x-min_x)), int(np.ceil(max_y-min_y))),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return corrected


def run_phase1(img):
    feats   = _extract(img, max_kp=SP_QUERY_MAX_KEYPOINTS)
    desc_np = feats['descriptors'].cpu().numpy()  # (N, 256)
    print(f'  SuperPoint: {len(desc_np)} descriptors')
    ranked  = flann_vote(desc_np, index)
    return ranked


orig = cv2.imread(str(PHOTO))

# Best angles from debug_p8_force
test_cases = [
    (-118, -10, 'yaw=-118 pitch=-10  [best inliers=56]'),
    (-125, -10, 'yaw=-125 pitch=-10  [inliers=53]'),
    (-110,   0, 'yaw=-110 pitch=0    [inliers=43]'),
    (-118,   0, 'yaw=-118 pitch=0    [inliers=32]'),
]

for yaw, pitch, label in test_cases:
    print('─' * 70)
    print(f'Case: {label}')
    corrected = large_canvas_correct(orig, pitch_deg=pitch, roll_deg=0, yaw_deg=yaw)

    # Phase 1: FLANN vote
    ranked = run_phase1(corrected)
    print(f'  Phase 1: {len(ranked)} tiles voted.')
    true_rank = None
    for rank, (ti, votes) in enumerate(ranked[:TOP_CANDIDATES]):
        tx, ty = index[ti]['tile_x'], index[ti]['tile_y']
        dist   = abs(tx - TRUE_TX) + abs(ty - TRUE_TY)   # L1 distance in tiles
        marker = ' ← TRUE' if (tx, ty) == (TRUE_TX, TRUE_TY) else ''
        if (tx, ty) == (TRUE_TX, TRUE_TY):
            true_rank = rank + 1
        if rank < 5 or (tx, ty) == (TRUE_TX, TRUE_TY):
            print(f'    #{rank+1:2d}: tile({tx},{ty}) votes={votes}  Δ={dist}{marker}')

    if true_rank is None:
        # Check if true tile appears anywhere in all votes
        for ti, votes in ranked:
            if index[ti]['tile_x'] == TRUE_TX and index[ti]['tile_y'] == TRUE_TY:
                true_rank = ranked.index((ti, votes)) + 1
                print(f'    True tile found at rank #{true_rank} (votes={votes})')
                break
        else:
            print(f'    True tile NOT in FLANN votes at all.')

    # Phase 2: forced on true tile
    result = lightglue_match(corrected, TRUE_TX, TRUE_TY, ZOOM, TILE_DIR_P,
                             radius=1, min_inliers=4)
    if result:
        lat, lon, inliers, *_ = result
        print(f'  Phase 2 (true tile forced): inliers={inliers}  → ({lat:.5f}, {lon:.5f})')
    else:
        print(f'  Phase 2 (true tile forced): FAIL (< 4 inliers or geometry bad)')

    # Phase 2: on top FLANN candidate
    if ranked:
        top_ti, top_votes = ranked[0]
        top_tx, top_ty = index[top_ti]['tile_x'], index[top_ti]['tile_y']
        result2 = lightglue_match(corrected, top_tx, top_ty, ZOOM, TILE_DIR_P,
                                  radius=1, min_inliers=4)
        if result2:
            lat2, lon2, inl2, *_ = result2
            print(f'  Phase 2 (top FLANN tile ({top_tx},{top_ty})): inliers={inl2}  → ({lat2:.5f}, {lon2:.5f})')
        else:
            print(f'  Phase 2 (top FLANN tile ({top_tx},{top_ty})): FAIL')

    print()

torch.cuda.empty_cache()
