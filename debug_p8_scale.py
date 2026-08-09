"""
Test whether rescaling the large-canvas corrected image back to original
dimensions fixes the Phase 1 scale mismatch.

Three variants:
  A: large canvas (as-is)       → scale mismatch → Phase 1 scatters
  B: large canvas + resize 1280×720              → same ground coverage as original
  C: original canvas + BORDER_CONSTANT=0         → valid content as rhombus, no fill

For each, report Phase 1 vote for true tile vs top-1 tile.
"""

import sys, os
from pathlib import Path
import cv2
import numpy as np
import torch

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
sys.path.insert(0, str(Path(__file__).parent))

from localize import _get_models, _extract, flann_vote, lightglue_match, load_index
from correct_tilt import correction_homography
from config import INDEX_DIR, SP_QUERY_MAX_KEYPOINTS, TOP_CANDIDATES

PHOTO    = Path(__file__).parent / 'photo_20260705/original/25.0431990_121.4743276-NOFIX_NOFIX__NAm__59.9m.png'
TRUE_TX, TRUE_TY = 439053, 224452
ZOOM     = 19
TILE_DIR_P = Path(__file__).parent.parent / 'gpsless_mapping/tiles'

print('Loading models + index …')
_get_models()
index = load_index(INDEX_DIR, ZOOM)
print(f'{len(index)} tiles loaded.\n')

orig = cv2.imread(str(PHOTO))
OH, OW = orig.shape[:2]   # 720, 1280

PITCH, ROLL, YAW = -10.0, 0.0, -118.0   # best from previous experiment


def apply_H(img, pitch, roll, yaw, output_size, border=cv2.BORDER_REPLICATE):
    H = correction_homography(img.shape, pitch, roll, yaw)
    return cv2.warpPerspective(img, H, output_size,
                               flags=cv2.INTER_LINEAR, borderMode=border,
                               borderValue=0)


def large_canvas(img, pitch, roll, yaw):
    h, w = img.shape[:2]
    H = correction_homography(img.shape, pitch, roll, yaw)
    corners = np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]], dtype=np.float64).T
    mapped  = H @ corners;  mapped /= mapped[2:3]
    min_x, max_x = mapped[0].min(), mapped[0].max()
    min_y, max_y = mapped[1].min(), mapped[1].max()
    new_w, new_h = int(np.ceil(max_x-min_x)), int(np.ceil(max_y-min_y))
    T   = np.array([[1,0,-min_x],[0,1,-min_y],[0,0,1]], dtype=np.float64)
    out = cv2.warpPerspective(img, T@H, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def phase1_report(label, img):
    feats   = _extract(img, max_kp=SP_QUERY_MAX_KEYPOINTS)
    desc_np = feats['descriptors'].cpu().numpy()
    ranked  = flann_vote(desc_np, index)

    top_tx, top_ty = index[ranked[0][0]]['tile_x'], index[ranked[0][0]]['tile_y']
    top_votes = ranked[0][1]

    true_rank, true_votes = None, 0
    for r, (ti, v) in enumerate(ranked):
        if index[ti]['tile_x'] == TRUE_TX and index[ti]['tile_y'] == TRUE_TY:
            true_rank, true_votes = r+1, v
            break

    print(f'  [{label}]  kp={len(desc_np):4d}  '
          f'top=({top_tx},{top_ty}) votes={top_votes}  '
          f'true rank=#{true_rank or "N/A"} votes={true_votes}')
    return ranked


print(f'pitch={PITCH}° roll={ROLL}° yaw={YAW}°   original={OW}×{OH}\n')

# ── Variant A: large canvas as-is ────────────────────────────────────────────
lc = large_canvas(orig, PITCH, ROLL, YAW)
print(f'Variant A (large canvas {lc.shape[1]}×{lc.shape[0]}):')
ranked_A = phase1_report('A', lc)

# Phase 2 forced
r = lightglue_match(lc, TRUE_TX, TRUE_TY, ZOOM, TILE_DIR_P, radius=1, min_inliers=4)
print(f'  [A] Phase 2 forced → inliers={r[2] if r else "FAIL"}')

# ── Variant B: large canvas → resize back to original WxH ────────────────────
lc_resized = cv2.resize(lc, (OW, OH), interpolation=cv2.INTER_LINEAR)
print(f'\nVariant B (large canvas resized to {OW}×{OH}):')
ranked_B = phase1_report('B', lc_resized)

r2 = lightglue_match(lc_resized, TRUE_TX, TRUE_TY, ZOOM, TILE_DIR_P, radius=1, min_inliers=4)
print(f'  [B] Phase 2 forced → inliers={r2[2] if r2 else "FAIL"}')

# Top FLANN candidate from B
top_ti_B = ranked_B[0][0]
top_tx_B, top_ty_B = index[top_ti_B]['tile_x'], index[top_ti_B]['tile_y']
r2b = lightglue_match(lc_resized, top_tx_B, top_ty_B, ZOOM, TILE_DIR_P, radius=1, min_inliers=4)
print(f'  [B] Phase 2 top FLANN ({top_tx_B},{top_ty_B}) → inliers={r2b[2] if r2b else "FAIL"}')

# ── Variant C: original canvas + BORDER_CONSTANT (no fill) ────────────────────
orig_const = apply_H(orig, PITCH, ROLL, YAW, (OW, OH), border=cv2.BORDER_CONSTANT)
print(f'\nVariant C (original size {OW}×{OH} + BORDER_CONSTANT=black):')
ranked_C = phase1_report('C', orig_const)

r3 = lightglue_match(orig_const, TRUE_TX, TRUE_TY, ZOOM, TILE_DIR_P, radius=1, min_inliers=4)
print(f'  [C] Phase 2 forced → inliers={r3[2] if r3 else "FAIL"}')

top_ti_C = ranked_C[0][0]
top_tx_C, top_ty_C = index[top_ti_C]['tile_x'], index[top_ti_C]['tile_y']
r3b = lightglue_match(orig_const, top_tx_C, top_ty_C, ZOOM, TILE_DIR_P, radius=1, min_inliers=4)
print(f'  [C] Phase 2 top FLANN ({top_tx_C},{top_ty_C}) → inliers={r3b[2] if r3b else "FAIL"}')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('SUMMARY  (pitch=-10° yaw=-118°):')
print(f'  A large_canvas        : top votes={ranked_A[0][1]}')
print(f'  B lc→resize 1280×720  : top votes={ranked_B[0][1]}')
print(f'  C orig+BORDER_CONST   : top votes={ranked_C[0][1]}')
print('(higher top-vote ≠ correct; check whether true tile rank improved)')

torch.cuda.empty_cache()
