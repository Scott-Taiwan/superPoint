"""
Diagnostic for photo #8: bypass Phase 1, force Phase 2 on the true tile.

Sweeps yaw values (-90 to -130) with large-canvas correction, then calls
lightglue_match() directly on tile (439053, 224452).  Reports keypoint
counts and inliers so we know whether the content is matchable at all.
"""

import math, sys
from pathlib import Path

import cv2
import numpy as np
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
os.environ.setdefault('FORCE_CPU', '0')

sys.path.insert(0, str(Path(__file__).parent))
from localize import lightglue_match, _get_models, _extract
from correct_tilt import correction_homography

PHOTO = Path(__file__).parent / 'photo_20260705/original/25.0431990_121.4743276-NOFIX_NOFIX__NAm__59.9m.png'
TRUE_TX, TRUE_TY = 439053, 224452
ZOOM   = 19
TILE_DIR = Path(__file__).parent.parent / 'gpsless_mapping/tiles'

# Preload models once
print('Loading SuperPoint + LightGlue …')
_get_models()
print('Models ready.\n')

orig = cv2.imread(str(PHOTO))
if orig is None:
    sys.exit(f'Cannot read {PHOTO}')
print(f'Original: {orig.shape[1]}×{orig.shape[0]}')


def large_canvas_correct(img, pitch_deg, roll_deg, yaw_deg, fov_h_deg=80.0):
    """Apply tilt correction to a canvas large enough to hold all content."""
    h, w = img.shape[:2]
    H = correction_homography(img.shape, pitch_deg, roll_deg, yaw_deg, fov_h_deg)

    # Map 4 corners through H to find bounding box
    corners = np.array([[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]],
                       dtype=np.float64).T          # 3×4
    mapped = H @ corners                             # 3×4
    mapped /= mapped[2:3, :]                         # normalise
    xs, ys = mapped[0], mapped[1]

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    # Shift so top-left → (0,0)
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0,      1]], dtype=np.float64)
    H2 = T @ H

    corrected = cv2.warpPerspective(img, H2, (new_w, new_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
    return corrected


print('=' * 60)
print(f'True tile: ({TRUE_TX}, {TRUE_TY})\n')

# Sweep key angles only
for yaw in [-100, -110, -118, -125, -130]:
    for pitch in [0, -10]:
        corrected = large_canvas_correct(orig, pitch_deg=pitch, roll_deg=0, yaw_deg=yaw)
        h, w = corrected.shape[:2]

        # Count non-black pixels (valid content)
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        valid_pct = 100 * np.count_nonzero(gray) / gray.size

        # Extract keypoints from corrected image
        feats = _extract(corrected, max_kp=1024)
        n_kp = feats['keypoints'].shape[0]

        # Force Phase 2 on true tile with relaxed min_inliers
        result = lightglue_match(corrected, TRUE_TX, TRUE_TY, ZOOM, TILE_DIR,
                                 radius=1, min_inliers=4)
        inliers = result[2] if result else 0
        lat_est = f'{result[0]:.5f}' if result else 'NOFIX'

        print(f'yaw={yaw:+d}° pitch={pitch:+d}° | '
              f'canvas {w}×{h} valid={valid_pct:.0f}% | '
              f'kp={n_kp:4d} | inliers={inliers:3d} | {lat_est}')

import torch, gc
torch.cuda.empty_cache(); gc.collect()
