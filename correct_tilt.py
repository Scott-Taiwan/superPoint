"""
Correct drone photo camera tilt to near-nadir (straight-down) view,
then optionally re-run SuperPoint localization on the corrected image.

Two modes:
  Auto  — reads companion .attitude.json saved by drone_localize
  Manual— use --pitch / --roll arguments

Usage:
    # With attitude JSON (saved automatically by new drone_localize):
    python correct_tilt.py photo.png --localize

    # Manual angles (estimate from visual inspection):
    python correct_tilt.py photo.png --pitch 30 --roll 0 --localize

    # Batch: correct all photos in a folder
    python correct_tilt.py photo_20260705/original/*.png --pitch 30 --localize

Convention:
    pitch > 0  camera tilts forward  (drone nose up  → scene stretches toward top)
    pitch < 0  camera tilts backward (drone nose down → scene stretches toward bottom)
    roll  > 0  right side low        (drone banks right)
    roll  < 0  left side low
"""

import argparse, json, math, re
from pathlib import Path

import cv2
import numpy as np


# ── Camera model ──────────────────────────────────────────────────────────────

def _build_K(w, h, fov_h_deg):
    """Pinhole camera intrinsic matrix (estimated from image size + FOV)."""
    f = w / (2 * math.tan(math.radians(fov_h_deg / 2)))
    return np.array([[f, 0, w / 2],
                     [0, f, h / 2],
                     [0, 0, 1    ]], dtype=np.float64)


def correction_homography(img_shape, pitch_deg, roll_deg, yaw_deg=0.0, fov_h_deg=80.0):
    """
    Compute the homography H that maps the tilted/yawed image to a nadir-corrected image.

    The drone camera points straight down when level and yaw=0 (North-up).
    When the drone pitches, rolls, or yaws, the image no longer matches the
    nadir satellite tiles.  We apply the inverse rotation through the camera
    model to produce a virtual nadir North-up view.

        H = K · R_correction · K⁻¹
        R_correction = Rx(-pitch) · Ry(-roll) · Rz(-yaw)

    IMPORTANT: SuperPoint descriptors are NOT rotation-invariant.  A drone
    yawed 40° produces descriptors that FLANN cannot match to the index.
    Always correct yaw first when the drone's heading is known.

    Convention (all angles: positive = clockwise when viewed from above/camera):
        pitch_deg  camera tilts forward (drone nose up → top of image looks far)
        roll_deg   camera tilts right   (drone banks right → right side low)
        yaw_deg    camera rotates CW    (drone heading 45° = top of image is NE)
    """
    h, w = img_shape[:2]
    K    = _build_K(w, h, fov_h_deg)

    p = math.radians(-pitch_deg)   # negate: we undo the rotation
    r = math.radians(-roll_deg)
    y = math.radians(-yaw_deg)

    # Pitch correction: rotate around horizontal axis (X)
    Rx = np.array([[1, 0,            0           ],
                   [0, math.cos(p), -math.sin(p) ],
                   [0, math.sin(p),  math.cos(p) ]], dtype=np.float64)

    # Roll correction: rotate around depth axis into scene (Y in image coords)
    Ry = np.array([[ math.cos(r), 0, math.sin(r)],
                   [0,            1, 0           ],
                   [-math.sin(r), 0, math.cos(r)]], dtype=np.float64)

    # Yaw correction: rotate around optical axis (Z).
    # Since Z is the optical axis, this is purely a 2D in-plane rotation.
    Rz = np.array([[ math.cos(y), -math.sin(y), 0],
                   [ math.sin(y),  math.cos(y), 0],
                   [0,             0,            1]], dtype=np.float64)

    R = Rx @ Ry @ Rz
    H = K @ R @ np.linalg.inv(K)
    return H


def correct_tilt(img, pitch_deg, roll_deg, yaw_deg=0.0, fov_h_deg=80.0):
    """Apply perspective correction for pitch, roll and yaw; return (corrected_img, H)."""
    h, w = img.shape[:2]
    H = correction_homography(img.shape, pitch_deg, roll_deg, yaw_deg, fov_h_deg)
    corrected = cv2.warpPerspective(img, H, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    return corrected, H


# ── GPS distance helper ───────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000; p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 +
         math.cos(lat1 * p) * math.cos(lat2 * p) *
         math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0, a)))


# ── Main ──────────────────────────────────────────────────────────────────────

def process_one(img_path: Path, pitch_deg, roll_deg, yaw_deg, fov_h_deg,
                out_dir: Path, do_localize: bool):

    # ── load attitude ────────────────────────────────────────────────────────
    json_path = img_path.with_suffix('.attitude.json')
    p, r, y = pitch_deg, roll_deg, yaw_deg

    if json_path.exists() and any(v is None for v in (p, r, y)):
        with open(json_path) as f:
            att = json.load(f)
        if p is None: p = float(att.get('pitch_deg', 0.0))
        if r is None: r = float(att.get('roll_deg',  0.0))
        if y is None: y = float(att.get('yaw_deg',   0.0))
        print(f'  Attitude from JSON: pitch={p:.1f}°  roll={r:.1f}°  yaw={y:.1f}°')

    if p is None or r is None:
        print(f'  SKIP {img_path.name} — no attitude data '
              f'(provide --pitch/--roll or companion .attitude.json)')
        return

    p = p or 0.0
    r = r or 0.0
    y = y or 0.0

    # ── load & correct image ─────────────────────────────────────────────────
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'  ERROR: cannot read {img_path}')
        return

    corrected, _ = correct_tilt(img, p, r, y, fov_h_deg)

    out_name = (img_path.stem
                + f'_corrected_p{p:.0f}r{r:.0f}y{y:.0f}'
                + img_path.suffix)
    out_path = out_dir / out_name
    cv2.imwrite(str(out_path), corrected)
    print(f'  Saved → {out_path.name}')

    # ── optional localization ────────────────────────────────────────────────
    if not do_localize:
        return

    import os, sys, gc, torch
    sys.path.insert(0, str(Path(__file__).parent))
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
    from localize import localize

    print(f'  Running localization…')
    result = localize(str(out_path), zoom=19)
    torch.cuda.empty_cache(); gc.collect()

    if result:
        est_lat, est_lon = result
        m = re.match(r'([\d.]+)_([\d.]+)', img_path.name)
        if m:
            true_lat, true_lon = float(m.group(1)), float(m.group(2))
            dist = haversine_m(true_lat, true_lon, est_lat, est_lon)
            print(f'  Result: {est_lat:.7f}, {est_lon:.7f}  誤差: {dist:.1f} m')
        else:
            print(f'  Result: {est_lat:.7f}, {est_lon:.7f}')
    else:
        print(f'  Result: NO FIX')


def main():
    parser = argparse.ArgumentParser(
        description='Correct drone photo tilt and optionally re-localize')
    parser.add_argument('images', nargs='+', help='Drone photo(s) to correct')
    parser.add_argument('--pitch',    type=float, default=None,
                        help='Camera pitch from nadir in degrees '
                             '(positive = forward tilt). Read from .attitude.json if omitted.')
    parser.add_argument('--roll',     type=float, default=None,
                        help='Camera roll from nadir in degrees '
                             '(positive = right side low). Read from .attitude.json if omitted.')
    parser.add_argument('--yaw',      type=float, default=None,
                        help='Camera yaw in degrees — drone heading clockwise from North '
                             '(positive = top of image faces NE). Read from .attitude.json if omitted.')
    parser.add_argument('--fov',      type=float, default=80.0,
                        help='Camera horizontal FOV in degrees (default: 80)')
    parser.add_argument('--localize', action='store_true',
                        help='Run SuperPoint localization on corrected image')
    parser.add_argument('--out-dir',  default=None,
                        help='Output directory (default: same as input image)')
    args = parser.parse_args()

    for img_str in args.images:
        img_path = Path(img_str)
        out_dir  = Path(args.out_dir) if args.out_dir else img_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n{img_path.name}')
        process_one(img_path, args.pitch, args.roll, args.yaw, args.fov,
                    out_dir, args.localize)


if __name__ == '__main__':
    main()
