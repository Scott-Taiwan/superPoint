"""
Batch test: run localize() on all photos in photo_without_legs/
and print a clean accuracy table.
"""
import re, math, gc
from pathlib import Path

import torch
torch.cuda.empty_cache()
gc.collect()

from localize import localize

PHOTO_DIR = Path('photo_without_legs')


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000; p = math.pi / 180
    a = (math.sin((lat2-lat1)*p/2)**2 +
         math.cos(lat1*p) * math.cos(lat2*p) *
         math.sin((lon2-lon1)*p/2)**2)
    return 2 * R * math.asin(math.sqrt(max(0, a)))


photos = sorted(PHOTO_DIR.glob('*.png'))
rows = []

for photo in photos:
    m = re.match(r'([\d.]+)_([\d.]+)', photo.name)
    exp_lat = float(m.group(1)) if m else None
    exp_lon = float(m.group(2)) if m else None
    alt_m   = re.search(r'__([\d.]+)m', photo.name)
    alt     = float(alt_m.group(1)) if alt_m else None
    edited  = '_edited' in photo.name

    out = localize(str(photo), zoom=19)

    if out is None:
        rows.append((photo.name, alt, edited, exp_lat, exp_lon, None, None, None))
    else:
        res_lat, res_lon = out
        err = haversine_m(exp_lat, exp_lon, res_lat, res_lon) if exp_lat else None
        rows.append((photo.name, alt, edited, exp_lat, exp_lon, res_lat, res_lon, err))

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Photo':<56} {'Alt':>5} {'Edit':>5}  {'Error(m)':>9}  Status")
print('-' * 90)
errors = []
for name, alt, ed, el, elo, rl, rlo, err in rows:
    alt_s = f'{alt:.0f}m' if alt else '?'
    ed_s  = 'yes' if ed else 'no'
    if err is None and rl is None:
        print(f"{name:<56} {alt_s:>5} {ed_s:>5}  {'—':>9}  NO FIX")
    elif err is None:
        print(f"{name:<56} {alt_s:>5} {ed_s:>5}  {'(no ref)':>9}  FIX")
    else:
        stat = 'OK' if err < 150 else ('WARN' if err < 500 else 'FAR')
        print(f"{name:<56} {alt_s:>5} {ed_s:>5}  {err:>9.1f}  {stat}")
        errors.append(err)

print('-' * 90)
fixed_all = [r for r in rows if r[5] is not None]
fixed_ref = [r for r in rows if r[7] is not None]
print(f"\nFixed : {len(fixed_all)}/21  |  With GPS ref: {len(fixed_ref)}")
if errors:
    errors_sorted = sorted(errors)
    print(f"Mean  : {sum(errors)/len(errors):.0f} m")
    print(f"Median: {errors_sorted[len(errors)//2]:.0f} m")
    print(f"Max   : {max(errors):.0f} m")
    print(f"< 150m: {sum(1 for e in errors if e < 150)}/{len(errors)}")
    print(f"< 50m : {sum(1 for e in errors if e < 50)}/{len(errors)}")
