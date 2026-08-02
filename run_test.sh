#!/bin/bash
# Run the batch test with the Jetson CUDA allocator fix.
# Must be set BEFORE Python starts — env var is read at driver init time.

cd "$(dirname "$0")"

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
python3 batch_test.py 2>&1 | grep -v \
  "NvMap\|NvMem\|libpng warning\|\[ WARN\|Loading SuperPoint\|Query image\|Query feat\|Loading index\|Tile index\|Total train\|Query kp voted\|Top candidate\|  tile (\|LightGlue match on\|Could not determine\|Suggestions:\|  •\|=====\|GPS Loc\|RANSAC\|Google Maps"
