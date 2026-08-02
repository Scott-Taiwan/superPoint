# SuperPoint + LightGlue 無 GPS 定位系統 — 開發注意事項

## 1. Index 與 Query 必須使用相同尺度

**問題**：SuperPoint 的預設 `preprocess_conf = {'resize': 1024}`，會把輸入圖片**放大**到最長邊 1024px 才提取特徵。

- Index 建立時（tile 256×256）→ 放大到 1024×1024 提特徵
- Query 時（drone 照片）→ `_img_to_tensor` 先縮到 640px

兩邊尺度不同，FLANN 投票只能拿到 ~7 票（正常應 30+），導致 Phase 1 定位失敗。

**解法**：`build_index.py` 和 `localize.py` 兩處都必須加上：
```python
extractor.preprocess_conf = {'resize': 640}
```
修改後必須**重新執行 `build_index.py` 重建 index**，兩邊才會對齊。

---

## 2. Jetson NvMap 崩潰（CUDACachingAllocator 錯誤）

**錯誤訊息**：
```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:838
```

**原因**：`preprocess_conf = {'resize': 1024}` 把 640×360 的輸入放大到 1024×576，SuperPoint encoder 的卷積層需要分配 ~225 MB 的連續 NvMap 記憶體，Jetson 無法完成。

**解法**：同上，設定 `preprocess_conf = {'resize': 640}` 即可避免放大。

---

## 3. Homography 投影中心點的尺度 Bug

**問題**：`lightglue_match` 計算 GPS 時，H（homography）是在 **縮放後的圖片空間**（640px）中建立的，但投影 query 中心點時卻使用**原始圖片尺寸**：
```python
h, w = query_bgr.shape[:2]  # 原始：1280×720
centre = cv2.perspectiveTransform(np.float32([[[w/2, h/2]]]), H)
# 錯誤：投影的是 (640, 360)，等於 640×360 圖片的右下角，不是中心點
```

**解法**：計算縮放後的尺寸，用縮放後的中心點投影：
```python
if max(h_orig, w_orig) > 640:
    sq = 640 / max(h_orig, w_orig)
    qw, qh = int(w_orig * sq), int(h_orig * sq)
else:
    qw, qh = w_orig, h_orig
centre = cv2.perspectiveTransform(np.float32([[[qw/2, qh/2]]]), H)
```

**效果**：此修正讓定位誤差從 **56.8m 降至 19.4m**（降低 66%）。

---

## 4. Composite Tile 的像素座標也需要換算回原始空間

**問題**：3×3 composite tile（768×768）同樣被 `_img_to_tensor` 縮到 640×640，H 的目標空間是 640px。投影後的 `cpx, cpy` 是在 640px 空間，但計算落在哪個 tile 時用的是原始的 `TILE_SIZE=256`：
```python
tile_dx = int(cpx / TILE_SIZE)  # 錯誤：cpx 在 640px 空間，不是 768px 空間
```

**解法**：先除以縮放比例，換算回原始空間：
```python
_sc = 640 / 768  # composite 縮放比
cpx_t = cpx / _sc  # 換算回 768px 空間
tile_dx = int(cpx_t / TILE_SIZE)
local_px = cpx_t - tile_dx * TILE_SIZE
```

---

## 5. GStreamer appsink 造成程式卡住

**問題**：`drone_localize.cpp` 的 GStreamer pipeline 沒有 `sync=false`，appsink 會等待 pipeline clock 同步而永久阻塞。

**解法**：appsink 加上以下參數：
```
appsink sync=false max-buffers=1 drop=true
```

---

## 6. 傾斜拍攝導致 Homography 失敗

**問題**：drone 傾斜約 35° 拍攝時，地面的透視形變無法用 2D projective transform（homography）正確建模，導致 NO FIX。

**解法**：在 `drone_localize.cpp` 加入 Attitude Guard，當 roll 或 pitch 超過 5° 時跳過拍照：
```cpp
const float MAX_TILT_DEG = 5.0f;
if (std::fabs(roll_deg) > MAX_TILT_DEG || std::fabs(pitch_deg) > MAX_TILT_DEG) {
    // 跳過此次拍攝，提前觸發下一次
}
```
需透過 MAVLink `MAV_DATA_STREAM_EXTRA1` 取得 ATTITUDE 資料。

---

## 7. 低紋理地面容易 NO FIX

- **草地、停車場、均勻鋪面**：特徵點少，Phase 1 投票分散，Phase 2 無法建立 homography
- **籃球場、跑道、建築物屋頂**：特徵豐富，成功率高

建議飛行路線應盡量涵蓋有明顯幾何特徵的區域。

---

## 8. Tile Server URL

使用 ESRI World Imagery（免費，無需 API key）：
```
https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
```
注意：URL 順序是 `z/y/x`，不是 `z/x/y`。

---

## 9. 兩階段定位流程摘要

```
Phase 1 — FLANN 粗定位
  Query SuperPoint 特徵 → 與所有 tile descriptor 做 top-1 最近鄰投票
  → 選出票數最高的前 N 個候選 tile

Phase 2 — LightGlue 精定位
  取候選 tile 及其 8 個鄰居拼成 3×3 composite
  → SuperPoint 提特徵 → LightGlue 匹配
  → RANSAC homography → 投影 query 中心點 → 換算 GPS
```

Inlier ratio guard：若 RANSAC inlier 比例 < 30%，視為誤匹配，捨棄。

---

## 10. 目前已知限制

| 限制 | 說明 |
|---|---|
| 誤差約 19m | 主要來自 drone 輕微傾斜、衛星影像與空拍視角差異 |
| 夜間不可用 | 衛星影像為日間可見光，無紅外線資料庫 |
| 需預先下載 tiles | 離線使用，需事先覆蓋飛行區域 |
| 低紋理地面成功率低 | 草地等區域特徵不足 |
