# 大角度 Yaw 下的定位方法筆記

## 背景

測試照片 #8（`25.0431990_121.4743276-NOFIX_NOFIX__NAm__59.9m.png`）
- 拍攝時 yaw ≈ 118°（機頭朝向 ESE，非北方）
- 現有 SuperPoint + LightGlue 定位流程完全失敗（NO FIX）
- 真實 tile：(439053, 224452)，zoom 19

---

## 根本原因

### Phase 1 (FLANN 投票) 為何失敗

SuperPoint 描述子有兩個弱點：

1. **非旋轉不變**  
   yaw 118° 的影像，其 SuperPoint 描述子和垂直俯視衛星 tile 的描述子
   完全不同。即使做了透視校正（H = K·R·K⁻¹），仍然失敗。

2. **非縮放不變（scale mismatch）**  
   無人機 60m 高度 + FOV 80°：地面解析度 ≈ 0.08 m/px  
   衛星 tile（zoom 19）：地面解析度 ≈ 0.27 m/px  
   差距約 3.4 倍 → SuperPoint FLANN 投票結果散亂，正確 tile 永遠在前 300 名之後

### Phase 2 (LightGlue) 為何本來可以成功

強制對真實 tile 做 Phase 2（繞過 Phase 1）：
- 大 canvas 校正（yaw=-118°, pitch=-10°）+ 真實 tile (439053, 224452)
- → **58 inliers** ← 幾何內容完全可以比對

**結論：影像的幾何內容足夠，問題只出在 Phase 1 找不到正確的候選 tile**

---

## 可行的解決方法

### 核心：SIFT Phase 1 ＋ LightGlue Phase 2（混合模式）

```
1. large_canvas_correct(img, pitch, roll, yaw)
   ├─ H = K · Rx(-pitch) · Ry(-roll) · Rz(-yaw) · K⁻¹
   ├─ 計算四角映射後的 bounding box
   ├─ 用 BORDER_CONSTANT=0（黑色），不用 BORDER_REPLICATE
   └─ 輸出比原圖大的 canvas，避免有效內容被裁切

2. SIFT Phase 1（旋轉+縮放不變）
   ├─ 用 cv2.SIFT_create(nfeatures=2000) 提取描述子
   ├─ 對 SIFT index（sift_index_z19.pkl）做 FLANN knnMatch（ratio=0.75）
   └─ 取前 30 個候選 tile

3. LightGlue Phase 2（radius=2，比預設 radius=1 大）
   ├─ 對每個候選 tile 建 5×5 composite（涵蓋範圍更大）
   ├─ SuperPoint + LightGlue 比對
   ├─ RANSAC → GPS
   └─ min_inliers=8（可接受，因大 canvas 特徵較少）
```

### 測試結果（photo #8，yaw 約 118°，無實測 attitude）

| 角度假設 | Phase 1 | radius | inliers | 估計誤差 |
|----------|---------|--------|---------|---------|
| yaw=-118°, pitch=-10° | SIFT rank#9 | r=2 | **65** | **33m** |
| yaw=-118°, pitch=0°  | SIFT top-20 | r=1 | 失敗 | — |
| yaw=-110°, pitch=0°  | SIFT top-20 | r=1 | 失敗 | — |

最佳角度組合：**pitch=-10°, yaw=-118°**（目視估計，非量測值）

### 33m 誤差的來源

yaw 和 pitch 都是目視估計，不是 IMU 量測值。
實際飛行有 `.attitude.json`（±1° 精度），誤差預計 < 10m。

---

## 實作時需要改動的地方

### 1. `correct_tilt.py`
已有 `correction_homography(img, pitch, roll, yaw)` — 不需改動。  
需加 `large_canvas_correct()` 函式（現有 debug 腳本中有實作）。

### 2. `localize.py`
目前只有 SuperPoint Phase 1。需加入：
- 接受 `phase1='sift'` 參數切換模式
- 或在 `localize()` 函式前先做 large canvas 校正（傳入 pitch/roll/yaw）
- `radius=2` 選項（現在是寫死 radius=1）

### 3. SIFT index 路徑
```
/home/scott/claude-project/gpsless_superpoint/gpsless_SIFT/index/sift_index_z19.pkl
```
547 tiles，123,123 keypoints，描述子維度 128（SIFT 標準）

### 4. 觸發條件（何時啟用 SIFT 模式）
```python
if abs(yaw_deg) > 30:   # 大角度 yaw，SuperPoint Phase 1 無效
    use_large_canvas = True
    phase1 = 'sift'
    radius = 2
else:                    # 正常前進飛行（pitch ≤ 15°，yaw 偏移小）
    use_large_canvas = False   # 或用原始 BORDER_REPLICATE
    phase1 = 'superpoint'
    radius = 1
```

---

## 診斷用腳本（保留於 gpsless_superpoint/）

| 腳本 | 用途 |
|------|------|
| `debug_p8_force.py` | 強制 Phase 2 對真實 tile → 確認影像可 match |
| `debug_p8_full.py` | 完整 Phase 1+2 診斷，顯示 true tile 的投票排名 |
| `debug_p8_scale.py` | 測試三種縮放策略對 Phase 1 的影響 |
| `debug_p8_sift_phase1.py` | SIFT Phase 1 + LightGlue Phase 2 混合測試 |
| `debug_p8_radius2.py` | 最終驗證：SIFT Phase 1 + radius=2 → 65 inliers, 33m |
