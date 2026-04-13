# Technical Details: Demographic Variations Generation

## 1. Visual Difference Calculation (5.89%)

### Metric Definition
The "5.89% visual difference" is calculated as:

```python
# Step 1: Compute MSE in normalized space [-1, 1]
diversity = F.mse_loss(synthetic_video, original_video).item()
# Example: diversity = 0.013877

# Step 2: Convert to pixel difference
# Videos are normalized: pixel = (normalized_value + 1) * 127.5
# Difference in normalized space: sqrt(MSE)
diff_normalized = sqrt(diversity)  # sqrt(0.013877) ≈ 0.1178

# Step 3: Convert to pixel space
diff_pixels = diff_normalized * 127.5  # ≈ 15.02 pixels

# Step 4: Convert to percentage
percentage = (diff_pixels / 255) * 100  # (15.02 / 255) * 100 ≈ 5.89%
```

**Formula**: `percentage = (sqrt(MSE) * 127.5 / 255) * 100`

**For 5.89%**: MSE ≈ 0.013877 in normalized space
**For 1.3%**: MSE ≈ 0.000676 in normalized space

### Why This Metric?
- Videos are normalized to [-1, 1] range
- Pixel values are: `pixel = (normalized + 1) * 127.5`
- MSE captures average squared difference across all pixels and frames
- Percentage makes it interpretable (5.89% = ~15 pixels difference per pixel on average)

---

## 2. Initial 1.3% Difference (Original Approach)

### Original Script (`generate_demographic_variations.py`)
The original approach was **simple direct generation**:

```python
# Original approach - no diversity injection
synthetic = generator(original_video, target_demographics)
diversity = F.mse_loss(synthetic, original_video).item()
# Result: ~0.000676 MSE = 1.3% difference
```

**Why so low?**
- Generator was trained for **perfect reconstruction** (Use Case 3)
- Demographic conditioning was weak (λ_demo = 5.0)
- Generator prioritized reconstruction over demographic variation
- No diversity constraints or feature mixing
- Result: Near-perfect copies with different demographic labels

**Problem**: Variations were visually identical, defeating the purpose of demographic augmentation.

---

## 3. Age Bin Boundary Cycling

### Age Bins
```python
age_bins = [0, 5, 10, 15, 18]  # 5 bins: [0-5), [5-10), [10-15), [15-18), [18+]
```

### Cycling Logic
```python
age_bin_idx = np.digitize(original_age, age_bins)  # Get current bin index (0-4)
new_age_bin = (age_bin_idx + 1) % 5  # Cycle to next bin (wraps around)

# Calculate new age as midpoint of new bin
if new_age_bin < 4:
    new_age = (age_bins[new_age_bin] + age_bins[new_age_bin + 1]) / 2
else:  # Last bin [15-18]
    new_age = 16.5  # Midpoint of 15-18
```

### Examples
- **Original age 3.5** (bin 0: [0-5)) → **New age 7.5** (bin 1: [5-10))
- **Original age 8.0** (bin 1: [5-10)) → **New age 12.5** (bin 2: [10-15))
- **Original age 16.0** (bin 3: [15-18)) → **New age 16.5** (bin 4: [18+))
- **Original age 17.5** (bin 4: [18+)) → **New age 2.5** (bin 0: [0-5), wraps around)

**Note**: Wrapping means a 17-year-old gets changed to a 2.5-year-old, which is a large demographic jump. This is intentional to maximize diversity, but may be unrealistic.

---

## 4. Multi-Candidate Selection Criterion

### Process
1. **Generate 15 candidates** with varying noise levels:
   ```python
   for i in range(15):
       noise_scale = base_noise * (0.6 + i / 14.0 * 0.8)  # Range: 0.6x to 1.4x
       noisy_video = original_video + torch.randn_like(original_video) * noise_scale
       candidate = generator(noisy_video, target_demo)
       diversity = F.mse_loss(candidate, original_video).item()
   ```

2. **Score each candidate** based on target diversity range:
   ```python
   target_diversity_low = 0.010   # ~10% pixel difference (MSE in normalized space)
   target_diversity_high = 0.018   # ~13% pixel difference
   
   if target_diversity_low <= diversity <= target_diversity_high:
       # In target range: score decreases with distance from center
       center = (target_diversity_low + target_diversity_high) / 2  # 0.014
       score = 2.0 - abs(diversity - center) * 20
       # Best score: 2.0 (exactly at center)
       # Worst in range: ~1.92 (at boundaries)
   
   elif diversity < target_diversity_low:
       # Too similar: score proportional to diversity
       score = (diversity / target_diversity_low) * 0.5
       # Range: 0.0 (MSE=0) to 0.5 (MSE=0.010)
   
   else:  # diversity > target_diversity_high
       # Too different: score decreases linearly
       score = max(0, 1.5 - (diversity - target_diversity_high) * 10)
       # At MSE=0.018: score = 1.5
       # At MSE=0.033: score = 0.0
   ```

3. **Sort by score** (highest first):
   ```python
   candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by score
   ```

4. **Blend top 3 candidates**:
   ```python
   if len(candidates) >= 3:
       final_output = 0.5 * candidates[0][0] + 0.3 * candidates[1][0] + 0.2 * candidates[2][0]
   ```

### Selection Criterion Summary
- **Target**: MSE between 0.010-0.018 (10-13% pixel difference)
- **Scoring**: 
  - Best: MSE exactly at 0.014 (center of target range) → score = 2.0
  - In range: Score decreases with distance from center
  - Too similar (MSE < 0.010): Low score (0.0-0.5)
  - Too different (MSE > 0.018): Score decreases linearly
- **Final output**: Weighted blend of top 3 candidates (50% best, 30% second, 20% third)

### Why This Approach?
- **Stability**: Blending multiple candidates reduces artifacts
- **Diversity**: Targets realistic variation range (10-13% matches real videos)
- **Quality**: Prefers candidates in optimal range, not too similar or too different

---

## Summary

| Metric | Calculation | Value |
|--------|-------------|-------|
| **5.89% visual difference** | `(sqrt(MSE) * 127.5 / 255) * 100` where MSE ≈ 0.013877 | 5.89% (~15 pixels) |
| **1.3% original** | Same formula, MSE ≈ 0.000676 | 1.3% (~3.3 pixels) |
| **Age cycling** | `(age_bin_idx + 1) % 5`, new_age = midpoint of new bin | Wraps around at boundaries |
| **Selection criterion** | Score based on MSE in [0.010, 0.018] range, blend top 3 | Targets 10-13% difference |
