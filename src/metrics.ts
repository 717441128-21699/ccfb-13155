/** 质量指标:等效视数 ENL、边缘锐度(10–90% 上升沿宽度)、分类精度 */

/** 等效视数 ENL = mean² / var(均匀区域,线性强度) */
export function enl(data: ArrayLike<number>, indices: ArrayLike<number>): number {
  let sum = 0, sum2 = 0, n = 0;
  for (let k = 0; k < indices.length; k++) {
    const v = data[indices[k]];
    sum += v; sum2 += v * v; n++;
  }
  if (n < 2) return 0;
  const mean = sum / n;
  const varr = sum2 / n - mean * mean;
  return varr > 0 ? (mean * mean) / varr : Infinity;
}

/** 在 w×h 图中取以 (cx,cy) 为中心、边长 size 的方形区域像素索引 */
export function regionIndices(w: number, h: number, cx: number, cy: number, size: number): Uint32Array {
  const x0 = Math.max(0, Math.round(cx - size / 2)), y0 = Math.max(0, Math.round(cy - size / 2));
  const x1 = Math.min(w - 1, Math.round(cx + size / 2)), y1 = Math.min(h - 1, Math.round(cy + size / 2));
  const out = new Uint32Array((x1 - x0 + 1) * (y1 - y0 + 1));
  let k = 0;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) out[k++] = y * w + x;
  return out;
}

/**
 * 边缘锐度:过 (cx,cy) 沿水平方向取长度 len 的剖面,找最强阶跃,
 * 用 10%–90% 过渡带宽度(像素)度量锐度。自动适配上升/下降沿。
 * 返回 null 表示剖面内没有明显阶跃。
 */
export function edgeRiseWidth(data: ArrayLike<number>, w: number, h: number, cx: number, cy: number, len = 41): number | null {
  const half = Math.floor(len / 2);
  const x0 = Math.max(1, cx - half), x1 = Math.min(w - 2, cx + half);
  if (x1 - x0 < 8 || cy < 0 || cy >= h) return null;
  const row = cy * w;
  // 找剖面内最大一阶差分位置作为边缘中心
  let best = -1, bestMag = 0;
  for (let x = x0; x < x1; x++) {
    const d = Math.abs(data[row + x + 1] - data[row + x]);
    if (d > bestMag) { bestMag = d; best = x; }
  }
  if (best < 0) return null;
  // 两侧平台电平:边缘左右各取 1/4 窗长的中位数近似(用均值即可,剖面已较短)
  const win = Math.max(3, Math.floor((x1 - x0) / 4));
  let loSum = 0, loN = 0, hiSum = 0, hiN = 0;
  for (let x = Math.max(x0, best - win); x < best; x++) { loSum += data[row + x]; loN++; }
  for (let x = best + 1; x <= Math.min(x1, best + 1 + win); x++) { hiSum += data[row + x]; hiN++; }
  if (loN === 0 || hiN === 0) return null;
  let a = loSum / loN, b = hiSum / hiN;
  if (Math.abs(b - a) < 1e-12) return null;
  const rising = b > a;
  const lo = rising ? a : b, hi = rising ? b : a;
  const t10 = lo + 0.1 * (hi - lo), t90 = lo + 0.9 * (hi - lo);
  // 从中心向两侧找首次穿过阈值的像素
  let xLo = best, xHi = best + 1;
  for (let x = best; x >= x0; x--) { if (rising ? data[row + x] <= t10 : data[row + x] >= t90) { xLo = x; break; } xLo = x; }
  for (let x = best + 1; x <= x1; x++) { if (rising ? data[row + x] >= t90 : data[row + x] <= t10) { xHi = x; break; } xHi = x; }
  return Math.max(1, xHi - xLo);
}

/** 分类精度:与真值逐像元比较(冰脊真值归入最近冰类时不计) */
export function classificationAccuracy(classes: Uint8Array, truth: Uint8Array): number {
  let ok = 0, n = 0;
  for (let i = 0; i < classes.length; i++) {
    const t = truth[i];
    if (t === 4) continue; // 冰脊:线状目标,不计入面状分类精度
    n++;
    if (classes[i] === t) ok++;
  }
  return n > 0 ? ok / n : 0;
}
