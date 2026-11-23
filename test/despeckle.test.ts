/** 算法指标验证:斑点统计、ENL 提升、边缘锐度保持 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { generateScene, gammaSpeckle, mulberry32 } from '../src/sim/sceneGenerator';
import { despeckleCPU } from '../src/pipeline/cpuPipeline';
import { enl, regionIndices, edgeRiseWidth } from '../src/metrics';
import { IceClass } from '../src/types';

/** 在真值图中找一个全为指定类别的 size×size 均匀块中心 */
function findHomogeneous(truth: Uint8Array, w: number, h: number, cls: number, size: number): [number, number] {
  for (let y = size; y < h - size; y += 7) {
    outer: for (let x = size; x < w - size; x += 7) {
      for (let dy = -size / 2; dy < size / 2; dy += 4)
        for (let dx = -size / 2; dx < size / 2; dx += 4)
          if (truth[Math.round(y + dy) * w + Math.round(x + dx)] !== cls) continue outer;
      return [x, y];
    }
  }
  throw new Error('找不到均匀区');
}

test('斑点噪声统计:均匀区 ENL ≈ 视数 L', () => {
  const rand = mulberry32(7);
  const n = 200000;
  const data = new Float32Array(n);
  const L = 4;
  for (let i = 0; i < n; i++) data[i] = gammaSpeckle(rand, L);
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  const e = enl(data, idx);
  assert.ok(Math.abs(e - L) / L < 0.1, `ENL=${e.toFixed(2)} 应接近 L=${L}`);
});

test('去斑:均匀区 ENL 提升 ≥ 8 倍', () => {
  const scene = generateScene({ width: 512, height: 512, looks: 4, windSpeed: 6, scenario: 'pack', seed: 11 });
  const [cx, cy] = findHomogeneous(scene.truth, 512, 512, IceClass.FirstYearIce, 41);
  const idx = regionIndices(512, 512, cx, cy, 41);
  const e0 = enl(scene.hh, idx);
  const out = despeckleCPU(scene.hh, scene.hv, 512, 512, 4);
  const e1 = enl(out.hh, idx);
  const gain = e1 / e0;
  console.log(`  ENL: ${e0.toFixed(1)} → ${e1.toFixed(1)}(提升 ${gain.toFixed(1)}×)`);
  assert.ok(gain >= 8, `ENL 提升 ${gain.toFixed(1)}× 应 ≥ 8×`);
});

test('去斑:阶跃边缘锐度损失 ≤ 10%(无噪声本征模糊)', () => {
  // 无斑点阶跃边:滤波器本身不应把边缘抹宽超过 10%
  const w = 200, h = 60;
  const hh = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) hh[y * w + x] = x < 100 ? 0.05 : 0.005;
  const out = despeckleCPU(hh, hh.slice(), w, h, 4);
  const r0 = edgeRiseWidth(hh, w, h, 100, 30)!;
  const r1 = edgeRiseWidth(out.hh, w, h, 100, 30)!;
  const loss = (r1 - r0) / r0;
  console.log(`  阶跃边缘: ${r0} → ${r1} px(损失 ${(loss * 100).toFixed(1)}%)`);
  assert.ok(loss <= 0.1, `锐度损失 ${(loss * 100).toFixed(1)}% 应 ≤ 10%`);
});

/** 多行平均剖面的 10–90% 上升沿宽度(抑制斑点后的集成测量) */
function avgProfileRiseWidth(data: Float32Array, w: number, h: number, cx: number, cy: number, rows: number, len: number): number {
  const prof = new Float32Array(len);
  const start = Math.round(cx - len / 2);
  for (let r = -Math.floor(rows / 2); r <= Math.floor(rows / 2); r++) {
    const y = Math.min(h - 1, Math.max(0, cy + r));
    for (let i = 0; i < len; i++) prof[i] += data[y * w + start + i];
  }
  for (let i = 0; i < len; i++) prof[i] /= rows;
  // 复用 edgeRiseWidth:把剖面放进单行图
  return edgeRiseWidth(prof, len, 1, Math.floor(len / 2), 0, len)!;
}

test('去斑:含斑点冰缝边缘锐度损失 ≤ 10%', () => {
  // 一年冰背景 + 4px 宽开阔水冰缝(垂直),L=4 斑点
  const w = 240, h = 240, leadX = 120, leadW = 4;
  const rand = mulberry32(99);
  const hh = new Float32Array(w * h);
  const iceMean = Math.pow(10, -13 / 10), waterMean = Math.pow(10, -24 / 10);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const m = Math.abs(x - leadX) < leadW / 2 ? waterMean : iceMean;
    hh[y * w + x] = m * gammaSpeckle(rand, 4);
  }
  const hv = hh.slice();
  const out = despeckleCPU(hh, hv, w, h, 4);
  const r0 = avgProfileRiseWidth(hh, w, h, leadX, 120, 80, 61);
  const r1 = avgProfileRiseWidth(out.hh, w, h, leadX, 120, 80, 61);
  const loss = (r1 - r0) / r0;
  console.log(`  冰缝边缘(集成剖面): ${r0.toFixed(2)} → ${r1.toFixed(2)} px(损失 ${(loss * 100).toFixed(1)}%)`);
  assert.ok(loss <= 0.1, `冰缝锐度损失 ${(loss * 100).toFixed(1)}% 应 ≤ 10%`);
});
