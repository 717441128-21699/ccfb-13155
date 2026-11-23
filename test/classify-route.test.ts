/** 冰型分类与航路规划验证 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { generateScene } from '../src/sim/sceneGenerator';
import { despeckleCPU, classifyCPU } from '../src/pipeline/cpuPipeline';
import { classificationAccuracy } from '../src/metrics';
import { planRoute, CLASS_COST } from '../src/route/astar';
import { IceClass } from '../src/types';

test('冰型分类:边缘冰区(含风浪开阔水)精度 ≥ 80%', () => {
  const scene = generateScene({ width: 512, height: 512, looks: 4, windSpeed: 10, scenario: 'miz', seed: 23 });
  const dsp = despeckleCPU(scene.hh, scene.hv, 512, 512, 4);
  const cls = classifyCPU(dsp.hh, dsp.hv, 512, 512);
  const acc = classificationAccuracy(cls.classes, scene.truth);
  // 风浪开阔水 vs 冰的混淆统计
  let waterAsIce = 0, waterTotal = 0;
  for (let i = 0; i < cls.classes.length; i++) {
    if (scene.truth[i] === IceClass.OpenWater) {
      waterTotal++;
      if (cls.classes[i] !== IceClass.OpenWater) waterAsIce++;
    }
  }
  console.log(`  分类精度 ${(acc * 100).toFixed(1)}%,开阔水误判率 ${(waterAsIce / waterTotal * 100).toFixed(1)}%`);
  assert.ok(acc >= 0.8, `精度 ${(acc * 100).toFixed(1)}% 应 ≥ 80%`);
  assert.ok(waterAsIce / waterTotal < 0.15, '风浪开阔水不应大量误判为冰');
});

test('冰型分类:密集冰区精度 ≥ 80%', () => {
  const scene = generateScene({ width: 512, height: 512, looks: 4, windSpeed: 5, scenario: 'pack', seed: 31 });
  const dsp = despeckleCPU(scene.hh, scene.hv, 512, 512, 4);
  const cls = classifyCPU(dsp.hh, dsp.hv, 512, 512);
  const acc = classificationAccuracy(cls.classes, scene.truth);
  console.log(`  密集冰区分类精度 ${(acc * 100).toFixed(1)}%`);
  assert.ok(acc >= 0.8);
});

test('航路规划:绕开多年冰,走开阔水/薄冰', () => {
  // 构造:左水右水,中间多年冰墙,墙上一个缺口
  const w = 400, h = 200;
  const cls = new Uint8Array(w * h).fill(IceClass.OpenWater);
  for (let y = 0; y < h; y++) for (let x = 180; x < 220; x++)
    if (y < 120 || y > 140) cls[y * w + x] = IceClass.MultiYearIce;
  const r = planRoute(cls, w, h, [20, 100], [380, 100], 4);
  assert.ok(r.found, '应找到航路');
  // 航路应穿过缺口(y≈130)而不是硬闯冰墙
  for (const [x, y] of r.path) {
    const c = cls[Math.min(h - 1, Math.round(y)) * w + Math.min(w - 1, Math.round(x))];
    assert.notEqual(c, IceClass.MultiYearIce, `航路不应穿过多年冰 (${x},${y})`);
  }
  console.log(`  航路 ${r.path.length} 点,代价 ${r.cost.toFixed(0)},耗时 ${r.timeMs.toFixed(1)} ms`);
});

test('航路规划:全冰场景仍给出最小代价路径', () => {
  const w = 300, h = 300;
  const cls = new Uint8Array(w * h).fill(IceClass.FirstYearIce);
  // 一条新冰水道
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (Math.abs(x - y) < 12) cls[y * w + x] = IceClass.NewIce;
  const r = planRoute(cls, w, h, [10, 10], [290, 290], 4);
  assert.ok(r.found);
  // 对角直穿代价 ≈ 距离×FYI 代价;走水道应显著更低
  const straight = Math.hypot(280, 280) * CLASS_COST[IceClass.FirstYearIce];
  console.log(`  代价 ${r.cost.toFixed(0)} vs 直穿 ${straight.toFixed(0)}`);
  assert.ok(r.cost < straight * 0.75, '应利用新冰水道降低代价');
});

test('性能:CPU 参考流水线 1024² 全链路', () => {
  const scene = generateScene({ width: 1024, height: 1024, looks: 4, windSpeed: 8, scenario: 'pack', seed: 5 });
  const dsp = despeckleCPU(scene.hh, scene.hv, 1024, 1024, 4);
  const cls = classifyCPU(dsp.hh, dsp.hv, 1024, 1024);
  const r = planRoute(cls.classes, 1024, 1024, [10, 10], [1000, 1000], 8);
  console.log(`  1024² CPU:去斑 ${dsp.timeMs.toFixed(0)} ms,分类 ${cls.timeMs.toFixed(0)} ms,航路 ${r.timeMs.toFixed(0)} ms`);
  assert.ok(dsp.timeMs + cls.timeMs < 60000, 'CPU 参考实现不应病态缓慢');
  assert.ok(r.found);
});
