/**
 * SAR 海冰场景模拟器。
 * 生成 HH/HV 双极化多视强度图,含 Gamma 斑点噪声、多种冰情场景:
 *  - 'pack'   密集冰区:一年冰/多年冰混合,含开阔冰缝(暗线)与冰脊(亮线)
 *  - 'miz'    边缘冰区:开阔水(含风浪粗糙水面)+ 破碎浮冰
 *  - 'newice' 新冰区:大面积平整新冰(尼罗冰)夹冰缝与浮冰群
 * 真值类别图与无斑点均值图一并输出,用于算法验证。
 */
import { IceClass, SarScene } from '../types';

export interface SimOptions {
  width: number;
  height: number;
  looks: number;        // 视数 L
  windSpeed: number;    // m/s,影响开阔水后向散射
  scenario: 'pack' | 'miz' | 'newice';
  seed?: number;
}

/** 各冰型后向散射系数 (dB)。文献量级:C 波段、中等入射角 */
const SIGMA0 = {
  newIce:      { hh: -17.0, hv: -29.0 },
  firstYear:   { hh: -13.0, hv: -24.0 },
  multiYear:   { hh: -9.0,  hv: -18.0 }, // 体散射强,去极化强
  ridge:       { hh: -6.0,  hv: -14.0 },
};

/** 开阔水 HH 随风速变化(经验近似):风速大时粗糙水面亮度可接近多年冰 */
function waterSigma(wind: number): { hh: number; hv: number } {
  const hh = -26 + 1.35 * Math.min(wind, 14); // 2 m/s ≈ -23.3 dB;12 m/s ≈ -9.8 dB
  const hv = hh - 13 - Math.min(wind, 14) * 0.15; // 水面去极化很弱,HV 始终很低
  return { hh, hv };
}

/* ---------------- 随机数与噪声场 ---------------- */

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Gamma(shape=L, scale=1/L) 采样 —— L 视强度斑点噪声,Marsaglia–Tsang */
export function gammaSpeckle(rand: () => number, L: number): number {
  if (L >= 1) {
    // 整数视数:指数变量求和,精确且快
    let s = 0;
    const Li = Math.max(1, Math.round(L));
    for (let i = 0; i < Li; i++) s += -Math.log(1 - rand());
    return s / Li;
  }
  return 1;
}

/** 平滑值噪声场(双线性插值的网格随机值),用于冰型分区与后向散射起伏 */
function makeNoiseField(w: number, h: number, cell: number, rand: () => number): Float32Array {
  const gw = Math.ceil(w / cell) + 2;
  const gh = Math.ceil(h / cell) + 2;
  const grid = new Float32Array(gw * gh);
  for (let i = 0; i < grid.length; i++) grid[i] = rand();
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const gy = y / cell, y0 = Math.floor(gy), fy = gy - y0;
    const sy = fy * fy * (3 - 2 * fy);
    for (let x = 0; x < w; x++) {
      const gx = x / cell, x0 = Math.floor(gx), fx = gx - x0;
      const sx = fx * fx * (3 - 2 * fx);
      const i00 = y0 * gw + x0;
      const v0 = grid[i00] + (grid[i00 + 1] - grid[i00]) * sx;
      const v1 = grid[i00 + gw] + (grid[i00 + gw + 1] - grid[i00 + gw]) * sx;
      out[y * w + x] = v0 + (v1 - v0) * sy;
    }
  }
  return out;
}

/** 多倍频 fBm 噪声 */
function fbm(w: number, h: number, baseCell: number, octaves: number, rand: () => number): Float32Array {
  const acc = new Float32Array(w * h);
  let amp = 1, total = 0, cell = baseCell;
  for (let o = 0; o < octaves; o++) {
    const n = makeNoiseField(w, h, Math.max(4, cell), rand);
    for (let i = 0; i < acc.length; i++) acc[i] += n[i] * amp;
    total += amp; amp *= 0.5; cell *= 0.5;
  }
  for (let i = 0; i < acc.length; i++) acc[i] /= total;
  return acc;
}

/* ---------------- 场景构建 ---------------- */

interface Line { x0: number; y0: number; x1: number; y1: number; width: number; }

/** 生成随机折线(冰缝/冰脊):从一条边出发,带随机游走地横穿图像 */
function randomCrack(w: number, h: number, rand: () => number): Line[] {
  const segs: Line[] = [];
  let x = rand() * w, y = rand() < 0.5 ? 0 : h;
  let ang = y === 0 ? Math.PI / 2 : -Math.PI / 2;
  ang += (rand() - 0.5) * 0.8;
  const step = Math.min(w, h) / 14;
  const width = 1.5 + rand() * 3;
  while (x > -20 && x < w + 20 && y > -20 && y < h + 20) {
    const nx = x + Math.cos(ang) * step;
    const ny = y + Math.sin(ang) * step;
    segs.push({ x0: x, y0: y, x1: nx, y1: ny, width });
    x = nx; y = ny;
    ang += (rand() - 0.5) * 0.55;
  }
  return segs;
}

function buildClassMap(opt: SimOptions, rand: () => number): Uint8Array {
  const { width: w, height: h } = opt;
  const cls = new Uint8Array(w * h);
  const region = fbm(w, h, Math.min(w, h) / 3, 4, rand);   // 大尺度冰型分区
  const detail = fbm(w, h, Math.min(w, h) / 12, 3, rand);  // 中尺度起伏

  for (let i = 0; i < w * h; i++) {
    const v = region[i] * 0.72 + detail[i] * 0.28;
    if (opt.scenario === 'pack') {
      // 密集冰区:FYI 与 MYI 交错
      cls[i] = v < 0.46 ? IceClass.FirstYearIce : v < 0.9 ? IceClass.MultiYearIce : IceClass.FirstYearIce;
    } else if (opt.scenario === 'miz') {
      // 边缘冰区:大片开阔水 + 浮冰(FYI 为主,少量 MYI)
      cls[i] = v < 0.42 ? IceClass.OpenWater : v < 0.82 ? IceClass.FirstYearIce : IceClass.MultiYearIce;
    } else {
      // 新冰区:新冰为主,嵌 FYI 浮冰
      cls[i] = v < 0.55 ? IceClass.NewIce : v < 0.93 ? IceClass.FirstYearIce : IceClass.MultiYearIce;
    }
  }

  // 冰缝(开阔水暗线)与冰脊(亮线)
  const nCracks = opt.scenario === 'pack' ? 7 : opt.scenario === 'newice' ? 5 : 3;
  const nRidges = opt.scenario === 'pack' ? 9 : 4;
  for (let c = 0; c < nCracks + nRidges; c++) {
    const isLead = c < nCracks;
    const segs = randomCrack(w, h, rand);
    const target = isLead
      ? (opt.scenario === 'newice' && rand() < 0.5 ? IceClass.NewIce : IceClass.OpenWater)
      : IceClass.Ridge;
    // 沿折线栅格化(步进 0.5px 保证连续)
    for (const s of segs) {
      const len = Math.hypot(s.x1 - s.x0, s.y1 - s.y0);
      const steps = Math.ceil(len * 2);
      for (let k = 0; k <= steps; k++) {
        const t = k / steps;
        const px = s.x0 + (s.x1 - s.x0) * t, py = s.y0 + (s.y1 - s.y0) * t;
        const r = s.width * (isLead ? 1 : 0.8);
        const x0 = Math.max(0, Math.floor(px - r - 1)), x1 = Math.min(w - 1, Math.ceil(px + r + 1));
        const y0 = Math.max(0, Math.floor(py - r - 1)), y1 = Math.min(h - 1, Math.ceil(py + r + 1));
        for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) {
          if (Math.hypot(xx - px, yy - py) <= r) {
            const idx = yy * w + xx;
            if (isLead || cls[idx] !== IceClass.OpenWater) cls[idx] = target;
          }
        }
      }
    }
  }
  return cls;
}

/** 主入口:生成一景模拟 SAR 影像 */
export function generateScene(opt: SimOptions): SarScene {
  const rand = mulberry32(opt.seed ?? 42);
  const { width: w, height: h, looks: L } = opt;
  const n = w * h;
  const truth = buildClassMap(opt, rand);

  // 各类型的均值后向散射(线性)
  const water = waterSigma(opt.windSpeed);
  const db2lin = (db: number) => Math.pow(10, db / 10);
  const lutHH = new Float32Array(8), lutHV = new Float32Array(8);
  lutHH[IceClass.OpenWater] = db2lin(water.hh); lutHV[IceClass.OpenWater] = db2lin(water.hv);
  lutHH[IceClass.NewIce] = db2lin(SIGMA0.newIce.hh); lutHV[IceClass.NewIce] = db2lin(SIGMA0.newIce.hv);
  lutHH[IceClass.FirstYearIce] = db2lin(SIGMA0.firstYear.hh); lutHV[IceClass.FirstYearIce] = db2lin(SIGMA0.firstYear.hv);
  lutHH[IceClass.MultiYearIce] = db2lin(SIGMA0.multiYear.hh); lutHV[IceClass.MultiYearIce] = db2lin(SIGMA0.multiYear.hv);
  lutHH[IceClass.Ridge] = db2lin(SIGMA0.ridge.hh); lutHV[IceClass.Ridge] = db2lin(SIGMA0.ridge.hv);

  // 后向散射的空间缓变(±1.5 dB),模拟真实冰面非均匀性
  const rough = fbm(w, h, Math.min(w, h) / 8, 3, rand);
  // 风浪区:开阔水上的阵风斑块,进一步提高局部 HH(考验水/冰区分)
  const gust = fbm(w, h, Math.min(w, h) / 6, 3, rand);

  const hh = new Float32Array(n), hv = new Float32Array(n);
  const hhMean = new Float32Array(n), hvMean = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const c = truth[i];
    let modDb = (rough[i] - 0.5) * 3.0;
    let gustDb = 0;
    if (c === IceClass.OpenWater) {
      // 阵风:局部等效风速抬升 HH 最多约 6 dB。
      // 物理上布拉格散射有极化选择性:HH 抬升强,HV 仅轻微(×0.3),
      // 因此风浪水面的去极化差进一步拉大 —— 这是区分水与冰的关键。
      gustDb = Math.max(0, gust[i] - 0.55) * 15;
    }
    const m = db2lin(modDb);
    const mhh = lutHH[c] * m * db2lin(gustDb);
    const mhv = lutHV[c] * m * db2lin(gustDb * 0.3);
    hhMean[i] = mhh; hvMean[i] = mhv;
    // 乘性 Gamma 斑点(多视强度)
    hh[i] = mhh * gammaSpeckle(rand, L);
    hv[i] = mhv * gammaSpeckle(rand, L);
  }

  return {
    width: w, height: h, looks: L, hh, hv, truth,
    meta: { scenario: opt.scenario, windSpeed: opt.windSpeed, noiseFree: { hh: hhMean, hv: hvMean } },
  };
}
