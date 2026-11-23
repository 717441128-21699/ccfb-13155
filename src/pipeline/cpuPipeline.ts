/**
 * CPU 参考流水线:方向自适应精制 Lee 去斑 + 高斯冰型分类。
 * 与 GPU(WebGL2)实现算法一致,用于:
 *  1) 单元测试验证指标(ENL 提升 ≥8×、边缘锐度损失 ≤10%)
 *  2) WebGL2 不可用时的回退路径
 */
import { IceClass } from '../types';

/* ---------------- 去斑:方向自适应精制 Lee ---------------- */

const WIN_R = 3;          // 7×7 局部统计窗半径
const DIR_LEN = 4;        // 方向线窗半长(全长 9)
// 4 个方向:水平、垂直、两条对角线
const DIRS: ReadonlyArray<readonly [number, number]> = [[1, 0], [0, 1], [1, 1], [1, -1]];

/** 可分离盒式均值(半径 r),边界截断 */
export function boxMean(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const win = 2 * r + 1;
  // 水平
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let acc = 0;
    for (let x = -r; x <= r; x++) acc += src[row + Math.min(w - 1, Math.max(0, x))];
    tmp[row] = acc / win;
    for (let x = 1; x < w; x++) {
      const xa = Math.min(w - 1, x + r), xs = Math.max(0, x - r - 1);
      acc += src[row + xa] - src[row + xs];
      tmp[row + x] = acc / win;
    }
  }
  // 垂直
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -r; y <= r; y++) acc += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    out[x] = acc / win;
    for (let y = 1; y < h; y++) {
      const ya = Math.min(h - 1, y + r), ys = Math.max(0, y - r - 1);
      acc += tmp[ya * w + x] - tmp[ys * w + x];
      out[y * w + x] = acc / win;
    }
  }
  return out;
}

/** 单通道方向自适应精制 Lee 滤波 */
function leeChannel(x: Float32Array, w: number, h: number, L: number, out: Float32Array): void {
  const mean = boxMean(x, w, h, WIN_R);
  const sq = new Float32Array(w * h);
  for (let i = 0; i < sq.length; i++) sq[i] = x[i] * x[i];
  const meanSq = boxMean(sq, w, h, WIN_R);
  const cu2 = 1 / L; // 斑点变异系数平方(理论值)

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let xx = 0; xx < w; xx++) {
      const i = row + xx;
      const mu = mean[i];
      const varr = Math.max(0, meanSq[i] - mu * mu);
      const cx2 = varr / (mu * mu + 1e-20);
      // Lee 权重:均匀区 k→0(输出均值),边缘/线目标 k→1(保留原值)
      let k = cx2 > cu2 ? 1 - cu2 / cx2 : 0;
      if (k < 0.25) {
        out[i] = mu; // 均匀区直接用 7×7 均值,ENL 最大化
        continue;
      }
      // 边缘附近:选方差最小的方向线窗均值,沿边缘平滑而不跨边缘
      let bestMu = mu, bestVar = Infinity;
      for (const [dx, dy] of DIRS) {
        let s = 0, s2 = 0, cnt = 0;
        for (let t = -DIR_LEN; t <= DIR_LEN; t++) {
          const px = Math.min(w - 1, Math.max(0, xx + dx * t));
          const py = Math.min(h - 1, Math.max(0, y + dy * t));
          const v = x[py * w + px];
          s += v; s2 += v * v; cnt++;
        }
        const m = s / cnt;
        const v = s2 / cnt - m * m;
        if (v < bestVar) { bestVar = v; bestMu = m; }
      }
      out[i] = bestMu + k * (x[i] - bestMu);
    }
  }
}

export interface DespeckleOut { hh: Float32Array; hv: Float32Array; timeMs: number; }

export function despeckleCPU(hh: Float32Array, hv: Float32Array, w: number, h: number, L: number): DespeckleOut {
  const t0 = performance.now();
  const oh = new Float32Array(w * h), ov = new Float32Array(w * h);
  leeChannel(hh, w, h, L, oh);
  leeChannel(hv, w, h, L, ov);
  return { hh: oh, hv: ov, timeMs: performance.now() - t0 };
}

/* ---------------- 冰型分类:特征 + 对角高斯 ---------------- */

export interface ClassGauss { mu: [number, number, number, number]; sd: [number, number, number, number]; }

/**
 * 特征:[HH_dB, HV_dB, 去极化差 HH_dB-HV_dB, HH_dB 局部纹理标准差(5×5)]
 * 关键判别:风浪粗糙的开阔水 HH 可升至多年冰量级,但水面去极化弱,
 * HV 始终远低于冰 —— 用 HV 与去极化差把风吹水与冰分开。
 * 开阔水用两个子模型(平静/风浪)覆盖随风速的大动态范围。
 */
export const MODEL_PARAMS: ClassGauss[] = [
  { mu: [-23, -35, 12.5, 0.5], sd: [2.5, 2.5, 2.0, 0.35] }, // 0: 平静开阔水
  { mu: [-11, -27, 16.0, 0.9], sd: [3.0, 2.5, 2.0, 0.5] },  // 1: 风浪开阔水
  { mu: [-17, -29, 12.0, 0.45], sd: [2.0, 2.0, 1.5, 0.3] }, // 2: 新冰
  { mu: [-13, -24, 11.0, 0.8], sd: [2.0, 2.0, 1.5, 0.4] },  // 3: 一年冰
  { mu: [-9, -18, 9.0, 1.2], sd: [2.2, 2.2, 1.5, 0.5] },    // 4: 多年冰
];
/** 子模型 → 输出类别(两个水子模型都映射到开阔水) */
export const MODEL_TO_CLASS = [
  IceClass.OpenWater, IceClass.OpenWater, IceClass.NewIce, IceClass.FirstYearIce, IceClass.MultiYearIce,
];
export const N_MODELS = MODEL_PARAMS.length;

/** 纹理:HH_dB 的 5×5 局部标准差 */
function localStdDb(hh: Float32Array, w: number, h: number, r = 2): Float32Array {
  const n = w * h;
  const db = new Float32Array(n);
  for (let i = 0; i < n; i++) db[i] = 10 * Math.log10(Math.max(hh[i], 1e-12));
  const mean = boxMean(db, w, h, r);
  const sq = new Float32Array(n);
  for (let i = 0; i < n; i++) sq[i] = db[i] * db[i];
  const meanSq = boxMean(sq, w, h, r);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sqrt(Math.max(0, meanSq[i] - mean[i] * mean[i]));
  return out;
}

export interface ClassifyOut { classes: Uint8Array; confidence: Float32Array; timeMs: number; }

export function classifyCPU(hh: Float32Array, hv: Float32Array, w: number, h: number): ClassifyOut {
  const t0 = performance.now();
  const n = w * h;
  const tex = localStdDb(hh, w, h);
  const classes = new Uint8Array(n);
  const confidence = new Float32Array(n);
  const logDet = MODEL_PARAMS.map(p =>
    Math.log(p.sd[0]) + Math.log(p.sd[1]) + Math.log(p.sd[2]) + Math.log(p.sd[3]));
  for (let i = 0; i < n; i++) {
    const hhDb = 10 * Math.log10(Math.max(hh[i], 1e-12));
    const hvDb = 10 * Math.log10(Math.max(hv[i], 1e-12));
    const f = [hhDb, hvDb, hhDb - hvDb, tex[i]];
    let bestC = 0, bestD = Infinity, second = Infinity;
    for (let c = 0; c < N_MODELS; c++) {
      const p = MODEL_PARAMS[c];
      let d = logDet[c];
      for (let k = 0; k < 4; k++) {
        const z = (f[k] - p.mu[k]) / p.sd[k];
        d += z * z;
      }
      if (d < bestD) { second = bestD; bestD = d; bestC = c; }
      else if (d < second) { second = d; }
    }
    classes[i] = MODEL_TO_CLASS[bestC];
    confidence[i] = Math.max(0, Math.min(1, 1 - bestD / (second + 1e-9)));
  }
  return { classes, confidence, timeMs: performance.now() - t0 };
}
