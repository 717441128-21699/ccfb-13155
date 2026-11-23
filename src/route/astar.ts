/** 航路规划:冰型代价图上的 A* 搜索 */
import { IceClass } from '../types';

/** 各冰型的通行代价(每像素步) */
export const CLASS_COST: Record<number, number> = {
  [IceClass.OpenWater]: 1.0,
  [IceClass.NewIce]: 2.5,
  [IceClass.FirstYearIce]: 8.0,
  [IceClass.MultiYearIce]: 40.0,
  [IceClass.Ridge]: 60.0,
};

/** 二叉堆(最小堆) */
class MinHeap {
  keys: number[] = [];   // f 值
  vals: number[] = [];   // 节点索引
  get size() { return this.keys.length; }
  push(k: number, v: number) {
    this.keys.push(k); this.vals.push(v);
    let i = this.keys.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.keys[p] <= this.keys[i]) break;
      this.swap(i, p); i = p;
    }
  }
  pop(): number {
    const top = this.vals[0];
    const lk = this.keys.pop()!, lv = this.vals.pop()!;
    if (this.keys.length > 0) {
      this.keys[0] = lk; this.vals[0] = lv;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < this.keys.length && this.keys[l] < this.keys[m]) m = l;
        if (r < this.keys.length && this.keys[r] < this.keys[m]) m = r;
        if (m === i) break;
        this.swap(i, m); i = m;
      }
    }
    return top;
  }
  private swap(a: number, b: number) {
    [this.keys[a], this.keys[b]] = [this.keys[b], this.keys[a]];
    [this.vals[a], this.vals[b]] = [this.vals[b], this.vals[a]];
  }
}

export interface RouteResult {
  path: Array<[number, number]>;  // [x,y] 序列(原图坐标系,已乘回 scale)
  cost: number;
  explored: number;
  timeMs: number;
  found: boolean;
}

/**
 * 在冰型分类图上做 A*。内部按 factor 降采样以加速(默认 8)。
 * 代价 = 类别代价 × 步长;启发函数 = 八向 octile 距离 × 最小类别代价。
 */
export function planRoute(
  classes: Uint8Array, w: number, h: number,
  start: [number, number], goal: [number, number],
  factor = 8,
): RouteResult {
  const t0 = performance.now();
  const gw = Math.max(2, Math.floor(w / factor)), gh = Math.max(2, Math.floor(h / factor));
  // 降采样代价图:块内取平均类别代价(含冰脊惩罚)
  const cost = new Float32Array(gw * gh);
  const cnt = new Float32Array(gw * gh);
  for (let y = 0; y < h; y++) {
    const gy = Math.min(gh - 1, Math.floor(y / factor));
    for (let x = 0; x < w; x++) {
      const gx = Math.min(gw - 1, Math.floor(x / factor));
      const gi = gy * gw + gx;
      cost[gi] += CLASS_COST[classes[y * w + x]] ?? 10;
      cnt[gi]++;
    }
  }
  for (let i = 0; i < cost.length; i++) cost[i] = cnt[i] > 0 ? cost[i] / cnt[i] : 10;

  const sx = Math.min(gw - 1, Math.floor(start[0] / factor));
  const sy = Math.min(gh - 1, Math.floor(start[1] / factor));
  const tx = Math.min(gw - 1, Math.floor(goal[0] / factor));
  const ty = Math.min(gh - 1, Math.floor(goal[1] / factor));
  const startI = sy * gw + sx, goalI = ty * gw + tx;

  const g = new Float32Array(gw * gh).fill(Infinity);
  const came = new Int32Array(gw * gh).fill(-1);
  const closed = new Uint8Array(gw * gh);
  const heap = new MinHeap();
  g[startI] = 0;
  const hCost = (i: number) => {
    const dx = Math.abs((i % gw) - tx), dy = Math.abs(Math.floor(i / gw) - ty);
    const mn = Math.min(dx, dy), mx = Math.max(dx, dy);
    return (mx - mn + mn * Math.SQRT2) * 1.0; // × 最小类别代价(可采纳)
  };
  heap.push(hCost(startI), startI);
  let explored = 0;
  const NB = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]] as const;

  let found = false;
  while (heap.size > 0) {
    const cur = heap.pop();
    if (closed[cur]) continue;
    closed[cur] = 1;
    explored++;
    if (cur === goalI) { found = true; break; }
    const cx = cur % gw, cy = Math.floor(cur / gw);
    for (const [dx, dy, wgt] of NB) {
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
      const ni = ny * gw + nx;
      if (closed[ni]) continue;
      const step = (cost[cur] + cost[ni]) * 0.5 * wgt;
      const ng = g[cur] + step;
      if (ng < g[ni]) {
        g[ni] = ng; came[ni] = cur;
        heap.push(ng + hCost(ni), ni);
      }
    }
  }

  const path: Array<[number, number]> = [];
  if (found) {
    let cur = goalI;
    while (cur >= 0) {
      path.push([(cur % gw) * factor + factor / 2, Math.floor(cur / gw) * factor + factor / 2]);
      cur = came[cur];
    }
    path.reverse();
  }
  return { path, cost: g[goalI], explored, timeMs: performance.now() - t0, found };
}
