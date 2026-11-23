/** 应用主程序:界面编排、影像显示、交互、GPU/CPU 流水线调度 */
import { generateScene } from './sim/sceneGenerator';
import { despeckleCPU, classifyCPU } from './pipeline/cpuPipeline';
import { GpuPipeline } from './pipeline/gpuPipeline';
import { enl, regionIndices, edgeRiseWidth, classificationAccuracy } from './metrics';
import { planRoute } from './route/astar';
import { SarScene, ClassifyResult, CLASS_NAMES } from './types';

/* ---------------- 状态 ---------------- */
interface AppState {
  scene: SarScene | null;
  filtHh: Float32Array | null;
  filtHv: Float32Array | null;
  classes: Uint8Array | null;
  layer: 'hh' | 'hv' | 'filt' | 'class';
  ship: [number, number] | null;
  goal: [number, number] | null;
  route: Array<[number, number]> | null;
  enlRect: { x0: number; y0: number; x1: number; y1: number } | null;
  edgePt: { x: number; y: number } | null;
  zoom: number; panX: number; panY: number;
}
const S: AppState = {
  scene: null, filtHh: null, filtHv: null, classes: null,
  layer: 'hh', ship: null, goal: null, route: null,
  enlRect: null, edgePt: null, zoom: 1, panX: 0, panY: 0,
};

let gpu: GpuPipeline | null = null;
let gpuError: string | null = null;
try { gpu = new GpuPipeline(); } catch (e) { gpuError = String(e); }

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('view');
const ctx = canvas.getContext('2d')!;

const CLASS_COLORS: Array<[number, number, number]> = [
  [27, 95, 174], [168, 216, 230], [232, 230, 223], [196, 78, 82],
];

const tick = () => new Promise(r => setTimeout(r, 0));

/* ---------------- 视图渲染 ---------------- */
function fitView(): void {
  if (!S.scene) return;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  S.zoom = Math.min(cw / S.scene.width, ch / S.scene.height);
  S.panX = (cw - S.scene.width * S.zoom) / 2;
  S.panY = (ch - S.scene.height * S.zoom) / 2;
}

function render(): void {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  ctx.fillStyle = '#0f1418';
  ctx.fillRect(0, 0, cw, ch);
  if (!S.scene) return;
  const { width: w, height: h } = S.scene;
  const img = ctx.createImageData(cw, ch);
  const d = img.data;
  const layer = S.layer;
  for (let cy = 0; cy < ch; cy++) {
    const iy = Math.floor((cy - S.panY) / S.zoom);
    if (iy < 0 || iy >= h) continue;
    const row = iy * w;
    for (let cx = 0; cx < cw; cx++) {
      const ix = Math.floor((cx - S.panX) / S.zoom);
      if (ix < 0 || ix >= w) continue;
      const o = (cy * cw + cx) * 4;
      if (layer === 'class' && S.classes) {
        const c = CLASS_COLORS[S.classes[row + ix]] ?? [0, 0, 0];
        d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
      } else {
        let v = 0;
        if (layer === 'hh') v = S.scene.hh[row + ix];
        else if (layer === 'hv') v = S.scene.hv[row + ix];
        else if (layer === 'filt' && S.filtHh) v = S.filtHh[row + ix];
        else v = S.scene.hh[row + ix];
        const db = 10 * Math.log10(Math.max(v, 1e-12));
        const g = Math.max(0, Math.min(255, ((db + 32) / 28) * 255));
        d[o] = d[o + 1] = d[o + 2] = g; d[o + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  drawOverlays(cw, ch);
}

function toCanvas(p: [number, number]): [number, number] {
  return [p[0] * S.zoom + S.panX, p[1] * S.zoom + S.panY];
}

function drawOverlays(cw: number, ch: number): void {
  // ENL 选区
  if (S.enlRect) {
    const [ax, ay] = toCanvas([S.enlRect.x0, S.enlRect.y0]);
    const [bx, by] = toCanvas([S.enlRect.x1, S.enlRect.y1]);
    ctx.strokeStyle = '#ffd54a'; ctx.lineWidth = 1.5;
    ctx.strokeRect(ax, ay, bx - ax, by - ay);
  }
  if (S.edgePt) {
    const [x, y] = toCanvas([S.edgePt.x, S.edgePt.y]);
    ctx.strokeStyle = '#ff6e6e'; ctx.lineWidth = 1.5;
    ctx.strokeRect(x - 20 * S.zoom, y - 2, 40 * S.zoom, 4);
  }
  if (S.route) {
    ctx.strokeStyle = '#37e08b'; ctx.lineWidth = 2.5;
    ctx.beginPath();
    S.route.forEach((p, i) => {
      const [x, y] = toCanvas(p);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  if (S.ship) {
    const [x, y] = toCanvas(S.ship);
    ctx.fillStyle = '#37e08b';
    ctx.beginPath(); ctx.moveTo(x, y - 9); ctx.lineTo(x + 7, y + 7); ctx.lineTo(x - 7, y + 7); ctx.closePath(); ctx.fill();
    ctx.fillText('船位', x + 10, y + 4);
  }
  if (S.goal) {
    const [x, y] = toCanvas(S.goal);
    ctx.fillStyle = '#ff5252';
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillText('目标', x + 10, y + 4);
  }
  void cw; void ch;
}

/* ---------------- 指标 ---------------- */
function updateMetrics(): void {
  if (!S.scene) return;
  const w = S.scene.width, h = S.scene.height;
  if (S.enlRect) {
    const r = S.enlRect;
    const cx = (r.x0 + r.x1) / 2, cy = (r.y0 + r.y1) / 2;
    const size = Math.max(4, Math.min(Math.abs(r.x1 - r.x0), Math.abs(r.y1 - r.y0)));
    const idx = regionIndices(w, h, cx, cy, size);
    const e0 = enl(S.scene.hh, idx);
    $('enlBefore').textContent = e0.toFixed(1);
    if (S.filtHh) {
      const e1 = enl(S.filtHh, idx);
      $('enlAfter').textContent = e1.toFixed(1);
      const gain = e1 / e0;
      $('enlGain').textContent = `${gain.toFixed(1)} × ${gain >= 8 ? '✓ 达标(≥8×)' : '✗ 未达 8×'}`;
      $('enlGain').style.color = gain >= 8 ? '#8fd0a0' : '#ff8a80';
    }
  }
  if (S.edgePt) {
    const { x, y } = S.edgePt;
    const r0 = edgeRiseWidth(S.scene.hh, w, h, Math.round(x), Math.round(y));
    $('edgeBefore').textContent = r0 === null ? '无阶跃' : r0.toFixed(1);
    if (S.filtHh && r0 !== null) {
      const r1 = edgeRiseWidth(S.filtHh, w, h, Math.round(x), Math.round(y));
      if (r1 !== null) {
        $('edgeAfter').textContent = r1.toFixed(1);
        const loss = (r1 - r0) / r0 * 100;
        $('edgeLoss').textContent = `${loss.toFixed(1)}% ${loss <= 10 ? '✓ 达标(≤10%)' : '✗ 超过 10%'}`;
        $('edgeLoss').style.color = loss <= 10 ? '#8fd0a0' : '#ff8a80';
      }
    }
  }
}

/* ---------------- 事件:导入 ---------------- */
$('btnGen').addEventListener('click', async () => {
  const size = parseInt($<HTMLSelectElement>('size').value, 10);
  const status = $('genStatus');
  status.textContent = '正在生成模拟 SAR 影像…';
  await tick();
  const t0 = performance.now();
  S.scene = generateScene({
    width: size, height: size,
    looks: parseInt($<HTMLInputElement>('looks').value, 10),
    windSpeed: parseFloat($<HTMLInputElement>('wind').value),
    scenario: $<HTMLSelectElement>('scenario').value as 'pack' | 'miz' | 'newice',
    seed: (Math.random() * 1e9) | 0,
  });
  S.filtHh = S.filtHv = null; S.classes = null; S.route = null;
  S.ship = null; S.goal = null; S.enlRect = null; S.edgePt = null;
  status.textContent = `已生成 ${size}×${size},视数 ${S.scene.looks},耗时 ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  $('btnDespeckle').removeAttribute('disabled');
  $('btnClassify').setAttribute('disabled', '');
  $('btnRoute').setAttribute('disabled', '');
  fitView(); render(); updateMetrics();
});

$('fileInput').addEventListener('change', async () => {
  const f = $<HTMLInputElement>('fileInput').files?.[0];
  if (!f) return;
  const bmp = await createImageBitmap(f);
  const w = bmp.width, h = bmp.height;
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const octx = off.getContext('2d')!;
  octx.drawImage(bmp, 0, 0);
  const px = octx.getImageData(0, 0, w, h).data;
  const hh = new Float32Array(w * h), hv = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const g = (px[i * 4] + px[i * 4 + 1] + px[i * 4 + 2]) / 3 / 255;
    const v = Math.pow(10, ((g * 30 - 30) / 10)); // 灰度 → -30..0 dB → 线性
    hh[i] = v; hv[i] = v * Math.pow(10, -13 / 10); // 单通道导入:HV 近似
  }
  S.scene = {
    width: w, height: h, looks: parseInt($<HTMLInputElement>('looks').value, 10),
    hh, hv, truth: new Uint8Array(w * h),
    meta: { scenario: 'import', windSpeed: 0 },
  };
  S.filtHh = S.filtHv = null; S.classes = null; S.route = null;
  $('genStatus').textContent = `已导入 ${w}×${h}(单通道,HV 按 -13 dB 近似)`;
  $('btnDespeckle').removeAttribute('disabled');
  fitView(); render();
});

/* ---------------- 事件:去斑 / 分类 / 航路 ---------------- */
$('btnDespeckle').addEventListener('click', async () => {
  if (!S.scene) return;
  const { hh, hv, width: w, height: h, looks } = S.scene;
  const t0 = performance.now();
  $('despeckleTime').textContent = gpu ? 'GPU 处理中…' : 'CPU 处理中(GPU 不可用)';
  await tick();
  let out: { hh: Float32Array; hv: Float32Array; timeMs: number };
  if (gpu) out = gpu.despeckle(hh, hv, w, h, looks);
  else out = despeckleCPU(hh, hv, w, h, looks);
  S.filtHh = out.hh; S.filtHv = out.hv;
  const total = ((performance.now() - t0) / 1000).toFixed(2);
  $('despeckleTime').textContent = `${total} s(${gpu ? 'WebGL2' : 'CPU'}${gpuError ? ',GPU 回退原因: ' + gpuError : ''})`;
  S.layer = 'filt';
  (document.querySelector('input[name=layer][value=filt]') as HTMLInputElement).checked = true;
  $('btnClassify').removeAttribute('disabled');
  render(); updateMetrics();
});

$('btnClassify').addEventListener('click', async () => {
  if (!S.scene || !S.filtHh || !S.filtHv) return;
  const { width: w, height: h } = S.scene;
  $('classInfo').textContent = '分类中…';
  await tick();
  let out: ClassifyResult | { classes: Uint8Array; confidence: Float32Array; timeMs: number };
  if (gpu) out = gpu.classify(S.filtHh, S.filtHv, w, h);
  else out = classifyCPU(S.filtHh, S.filtHv, w, h);
  S.classes = out.classes;
  let msg = `耗时 ${(out.timeMs / 1000).toFixed(2)} s(${gpu ? 'WebGL2' : 'CPU'})`;
  if (S.scene.meta.scenario !== 'import') {
    const acc = classificationAccuracy(out.classes, S.scene.truth);
    msg += `\n分类精度(vs 真值):${(acc * 100).toFixed(1)}%`;
  }
  $('classInfo').textContent = msg;
  S.layer = 'class';
  (document.querySelector('input[name=layer][value=class]') as HTMLInputElement).checked = true;
  $('btnRoute').removeAttribute('disabled');
  render();
});

$('btnRoute').addEventListener('click', async () => {
  if (!S.classes || !S.scene || !S.ship || !S.goal) {
    $('routeInfo').textContent = '请先分类,并用 S/T + 单击设置船位与目标点';
    return;
  }
  $('routeInfo').textContent = '规划中…';
  await tick();
  const { width: w, height: h } = S.scene;
  const factor = Math.max(1, Math.round(w / 1250));
  const r = planRoute(S.classes, w, h, S.ship, S.goal, factor);
  S.route = r.found ? r.path : null;
  $('routeInfo').textContent = r.found
    ? `航路 ${r.path.length} 个航点,总代价 ${r.cost.toFixed(0)},搜索 ${r.explored} 格,耗时 ${r.timeMs.toFixed(0)} ms`
    : '未找到可行航路';
  render();
});

/* ---------------- 画布交互 ---------------- */
document.querySelectorAll('input[name=layer]').forEach(el => {
  el.addEventListener('change', e => {
    S.layer = (e.target as HTMLInputElement).value as AppState['layer'];
    render();
  });
});

let drag: { sx: number; sy: number; panX: number; panY: number; mode: 'pan' | 'rect' } | null = null;

canvas.addEventListener('mousedown', e => {
  if (e.shiftKey) drag = { sx: e.offsetX, sy: e.offsetY, panX: 0, panY: 0, mode: 'rect' };
  else drag = { sx: e.offsetX, sy: e.offsetY, panX: S.panX, panY: S.panY, mode: 'pan' };
});
canvas.addEventListener('mousemove', e => {
  if (drag?.mode === 'pan') {
    S.panX = drag.panX + e.offsetX - drag.sx;
    S.panY = drag.panY + e.offsetY - drag.sy;
    render();
  } else if (drag?.mode === 'rect' && S.scene) {
    const x0 = (drag.sx - S.panX) / S.zoom, y0 = (drag.sy - S.panY) / S.zoom;
    const x1 = (e.offsetX - S.panX) / S.zoom, y1 = (e.offsetY - S.panY) / S.zoom;
    S.enlRect = { x0, y0, x1, y1 };
    render(); updateMetrics();
  }
  if (S.scene) {
    const ix = Math.floor((e.offsetX - S.panX) / S.zoom), iy = Math.floor((e.offsetY - S.panY) / S.zoom);
    if (ix >= 0 && iy >= 0 && ix < S.scene.width && iy < S.scene.height) {
      const i = iy * S.scene.width + ix;
      const db = 10 * Math.log10(Math.max(S.scene.hh[i], 1e-12));
      const cls = S.classes ? CLASS_NAMES[S.classes[i]] : '';
      $('cursorInfo').textContent = `(${ix}, ${iy}) HH ${db.toFixed(1)} dB ${cls}`;
    }
  }
});
canvas.addEventListener('mouseup', () => { drag = null; });
canvas.addEventListener('click', e => {
  if (!S.scene) return;
  const ix = Math.round((e.offsetX - S.panX) / S.zoom), iy = Math.round((e.offsetY - S.panY) / S.zoom);
  if (ix < 0 || iy < 0 || ix >= S.scene.width || iy >= S.scene.height) return;
  if (e.getModifierState?.('Shift') || e.shiftKey) return; // 框选已处理
  if (e.altKey || keyState['KeyE']) { S.edgePt = { x: ix, y: iy }; updateMetrics(); render(); }
  else if (keyState['KeyS']) { S.ship = [ix, iy]; render(); }
  else if (keyState['KeyT']) { S.goal = [ix, iy]; render(); }
});
const keyState: Record<string, boolean> = {};
window.addEventListener('keydown', e => { keyState[e.code] = true; });
window.addEventListener('keyup', e => { keyState[e.code] = false; });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  if (!S.scene) return;
  const k = e.deltaY < 0 ? 1.2 : 1 / 1.2;
  const nx = S.zoom * k;
  S.panX = e.offsetX - (e.offsetX - S.panX) * (nx / S.zoom);
  S.panY = e.offsetY - (e.offsetY - S.panY) * (nx / S.zoom);
  S.zoom = nx;
  render();
}, { passive: false });
window.addEventListener('resize', () => render());

render();

// 测试钩子
(window as unknown as { __app: AppState }).__app = S;
