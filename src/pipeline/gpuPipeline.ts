/**
 * GPU 流水线:分块(tiling)在 WebGL2 中执行去斑与分类。
 * 大图按 2048×2048 内块 + halo 重叠分块处理,结果读回 CPU 大数组。
 * 算法与 cpuPipeline 完全一致。
 */
import { GLEngine, GpuTexture } from '../gl/engine';
import { FS_BLUR, FS_FILTER, FS_CLASSIFY } from '../gl/shaders';
import { MODEL_PARAMS, N_MODELS } from './cpuPipeline';

const TILE = 2048;
const HALO_FILTER = 16;
const HALO_CLASSIFY = 8;

interface Tile { x0: number; y0: number; w: number; h: number; px0: number; py0: number; pw: number; ph: number; }

function splitTiles(w: number, h: number, tile: number, halo: number): Tile[] {
  const out: Tile[] = [];
  for (let y0 = 0; y0 < h; y0 += tile) {
    for (let x0 = 0; x0 < w; x0 += tile) {
      const tw = Math.min(tile, w - x0), th = Math.min(tile, h - y0);
      const px0 = Math.max(0, x0 - halo), py0 = Math.max(0, y0 - halo);
      const px1 = Math.min(w, x0 + tw + halo), py1 = Math.min(h, y0 + th + halo);
      out.push({ x0, y0, w: tw, h: th, px0, py0, pw: px1 - px0, ph: py1 - py0 });
    }
  }
  return out;
}

export class GpuPipeline {
  private eng: GLEngine;
  private progBlur3H: WebGLProgram;
  private progBlur3V: WebGLProgram;
  private progBlur2H: WebGLProgram;
  private progBlur2V: WebGLProgram;
  private progFilter: WebGLProgram;
  private progClassify: WebGLProgram;
  private pool = new Map<string, GpuTexture[]>();
  private locCache = new Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>();

  constructor() {
    this.eng = new GLEngine();
    this.progBlur3H = this.eng.compile(FS_BLUR, { RADIUS: '3' });
    this.progBlur3V = this.eng.compile(FS_BLUR, { RADIUS: '3' });
    this.progBlur2H = this.eng.compile(FS_BLUR, { RADIUS: '2' });
    this.progBlur2V = this.eng.compile(FS_BLUR, { RADIUS: '2' });
    this.progFilter = this.eng.compile(FS_FILTER);
    this.progClassify = this.eng.compile(FS_CLASSIFY);
    this.checkFloatReadback();
  }

  private checkFloatReadback(): void {
    const gl = this.eng.gl;
    const t = this.eng.createTarget(4, 4, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    this.eng.readPixels(t, 0, 0, 4, 4, gl.RGBA, gl.FLOAT, new Float32Array(64));
    const err = gl.getError();
    this.eng.dispose(t);
    if (err !== gl.NO_ERROR) throw new Error('浮点纹理读回不可用,改用 CPU 流水线');
  }

  private acquire(w: number, h: number, internal: number, format: number, type: number): GpuTexture {
    const key = `${w}x${h}x${internal}`;
    const arr = this.pool.get(key);
    if (arr && arr.length > 0) return arr.pop()!;
    return this.eng.createTarget(w, h, internal, format, type);
  }

  private release(t: GpuTexture): void {
    const key = `${t.w}x${t.h}x${t.internal}`;
    let arr = this.pool.get(key);
    if (!arr) { arr = []; this.pool.set(key, arr); }
    arr.push(t);
  }

  private loc(prog: WebGLProgram, name: string): WebGLUniformLocation | null {
    let m = this.locCache.get(prog);
    if (!m) { m = new Map(); this.locCache.set(prog, m); }
    if (!m.has(name)) m.set(name, this.eng.gl.getUniformLocation(prog, name));
    return m.get(name)!;
  }

  /** 打包 (hh, hv, hh², hv²) 的填充块数据 */
  private static packTile(
    hh: Float32Array, hv: Float32Array, w: number, t: Tile, buf: Float32Array,
  ): Float32Array {
    let k = 0;
    for (let y = 0; y < t.ph; y++) {
      const sy = Math.min(t.py0 + y, t.py0 + t.ph - 1); // 已 clamp,无需再处理
      const row = sy * w;
      for (let x = 0; x < t.pw; x++) {
        const i = row + t.px0 + x;
        const a = hh[i], b = hv[i];
        buf[k++] = a; buf[k++] = b; buf[k++] = a * a; buf[k++] = b * b;
      }
    }
    return buf.subarray(0, t.pw * t.ph * 4);
  }

  /** 去斑:方向自适应精制 Lee,返回 hh'/hv' 全图 */
  despeckle(hh: Float32Array, hv: Float32Array, w: number, h: number, L: number): { hh: Float32Array; hv: Float32Array; timeMs: number } {
    const gl = this.eng.gl;
    const t0 = performance.now();
    const outH = new Float32Array(w * h), outV = new Float32Array(w * h);
    const tiles = splitTiles(w, h, TILE, HALO_FILTER);
    const packBuf = new Float32Array((TILE + 2 * HALO_FILTER) ** 2 * 4);
    const readBuf = new Float32Array(TILE * TILE * 4);

    for (const t of tiles) {
      const packed = GpuPipeline.packTile(hh, hv, w, t, packBuf);
      const raw = this.acquire(t.pw, t.ph, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.eng.upload(raw, packed as Float32Array, gl.RGBA, gl.FLOAT);
      const b1 = this.acquire(t.pw, t.ph, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
      const b2 = this.acquire(t.pw, t.ph, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
      const filt = this.acquire(t.pw, t.ph, gl.RGBA32F, gl.RGBA, gl.FLOAT);

      const stepH = [1 / t.pw, 0], stepV = [0, 1 / t.ph];
      this.eng.run(this.progBlur3H, b1, [{ name: 'uTex', tex: raw }], (g, p) => {
        g.uniform2f(this.loc(p, 'uStep'), stepH[0], stepH[1]);
        g.uniform2f(this.loc(p, 'uUvScale'), 1, 1);
      });
      this.eng.run(this.progBlur3V, b2, [{ name: 'uTex', tex: b1 }], (g, p) => {
        g.uniform2f(this.loc(p, 'uStep'), stepV[0], stepV[1]);
        g.uniform2f(this.loc(p, 'uUvScale'), 1, 1);
      });
      this.eng.run(this.progFilter, filt, [
        { name: 'uRaw', tex: raw }, { name: 'uBlur', tex: b2 },
      ], (g, p) => {
        g.uniform2f(this.loc(p, 'uTexel'), 1 / t.pw, 1 / t.ph);
        g.uniform2f(this.loc(p, 'uUvScale'), 1, 1);
        g.uniform1f(this.loc(p, 'uCu2'), 1 / L);
      });

      // 读回内块
      const ix = t.x0 - t.px0, iy = t.y0 - t.py0;
      this.eng.readPixels(filt, ix, iy, t.w, t.h, gl.RGBA, gl.FLOAT, readBuf);
      for (let y = 0; y < t.h; y++) {
        const dstRow = (t.y0 + y) * w + t.x0;
        const srcRow = y * t.w * 4;
        for (let x = 0; x < t.w; x++) {
          outH[dstRow + x] = readBuf[srcRow + x * 4];
          outV[dstRow + x] = readBuf[srcRow + x * 4 + 1];
        }
      }
      this.release(raw); this.release(b1); this.release(b2); this.release(filt);
    }
    this.eng.finish();
    return { hh: outH, hv: outV, timeMs: performance.now() - t0 };
  }

  /** 冰型分类:返回类别图(0..3) */
  classify(hh: Float32Array, hv: Float32Array, w: number, h: number): { classes: Uint8Array; confidence: Float32Array; timeMs: number } {
    const gl = this.eng.gl;
    const t0 = performance.now();
    const classes = new Uint8Array(w * h);
    const confidence = new Float32Array(w * h);
    const tiles = splitTiles(w, h, TILE, HALO_CLASSIFY);
    const packBuf = new Float32Array((TILE + 2 * HALO_CLASSIFY) ** 2 * 4);
    const readBuf = new Uint8Array(TILE * TILE * 4);

    // 高斯参数 → uniform 数组
    const mu = new Float32Array(N_MODELS * 4), sdRcp = new Float32Array(N_MODELS * 4), logDet = new Float32Array(N_MODELS);
    for (let c = 0; c < N_MODELS; c++) {
      const p = MODEL_PARAMS[c];
      for (let k = 0; k < 4; k++) {
        mu[c * 4 + k] = p.mu[k];
        sdRcp[c * 4 + k] = 1 / p.sd[k];
      }
      logDet[c] = p.sd.reduce((a, s) => a + Math.log(s), 0);
    }

    for (const t of tiles) {
      // 打包 (hh', hv', HH_dB, HH_dB²)
      let k = 0;
      for (let y = 0; y < t.ph; y++) {
        const row = (t.py0 + y) * w;
        for (let x = 0; x < t.pw; x++) {
          const i = row + t.px0 + x;
          const a = hh[i], b = hv[i];
          const db = 10 * Math.log10(Math.max(a, 1e-12));
          packBuf[k++] = a; packBuf[k++] = b; packBuf[k++] = db; packBuf[k++] = db * db;
        }
      }
      const filt = this.acquire(t.pw, t.ph, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.eng.upload(filt, packBuf.subarray(0, t.pw * t.ph * 4) as Float32Array, gl.RGBA, gl.FLOAT);
      const b1 = this.acquire(t.pw, t.ph, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
      const b2 = this.acquire(t.pw, t.ph, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
      const colorT = this.acquire(t.pw, t.ph, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
      const idT = this.acquire(t.pw, t.ph, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);

      this.eng.run(this.progBlur2H, b1, [{ name: 'uTex', tex: filt }], (g, p) => {
        g.uniform2f(this.loc(p, 'uStep'), 1 / t.pw, 0);
        g.uniform2f(this.loc(p, 'uUvScale'), 1, 1);
      });
      this.eng.run(this.progBlur2V, b2, [{ name: 'uTex', tex: b1 }], (g, p) => {
        g.uniform2f(this.loc(p, 'uStep'), 0, 1 / t.ph);
        g.uniform2f(this.loc(p, 'uUvScale'), 1, 1);
      });
      this.eng.run(this.progClassify, colorT, [
        { name: 'uFilt', tex: filt }, { name: 'uDbBlur', tex: b2 },
      ], (g, p) => {
        g.uniform2f(this.loc(p, 'uUvScale'), 1, 1);
        g.uniform4fv(this.loc(p, 'uMu[0]'), mu);
        g.uniform4fv(this.loc(p, 'uSdRcp[0]'), sdRcp);
        g.uniform1fv(this.loc(p, 'uLogDet[0]'), logDet);
      }, [idT]);

      const ix = t.x0 - t.px0, iy = t.y0 - t.py0;
      this.eng.readPixels(idT, ix, iy, t.w, t.h, gl.RGBA, gl.UNSIGNED_BYTE, readBuf);
      for (let y = 0; y < t.h; y++) {
        const dstRow = (t.y0 + y) * w + t.x0;
        const srcRow = y * t.w * 4;
        for (let x = 0; x < t.w; x++) {
          const j = srcRow + x * 4;
          classes[dstRow + x] = Math.round(readBuf[j] / 85);
          confidence[dstRow + x] = readBuf[j + 1] / 255;
        }
      }
      this.release(filt); this.release(b1); this.release(b2); this.release(colorT); this.release(idT);
    }
    this.eng.finish();
    return { classes, confidence, timeMs: performance.now() - t0 };
  }
}
