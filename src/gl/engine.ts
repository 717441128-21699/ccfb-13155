/** WebGL2 引擎封装:离屏渲染、纹理/帧缓冲管理、全屏 pass 执行 */
import { VS_QUAD } from './shaders';

export interface GpuTexture {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
  w: number;
  h: number;
  internal: number;
}

export class GLEngine {
  readonly gl: WebGL2RenderingContext;
  private quadBuf: WebGLBuffer;
  private vao: WebGLVertexArrayObject;

  constructor() {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2', {
      antialias: false, depth: false, stencil: false, preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('WebGL2 不可用');
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('缺少 EXT_color_buffer_float 扩展,无法运行 GPU 流水线');
    }
    this.gl = gl;
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  }

  compile(fsSrc: string, defines: Record<string, string> = {}): WebGLProgram {
    const gl = this.gl;
    const def = Object.entries(defines).map(([k, v]) => `#define ${k} ${v}`).join('\n');
    // #version 必须位于最前:#define 注入到 #version 行之后
    const src0 = def ? fsSrc.replace('#version 300 es', m => m + '\n' + def) : fsSrc;
    const mk = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        throw new Error('着色器编译失败: ' + gl.getShaderInfoLog(s));
      }
      return s;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, mk(gl.VERTEX_SHADER, VS_QUAD));
    gl.attachShader(prog, mk(gl.FRAGMENT_SHADER, src0));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('着色器链接失败: ' + gl.getProgramInfoLog(prog));
    }
    return prog;
  }

  createTarget(w: number, h: number, internal: number, format: number, type: number): GpuTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    const fbo = gl.createFramebuffer()!;
    return { tex, fbo, w, h, internal };
  }

  upload(t: GpuTexture, data: Float32Array | Uint8Array, format: number, type: number): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, t.internal, t.w, t.h, 0, format, type, data as never);
  }

  /** 执行一个全屏 pass。inputs 按 uTex0.. 顺序绑定;uniforms 回调设置其余 uniform。 */
  run(
    prog: WebGLProgram,
    target: GpuTexture,
    inputs: Array<{ name: string; tex: GpuTexture }>,
    setUniforms: (gl: WebGL2RenderingContext, prog: WebGLProgram) => void,
    extraTargets: GpuTexture[] = [],
  ): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    const bufs: number[] = [gl.COLOR_ATTACHMENT0];
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target.tex, 0);
    extraTargets.forEach((t, i) => {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1 + i, gl.TEXTURE_2D, t.tex, 0);
      bufs.push(gl.COLOR_ATTACHMENT1 + i);
    });
    gl.drawBuffers(bufs);
    gl.viewport(0, 0, target.w, target.h);
    gl.useProgram(prog);
    inputs.forEach((inp, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, inp.tex.tex);
      gl.uniform1i(gl.getUniformLocation(prog, inp.name), i);
    });
    setUniforms(gl, prog);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  readPixels(t: GpuTexture, x: number, y: number, w: number, h: number, format: number, type: number, dst: ArrayBufferView): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(x, y, w, h, format as never, type as never, dst as never);
  }

  dispose(t: GpuTexture): void {
    this.gl.deleteTexture(t.tex);
    this.gl.deleteFramebuffer(t.fbo);
  }

  finish(): void { this.gl.finish(); }
}
