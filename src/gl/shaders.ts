/** GLSL 着色器源码(WebGL2 / GLSL ES 3.00)。算法与 cpuPipeline 一致。 */

export const VS_QUAD = `#version 300 es
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){ vUV = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }
`;

/** 可分离盒式模糊,RADIUS 由 define 注入,uStep = texel * 方向 */
export const FS_BLUR = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uStep;
uniform vec2 uUvScale;
in vec2 vUV;
out vec4 o;
void main(){
  vec2 uv = vUV * uUvScale;
  vec4 acc = vec4(0.0);
  for (int i = -RADIUS; i <= RADIUS; i++) acc += texture(uTex, uv + uStep * float(i));
  o = acc / float(2 * RADIUS + 1);
}
`;

/**
 * 方向自适应精制 Lee 滤波。
 * uRaw: (hh, hv, hh², hv²);uBlur: 上者的 7×7 均值。
 * 输出: (hh', hv', HH'_dB, (HH'_dB)²) —— 后两个通道供纹理统计。
 */
export const FS_FILTER = `#version 300 es
precision highp float;
uniform sampler2D uRaw;
uniform sampler2D uBlur;
uniform vec2 uTexel;
uniform vec2 uUvScale;
uniform float uCu2;   // 斑点变异系数平方 = 1/L
in vec2 vUV;
out vec4 o;

const vec2 DIRS[4] = vec2[4](vec2(1.0,0.0), vec2(0.0,1.0), vec2(1.0,1.0), vec2(1.0,-1.0));

float lee1(float x, float mu, float varr, vec2 uv, int c) {
  float cx2 = varr / (mu * mu + 1e-20);
  float k = cx2 > uCu2 ? 1.0 - uCu2 / cx2 : 0.0;
  if (k < 0.25) return mu;
  float bestVar = 1e30;
  float bestMu = mu;
  for (int d = 0; d < 4; d++) {
    vec2 dir = DIRS[d] * uTexel;
    float s = 0.0, s2 = 0.0;
    for (int t = -4; t <= 4; t++) {
      float v = texture(uRaw, uv + dir * float(t))[c];
      s += v; s2 += v * v;
    }
    float m = s / 9.0;
    float vv = s2 / 9.0 - m * m;
    if (vv < bestVar) { bestVar = vv; bestMu = m; }
  }
  return bestMu + k * (x - bestMu);
}

void main(){
  vec2 uv = vUV * uUvScale;
  vec4 raw = texture(uRaw, uv);
  vec4 st = texture(uBlur, uv);
  float muH = st.r, muV = st.g;
  float varH = max(st.b - muH * muH, 0.0);
  float varV = max(st.a - muV * muV, 0.0);
  float hh = lee1(raw.r, muH, varH, uv, 0);
  float hv = lee1(raw.g, muV, varV, uv, 1);
  float db = 10.0 * log(max(hh, 1e-12)) * 0.4342944819032518;
  o = vec4(hh, hv, db, db * db);
}
`;

/**
 * 冰型分类:对角协方差高斯最大似然。
 * uFilt: (hh', hv', HH'_dB, ·);uDbBlur: uFilt 的 5×5 均值(用 .z/.w 得纹理)。
 * MRT 输出:oColor = 类别色 + 置信度 alpha;oId = 类别 id(R,按 /3 编码)。
 */
export const FS_CLASSIFY = `#version 300 es
precision highp float;
uniform sampler2D uFilt;
uniform sampler2D uDbBlur;
uniform vec2 uUvScale;
uniform vec4 uMu[5];
uniform vec4 uSdRcp[5];
uniform float uLogDet[5];
in vec2 vUV;
layout(location=0) out vec4 oColor;
layout(location=1) out vec4 oId;

const vec3 COLORS[4] = vec3[4](
  vec3(0.11, 0.37, 0.68),  // 开阔水 蓝
  vec3(0.66, 0.85, 0.90),  // 新冰   浅青
  vec3(0.91, 0.90, 0.87),  // 一年冰 灰白
  vec3(0.77, 0.31, 0.32)   // 多年冰 红
);

void main(){
  vec2 uv = vUV * uUvScale;
  vec4 f = texture(uFilt, uv);
  vec4 dbs = texture(uDbBlur, uv);
  float hhDb = f.z;
  float hvDb = 10.0 * log(max(f.g, 1e-12)) * 0.4342944819032518;
  float tex = sqrt(max(dbs.w - dbs.z * dbs.z, 0.0));
  vec4 feat = vec4(hhDb, hvDb, hhDb - hvDb, tex);
  float best = 1e30, second = 1e30;
  int bestC = 0;
  for (int c = 0; c < 5; c++) {
    vec4 z = (feat - uMu[c]) * uSdRcp[c];
    float d = dot(z, z) + uLogDet[c];
    if (d < best) { second = best; best = d; bestC = c; }
    else if (d < second) { second = d; }
  }
  // 子模型 0/1 都是开阔水,2..4 依次为新冰/一年冰/多年冰
  int cls = bestC < 2 ? 0 : bestC - 1;
  float conf = clamp(1.0 - best / (second + 1e-9), 0.0, 1.0);
  oColor = vec4(COLORS[cls], conf);
  oId = vec4(float(cls) / 3.0, conf, 0.0, 1.0);
}
`;
