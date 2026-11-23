/** 冰型类别 */
export enum IceClass {
  OpenWater = 0,
  NewIce = 1,
  FirstYearIce = 2,
  MultiYearIce = 3,
  Ridge = 4, // 冰脊(作为冰的亚类,分类时并入 MYI 代价)
}

export const CLASS_NAMES: Record<number, string> = {
  [IceClass.OpenWater]: '开阔水',
  [IceClass.NewIce]: '新冰',
  [IceClass.FirstYearIce]: '一年冰',
  [IceClass.MultiYearIce]: '多年冰',
  [IceClass.Ridge]: '冰脊',
};

/** 分类输出的 4 类(冰脊并入多年冰代价图) */
export const N_CLASSES = 4;

/** 一景 SAR 影像:HH/HV 线性强度的多视强度图 + 真值类别图 */
export interface SarScene {
  width: number;
  height: number;
  looks: number;          // 雷达视数 L
  hh: Float32Array;       // HH 线性强度的多视强度图
  hv: Float32Array;       // HV 线性强度
  truth: Uint8Array;      // 真值冰型(模拟数据才有)
  meta: SceneMeta;
}

export interface SceneMeta {
  scenario: string;
  windSpeed: number;      // m/s
  noiseFree?: { hh: Float32Array; hv: Float32Array }; // 无斑点均值图(供算法验证)
}

export interface DespeckleResult {
  hh: Float32Array;
  hv: Float32Array;
  width: number;
  height: number;
  timeMs: number;
}

export interface ClassifyResult {
  classes: Uint8Array;    // 0..3
  confidence: Float32Array;
  width: number;
  height: number;
  timeMs: number;
}
