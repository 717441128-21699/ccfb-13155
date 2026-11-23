/**
 * 端到端验证:真实浏览器(Chromium + WebGL2)中运行完整流程。
 * 用法:node test/e2e.mjs [尺寸]
 * 检查:页面加载无报错 → 生成样例 → GPU 去斑 → GPU 分类(精度)→ 航路规划。
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';

const SIZE = process.argv[2] || '2048';
const PORT = 5200 + Math.floor(Math.random() * 500);

const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--port', String(PORT), '--strictPort'], {
  cwd: new URL('..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'),
  stdio: 'pipe',
});
await new Promise((resolve, reject) => {
  const to = setTimeout(() => reject(new Error('vite preview 启动超时')), 20000);
  server.stdout.on('data', d => { if (String(d).includes('Local:')) { clearTimeout(to); resolve(); } });
  server.stderr.on('data', d => process.stderr.write(d));
});

const browser = await chromium.launch({
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'], // 软渲染 WebGL2,验证 GPU 路径正确性
});
const page = await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

let failed = false;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? '✔' : '✖'} ${name}${detail ? ' — ' + detail : ''}`);
  if (!ok) failed = true;
};

try {
  await page.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle' });
  check('页面加载', true);

  // 生成样例图
  await page.selectOption('#size', SIZE);
  await page.click('#btnGen');
  await page.waitForFunction(() => document.getElementById('genStatus').textContent.includes('已生成'), null, { timeout: 120000 });
  const genStatus = await page.textContent('#genStatus');
  check('生成样例雷达图', true, genStatus.trim());

  // GPU 去斑
  await page.click('#btnDespeckle');
  await page.waitForFunction(() => !document.getElementById('despeckleTime').textContent.includes('处理中'), null, { timeout: 600000 });
  const dspTime = await page.textContent('#despeckleTime');
  check('去斑完成', !dspTime.includes('CPU'), dspTime.trim());

  // ENL 指标:用真值图找一块全为一年冰的 41×41 均匀区,直接在数据上算 ENL
  const enlResult = await page.evaluate(() => {
    const S = window.__app;
    const { width: w, height: h, truth, hh } = S.scene;
    const filt = S.filtHh;
    const enlOf = (data, cx, cy, size) => {
      let s = 0, s2 = 0, n = 0;
      for (let y = cy - size / 2; y < cy + size / 2; y++)
        for (let x = cx - size / 2; x < cx + size / 2; x++) {
          const v = data[Math.round(y) * w + Math.round(x)];
          s += v; s2 += v * v; n++;
        }
      const m = s / n;
      return m * m / (s2 / n - m * m);
    };
    outer: for (let y = 60; y < h - 60; y += 13) {
      for (let x = 60; x < w - 60; x += 13) {
        let ok = true;
        for (let dy = -20; dy < 20 && ok; dy += 5)
          for (let dx = -20; dx < 20 && ok; dx += 5)
            if (truth[(y + dy) * w + x + dx] !== 2) ok = false;
        if (ok) {
          const e0 = enlOf(hh, x, y, 41), e1 = enlOf(filt, x, y, 41);
          return { e0, e1, gain: e1 / e0 };
        }
        continue outer;
      }
    }
    return null;
  });
  check('ENL 提升 ≥ 8×(均匀一年冰区)', enlResult && enlResult.gain >= 8,
    enlResult ? `ENL ${enlResult.e0.toFixed(1)} → ${enlResult.e1.toFixed(1)}(${enlResult.gain.toFixed(1)}×)` : '未找到均匀区');

  // 分类
  await page.click('#btnClassify');
  await page.waitForFunction(() => !document.getElementById('classInfo').textContent.includes('分类中'), null, { timeout: 600000 });
  const classInfo = await page.textContent('#classInfo');
  const accMatch = classInfo.match(/(\d+\.\d+)%/);
  check('冰型分类完成', true, classInfo.trim());
  check('分类精度 ≥ 80%', accMatch && parseFloat(accMatch[1]) >= 80, accMatch?.[0]);

  // 航路:S+单击设船位,T+单击设目标
  const box = await page.locator('#view').boundingBox();
  await page.keyboard.down('s');
  await page.mouse.click(box.x + box.width * 0.15, box.y + box.height * 0.5);
  await page.keyboard.up('s');
  await page.keyboard.down('t');
  await page.mouse.click(box.x + box.width * 0.85, box.y + box.height * 0.5);
  await page.keyboard.up('t');
  await page.click('#btnRoute');
  await page.waitForFunction(() => !document.getElementById('routeInfo').textContent.includes('规划中'), null, { timeout: 120000 });
  const routeInfo = await page.textContent('#routeInfo');
  check('航路规划', routeInfo.includes('航路') && !routeInfo.includes('未找到'), routeInfo.trim());

  // 处理总耗时(去斑+分类)对 10000×10000 的 20 s 指标
  const dspMs = parseFloat((dspTime.match(/([\d.]+) s/) || [])[1]) * 1000;
  const clsMs = parseFloat((classInfo.match(/([\d.]+) s/) || [])[1]) * 1000;
  if (SIZE === '10000') {
    check('处理耗时 ≤ 20 s(去斑+分类)', dspMs + clsMs <= 20000, `${((dspMs + clsMs) / 1000).toFixed(1)} s(注意:软渲染 SwiftShader)`);
  } else {
    console.log(`  处理耗时:去斑 ${(dspMs / 1000).toFixed(2)} s + 分类 ${(clsMs / 1000).toFixed(2)} s(${SIZE}²,SwiftShader 软渲染)`);
  }

  check('无页面 JS 错误', errors.length === 0, errors.join('; '));
} finally {
  await browser.close();
  spawn('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
}
process.exit(failed ? 1 : 0);
