const pad = document.getElementById('pad');
const ctx = pad.getContext('2d', { willReadFrequently: true });
const preview = document.getElementById('preview');
const pctx = preview.getContext('2d');
const digitEl = document.getElementById('digit');
const barsEl = document.getElementById('bars');

ctx.fillStyle = '#000'; ctx.fillRect(0, 0, pad.width, pad.height);
ctx.strokeStyle = '#fff';
ctx.lineWidth = 22;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

let drawing = false, last = null;

function pos(e) {
  const r = pad.getBoundingClientRect();
  const t = e.touches ? e.touches[0] : e;
  return { x: t.clientX - r.left, y: t.clientY - r.top };
}
function start(e) { drawing = true; last = pos(e); e.preventDefault(); }
function move(e) {
  if (!drawing) return;
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  last = p;
  schedulePredict();
  e.preventDefault();
}
function end() { drawing = false; last = null; schedulePredict(); }

pad.addEventListener('mousedown', start);
pad.addEventListener('mousemove', move);
window.addEventListener('mouseup', end);
pad.addEventListener('touchstart', start);
pad.addEventListener('touchmove', move);
pad.addEventListener('touchend', end);

document.getElementById('clear').onclick = () => {
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, pad.width, pad.height);
  pctx.fillStyle = '#000'; pctx.fillRect(0, 0, 28, 28);
  clearPrediction();
};

// MNIST-style preprocessing: crop to bbox, fit in 20x20, center in 28x28.
function preprocess() {
  const src = ctx.getImageData(0, 0, pad.width, pad.height).data;
  const W = pad.width, H = pad.height;
  let minX = W, minY = H, maxX = -1, maxY = -1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (src[(y * W + x) * 4] > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;

  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const sw = Math.max(1, Math.round(bw * scale));
  const sh = Math.max(1, Math.round(bh * scale));

  pctx.fillStyle = '#000'; pctx.fillRect(0, 0, 28, 28);
  pctx.imageSmoothingEnabled = true;
  const ox = Math.floor((28 - sw) / 2);
  const oy = Math.floor((28 - sh) / 2);
  pctx.drawImage(pad, minX, minY, bw, bh, ox, oy, sw, sh);

  const out = pctx.getImageData(0, 0, 28, 28).data;
  const pixels = new Array(784);
  for (let i = 0; i < 784; i++) pixels[i] = out[i * 4] / 255;
  return pixels;
}

const barEls = [], pctEls = [];
for (let i = 0; i < 10; i++) {
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `<span class="label">${i}</span>` +
                  `<div class="bar-wrap"><div class="bar" style="width:0px;"></div></div>` +
                  `<span class="pct">0.0%</span>`;
  barsEl.appendChild(row);
  barEls.push(row.querySelector('.bar'));
  pctEls.push(row.querySelector('.pct'));
}

function renderPrediction(digit, probs) {
  digitEl.textContent = digit;
  probs.forEach((p, i) => {
    barEls[i].style.width = `${Math.round(p * 160)}px`;
    pctEls[i].textContent = `${(p * 100).toFixed(1)}%`;
  });
}

function clearPrediction() {
  digitEl.textContent = '–';
  for (let i = 0; i < 10; i++) {
    barEls[i].style.width = '0px';
    pctEls[i].textContent = '0.0%';
  }
}

let inFlight = false, dirty = false;
async function schedulePredict() {
  if (inFlight) { dirty = true; return; }
  inFlight = true;
  try {
    do {
      dirty = false;
      const pixels = preprocess();
      if (!pixels) { clearPrediction(); break; }
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixels })
      });
      const { digit, probs } = await res.json();
      renderPrediction(digit, probs);
    } while (dirty);
  } finally {
    inFlight = false;
  }
}

document.getElementById('predict').onclick = schedulePredict;
