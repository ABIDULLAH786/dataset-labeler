/* Drone Labeller — client.
 *
 * Three behaviours worth knowing before reading the code:
 *   - there is no unsaved-changes prompt. Navigating saves. A prompt on every frame is friction
 *     in a job that is hundreds of frames long.
 *   - next/prev walk the FILTERED list. With "Blank" selected, Next goes to the next blank
 *     frame, not the next frame in the folder.
 *   - a selected box can be dragged to move or pulled by one of eight handles to resize, the
 *     way an annotation tool is expected to behave.
 */
const $ = s => document.querySelector(s);
const cv = $('#cv'), ctx = cv.getContext('2d');
const lp = $('#lp'), lctx = lp.getContext('2d');

let folder = null, frames = [], cur = null, filter = 'all';
let img = new Image(), iw = 0, ih = 0;
let view = {x: 0, y: 0, s: 1};
let boxes = [], sel = -1, cands = [], candHi = -1;
let draw_ = null, pan = null, grab = null, space = false;
let enhance = false, dirty = false, busy = false;

const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
const HS = 7;                                   // handle hit size, screen px

/* ---------------------------------------------------------------- util */
function toast(msg, kind) {
  const t = document.createElement('div');
  t.className = 'toast ' + (kind || '');
  t.textContent = msg;
  $('#toasts').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .25s'; }, 2200);
  setTimeout(() => t.remove(), 2550);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.text()).slice(0, 180));
  return r.json();
}

function confirmDialog({title, body, okLabel = 'Delete', danger = true}) {
  return new Promise(resolve => {
    $('#mt').textContent = title;
    $('#mb').innerHTML = body;
    const ok = $('#mok'), cancel = $('#mcancel');
    ok.textContent = okLabel;
    ok.className = 'btn ' + (danger ? 'destroy' : 'pri');
    $('#veil').classList.add('on');
    const done = v => {
      $('#veil').classList.remove('on');
      ok.onclick = cancel.onclick = null;
      document.removeEventListener('keydown', esc, true);
      resolve(v);
    };
    const esc = e => {
      if (e.key === 'Escape') { e.stopPropagation(); done(false); }
      if (e.key === 'Enter') { e.stopPropagation(); done(true); }
    };
    ok.onclick = () => done(true);
    cancel.onclick = () => done(false);
    document.addEventListener('keydown', esc, true);
    ok.focus();
  });
}

/* ---------------------------------------------------------------- data */
async function loadFolders() {
  const fs = await api('/api/folders');
  $('#folders').innerHTML = fs.map(f =>
    `<div class="fo" data-n="${f.name}">${f.name}<small>${f.count} frames</small></div>`).join('');
  $('#folders').querySelectorAll('.fo').forEach(d =>
    d.onclick = () => selectFolder(d.dataset.n));
  if (fs.length === 1) selectFolder(fs[0].name);
}

async function selectFolder(name) {
  folder = name;
  $('#folders').querySelectorAll('.fo').forEach(d => d.classList.toggle('on', d.dataset.n === name));
  frames = await api('/api/frames?folder=' + encodeURIComponent(name));
  cur = null;
  renderFrames();
  const list = view_list();
  if (list.length) open(list[0].name);
}

function isCheck(f) { return /CHECK/i.test(f.note || ''); }

function view_list() {
  return frames.filter(f =>
    filter === 'all' ? true :
    filter === 'yes' ? f.boxes > 0 :
    filter === 'no' ? f.has_label && f.boxes === 0 :
    isCheck(f));
}

function counts() {
  $('#c-all').textContent = frames.length;
  $('#c-yes').textContent = frames.filter(f => f.boxes > 0).length;
  $('#c-no').textContent = frames.filter(f => f.has_label && f.boxes === 0).length;
  $('#c-check').textContent = frames.filter(isCheck).length;
}

function renderFrames() {
  const list = view_list();
  counts();
  $('#fcount').textContent = list.length + (filter === 'all' ? '' : ' of ' + frames.length);
  $('#frames').innerHTML = list.length ? list.map(f => {
    const cls = !f.has_label ? 'u' : f.boxes ? 'y' : 'n';
    return `<div class="fr${f.name === cur ? ' on' : ''}" data-n="${f.name}">` +
      `<span class="dot ${cls}"></span>${String(f.n).padStart(5, '0')}` +
      `${isCheck(f) ? '<span class="chk">!</span>' : ''}` +
      `<span class="cnt">${f.boxes || ''}</span></div>`;
  }).join('') : '<div class="empty">No frames match this filter.</div>';
  $('#frames').querySelectorAll('.fr').forEach(d => d.onclick = () => open(d.dataset.n));
  const on = $('#frames .fr.on');
  if (on) on.scrollIntoView({block: 'nearest'});
  const i = list.findIndex(f => f.name === cur);
  $('#pos').textContent = i >= 0 ? `${i + 1} / ${list.length}` : '';
  $('#prev').disabled = i <= 0;
  $('#next').disabled = i < 0 || i >= list.length - 1;
}

async function open(name, skipSave) {
  if (!skipSave && dirty && cur) { await save(true); }
  cur = name; sel = -1; cands = []; candHi = -1; dirty = false;
  renderCands();
  $('#title').textContent = name;
  const d = await api(`/api/label?folder=${encodeURIComponent(folder)}&name=${encodeURIComponent(name)}`);
  iw = d.w; ih = d.h; boxes = d.boxes.map(b => b.slice());
  img = new Image();
  img.onload = () => { fit(); render(); };
  img.src = imgUrl(name);
  renderBoxes(); renderFrames();
}

function imgUrl(name) {
  return `/api/image?folder=${encodeURIComponent(folder)}&name=${encodeURIComponent(name)}` +
         `&enhance=${enhance ? 1 : 0}&t=${Date.now()}`;
}

function step(delta) {
  const list = view_list();
  const i = list.findIndex(f => f.name === cur);
  if (i < 0) { if (list.length) open(list[0].name); return; }
  const j = i + delta;
  if (j < 0 || j >= list.length) { toast(delta > 0 ? 'Last frame in this filter' : 'First frame in this filter'); return; }
  open(list[j].name);
}

/* ---------------------------------------------------------------- canvas */
function stageSize() { const r = $('#stage').getBoundingClientRect(); return [r.width, r.height]; }

function fit() {
  const [W, H] = stageSize();
  cv.width = W; cv.height = H;
  view.s = Math.min(W / iw, H / ih) * 0.96;
  view.x = (W - iw * view.s) / 2;
  view.y = (H - ih * view.s) / 2;
}

const toImg = (px, py) => [(px - view.x) / view.s, (py - view.y) / view.s];
const toScr = (ix, iy) => [ix * view.s + view.x, iy * view.s + view.y];

function norm(b) {
  return [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
}

function handlePoints(b) {
  const [x0, y0] = toScr(b[0], b[1]), [x1, y1] = toScr(b[2], b[3]);
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  return {nw: [x0, y0], n: [mx, y0], ne: [x1, y0], e: [x1, my],
          se: [x1, y1], s: [mx, y1], sw: [x0, y1], w: [x0, my]};
}

function hitHandle(px, py) {
  if (sel < 0) return null;
  const hp = handlePoints(boxes[sel]);
  for (const k of HANDLES) {
    const [hx, hy] = hp[k];
    if (Math.abs(px - hx) <= HS && Math.abs(py - hy) <= HS) return k;
  }
  return null;
}

function render() {
  const [W, H] = stageSize();
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  ctx.fillStyle = '#07090c'; ctx.fillRect(0, 0, W, H);
  if (!img.complete || !iw) return;
  ctx.imageSmoothingEnabled = view.s < 2;
  ctx.drawImage(img, view.x, view.y, iw * view.s, ih * view.s);

  cands.forEach((c, i) => {
    const [x0, y0] = toScr(c.box[0], c.box[1]), [x1, y1] = toScr(c.box[2], c.box[3]);
    ctx.strokeStyle = i === candHi ? '#efa83e' : 'rgba(239,168,62,.5)';
    ctx.lineWidth = i === candHi ? 2 : 1;
    ctx.setLineDash(i === candHi ? [] : [3, 3]);
    ctx.strokeRect(x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4);
  });
  ctx.setLineDash([]);

  boxes.forEach((b, i) => {
    const [x0, y0] = toScr(b[0], b[1]), [x1, y1] = toScr(b[2], b[3]);
    ctx.strokeStyle = i === sel ? '#52c08a' : '#ff4d4d';
    ctx.lineWidth = i === sel ? 2 : 1.5;
    ctx.strokeRect(x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = '11px ui-monospace,monospace';
    ctx.fillText(`${Math.round(b[2] - b[0])}×${Math.round(b[3] - b[1])}`, x0 - 2, y0 - 8);
    if (i === sel) {
      const hp = handlePoints(b);
      ctx.fillStyle = '#52c08a';
      ctx.strokeStyle = '#0e1116'; ctx.lineWidth = 1;
      for (const k of HANDLES) {
        const [hx, hy] = hp[k];
        ctx.fillRect(hx - 4, hy - 4, 8, 8);
        ctx.strokeRect(hx - 4, hy - 4, 8, 8);
      }
    }
  });

  if (draw_) {
    ctx.strokeStyle = '#52c08a'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    ctx.strokeRect(draw_.sx, draw_.sy, draw_.cx - draw_.sx, draw_.cy - draw_.sy);
    ctx.setLineDash([]);
  }
}

function loupe(px, py) {
  const el = $('#loupe');
  if (!img.complete || !iw || draw_ || grab) { el.style.display = 'none'; return; }
  const [ix, iy] = toImg(px, py), Z = 8, R = 210 / Z;
  el.style.display = 'block';
  el.style.left = Math.min(px + 24, cv.width - 218) + 'px';
  el.style.top = Math.max(py - 230, 8) + 'px';
  lctx.imageSmoothingEnabled = false;
  lctx.fillStyle = '#000'; lctx.fillRect(0, 0, 210, 210);
  lctx.drawImage(img, ix - R / 2, iy - R / 2, R, R, 0, 0, 210, 210);
  boxes.forEach(b => {
    lctx.strokeStyle = '#ff4d4d'; lctx.lineWidth = 1;
    lctx.strokeRect((b[0] - (ix - R / 2)) * Z, (b[1] - (iy - R / 2)) * Z,
                    (b[2] - b[0]) * Z, (b[3] - b[1]) * Z);
  });
  lctx.strokeStyle = 'rgba(239,168,62,.85)'; lctx.lineWidth = 1;
  lctx.beginPath();
  lctx.moveTo(105, 93); lctx.lineTo(105, 117);
  lctx.moveTo(93, 105); lctx.lineTo(117, 105);
  lctx.stroke();
}

/* ---------------------------------------------------------------- mouse */
const stage = $('#stage');

stage.addEventListener('mousedown', e => {
  if (busy) return;
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  if (e.button === 1 || space) { pan = {x: px, y: py}; e.preventDefault(); return; }
  if (e.button !== 0) return;

  const h = hitHandle(px, py);
  if (h) { grab = {mode: 'resize', h, start: boxes[sel].slice()}; return; }

  const [ix, iy] = toImg(px, py);
  const hit = boxes.findIndex(b => ix >= b[0] - 3 && ix <= b[2] + 3 && iy >= b[1] - 3 && iy <= b[3] + 3);
  if (hit >= 0) {
    sel = hit; renderBoxes(); render();
    grab = {mode: 'move', ox: ix, oy: iy, start: boxes[sel].slice()};
    return;
  }
  sel = -1; renderBoxes();
  draw_ = {sx: px, sy: py, cx: px, cy: py};
});

stage.addEventListener('mousemove', e => {
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;

  if (pan) { view.x += px - pan.x; view.y += py - pan.y; pan = {x: px, y: py}; render(); return; }

  if (grab) {
    const [ix, iy] = toImg(px, py);
    const b = boxes[sel], s = grab.start;
    if (grab.mode === 'move') {
      const dx = ix - grab.ox, dy = iy - grab.oy;
      boxes[sel] = [s[0] + dx, s[1] + dy, s[2] + dx, s[3] + dy];
    } else {
      let [x0, y0, x1, y1] = s;
      if (grab.h.includes('n')) y0 = iy;
      if (grab.h.includes('s')) y1 = iy;
      if (grab.h.includes('w')) x0 = ix;
      if (grab.h.includes('e')) x1 = ix;
      boxes[sel] = norm([x0, y0, x1, y1]);
    }
    dirty = true; renderBoxes(); render();
    return;
  }

  if (draw_) { draw_.cx = px; draw_.cy = py; render(); return; }

  const h = hitHandle(px, py);
  if (h) stage.style.cursor = (h === 'n' || h === 's') ? 'ns-resize' :
                              (h === 'e' || h === 'w') ? 'ew-resize' :
                              (h === 'nw' || h === 'se') ? 'nwse-resize' : 'nesw-resize';
  else if (space) stage.style.cursor = 'grab';
  else {
    const [ix, iy] = toImg(px, py);
    const over = boxes.some(b => ix >= b[0] && ix <= b[2] && iy >= b[1] && iy <= b[3]);
    stage.style.cursor = over ? 'move' : 'crosshair';
  }
  loupe(px, py);
});

stage.addEventListener('mouseleave', () => { $('#loupe').style.display = 'none'; });

stage.addEventListener('mouseup', async e => {
  if (pan) { pan = null; return; }
  if (grab) { grab = null; boxes[sel] = norm(boxes[sel]); renderBoxes(); render(); return; }
  if (!draw_) return;

  const d = draw_; draw_ = null;
  const moved = Math.abs(d.cx - d.sx) > 4 || Math.abs(d.cy - d.sy) > 4;
  if (moved) {
    const [x0, y0] = toImg(Math.min(d.sx, d.cx), Math.min(d.sy, d.cy));
    const [x1, y1] = toImg(Math.max(d.sx, d.cx), Math.max(d.sy, d.cy));
    boxes.push([x0, y0, x1, y1]); sel = boxes.length - 1; dirty = true;
    renderBoxes(); render();
  } else {
    const [ix, iy] = toImg(d.sx, d.sy);
    busy = true;
    try {
      const r = await api('/api/snap', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({folder, name: cur, x: ix, y: iy, rad: 70})});
      if (r.box) {
        boxes.push(r.box); sel = boxes.length - 1; dirty = true;
        toast(`Snapped ${Math.round(r.box[2] - r.box[0])}×${Math.round(r.box[3] - r.box[1])} px`, 'ok');
      } else toast('Nothing found there — drag to draw by hand', 'err');
    } catch (err) { toast('Snap failed: ' + err.message, 'err'); }
    busy = false;
    renderBoxes(); render();
  }
});

stage.addEventListener('wheel', e => {
  e.preventDefault();
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const [ix, iy] = toImg(px, py);
  view.s = Math.max(0.05, Math.min(80, view.s * (e.deltaY < 0 ? 1.18 : 1 / 1.18)));
  view.x = px - ix * view.s; view.y = py - iy * view.s;
  render(); loupe(px, py);
}, {passive: false});

/* ---------------------------------------------------------------- panels */
function renderBoxes() {
  $('#bcount').textContent = boxes.length || '';
  $('#boxes').innerHTML = boxes.length ? boxes.map((b, i) =>
    `<div class="bx${i === sel ? ' sel' : ''}" data-i="${i}">` +
    `<span class="sz">${Math.round(b[2] - b[0])}×${Math.round(b[3] - b[1])}</span>` +
    `<span class="at">(${Math.round((b[0] + b[2]) / 2)}, ${Math.round((b[1] + b[3]) / 2)})</span>` +
    `<button class="x" data-d="${i}" title="Remove">×</button></div>`).join('')
    : '<div class="empty">None. Click the drone to snap a box, or drag one.</div>';
  $('#boxes').querySelectorAll('.bx').forEach(el => el.onclick = ev => {
    const d = ev.target.dataset.d;
    if (d !== undefined) {
      boxes.splice(+d, 1); sel = -1; dirty = true; renderBoxes(); render();
      return;
    }
    sel = +el.dataset.i; renderBoxes(); render();
  });
}

function renderCands() {
  $('#cands').innerHTML = cands.length ? cands.map((c, i) =>
    `<div class="cd" data-i="${i}"><span class="tag ${c.src}">${c.src}</span>` +
    `${Math.round(c.box[2] - c.box[0])}×${Math.round(c.box[3] - c.box[1])} ` +
    `<span class="at">(${Math.round((c.box[0] + c.box[2]) / 2)}, ` +
    `${Math.round((c.box[1] + c.box[3]) / 2)})</span></div>`).join('')
    : '<div class="empty">Press <b>A</b> to run the detectors.</div>';
  $('#cands').querySelectorAll('.cd').forEach(el => {
    el.onmouseenter = () => { candHi = +el.dataset.i; render(); };
    el.onmouseleave = () => { candHi = -1; render(); };
    el.onclick = () => {
      const c = cands[+el.dataset.i];
      boxes.push(c.box.slice()); sel = boxes.length - 1; dirty = true;
      renderBoxes(); render(); toast('Candidate accepted', 'ok');
    };
  });
}

/* ---------------------------------------------------------------- actions */
async function save(quiet) {
  if (!cur) return;
  try {
    const r = await api('/api/label', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({folder, name: cur, boxes})});
    const f = frames.find(x => x.name === cur);
    if (f) { f.boxes = r.boxes; f.has_label = true; }
    dirty = false;
    renderFrames();
    if (!quiet) toast(r.boxes ? `Saved ${r.boxes} box${r.boxes > 1 ? 'es' : ''}` : 'Saved as no-drone', 'ok');
  } catch (e) { toast('Save failed: ' + e.message, 'err'); }
}

async function runAI() {
  if (!cur || busy) return;
  busy = true; toast('Running detectors…');
  try {
    const r = await api('/api/ai', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({folder, name: cur})});
    cands = r.candidates; renderCands(); render();
    (r.notes || []).forEach(n => toast(n, 'err'));
    toast(`${cands.length} candidate${cands.length === 1 ? '' : 's'}`, cands.length ? 'ok' : 'err');
  } catch (e) { toast('Detectors failed: ' + e.message, 'err'); }
  busy = false;
}

$('#prev').onclick = () => step(-1);
$('#next').onclick = () => step(1);
$('#save').onclick = () => save(false);
$('#ai').onclick = runAI;
$('#fit').onclick = () => { fit(); render(); };

$('#blank').onclick = async () => {
  if (!cur) return;
  boxes = []; sel = -1; dirty = true; renderBoxes(); render();
  await save(true);
  toast('Marked as no drone — empty label written', 'ok');
};

$('#enh').onclick = () => {
  enhance = !enhance;
  $('#enh').classList.toggle('on', enhance);
  if (!cur) return;
  const keep = {...view};
  img = new Image();
  img.onload = () => { view = keep; render(); };
  img.src = imgUrl(cur);
};

$('#del').onclick = async () => {
  if (!cur) return;
  const f = frames.find(x => x.name === cur);
  const ok = await confirmDialog({
    title: 'Delete this frame?',
    body: `<b>${cur}</b><p>This removes the frame and everything derived from it:</p>` +
          `<ul><li>images/ — the original frame</li><li>labels/ — its label file</li>` +
          `<li>annotated/ — the boxed copy</li><li>crops/ — the review crop</li>` +
          `<li>REVIEW.csv — its row</li></ul>` +
          `<div class="warn">This cannot be undone from here. The frame can only come back ` +
          `from the source folder it was cropped from.</div>`,
    okLabel: 'Delete frame'});
  if (!ok) return;
  try {
    const list = view_list();
    const i = list.findIndex(x => x.name === cur);
    const nextName = (list[i + 1] || list[i - 1] || {}).name;
    await api('/api/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({folder, name: cur})});
    frames = frames.filter(x => x.name !== cur);
    dirty = false;
    toast('Frame deleted', 'ok');
    if (nextName) open(nextName, true); else { cur = null; renderFrames(); }
  } catch (e) { toast('Delete failed: ' + e.message, 'err'); }
};

$('#filters').querySelectorAll('button').forEach(b => b.onclick = () => {
  filter = b.dataset.f;
  $('#filters').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  renderFrames();
  const list = view_list();
  if (list.length && !list.some(f => f.name === cur)) open(list[0].name);
});

/* ---------------------------------------------------------------- keys */
addEventListener('keydown', e => {
  if ($('#veil').classList.contains('on')) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.code === 'Space') { space = true; stage.style.cursor = 'grab'; e.preventDefault(); return; }
  if (e.ctrlKey || e.metaKey) return;
  const k = e.key.toLowerCase();
  if (e.key === 'ArrowRight' || k === 'd') { e.preventDefault(); step(1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
  else if (k === 's') { e.preventDefault(); save(false); }
  else if (k === 'b') $('#blank').click();
  else if (k === 'a') runAI();
  else if (k === 'e') $('#enh').click();
  else if (k === 'f') { fit(); render(); }
  else if (e.key === 'Escape') { sel = -1; renderBoxes(); render(); }
  else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (sel >= 0) { boxes.splice(sel, 1); sel = -1; dirty = true; renderBoxes(); render(); }
  }
});
addEventListener('keyup', e => {
  if (e.code === 'Space') { space = false; stage.style.cursor = 'crosshair'; }
});
addEventListener('resize', () => render());
addEventListener('beforeunload', e => { if (dirty) { save(true); } });

loadFolders().then(() => toast('Pick a folder to begin'));
