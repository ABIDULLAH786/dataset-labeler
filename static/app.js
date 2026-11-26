/*
 * Dataset Labeler - browser side.
 *
 * Three views, switched by setView: browse (pick a folder), gallery (cards, tabs, pages) and
 * annotate (canvas). The gallery exists because scanning a folder one frame at a time tells you
 * nothing about the folder; the annotate view exists because an 8 px target cannot be judged
 * from a card.
 *
 * A box is [x0, y0, x1, y1, cls] in IMAGE pixels. The first four indices are the geometry the
 * canvas code works in; the fifth is the class index, which is what goes in column one of the
 * YOLO label. Conversion to and from the server's object form happens only at the boundary.
 */
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const cv = $('#cv'), ctx = cv.getContext('2d');
const lp = $('#lp'), lctx = lp.getContext('2d');

let base = null, root = null;
let browsePath = null, browseSelf = null;
let folder = null, folders = [], frames = [], cur = null;
let status_ = 'all', page = 1, per = 60, query = '';
let counts = {}, pages = 1, selected = new Set();
let total = 0, pagePer = 60;                     // frames in the whole filtered list, page size
let wasF = '', wasCounts = {};                   // Deleted tab: "was annotated" etc. filter
let classes = [], activeClass = 0, classHi = 0;

let img = new Image(), iw = 0, ih = 0;
let viewport = {x: 0, y: 0, s: 1};
let boxes = [], sel = -1, cands = [], candHi = -1;
let draw_ = null, pan = null, grab = null, space = false;
let enhance = false, dirty = false, busy = false;
let gridMode = localStorage.getItem('dl-view') || 'grid';
let bright = 0;
let loupeOn = localStorage.getItem('dl-loupe') !== 'off';
// Magnifier strength. 100% is the full 8x; 75/50/25% are 6x/4x/2x, wider and less enlarged.
const LOUPE_MAX = 8, LOUPE_STEPS = [25, 50, 75, 100];
let loupePct = (() => {
  try { const v = +localStorage.getItem('dl-loupe-pct'); return LOUPE_STEPS.includes(v) ? v : 100; }
  catch { return 100; }
})();
// Undo stack of whole box-lists. A frame holds a handful of boxes, so snapshotting the lot is
// simpler and less error-prone than recording deltas, and the memory is nothing.
let past = [], future = [];

const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
const HS = 7;                                   // handle hit size, screen px
const DRAG_MIN = 5;                             // px of drag before a box is created at all
const STATUSES = [['all', 'All'], ['unassigned', 'Unassigned'], ['annotated', 'Annotated'],
                  ['null', 'Null'], ['rejected', 'Rejected'], ['deleted', 'Deleted']];
// shown as words in the list view, where there is room to say what the dot means
const STATUS_LABEL = {unassigned: 'Unassigned', annotated: 'Annotated', null: 'Null',
                      rejected: 'Rejected', deleted: 'Deleted'};
const WAS = [['', 'any state'], ['annotated', 'annotated'], ['unassigned', 'unassigned'],
             ['null', 'null'], ['rejected', 'rejected']];
// must match PALETTE in server.py, so a class is the same colour on a card and on the canvas
const PALETTE = ['#f5c518', '#2fb673', '#3c90e0', '#9b5fd0', '#e0614f', '#17a8a0', '#e0699f',
                 '#6b7fa8', '#c08a3e', '#7cb342'];

/* ---------------------------------------------------------------- plumbing */
function toast(msg, kind) {
  // the same message already on screen is not shown again - a key pressed ten times against
  // a failing server used to stack ten identical errors down the side of the page
  for (const t of $('#toasts').children) if (t.textContent === msg) return;
  const el = document.createElement('div');
  el.className = 'toast' + (kind ? ' ' + kind : '');
  el.textContent = msg;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), 3600);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const detail = (await r.json().catch(() => ({}))).detail;
    // A 404 on an endpoint itself (not "no such frame") means the page is newer than the
    // server: the browser reloads the script, but Python keeps running the old server.py.
    if (r.status === 404 && (!detail || detail === 'Not Found'))
      throw new Error('the server is running older code than this page - ' +
                      'stop labeller (Ctrl+C), start it again, then reload');
    throw new Error(detail || r.statusText);
  }
  return r.json();
}

const post = (path, body) => api(path, {
  method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});

function confirmDialog({title, body, okLabel = 'Delete', danger = true}) {
  return new Promise(resolve => {
    $('#mt').textContent = title;
    $('#mb').innerHTML = body;
    const ok = $('#mok');
    ok.textContent = okLabel;
    ok.className = 'btn ' + (danger ? 'destroy' : 'pri');
    $('#veil').classList.add('on');
    const done = v => {
      $('#veil').classList.remove('on');
      ok.onclick = null; $('#mcancel').onclick = null;
      document.removeEventListener('keydown', key, true);
      resolve(v);
    };
    const key = e => {
      if (e.key === 'Escape') { e.stopPropagation(); done(false); }
      if (e.key === 'Enter') { e.stopPropagation(); done(true); }
    };
    ok.onclick = () => done(true);
    $('#mcancel').onclick = () => done(false);
    document.addEventListener('keydown', key, true);
    ok.focus();
  });
}

function setView(v) {
  const prev = document.body.dataset.view;
  $$('.view').forEach(s => s.classList.remove('on'));
  $('#view-' + v).classList.add('on');
  document.body.dataset.view = v;
  if (v === 'annotate') requestAnimationFrame(() => { fitView(); render(); });
  // Switching between gallery and annotate is a new history entry, so Back returns to it.
  if (v === 'gallery' || v === 'annotate')
    syncUrl(prev !== v && (prev === 'gallery' || prev === 'annotate'));
}

/* ---------------------------------------------------------------- address bar
 * The URL carries where you are - view, folder, tab, page, open frame - so a reload (even a
 * hard one) comes back to the same frame instead of the gallery:
 *   #annotate?folder=1_fires&status=all&page=2&per=60&frame=f4_b1_00126.jpg          */
let restoring = false;
function syncUrl(push) {
  if (!folder) return;
  const view = document.body.dataset.view === 'annotate' ? 'annotate' : 'gallery';
  const p = new URLSearchParams({folder, status: status_, page, per});
  if (query) p.set('q', query);
  if (wasF && status_ === 'deleted') p.set('was', wasF);
  if (view === 'annotate' && cur) p.set('frame', cur);
  const url = `#${view}?${p}`;
  if (location.hash === url) return;
  history[push && !restoring ? 'pushState' : 'replaceState'](null, '', url);
}

async function restoreFromUrl() {
  const [view, qs] = location.hash.slice(1).split('?');
  const u = new URLSearchParams(qs || '');
  restoring = true;
  try {
    if (dirty && cur) await save(true);
    const want = folders.find(f => f.name === u.get('folder'));
    folder = want ? want.name : folders[0].name;
    status_ = STATUSES.some(s => s[0] === u.get('status')) ? u.get('status') : 'all';
    per = +u.get('per') || per;
    page = +u.get('page') || 1;
    query = u.get('q') || '';
    wasF = WAS.some(w => w[0] === u.get('was')) ? u.get('was') : '';
    $('#search').value = query;
    $('#per').value = String(per);
    selected.clear();
    renderFolders();
    await loadFrames();
    if (view === 'annotate' && frames.length) {
      const f = frames.find(x => x.name === u.get('frame')) || frames[0];
      await openFrame(f.name, true);
    } else {
      setView('gallery');
    }
  } finally { restoring = false; }
}
window.addEventListener('popstate', () => { if (folders.length) restoreFromUrl(); });

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

/* ---------------------------------------------------------------- browse */
async function openBrowse(path) {
  let r;
  try {
    r = await api('/api/browse?path=' + encodeURIComponent(path || ''));
  } catch (e) {
    // Browsing is bounded by the base so a stray URL cannot walk the disk. A path chosen from
    // the desktop dialog, or typed in, is deliberate - move the base to it and try again.
    const w = await post('/api/base', {path});
    base = w.base;
    r = await api('/api/browse?path=' + encodeURIComponent(path));
  }
  browsePath = r.path; browseSelf = r.self;
  setView('browse');

  const parts = r.path.split('/').filter(Boolean);
  let acc = '';
  const crumbs = [`<a data-p="/">/</a>`].concat(parts.map(p => {
    acc += '/' + p;
    return `<a data-p="${esc(acc)}">${esc(p)}</a>`;
  }));
  $('#crumbs').innerHTML = crumbs.join('<i>›</i>');
  $('#crumbs').querySelectorAll('a').forEach(a =>
    a.onclick = () => openBrowse(a.dataset.p));

  const up = r.parent ? `<div class="dir up" data-p="${esc(r.parent)}">
      <div class="dn">↑ up one level</div></div>` : '';
  $('#dirlist').innerHTML = up + (r.dirs.length ? r.dirs.map(d => {
    if (d.role)                                   // part of this folder's layout, not a target
      return `<div class="dir part"><div class="dn">${esc(d.name)}</div>
                <div class="dt"><span class="tag dimtag">${esc(d.role)}</span></div></div>`;
    const badge = d.layout
      ? `<span class="tag ok">${d.images} images</span>
         <span class="tag">${d.labelled} labelled</span>
         <span class="tag dimtag">${esc(d.layout)}</span>`
      : (d.deep_images ? `<span class="tag dimtag">${d.deep_images} images deeper in</span>`
                       : `<span class="tag dimtag">no images</span>`);
    return `<div class="dir${d.layout ? ' has' : ''}" data-p="${esc(d.path)}">
              <div class="dn">${esc(d.name)}</div><div class="dt">${badge}</div></div>`;
  }).join('') : '<div class="empty">Nothing in here.</div>');

  $('#dirlist').querySelectorAll('.dir[data-p]').forEach(el =>
    el.onclick = () => openBrowse(el.dataset.p));

  const s = r.self;
  $('#selfinfo').textContent = s.layout
    ? `${r.path} — ${s.images} images, ${s.labelled} already labelled, ${s.layout} layout`
    : `${r.path} — no images directly here, but subfolders may hold them`;
  $('#usehere').textContent = s.layout ? 'Use this folder' : 'Use this folder anyway';
}

$('#usehere').onclick = async () => {
  const r = await post('/api/root', {path: browsePath});
  root = r.root; folders = r.folders;
  $('#rootpath').textContent = root;
  await loadClasses();
  if (!folders.length) {
    toast('No images found under that folder', 'err');
    return;
  }
  await selectFolder(folders[0].name);
};

$('#pick').onclick = () => openBrowse(root || base);

$('#osdialog').onclick = async () => {
  $('#osdialog').disabled = true;
  $('#osdialog').textContent = 'Dialog open…';
  try {
    const r = await post('/api/pick', {});
    if (r.path) await openBrowse(r.path);
  } catch (e) {
    toast(e.message, 'err');
  } finally {
    $('#osdialog').disabled = false;
    $('#osdialog').textContent = 'Browse…';
  }
};

$('#pathbox').addEventListener('keydown', async e => {
  if (e.key !== 'Enter') return;
  const p = $('#pathbox').value.trim();
  if (!p) return;
  try { await openBrowse(p); $('#pathbox').value = ''; }
  catch (err) { toast(err.message, 'err'); }
});

/* ---------------------------------------------------------------- classes */
/* The class new boxes are drawn in is remembered per folder. It used to start at class 0 on
 * every load, so after a reload a run of birds was quietly saved as "drone" until someone
 * noticed. */
const classKey = () => 'dl-class:' + (root || '');
function savedClass() {
  try { const v = localStorage.getItem(classKey()); return v === null ? null : +v; } catch { return null; }
}
async function loadClasses() {
  const r = await api('/api/classes');
  classes = r.classes;
  const kept = savedClass();
  if (kept !== null && kept >= 0 && kept < classes.length) activeClass = kept;
  if (activeClass >= classes.length) activeClass = 0;
  classHi = activeClass;
  renderClassBar();
  renderClassList();
}

/* ---- recent classes: the last few picked, per folder, shown first wherever a class is chosen */
const recentKey = () => 'dl-recent:' + (root || '');
function recentClasses() {
  try { return (JSON.parse(localStorage.getItem(recentKey()) || '[]')).filter(i => i < classes.length); }
  catch { return []; }
}
function pushRecent(id) {
  const r = [id, ...recentClasses().filter(x => x !== id)].slice(0, 5);
  try { localStorage.setItem(recentKey(), JSON.stringify(r)); } catch {}
}

/* The classes to show for a search, as [{section, ids}]. With a handful of classes there is one
 * plain list in class order; with many, recent ones come first so the usual pick is near the
 * top, and typing narrows the whole list by name. */
const MANY = 5;                                  // up to 5 classes: a plain list; from 6, search + recent
function classSections(q) {
  q = (q || '').trim().toLowerCase();
  if (q) {
    const hit = classes.filter(c => c.name.toLowerCase().includes(q))
      .sort((a, b) => a.name.toLowerCase().indexOf(q) - b.name.toLowerCase().indexOf(q) || a.id - b.id);
    return [{section: `${hit.length} match${hit.length === 1 ? '' : 'es'}`, ids: hit.map(c => c.id)}];
  }
  if (classes.length <= MANY) return [{section: '', ids: classes.map(c => c.id)}];
  const rec = recentClasses();
  return [...(rec.length ? [{section: 'Recent', ids: rec}] : []),
          {section: `All ${classes.length} classes`, ids: classes.map(c => c.id)}];
}
const keyHint = id => id < 8 ? String(id + 2) : '';

/* Header: one class picker beside "Classes", however many classes there are. It shows the class
 * new boxes get, and opens a searchable list (also on the C key while annotating). */
function renderClassBar() {
  if (!classes.length) {
    $('#classbar').innerHTML =
      `<button class="chip need" id="firstclass">No classes yet - create one</button>`;
    $('#firstclass').onclick = () => {
      setView(document.body.dataset.view);
      $('#classfilter') && $('#classfilter').focus();
      $('#editclasses').click();
    };
    renderDrawAs();
    return;
  }
  const c = classes[activeClass] || classes[0];
  if (!$('#cp-btn')) {
    $('#classbar').innerHTML = `
      <div class="classpick">
        <button id="cp-btn" class="cpbtn" aria-haspopup="listbox" aria-expanded="false"
                title="Class for new boxes - click or press C to change"><i></i><span></span>
          <svg class="caret" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg></button>
        <div id="cp-pop" class="cppop" hidden>
          <input id="cp-q" class="inp" placeholder="Search classes…" autocomplete="off" spellcheck="false">
          <div id="cp-list" role="listbox" aria-label="Classes" tabindex="0"></div>
          <div class="cpfoot"><span>↑↓ move · Enter pick · Esc close</span>
            <button id="cp-manage" type="button">Manage…</button></div>
        </div>
      </div>`;
    $('#cp-btn').onclick = () => $('#cp-pop').hidden ? openClassPick() : closeClassPick();
    $('#cp-q').oninput = () => { cpHi = 0; renderClassPick(); };
    $('#cp-q').onkeydown = classPickKeys;
    $('#cp-list').onkeydown = classPickKeys;
    $('#cp-manage').onclick = () => { closeClassPick(); $('#editclasses').click(); };
    document.addEventListener('mousedown', e => {
      if (!e.target.closest('.classpick')) closeClassPick();
    }, true);
  }
  const btn = $('#cp-btn');
  btn.style.setProperty('--c', c.color);
  btn.querySelector('span').textContent = c.name;
  if (!$('#cp-pop').hidden) renderClassPick();
  renderDrawAs();
}

let cpHi = 0;                                   // highlighted row in the open picker
function pickIds() { return [...new Set(classSections($('#cp-q').value).flatMap(s => s.ids))]; }
function renderClassPick() {
  const ids = pickIds();
  cpHi = Math.max(0, Math.min(cpHi, ids.length - 1));
  let n = 0;
  $('#cp-list').innerHTML = classSections($('#cp-q').value).map(sec =>
    (sec.section ? `<div class="cpsec">${esc(sec.section)}</div>` : '') +
    sec.ids.map(id => {
      const c = classes[id], k = keyHint(id), mine = n++;
      return `<div class="cprow${id === activeClass ? ' cur' : ''}${ids[cpHi] === id && mine === firstIndex(id) ? ' hi' : ''}"
                   role="option" aria-selected="${id === activeClass}" data-c="${id}">
                <i style="--c:${c.color}"></i><span>${esc(c.name)}</span>${k ? `<kbd>${k}</kbd>` : ''}</div>`;
    }).join('')).join('') || '<div class="cpnone">No class matches</div>';
  $('#cp-list').querySelectorAll('.cprow').forEach(r => r.onclick = () => {
    setActiveClass(+r.dataset.c); closeClassPick();
  });
  const hi = $('#cp-list .cprow.hi');
  if (hi) hi.scrollIntoView({block: 'nearest'});
}
// a class can appear twice (Recent and All); only its first row is highlighted
function firstIndex(id) {
  let n = 0;
  for (const sec of classSections($('#cp-q').value)) for (const x of sec.ids) { if (x === id) return n; n++; }
  return -1;
}
function openClassPick() {
  if (!classes.length || !$('#cp-pop')) return;
  closeBoxMenu();
  $('#cp-pop').hidden = false;
  $('#cp-btn').setAttribute('aria-expanded', 'true');
  $('#cp-q').value = '';
  $('#cp-q').classList.toggle('hidden', classes.length <= MANY);
  cpHi = Math.max(0, pickIds().indexOf(activeClass));
  renderClassPick();
  (classes.length > MANY ? $('#cp-q') : $('#cp-list')).focus();
}
function closeClassPick() {
  const pop = $('#cp-pop');
  if (!pop || pop.hidden) return;
  // focus left inside the hidden list would swallow every shortcut after it: give it back
  const hadFocus = pop.contains(document.activeElement);
  pop.hidden = true;
  $('#cp-btn').setAttribute('aria-expanded', 'false');
  if (hadFocus) $('#cp-btn').focus({preventScroll: true});
}
function classPickKeys(e) {
  e.stopPropagation();                           // keys belong to the picker while it is open
  const ids = pickIds();
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault(); e.stopPropagation();
    cpHi = (cpHi + (e.key === 'ArrowDown' ? 1 : -1) + ids.length) % Math.max(1, ids.length);
    renderClassPick();
  } else if (e.key === 'Enter') {
    e.preventDefault(); e.stopPropagation();
    if (ids.length) { setActiveClass(ids[cpHi]); closeClassPick(); }
  } else if (e.key === 'Escape') {
    e.preventDefault(); e.stopPropagation(); closeClassPick();
  }
}

// Always on screen, on the canvas toolbar: the class the next box will get.
function renderDrawAs() {
  const el = $('#drawas');
  if (!el) return;
  const c = classes[activeClass];
  el.classList.toggle('hidden', !c);
  if (!c) return;
  el.style.setProperty('--c', c.color);
  el.querySelector('span').textContent = c.name;
}

// `how` is 'key' when a shortcut did it: a change you did not look at is announced.
function setActiveClass(id, how) {
  const was = activeClass;
  activeClass = id; classHi = id;
  try { localStorage.setItem(classKey(), String(id)); } catch {}
  pushRecent(id);
  renderClassBar(); renderClassList();
  let boxChanged = false;
  if (sel >= 0 && boxes[sel][4] !== id) {
    remember(); boxes[sel][4] = id; dirty = true; renderBoxes(); render();
    boxChanged = true;
  }
  if (how === 'key' && (was !== id || boxChanged))
    toast(boxChanged ? `Selected box is now ${nameOf(id)} — new boxes too` : `New boxes: ${nameOf(id)}`);
  if (was !== id) {
    const el = $('#drawas');
    if (el) { el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); }
  }
}

function renderClassList() {
  const q = ($('#classfilter').value || '').toLowerCase();
  const hits = classes.filter(c => c.name.toLowerCase().includes(q));
  const mk = c => `<div class="cl${c.id === classHi ? ' hi' : ''}${c.id === activeClass ? ' act' : ''}"
       data-c="${c.id}" style="--c:${c.color}" title="class ${c.id}${c.id < 8 ? ` · key ${c.id + 2}` : ''}"><i></i>${esc(c.name)}</div>`;
  let html = hits.map(mk).join('');
  if (q && !classes.some(c => c.name.toLowerCase() === q))
    html += `<div class="cl create" data-new="1">+ create class “${esc($('#classfilter').value)}”</div>`;
  if (!classes.length && !q)
    html = '<div class="empty">No classes yet. Type a name above and press Enter.</div>';
  $('#classlist').innerHTML = html || '<div class="empty">No class matches.</div>';
  $('#classlist').querySelectorAll('.cl').forEach(el => el.onclick = () => {
    if (el.dataset.new) addClass($('#classfilter').value);
    else setActiveClass(+el.dataset.c);
  });
}

async function addClass(name) {
  name = (name || '').trim();
  if (!name) return;
  if (classes.some(c => c.name.toLowerCase() === name.toLowerCase())) {
    toast(`“${name}” already exists`, 'err');
    return;
  }
  // The endpoint takes {id, name}: the id is the index the class HELD BEFORE, and null means
  // new. Sending bare strings, as this used to, is rejected outright - which is why clicking
  // "create class" appeared to do nothing at all.
  const payload = classes.map(c => ({id: c.id, name: c.name})).concat([{id: null, name}]);
  try {
    const r = await post('/api/classes', {classes: payload, apply_to_labels: false});
    classes = r.classes;
    $('#classfilter').value = '';
    setActiveClass(classes.length - 1);
    renderClassBar();
    toast(`Class “${name}” created as index ${classes.length - 1}`, 'ok');
  } catch (e) { toast('Could not create the class: ' + e.message, 'err'); }
}

$('#classfilter').addEventListener('input', renderClassList);
$('#classfilter').addEventListener('keydown', e => {
  const hits = classes.filter(c =>
    c.name.toLowerCase().includes(($('#classfilter').value || '').toLowerCase()));
  if (e.key === 'Enter') {
    e.preventDefault();
    if (hits.length) setActiveClass(hits.some(c => c.id === classHi) ? classHi : hits[0].id);
    else addClass($('#classfilter').value);
  } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (!hits.length) return;
    let i = hits.findIndex(c => c.id === classHi);
    i = (i + (e.key === 'ArrowDown' ? 1 : -1) + hits.length) % hits.length;
    classHi = hits[i].id;
    renderClassList();
  }
  e.stopPropagation();
});

/* ---------------------------------------------------------------- class manager
 * Each row remembers the index its class HELD WHEN THE DIALOG OPENED. That is the whole trick:
 * with it, a renamed row and a deleted-then-added row are distinguishable, so the server can
 * rewrite labels exactly instead of guessing from names. */
let draft = [], usage = {};

function renderDraft() {
  $('#clsrows').innerHTML = draft.map((c, i) => `
    <div class="clsrow" data-i="${i}">
      <i style="--c:${PALETTE[i % PALETTE.length]}"></i>
      <b class="ix">${i}</b>
      <input class="inp nm" value="${esc(c.name)}" placeholder="class name">
      <span class="use">${c.id === null ? 'new'
        : (usage[c.id] ? usage[c.id] + ' boxes' : 'unused')}</span>
      <button class="mv" data-d="-1" title="Move up" ${i === 0 ? 'disabled' : ''}>↑</button>
      <button class="mv" data-d="1" title="Move down" ${i === draft.length - 1 ? 'disabled' : ''}>↓</button>
      <button class="x" title="Delete this class">×</button>
    </div>`).join('');

  $('#clsrows').querySelectorAll('.clsrow').forEach(row => {
    const i = +row.dataset.i;
    row.querySelector('.nm').oninput = e => { draft[i].name = e.target.value; previewImpact(); };
    row.querySelectorAll('.mv').forEach(b => b.onclick = () => {
      const j = i + (+b.dataset.d);
      if (j < 0 || j >= draft.length) return;
      [draft[i], draft[j]] = [draft[j], draft[i]];
      renderDraft(); previewImpact();
    });
    row.querySelector('.x').onclick = () => {
      if (draft.length === 1) { toast('At least one class is needed', 'err'); return; }
      draft.splice(i, 1); renderDraft(); previewImpact();
    };
  });
  previewImpact();
}

function previewImpact() {
  const moved = [], dropped = [];
  draft.forEach((c, i) => { if (c.id !== null && c.id !== i) moved.push(c.name); });
  classes.forEach(c => { if (!draft.some(d => d.id === c.id)) dropped.push(c); });
  const movedBoxes = draft.reduce((n, c, i) =>
    n + (c.id !== null && c.id !== i ? (usage[c.id] || 0) : 0), 0);
  const lostBoxes = dropped.reduce((n, c) => n + (usage[c.id] || 0), 0);

  const bits = [];
  if (movedBoxes) bits.push(`${movedBoxes} box(es) renumbered`);
  if (lostBoxes) bits.push(`${lostBoxes} box(es) deleted`);
  $('#clsimpact').textContent = bits.length ? '— ' + bits.join(', ') : '— nothing to rewrite';

  const w = $('#clswarn');
  if (!moved.length && !dropped.length) { w.classList.add('hidden'); return; }
  w.classList.remove('hidden');
  const apply = $('#clsapply').checked;
  w.innerHTML = apply
    ? `Labels will be rewritten: ${bits.join(', ') || 'no boxes affected'}.` +
      (lostBoxes ? ` Boxes of a deleted class are removed — that cannot be undone.` : '')
    : `Labels will be left alone, so ${movedBoxes + lostBoxes} box(es) will point at the wrong
       class. Tick the box above unless you know you want that.`;
}

$('#clsadd').onclick = () => { draft.push({id: null, name: ''}); renderDraft(); };
$('#clsapply').onchange = previewImpact;
$('#clscancel').onclick = () => $('#clsveil').classList.remove('on');

$('#editclasses').onclick = async () => {
  if (!root) { toast('Pick a folder first', 'err'); return; }
  draft = classes.length ? classes.map(c => ({id: c.id, name: c.name}))
                         : [{id: null, name: ''}];
  usage = {};
  $('#clsveil').classList.add('on');
  renderDraft();
  try {
    const u = await api('/api/class-usage');
    usage = u.usage; renderDraft();
  } catch (e) { /* counts are a nicety; the dialog works without them */ }
};

$('#clssave').onclick = async () => {
  const payload = draft.map(c => ({id: c.id, name: c.name}))
                       .filter(c => c.name.trim());
  if (!payload.length) { toast('At least one class is needed', 'err'); return; }
  try {
    const r = await post('/api/classes',
                         {classes: payload, apply_to_labels: $('#clsapply').checked});
    classes = r.classes;
    if (activeClass >= classes.length) activeClass = 0;
    renderClassBar(); renderClassList();
    $('#clsveil').classList.remove('on');
    const a = r.applied || {};
    if (a.files) toast(`Classes saved — ${a.moved} box(es) renumbered, ` +
                       `${a.dropped} removed, across ${a.files} file(s)`, 'ok');
    else toast(r.warning || 'Classes saved', r.warning ? 'err' : 'ok');
    if (folder) { await refreshFolderCounts(); await loadFrames(); }
    if (cur && document.body.dataset.view === 'annotate') openFrame(cur, true);
  } catch (e) { toast(e.message, 'err'); }
};

/* ---------------------------------------------------------------- folders */
/* The folder pane is a tree, at most three levels deep:
 *
 *     dataset  ▸  train  ▸  bird
 *
 * The first two levels are sections that open and close; the third is the folder itself. A path
 * that goes deeper is named in full at the third level ("c/d"), so the pane never indents past
 * readability. "images" is dropped from names - it is where frames sit, not a level of meaning.
 * Every section can be opened and closed on its own; that choice is remembered per browser. */
const MAX_DEPTH = 3;
const openGroups = (() => {                      // {section key: true|false}, only what was toggled
  try { return JSON.parse(localStorage.getItem('dl-groups') || '{}'); } catch { return {}; }
})();
const saveGroups = () => { try { localStorage.setItem('dl-groups', JSON.stringify(openGroups)); } catch {} };
const shownParts = name => name.split('/').filter(p => p !== 'images');
// "train/images/bird" reads as "train/bird"
const leafName = rel => shownParts(rel).join('/');
const inSection = (key, name) => name !== '.' && (shownParts(name).join('/') + '/').startsWith(key + '/');
// a section starts open only if it holds the folder being worked in; a click remembers otherwise
const isOpen = key => key in openGroups ? openGroups[key] : !!folder && inSection(key, folder);

function folderTree() {
  const root = {key: '', kids: [], items: []};
  for (const f of folders) {
    if (f.name === '.') { root.items.push({f, label: '(this folder)'}); continue; }
    const parts = shownParts(f.name);
    const secs = parts.slice(0, Math.min(parts.length - 1, MAX_DEPTH - 1));
    let node = root;
    for (const p of secs) {
      const key = node.key ? `${node.key}/${p}` : p;
      let k = node.kids.find(x => x.key === key);
      if (!k) { k = {key, name: p, kids: [], items: []}; node.kids.push(k); }
      node = k;
    }
    node.items.push({f, label: parts.slice(secs.length).join('/')});
  }
  // A folder that holds frames itself AND has folders under it goes inside its own section,
  // as "(this folder)", instead of appearing twice side by side.
  (function fold(n) {
    for (const k of n.kids) {
      const i = n.items.findIndex(it => it.label === k.name);
      if (i >= 0) k.items.unshift({...n.items.splice(i, 1)[0], label: '(this folder)'});
      fold(k);
    }
  })(root);
  return root;
}

const allIn = n => n.items.map(it => it.f).concat(...n.kids.map(allIn));
const allKeys = n => n.kids.flatMap(k => [k.key, ...allKeys(k)]);

const countsLine = fs => {
  const sum = k => fs.reduce((a, f) => a + (f[k] || 0), 0);
  return `${sum('count')} images · ${sum('annotated')} annotated · ${sum('unassigned')} unassigned${
    sum('deleted') ? ` · <span class="fdel">${sum('deleted')} deleted</span>` : ''}`;
};

function folderRow(f, label, depth) {
  return `<div class="fo${f.name === folder ? ' on' : ''}" data-n="${esc(f.name)}" style="--d:${depth}"
            title="${esc(f.name)} — ${esc(f.layout || '')}">
            <span class="fon">${esc(label)}</span>
            <small>${countsLine([f])}</small>
          </div>`;
}

function sectionHtml(n, depth) {
  const fs = allIn(n);
  const open = isOpen(n.key);
  const here = fs.some(f => f.name === folder);
  // closed, the section still names the open folder, so "where am I" never needs a click
  const rest = here && !open
    ? (folder === n.key ? '(this folder)' : shownParts(folder).slice(n.key.split('/').length).join('/') || '(this folder)')
    : '';
  const nFolders = fs.length;
  return `<div class="fg${open ? ' open' : ''}${here ? ' here' : ''}" data-g="${esc(n.key)}" style="--d:${depth}">
      <div class="fgh" title="${esc(n.key)}${here ? ' — you are working in here' : ''}"
           aria-expanded="${open}" role="button" tabindex="0">
        <svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>
        <span class="fgn">${esc(n.name)}</span>
        <span class="fgc" title="${nFolders} image folder${nFolders === 1 ? '' : 's'}">${nFolders}</span>
        ${rest ? `<span class="fgcur" title="The folder open in the gallery">${esc(rest)}</span>` : ''}
        <small>${countsLine(fs)}</small>
      </div>
      <div class="fgb">${nodeHtml(n, depth + 1)}</div>
    </div>`;
}

function nodeHtml(n, depth) {
  return n.items.map(it => folderRow(it.f, it.label, depth)).join('') +
         n.kids.map(k => sectionHtml(k, depth)).join('');
}

function renderFolders() {
  const tree = folderTree();
  const keys = allKeys(tree);
  const nTop = tree.kids.length;
  $('#folder-n').textContent = folders.length ? (nTop ? `${nTop} · ${folders.length}` : folders.length) : '';
  $('#folder-n').title = nTop ? `${nTop} dataset(s), ${folders.length} image folders` : '';
  $('#fo-all').classList.toggle('hidden', !keys.length);
  $('#fo-all').textContent = keys.some(k => !isOpen(k)) ? 'expand all' : 'collapse all';
  $('#folders').innerHTML = nodeHtml(tree, 0);
  $('#folders').querySelectorAll('.fo').forEach(d =>
    d.onclick = () => selectFolder(d.dataset.n));
  $('#folders').querySelectorAll('.fgh').forEach(h => {
    h.onclick = () => {
      const g = h.parentElement.dataset.g;
      openGroups[g] = !isOpen(g); saveGroups(); renderFolders();
    };
    h.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); h.click(); } };
  });
  const on = $('#folders .fo.on');
  if (on) on.scrollIntoView({block: 'nearest'});
}
$('#fo-all').onclick = () => {
  const keys = allKeys(folderTree());
  const open = keys.some(k => !isOpen(k));
  keys.forEach(k => { openGroups[k] = open; });
  saveGroups(); renderFolders();
};

async function selectFolder(name) {
  folder = name;
  page = 1; selected.clear();
  renderFolders();
  await loadFrames();
  setView('gallery');
}

/* ---------------------------------------------------------------- gallery */
function framesUrl(p) {
  return `/api/frames?folder=${encodeURIComponent(folder)}&status=${status_}` +
         `&page=${p}&per=${per}&q=${encodeURIComponent(query)}` +
         (status_ === 'deleted' && wasF ? `&was=${wasF}` : '');
}

async function loadFrames() {
  const r = await api(framesUrl(page));
  frames = r.items; counts = r.counts; pages = r.pages; page = r.page;
  total = r.total; pagePer = r.per; wasCounts = r.was_counts || {};
  syncUrl(false);
  renderTabs(); renderTrashBar(); renderGrid(); renderPager(); renderFramePager(); renderBatch();
  // the next page's pictures are fetched while this one is being looked at, so paging forward
  // lands on images the browser already has
  clearTimeout(window._pf); window._pf = setTimeout(prefetchNextPage, 900);
}

function renderTabs() {
  $('#tabs').innerHTML = STATUSES.map(([k, label]) =>
    `<button class="tab t-${k}${k === status_ ? ' on' : ''}" data-s="${k}">${label}
       <b>${counts[k] ?? 0}</b></button>`).join('');
  $('#tabs').querySelectorAll('.tab').forEach(b => b.onclick = () => {
    status_ = b.dataset.s; page = 1; wasF = ''; selected.clear(); loadFrames();
  });
}

// The Deleted tab's own toolbar: filter by what the frame was, and act on the whole tab.
function renderTrashBar() {
  const on = status_ === 'deleted';
  $('#trashbar').classList.toggle('hidden', !on);
  if (!on) return;
  const n = counts.deleted ?? 0;
  $('#wasf').innerHTML = WAS.map(([k, label]) =>
    `<option value="${k}"${k === wasF ? ' selected' : ''}>was ${label}${
      k ? ` (${wasCounts[k] ?? 0})` : ` (${n})`}</option>`).join('');
  const filtered = wasF || query;
  $('#t-restore-all').textContent = filtered ? `Restore these ${total}` : `Restore all ${total}`;
  $('#t-empty').textContent = filtered ? `Delete these ${total} forever` : `Delete all ${total} forever`;
  $('#t-restore-all').disabled = $('#t-empty').disabled = !total;
}
$('#wasf').onchange = e => { wasF = e.target.value; page = 1; selected.clear(); loadFrames(); };
$('#t-restore-all').onclick = async () => {
  const r = await post('/api/untrash', {folder, all_deleted: true, was: wasF, q: query});
  toast(`${r.restored} frame(s) restored`, 'ok');
  selected.clear();
  await refreshFolderCounts(); await loadFrames();
};
$('#t-empty').onclick = () => purge(null, total);

// Delete forever - the only step that touches the disk, so the only one that asks first.
async function purge(names, n) {
  const ok = await confirmDialog({
    title: `Delete ${n} frame(s) forever?`,
    body: `<p>This removes the image, its label and any annotated/crops copies from disk.</p>
           ${names ? `<ul>${names.slice(0, 8).map(x => `<li>${esc(x)}</li>`).join('')}</ul>
           ${names.length > 8 ? `<p class="dim">…and ${names.length - 8} more.</p>` : ''}` : ''}
           <div class="warn">This cannot be undone.</div>`,
    okLabel: 'Delete forever'});
  if (!ok) return;
  const r = await post('/api/delete', names ? {folder, names}
                                            : {folder, all_deleted: true, was: wasF, q: query});
  selected.clear();
  toast(`${r.removed} frame(s) deleted forever`, 'ok');
  await refreshFolderCounts(); await loadFrames();
}

// "Today, 14:00 – 15:00": the heading over one hour of deletions.
function hourLabel(t) {
  const d = new Date(t * 1000); d.setMinutes(0, 0, 0);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const day = new Date(d); day.setHours(0, 0, 0, 0);
  const diff = Math.round((today - day) / 864e5);
  const name = diff === 0 ? 'Today' : diff === 1 ? 'Yesterday'
    : d.toLocaleDateString(undefined, {weekday: 'short', day: 'numeric', month: 'short',
        ...(d.getFullYear() !== today.getFullYear() ? {year: 'numeric'} : {})});
  const hh = h => String(h % 24).padStart(2, '0') + ':00';
  return `${name}, ${hh(d.getHours())} – ${hh(d.getHours() + 1)}`;
}
// local hour, not UTC: a +5:30 timezone must still group on the hour you see on the clock
const hourKey = t => { const d = new Date(t * 1000); d.setMinutes(0, 0, 0); return d.getTime(); };
// "14:29" today, "Yesterday 14:29", "11 Nov 14:29" - 24-hour, like the hour headings
function clock(t) {
  const d = new Date(t * 1000);
  const hm = d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', hour12: false});
  const day = new Date(d); day.setHours(0, 0, 0, 0);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const diff = Math.round((today - day) / 864e5);
  if (diff === 0) return hm;
  if (diff === 1) return `Yesterday ${hm}`;
  return `${d.toLocaleDateString(undefined, {day: 'numeric', month: 'short'})} ${hm}`;
}

function thumbUrl(name, w) {
  return `/api/thumb?folder=${encodeURIComponent(folder)}&name=${encodeURIComponent(name)}` +
         `&w=${w || (gridMode === 'list' ? 200 : 320)}`;
}

function classChips(ids) {
  return ids.map(i => {
    const c = classes[i];
    return `<span class="mini" style="--c:${c ? c.color : '#888'}">${
      c ? esc(c.name) : 'class ' + i}</span>`;
  }).join('');
}

function renderGrid() {
  $('#grid').className = gridMode;
  if (!frames.length) {
    $('#grid').innerHTML = `<div class="empty big">${status_ === 'deleted'
      ? 'Nothing deleted. Press <b>Del</b> on a frame while annotating to send it here.'
      : 'Nothing in this tab.'}</div>`;
    return;
  }
  const card = f => {
    const gone = f.status === 'deleted';
    return `
    <div class="card${gone ? ' deleted' : ''}${selected.has(f.name) ? ' sel' : ''}" data-n="${esc(f.name)}">
      <div class="cbox"><input type="checkbox" ${selected.has(f.name) ? 'checked' : ''}></div>
      <div class="ci"><img loading="lazy" decoding="async" src="${thumbUrl(f.name)}" alt=""></div>
      <div class="cm">
        <span class="dot ${f.status}" title="${gone ? 'deleted - was ' + f.was : f.status}"></span>
        <span class="cn" title="${esc(f.name)}">${esc(f.name)}</span>
        <span class="cb">${f.boxes || ''}</span>
        <span class="lst ${f.status}">${STATUS_LABEL[f.status] || f.status}</span>
        <span class="lbx">${f.boxes ? `${f.boxes} box${f.boxes === 1 ? '' : 'es'}` : '—'}</span>
      </div>
      <div class="cc">${gone
        ? `<span class="mini gone" title="Deleted at ${clock(f.deleted_at)}">was ${f.was} · ${clock(f.deleted_at)}</span>` : ''}${
        classChips(f.classes)}${
        f.note && !gone ? `<span class="mini warn">${esc(f.note.slice(0, 40))}</span>` : ''}</div>
    </div>`;
  };

  if (status_ === 'deleted') {
    // newest first (the server sorts), one section per hour of deletions
    const groups = [];
    for (const f of frames) {
      const k = hourKey(f.deleted_at);
      if (!groups.length || groups[groups.length - 1].k !== k) groups.push({k, t: f.deleted_at, items: []});
      groups[groups.length - 1].items.push(f);
    }
    $('#grid').innerHTML = groups.map(g => {
      const all = g.items.every(f => selected.has(f.name));
      return `<div class="ghead" data-k="${g.k}">
          <label class="selall"><input type="checkbox"${all ? ' checked' : ''}>
            <b>${hourLabel(g.t)}</b></label>
          <span class="dim">${g.items.length} frame${g.items.length === 1 ? '' : 's'}</span>
          <span class="sp"></span>
          <button class="btn sm" data-act="restore">Restore</button>
          <button class="btn sm danger" data-act="purge">Delete forever</button>
        </div>` + g.items.map(card).join('');
    }).join('');
    $('#grid').querySelectorAll('.ghead').forEach(h => {
      const names = groups.find(g => String(g.k) === h.dataset.k).items.map(f => f.name);
      h.querySelector('input').onchange = e => {
        names.forEach(n => e.target.checked ? selected.add(n) : selected.delete(n));
        renderGrid(); renderBatch();
      };
      h.querySelector('[data-act=restore]').onclick = () => restoreNames(names);
      h.querySelector('[data-act=purge]').onclick = () => purge(names, names.length);
    });
  } else {
    $('#grid').innerHTML = frames.map(card).join('');
  }

  $('#grid').querySelectorAll('.card').forEach(el => {
    const name = el.dataset.n;
    el.querySelector('.cbox').onclick = e => {
      e.stopPropagation();
      selected.has(name) ? selected.delete(name) : selected.add(name);
      el.classList.toggle('sel', selected.has(name));
      el.querySelector('input').checked = selected.has(name);
      if (status_ === 'deleted') renderGrid();     // keeps the hour's checkbox in step
      renderBatch();
    };
    el.onclick = () => openFrame(name);
  });
}

function renderPager() {
  if (pages <= 1) { $('#pager').innerHTML = ''; return; }
  const btn = (p, label, on) =>
    `<button class="pg${on ? ' on' : ''}" data-p="${p}">${label}</button>`;
  const near = [];
  for (let p = Math.max(1, page - 2); p <= Math.min(pages, page + 2); p++)
    near.push(btn(p, p, p === page));
  $('#pager').innerHTML =
    btn(Math.max(1, page - 1), '‹ prev', false) +
    (page > 3 ? btn(1, '1', false) + '<i>…</i>' : '') +
    near.join('') +
    (page < pages - 2 ? '<i>…</i>' + btn(pages, pages, false) : '') +
    btn(Math.min(pages, page + 1), 'next ›', false) +
    `<label class="goto">go to <input type="number" min="1" max="${pages}" value="${page}"></label>` +
    `<span class="dim">page ${page} of ${pages}</span>`;
  $('#pager').querySelectorAll('.pg').forEach(b => b.onclick = () => {
    page = +b.dataset.p; loadFrames();
  });
  $('#pager .goto input').onkeydown = e => {
    if (e.key !== 'Enter') return;
    const p = clampPage(e.target.value);
    if (p !== page) { page = p; loadFrames(); }
  };
}

const clampPage = v => Math.max(1, Math.min(pages, Math.round(+v) || 1));

// The page controls under the frame list in the annotate view.
function renderFramePager() {
  $('#fpager').classList.toggle('hidden', pages <= 1);
  $('#fp-num').value = page; $('#fp-num').max = pages;
  $('#fp-of').textContent = `/ ${pages}`;
  $('#fp-first').disabled = $('#fp-prev').disabled = page <= 1;
  $('#fp-next').disabled = $('#fp-last').disabled = page >= pages;
}

// Jump to another page while annotating: save the open frame, then open the first frame there.
async function gotoPage(p, openLast) {
  p = clampPage(p);
  if (p === page || stepping) { renderFramePager(); return; }
  stepping = true;
  try {
    if (dirty && cur) await save(true);
    page = p;
    await loadFrames();
    if (frames.length) await openFrame(frames[openLast ? frames.length - 1 : 0].name, true);
  } finally { stepping = false; }
}
$('#fp-first').onclick = () => gotoPage(1);
$('#fp-prev').onclick = () => gotoPage(page - 1);
$('#fp-next').onclick = () => gotoPage(page + 1);
$('#fp-last').onclick = () => gotoPage(pages);
$('#fp-num').onkeydown = e => { if (e.key === 'Enter') { gotoPage(e.target.value); e.target.blur(); } };
$('#fp-num').onblur = e => { e.target.value = page; };

function renderBatch() {
  const n = selected.size;
  $('#batchbar').classList.toggle('hidden', !n);
  $('#nsel').textContent = n;
  const onPage = frames.filter(f => selected.has(f.name)).length;
  const box = $('#selall');
  box.checked = frames.length > 0 && onPage === frames.length;
  box.indeterminate = onPage > 0 && onPage < frames.length;
  $('#selall-label').textContent =
    onPage && onPage === frames.length ? 'Clear page' : `Select page (${frames.length})`;
  // In the Deleted tab the only things to do are bring back or remove for good.
  const inTrash = status_ === 'deleted';
  ['#b-null', '#b-reject', '#b-restore', '#b-clear', '#b-del'].forEach(s =>
    $(s).classList.toggle('hidden', inTrash));
  $('#b-untrash').classList.toggle('hidden', !(inTrash || status_ === 'all'));
  $('#b-purge').classList.toggle('hidden', !inTrash);
}

$('#search').addEventListener('input', () => {
  query = $('#search').value; page = 1;
  clearTimeout(window._q); window._q = setTimeout(loadFrames, 220);
});
$('#per').onchange = () => { per = +$('#per').value; page = 1; loadFrames(); };
// One checkbox for the page, with the three states a page can actually be in: none, some,
// all. A button that only ever adds cannot express "clear the page again", which is the thing
// you want straight after selecting it by mistake.
$('#selall').onchange = e => {
  if (e.target.checked) frames.forEach(f => selected.add(f.name));
  else frames.forEach(f => selected.delete(f.name));
  renderGrid(); renderBatch();
};
$('#b-none').onclick = () => { selected.clear(); renderGrid(); renderBatch(); };

function setMode(m) {
  gridMode = m;
  localStorage.setItem('dl-view', m);
  $('#grid').className = m;
  $('#v-grid').classList.toggle('on', m === 'grid');
  $('#v-list').classList.toggle('on', m === 'list');
}
$('#v-grid').onclick = () => { setMode('grid'); };
$('#v-list').onclick = () => { setMode('list'); };
$('#annotate-first').onclick = () => frames.length && openFrame(frames[0].name);

async function batch(status, label) {
  const names = Array.from(selected);
  await post('/api/status', {folder, names, status});
  selected.clear();
  toast(`${names.length} frame(s) ${label}`, 'ok');
  await refreshFolderCounts();
  await loadFrames();
}
$('#b-null').onclick = () => batch('null', 'marked null');
$('#b-reject').onclick = () => batch('rejected', 'rejected');
$('#b-restore').onclick = () => batch('unrejected', 'un-rejected');
$('#b-clear').onclick = async () => {
  const ok = await confirmDialog({
    title: `Clear ${selected.size} label(s)?`,
    body: `<p>The label files are deleted, so these frames go back to
           <b>Unassigned</b> — as if nobody had looked at them. The images are untouched.</p>`,
    okLabel: 'Clear labels'});
  if (ok) batch('unassigned', 'cleared');
};
// Delete is a soft delete now - the frames move to the Deleted tab - so it does not ask.
$('#b-del').onclick = async () => {
  const names = Array.from(selected);
  let r;
  try { r = await post('/api/trash', {folder, names}); }
  catch (e) { toast('Delete failed: ' + e.message, 'err'); return; }
  selected.clear();
  toast(`${r.moved} frame(s) moved to Deleted`, 'ok');
  await refreshFolderCounts();
  await loadFrames();
};
async function restoreNames(names) {
  let r;
  try { r = await post('/api/untrash', {folder, names}); }
  catch (e) { toast('Restore failed: ' + e.message, 'err'); return; }
  names.forEach(n => selected.delete(n));
  toast(`${r.restored} frame(s) restored`, 'ok');
  await refreshFolderCounts();
  await loadFrames();
}
$('#b-untrash').onclick = () => restoreNames(Array.from(selected));
$('#b-purge').onclick = () => { const n = Array.from(selected); purge(n, n.length); };

async function refreshFolderCounts() {
  folders = await api('/api/folders');
  renderFolders();
}

/* ---------------------------------------------------------------- annotate */
function isCheck(f) { return /CHECK/i.test(f.note || ''); }

/* Each row carries the colour of the class(es) labelled in it - a bird frame reads green, a
 * drone frame yellow - so a frame saved under the wrong class stands out in the list without
 * opening it. Several classes: the stripe is split between their colours and every class is
 * named. */
function frameRow(f) {
  const ids = f.status === 'deleted' ? [] : (f.classes || []);
  const cols = ids.map(colorOf);
  const stripe = cols.length > 1
    ? `--stripe:linear-gradient(${cols.map((c, i) =>
        `${c} ${i * 100 / cols.length}% ${(i + 1) * 100 / cols.length}%`).join(',')})` : '';
  const names = ids.map(nameOf);
  const tags = ids.slice(0, 2).map(i =>
    `<span class="ftag" style="--c:${colorOf(i)}">${esc(nameOf(i))}</span>`).join('') +
    (ids.length > 2 ? `<span class="ftag more">+${ids.length - 2}</span>` : '');
  return `<div class="fr${f.name === cur ? ' on' : ''}${f.status === 'deleted' ? ' deleted' : ''}${
           ids.length ? ' tinted' : ''}" data-n="${esc(f.name)}"
           style="${ids.length ? `--c:${cols[0]};` : ''}${stripe}"
           title="${esc(f.name)}${names.length ? ' — ' + names.join(', ') : ''}">
       <span class="dot ${f.status}"></span>
       <span class="fn">${esc(f.name)}</span>
       ${isCheck(f) ? '<span class="chk">!</span>' : ''}
       ${tags}
       <span class="cnt">${f.boxes || ''}</span>
     </div>`;
}

function renderFrameList() {
  $('#fcount').textContent = frames.length ? `${frames.length}` : '';
  $('#frames').innerHTML = frames.map(frameRow).join('');
  $('#frames').querySelectorAll('.fr').forEach(d =>
    d.onclick = () => openFrame(d.dataset.n));
  const on = $('#frames .fr.on');
  if (on) on.scrollIntoView({block: 'nearest'});
}

/* Every request to open a frame gets a number, and only the newest one may touch the screen.
 * Responses can come back out of order - the server answers in parallel - and a slow, older
 * frame used to land last: its boxes then sat under the newer frame's name, and the next save
 * wrote them into the wrong label file. */
let openSeq = 0;
async function openFrame(name, skipSave) {
  const seq = ++openSeq;
  if (dirty && cur && !skipSave) await save(true);
  if (seq !== openSeq) return;                   // a later frame was asked for meanwhile
  closeBoxMenu();
  cur = name;
  setView('annotate');
  renderFrameList();
  const d = await api(`/api/label?folder=${encodeURIComponent(folder)}&name=${encodeURIComponent(name)}`);
  if (seq !== openSeq || cur !== name) return;   // stale answer: drop it
  iw = d.w; ih = d.h;
  boxes = d.boxes.map(b => [b.x0, b.y0, b.x1, b.y1, b.cls]);
  sel = -1; cands = []; candHi = -1; dirty = false;
  past = []; future = []; syncUndo();
  $('#title').textContent = `${name}  ${iw}×${ih}`;
  showRejected(!!d.rejected);
  showDeleted(!!d.deleted);
  const i = frames.findIndex(f => f.name === name);
  // position on this page, then in the whole list: "1 / 60 · 61 / 2000"
  $('#pos').textContent = i >= 0
    ? `${i + 1} / ${frames.length} · ${(page - 1) * pagePer + i + 1} / ${total}` : '';
  $('#pos').title = i >= 0 ? `Page ${page} of ${pages}` : '';
  $('#cands').innerHTML = '<div class="empty">Press <b>A</b> to run the detectors.</div>';
  renderBoxes();
  img = new Image();
  img.onload = () => { if (seq === openSeq) { fitView(); render(); syncZoom(); } };
  img.src = imgUrl(name);
}

function imgUrl(name) {
  return `/api/image?folder=${encodeURIComponent(folder)}&name=${encodeURIComponent(name)}` +
         (enhance ? '&enhance=1' : '') + `&t=${Date.now()}`;
}

let stepping = false;
async function step(delta) {
  if (stepping) return;                          // a held arrow key must not skip pages
  const i = frames.findIndex(f => f.name === cur);
  const j = i + delta;
  if (j >= 0 && j < frames.length) { openFrame(frames[j].name); return; }

  // Off the end of this page: carry on into the next (or previous) one.
  const to = page + (delta > 0 ? 1 : -1);
  if (to < 1 || to > pages) {
    toast(delta > 0 ? 'Last frame' : 'First frame');
    return;
  }
  await gotoPage(to, delta < 0);
}

function stageSize() { const r = $('#stage').getBoundingClientRect(); return [r.width, r.height]; }

function fitView() {
  const [W, H] = stageSize();
  cv.width = W; cv.height = H;
  viewport.s = Math.min(W / iw, H / ih) * 0.96;
  viewport.x = (W - iw * viewport.s) / 2;
  viewport.y = (H - ih * viewport.s) / 2;
}

const toImg = (px, py) => [(px - viewport.x) / viewport.s, (py - viewport.y) / viewport.s];
const toScr = (ix, iy) => [ix * viewport.s + viewport.x, iy * viewport.s + viewport.y];

function norm(b) {
  return [Math.min(b[0], b[2]), Math.min(b[1], b[3]),
          Math.max(b[0], b[2]), Math.max(b[1], b[3]), b[4] ?? activeClass];
}

/* A label is stored as a fraction of the image, so a box reaching past the edge has no way to
 * be written down truthfully: the server clamps it and the file ends up describing something
 * other than what was on screen. Keeping the box inside the frame while it is being edited is
 * what makes the two agree. */
const lim = (v, hi) => Math.min(Math.max(0, v), hi);

function clampBox(b) {
  return [lim(b[0], iw), lim(b[1], ih), lim(b[2], iw), lim(b[3], ih), b[4]];
}

/* Moving keeps the box's size and stops it at the edge, rather than squashing it flat against
 * one, which is what clamping each corner on its own would do. */
function shiftInside(b, dx, dy) {
  const w = b[2] - b[0], h = b[3] - b[1];
  const x = Math.min(Math.max(0, b[0] + dx), Math.max(0, iw - w));
  const y = Math.min(Math.max(0, b[1] + dy), Math.max(0, ih - h));
  return [x, y, x + w, y + h, b[4]];
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

const colorOf = i => (classes[i] && classes[i].color) || '#ff4d4d';
const nameOf = i => (classes[i] && classes[i].name) || ('class ' + i);

/* Outlines sit 2 px OUTSIDE a box so the line never hides the target's own edge pixels - on a
 * 6 px drone that is half of it. At the image border there is no outside, and the line used to
 * land on the grey canvas, as if the box reached past the picture. So the padded rectangle is
 * held inside the image's on-screen bounds: the line stays on the picture, always. */
function imageRectOnScreen() {
  const [L, T] = toScr(0, 0), [R, B] = toScr(iw, ih);
  return {L, T, R, B};
}
function outline(x0, y0, x1, y1, lw, pad = 2) {
  const {L, T, R, B} = imageRectOnScreen(), h = lw / 2;
  const a = Math.max(x0 - pad, L + h), b = Math.max(y0 - pad, T + h);
  const c = Math.min(x1 + pad, R - h), d = Math.min(y1 + pad, B - h);
  ctx.strokeRect(a, b, Math.max(0, c - a), Math.max(0, d - b));
  return {a, b, c, d};
}

function render() {
  const [W, H] = stageSize();
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  ctx.fillStyle = '#eceff3'; ctx.fillRect(0, 0, W, H);
  if (!img.complete || !iw) return;
  ctx.imageSmoothingEnabled = viewport.s < 2;
  if (bright) ctx.filter = `brightness(${100 + bright}%)`;
  ctx.drawImage(img, viewport.x, viewport.y, iw * viewport.s, ih * viewport.s);
  ctx.filter = 'none';

  cands.forEach((c, i) => {
    const [x0, y0] = toScr(c.box[0], c.box[1]), [x1, y1] = toScr(c.box[2], c.box[3]);
    ctx.strokeStyle = i === candHi ? '#e0ae00' : 'rgba(224,174,0,.55)';
    ctx.lineWidth = i === candHi ? 2 : 1;
    ctx.setLineDash(i === candHi ? [] : [3, 3]);
    outline(x0, y0, x1, y1, ctx.lineWidth);
  });
  ctx.setLineDash([]);

  boxes.forEach((b, i) => {
    const [x0, y0] = toScr(b[0], b[1]), [x1, y1] = toScr(b[2], b[3]);
    const col = colorOf(b[4]);
    ctx.strokeStyle = i === sel ? '#1b2230' : col;
    ctx.lineWidth = i === sel ? 3 : 2.5;           // bold enough to read on any background
    const o = outline(x0, y0, x1, y1, ctx.lineWidth);
    ctx.font = 'bold 11px ui-monospace,monospace';
    const tag = `${nameOf(b[4])} ${Math.round(b[2] - b[0])}×${Math.round(b[3] - b[1])}`;
    const tw = ctx.measureText(tag).width + 6;
    // the tag sits above the box; when that would be above the picture, it tucks inside instead
    const {T: imgTop} = imageRectOnScreen();
    const ty = o.b - 16 >= imgTop ? o.b - 16 : o.b + 2;
    ctx.fillStyle = col;
    ctx.fillRect(o.a - 1, ty, tw, 14);
    ctx.fillStyle = '#10141a';
    ctx.fillText(tag, o.a + 2, ty + 11);
    if (i === sel) {
      const hp = handlePoints(b);
      ctx.fillStyle = '#1b2230';
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1;
      for (const k of HANDLES) {
        const [hx, hy] = hp[k];
        ctx.fillRect(hx - 4, hy - 4, 8, 8);
        ctx.strokeRect(hx - 4, hy - 4, 8, 8);
      }
    }
  });

  if (draw_) {
    ctx.strokeStyle = colorOf(activeClass); ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    // the preview shows the box that will be made, which is clipped to the picture
    const {L, T, R, B} = imageRectOnScreen();
    const cl = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
    const px0 = cl(Math.min(draw_.sx, draw_.cx), L, R), px1 = cl(Math.max(draw_.sx, draw_.cx), L, R);
    const py0 = cl(Math.min(draw_.sy, draw_.cy), T, B), py1 = cl(Math.max(draw_.sy, draw_.cy), T, B);
    ctx.strokeRect(px0 + .5, py0 + .5, Math.max(0, px1 - px0 - 1), Math.max(0, py1 - py0 - 1));
    ctx.setLineDash([]);
  }
  drawGuides();
}

// Dashed guide lines through the cursor, across the whole image, so it is clear where a box's
// edge will start. Two-tone dashes - light, with a faint dark gap - stay visible on sky and on
// shadow alike without being heavy. Only with the box tool, and not while panning or moving.
function drawGuides() {
  if (!hover || hover.pan || pan || grab || tool !== 'draw') return;
  const [bx, by] = toScr(0, 0), [ex, ey] = toScr(iw, ih);
  if (hover.x < bx || hover.x > ex || hover.y < by || hover.y > ey) return;   // off the image
  const x = Math.round(hover.x) + .5, y = Math.round(hover.y) + .5;          // crisp 1px lines
  const line = () => {
    ctx.beginPath();
    ctx.moveTo(x, Math.max(by, 0)); ctx.lineTo(x, Math.min(ey, cv.height));
    ctx.moveTo(Math.max(bx, 0), y); ctx.lineTo(Math.min(ex, cv.width), y);
    ctx.stroke();
  };
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]); ctx.lineDashOffset = -6;          // faint dark, in the light gaps
  ctx.strokeStyle = 'rgba(0,0,0,.18)'; line();
  ctx.setLineDash([6, 4]); ctx.lineDashOffset = 0;           // light dashes
  ctx.strokeStyle = 'rgba(255,255,255,.7)'; line();
  ctx.setLineDash([]); ctx.lineDashOffset = 0;
}

function loupe(px, py) {
  const el = $('#loupe');
  // Drawing and resizing are when magnification matters most: it is how you see where the
  // target's edge actually is. It used to switch off here, and because mousemove returned
  // before reaching this function it was not hidden either, so it froze in place over the
  // thing being labelled.
  if (!img.complete || !iw || !loupeOn) { el.style.display = 'none'; return; }
  const [ix, iy] = toImg(px, py), Z = LOUPE_MAX * loupePct / 100, R = 210 / Z;
  const S = 214, GAP = 26;                      // 210 canvas + 2px border each side
  // Put the loupe in the quadrant opposite the cursor. Clamping it to the edge instead, which
  // is what it used to do, parked it exactly over the corner of a full-size image - the one
  // place you cannot then see well enough to draw a box.
  const left = px > cv.width / 2 ? px - GAP - S : px + GAP;
  const top = py > cv.height / 2 ? py - GAP - S : py + GAP;
  el.style.display = 'block';
  el.style.left = Math.max(4, Math.min(left, cv.width - S - 4)) + 'px';
  el.style.top = Math.max(4, Math.min(top, cv.height - S - 4)) + 'px';
  lctx.imageSmoothingEnabled = false;
  lctx.fillStyle = '#fff'; lctx.fillRect(0, 0, 210, 210);
  // the loupe is a magnifier, not a second opinion: it has to show the same adjustment the
  // canvas is showing, or the two disagree about what is there
  if (bright) lctx.filter = `brightness(${100 + bright}%)`;
  lctx.drawImage(img, ix - R / 2, iy - R / 2, R, R, 0, 0, 210, 210);
  lctx.filter = 'none';
  const ox = ix - R / 2, oy = iy - R / 2;
  boxes.forEach(b => {
    lctx.strokeStyle = colorOf(b[4]); lctx.lineWidth = 2;
    lctx.strokeRect((b[0] - ox) * Z, (b[1] - oy) * Z, (b[2] - b[0]) * Z, (b[3] - b[1]) * Z);
  });
  if (draw_) {                       // the box being dragged right now, magnified with it
    const [ax, ay] = toImg(draw_.sx, draw_.sy), [bx, by] = toImg(draw_.cx, draw_.cy);
    lctx.strokeStyle = colorOf(activeClass); lctx.lineWidth = 1.5;
    lctx.setLineDash([4, 3]);
    lctx.strokeRect((Math.min(ax, bx) - ox) * Z, (Math.min(ay, by) - oy) * Z,
                    Math.abs(bx - ax) * Z, Math.abs(by - ay) * Z);
    lctx.setLineDash([]);
  }
  lctx.strokeStyle = 'rgba(224,174,0,.95)'; lctx.lineWidth = 1;
  lctx.beginPath();
  lctx.moveTo(105, 93); lctx.lineTo(105, 117);
  lctx.moveTo(93, 105); lctx.lineTo(117, 105);
  lctx.stroke();
}

/* ---------------------------------------------------------------- mouse */
const stage = $('#stage');

// The view bar and the loupe live inside #stage so they float over the picture. That also
// means their clicks arrive at the drawing handler, which is why pressing Contrast left a
// small box behind every time. Anything that starts on a control is not a drawing gesture.
const onChrome = e => !!(e.target.closest && e.target.closest('#viewbar, #boxmenu'));

// Two tools: 'draw' makes boxes, 'pan' moves the view. Holding Ctrl or Space pans with either
// one, and the middle button always pans.
let tool = 'draw';
let hover = null;                                // last pointer position on the stage, for guides
const panning = e => tool === 'pan' || space || !!(e && (e.ctrlKey || e.metaKey));
function idleCursor(e) {
  stage.style.cursor = panning(e) ? 'grab' : 'crosshair';
  if (hover) { hover.pan = panning(e); render(); }   // guides hide while Ctrl/Space pans
}
function setTool(t) {
  tool = t;
  $('#t-draw').classList.toggle('on', t === 'draw');
  $('#t-pan').classList.toggle('on', t === 'pan');
  idleCursor();
}
$('#t-draw').onclick = () => setTool('draw');
$('#t-pan').onclick = () => setTool('pan');

stage.addEventListener('mousedown', e => {
  if (busy || onChrome(e)) return;
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  if (e.button === 1 || (e.button === 0 && panning(e))) {
    pan = {x: px, y: py}; stage.style.cursor = 'grabbing'; e.preventDefault(); return;
  }
  if (e.button !== 0) return;
  if (!classes.length) {
    toast('Create a class first - every box has to say what it is', 'err');
    $('#classfilter').focus();
    return;
  }

  const h = hitHandle(px, py);
  if (h) { remember(); grab = {mode: 'resize', h, start: boxes[sel].slice()}; return; }

  // Only the selected box can be moved. An unselected box is inert: dragging over it draws a
  // new box, so a drone right beside one already labelled can still be boxed. Picking a box
  // up straight away used to drag the old label onto the neighbour instead.
  const [ix, iy] = toImg(px, py);
  if (sel >= 0 && boxAt(ix, iy).includes(sel)) {
    remember();
    grab = {mode: 'move', ox: ix, oy: iy, start: boxes[sel].slice()};
    return;
  }
  draw_ = {sx: px, sy: py, cx: px, cy: py, ix, iy};
});

// Indices of the boxes under an image point, smallest first: where boxes overlap, the small
// one is almost always the one being aimed at.
function boxAt(ix, iy) {
  return boxes.map((b, i) => i)
    .filter(i => { const b = boxes[i];
      return ix >= b[0] - 3 && ix <= b[2] + 3 && iy >= b[1] - 3 && iy <= b[3] + 3; })
    .sort((a, b) => (boxes[a][2] - boxes[a][0]) * (boxes[a][3] - boxes[a][1])
                  - (boxes[b][2] - boxes[b][0]) * (boxes[b][3] - boxes[b][1]));
}

stage.addEventListener('mousemove', e => {
  if (onChrome(e) && !draw_ && !grab && !pan) {
    $('#loupe').style.display = 'none';
    if (hover) { hover = null; render(); }        // no guides over the view bar
    return;
  }
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  hover = {x: px, y: py, pan: panning(e)};

  if (pan) {
    viewport.x += px - pan.x; viewport.y += py - pan.y;
    pan = {x: px, y: py}; render(); return;
  }

  if (grab) {
    const [ix, iy] = toImg(px, py);
    const s = grab.start;
    if (grab.mode === 'move') {
      boxes[sel] = shiftInside(s, ix - grab.ox, iy - grab.oy);
    } else {
      let [x0, y0, x1, y1] = s;
      if (grab.h.includes('n')) y0 = lim(iy, ih);
      if (grab.h.includes('s')) y1 = lim(iy, ih);
      if (grab.h.includes('w')) x0 = lim(ix, iw);
      if (grab.h.includes('e')) x1 = lim(ix, iw);
      boxes[sel] = norm([x0, y0, x1, y1, s[4]]);
    }
    dirty = true; renderBoxes(); render(); loupe(px, py);
    return;
  }

  if (draw_) {
    const [bx, by] = toScr(0, 0), [ex, ey] = toScr(iw, ih);
    draw_.cx = Math.min(Math.max(px, bx), ex);
    draw_.cy = Math.min(Math.max(py, by), ey);
    render(); loupe(px, py); return;
  }

  const h = hitHandle(px, py);
  if (h) stage.style.cursor = (h === 'n' || h === 's') ? 'ns-resize' :
                              (h === 'e' || h === 'w') ? 'ew-resize' :
                              (h === 'nw' || h === 'se') ? 'nwse-resize' : 'nesw-resize';
  else if (panning(e)) stage.style.cursor = 'grab';
  else {
    const [ix, iy] = toImg(px, py);
    const overSel = sel >= 0 && boxAt(ix, iy).includes(sel);
    stage.style.cursor = overSel ? 'move' : 'crosshair';
  }
  render();                                        // move the guide lines with the cursor
  loupe(px, py);
});

/* ---------------------------------------------------------------- box menu
 * Right-click on a box selects it - or keeps it selected - and opens a menu of what can be
 * done to it. Where boxes overlap, an already selected box under the pointer wins, then the
 * smallest. Right-click elsewhere on the picture does nothing (and no browser menu). */
const boxMenu = $('#boxmenu');

/* While the menu is open it is the only thing that responds. A veil over the rest of the page
 * dims it and takes the mouse: a click there only closes the menu (it can no longer start a box
 * or press a button on its way), and the wheel there does nothing - over the menu it scrolls
 * the class list instead of zooming the picture behind it. */
const menuVeil = document.createElement('div');
menuVeil.id = 'menuveil';
menuVeil.hidden = true;
document.body.appendChild(menuVeil);
const swallow = e => { e.preventDefault(); e.stopPropagation(); };
menuVeil.addEventListener('mousedown', e => { swallow(e); closeBoxMenu(); });
menuVeil.addEventListener('contextmenu', e => { swallow(e); closeBoxMenu(); });
menuVeil.addEventListener('wheel', swallow, {passive: false});

function closeBoxMenu() {
  if (boxMenu.hidden) return;
  boxMenu.hidden = true; boxMenu.innerHTML = '';
  menuVeil.hidden = true;
}

let menuBox = -1;                                // the box the open menu acts on

function menuClassRows() {
  const q = boxMenu.querySelector('.mq');
  const cur = boxes[menuBox] ? boxes[menuBox][4] : -1;
  const seen = new Set();
  return classSections(q ? q.value : '').map(sec =>
    (sec.section && classes.length > MANY ? `<div class="mls">${esc(sec.section)}</div>` : '') +
    sec.ids.map(ci => {
      const dup = seen.has(ci); seen.add(ci);
      return `<button data-a="cls" data-c="${ci}" class="${ci === cur ? 'cur' : ''}${dup ? ' dup' : ''}" role="menuitemradio">
                <span class="sw" style="--c:${colorOf(ci)}"></span><span class="mn">${esc(classes[ci].name)}</span></button>`;
    }).join('')).join('') || '<div class="mnone">No class matches</div>';
}

function openBoxMenu(i, px, py) {
  const b = boxes[i];
  menuBox = i;
  const many = classes.length > MANY;
  const item = (act, label, key, cls = '') =>
    `<button data-a="${act}" class="${cls}" role="menuitem">${label}${
      key ? `<span class="k">${key}</span>` : ''}</button>`;
  boxMenu.innerHTML =
    `<div class="mh">${esc(nameOf(b[4]))} ${Math.round(b[2] - b[0])}×${Math.round(b[3] - b[1])}</div>` +
    (classes.length > 1
      ? '<div class="ml">Class</div>' +
        (many ? `<input class="mq inp" placeholder="Search ${classes.length} classes…" autocomplete="off" spellcheck="false">` : '') +
        `<div class="mlist${many ? ' scroll' : ''}">${menuClassRows()}</div>` +
        '<div class="ms"></div>'
      : '') +
    item('dup', 'Duplicate') +
    item('zoom', 'Zoom to box') +
    item('desel', 'Deselect', 'Esc') +
    '<div class="ms"></div>' +
    item('del', 'Delete box', 'Del', 'danger');

  // Show it beside the pointer, flipped to stay inside the picture area.
  $('#loupe').style.display = 'none';
  closeClassPick();
  menuVeil.hidden = false;
  boxMenu.hidden = false;
  const W = cv.width, H = cv.height, mw = boxMenu.offsetWidth, mh = boxMenu.offsetHeight;
  boxMenu.style.left = Math.max(4, px + mw + 4 > W ? px - mw : px) + 'px';
  boxMenu.style.top = Math.max(4, Math.min(py + mh + 4 > H ? py - mh : py, H - mh - 4)) + 'px';

  const q = boxMenu.querySelector('.mq');
  if (q) {
    q.oninput = () => { boxMenu.querySelector('.mlist').innerHTML = menuClassRows(); };
    q.focus({preventScroll: true});              // many classes: type straight away
  } else {
    const first = boxMenu.querySelector('button');
    if (first) first.focus({preventScroll: true});
  }
}

// One handler for every row, including class rows redrawn by the search.
boxMenu.addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (!btn) return;
  e.stopPropagation();
  const a = btn.dataset.a, i = menuBox;
  closeBoxMenu();
  if (i < 0 || i >= boxes.length) return;
  if (a === 'del') {
    remember(); boxes.splice(i, 1); sel = -1; dirty = true;
  } else if (a === 'cls') {
    const c = +btn.dataset.c;
    pushRecent(c);
    if (c !== boxes[i][4]) { remember(); boxes[i][4] = c; dirty = true; classHi = c; renderClassList(); }
  } else if (a === 'dup') {
    // a copy nudged down-right so it is visible, kept inside the image
    remember(); boxes.push(shiftInside(boxes[i].slice(), 8 / viewport.s, 8 / viewport.s));
    sel = boxes.length - 1; dirty = true;
  } else if (a === 'zoom') {
    const [W2, H2] = stageSize(), bb = boxes[i];
    const s = Math.min(W2 / (bb[2] - bb[0]), H2 / (bb[3] - bb[1])) * 0.5;
    viewport.s = Math.max(0.02, Math.min(80, s));
    viewport.x = W2 / 2 - (bb[0] + bb[2]) / 2 * viewport.s;
    viewport.y = H2 / 2 - (bb[1] + bb[3]) / 2 * viewport.s;
    syncZoom();
  } else if (a === 'desel') {
    sel = -1;
  }
  renderBoxes(); render();
});

stage.addEventListener('contextmenu', e => {
  e.preventDefault();                              // never the browser's menu over the picture
  if (busy || onChrome(e) || e.target.closest('#boxmenu')) return;
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const [ix, iy] = toImg(px, py);
  const under = boxAt(ix, iy);
  if (!under.length) { closeBoxMenu(); return; }
  const i = under.includes(sel) ? sel : under[0];
  sel = i; classHi = boxes[i][4];
  renderBoxes(); renderClassList(); render();
  openBoxMenu(i, px, py);
});
// Any other click, scroll, Escape or frame change closes it.
document.addEventListener('mousedown', e => { if (!e.target.closest('#boxmenu')) closeBoxMenu(); }, true);
// Keys typed while the menu has focus belong to the menu, not to the frame shortcuts.
boxMenu.addEventListener('keydown', e => {
  e.stopPropagation();
  const inSearch = e.target.classList && e.target.classList.contains('mq');
  const items = [...boxMenu.querySelectorAll('button:not(.dup)')];
  const at = items.indexOf(document.activeElement);
  if (e.key === 'Escape') { e.preventDefault(); closeBoxMenu(); }
  else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    const next = inSearch ? (e.key === 'ArrowDown' ? 0 : items.length - 1)
                          : (at + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
    items[next].focus();
    items[next].scrollIntoView({block: 'nearest'});
  } else if (inSearch && e.key === 'Enter') {
    e.preventDefault();
    const hit = boxMenu.querySelector('[data-a="cls"]');
    if (hit) hit.click();
  } else if (!inSearch && (e.key === 'Delete' || e.key === 'Backspace')) {
    // typed into the search box, Backspace edits the text - it must never delete the box
    e.preventDefault(); boxMenu.querySelector('[data-a="del"]').click();
  }
});

stage.addEventListener('mouseleave', () => {
  $('#loupe').style.display = 'none';
  if (hover) { hover = null; render(); }
});

stage.addEventListener('mouseup', async e => {
  if (pan) { pan = null; idleCursor(e); return; }
  if (grab) { grab = null; boxes[sel] = clampBox(norm(boxes[sel])); renderBoxes(); render(); return; }
  if (!draw_) return;

  const d = draw_; draw_ = null;
  // A box only exists once the pointer has actually been dragged. Creating one on a bare
  // click meant every stray click on the picture left a label behind.
  const moved = Math.abs(d.cx - d.sx) > DRAG_MIN || Math.abs(d.cy - d.sy) > DRAG_MIN;
  if (moved) {
    const [x0, y0] = toImg(Math.min(d.sx, d.cx), Math.min(d.sy, d.cy));
    const [x1, y1] = toImg(Math.max(d.sx, d.cx), Math.max(d.sy, d.cy));
    const nb = clampBox([x0, y0, x1, y1, activeClass]);
    if (nb[2] - nb[0] < 1 || nb[3] - nb[1] < 1) {
      toast('That box falls outside the image', 'err');
      render(); return;
    }
    remember();
    boxes.push(nb);
    sel = boxes.length - 1; dirty = true;
    renderBoxes(); render();
  } else {
    // Just a click: select the box under it (the smallest, where they overlap), or deselect
    // on empty image. Never creates anything.
    const hit = boxAt(d.ix, d.iy)[0];
    sel = hit === undefined ? -1 : hit;
    if (sel >= 0) { classHi = boxes[sel][4]; renderClassList(); }
    renderBoxes(); render();
  }
});

/* No double-click snap: a box only ever comes from a deliberate drag. Snapping on double-click
 * dropped boxes onto whatever was near a fast pair of clicks. */

stage.addEventListener('wheel', e => {
  if (e.target.closest('#boxmenu')) return;       // the menu scrolls its own list
  e.preventDefault();
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const [ix, iy] = toImg(px, py);
  viewport.s = Math.max(0.05, Math.min(80, viewport.s * (e.deltaY < 0 ? 1.18 : 1 / 1.18)));
  viewport.x = px - ix * viewport.s; viewport.y = py - iy * viewport.s;
  render(); syncZoom(); loupe(px, py);
}, {passive: false});

/* ---------------------------------------------------------------- panels */
/* Every change to the box list goes through here first, so undo works everywhere without
 * each call site having to remember it. */
function remember() {
  past.push(boxes.map(b => b.slice()));
  if (past.length > 80) past.shift();
  future.length = 0;
  syncUndo();
}

function syncUndo() {
  const u = $('#undo'), r = $('#redo');
  if (u) u.disabled = !past.length;
  if (r) r.disabled = !future.length;
}

function undo() {
  if (!past.length) { toast('Nothing to undo'); return; }
  future.push(boxes.map(b => b.slice()));
  boxes = past.pop();
  sel = -1; dirty = true;
  syncUndo(); renderBoxes(); render();
}

function redo() {
  if (!future.length) { toast('Nothing to redo'); return; }
  past.push(boxes.map(b => b.slice()));
  boxes = future.pop();
  sel = -1; dirty = true;
  syncUndo(); renderBoxes(); render();
}

// The open frame's row follows the boxes on screen, so changing a box's class recolours it at once.
function syncCurRow() {
  const f = frames.find(x => x.name === cur);
  if (!f || f.status === 'deleted') return;
  f.classes = Array.from(new Set(boxes.map(b => b[4]))).sort((a, b) => a - b);
  f.boxes = boxes.length;
  const el = [...document.querySelectorAll('#frames .fr')].find(d => d.dataset.n === cur);
  if (!el) return;
  el.outerHTML = frameRow(f);
  const fresh = [...document.querySelectorAll('#frames .fr')].find(d => d.dataset.n === cur);
  if (fresh) fresh.onclick = () => openFrame(cur);
}

function renderBoxes() {
  syncCurRow();
  $('#bcount').textContent = boxes.length || '';
  $('#boxes').innerHTML = boxes.length ? boxes.map((b, i) =>
    `<div class="bx${i === sel ? ' sel' : ''}" data-i="${i}" style="--c:${colorOf(b[4])}">
       <i></i>
       <span class="bn">${esc(nameOf(b[4]))}</span>
       <span class="sz">${Math.round(b[2] - b[0])}×${Math.round(b[3] - b[1])}</span>
       <button class="x" title="Remove">×</button>
     </div>`).join('')
    : '<div class="empty">No boxes. Click the target, or drag to draw one.</div>';
  $('#boxes').querySelectorAll('.bx').forEach(el => {
    const i = +el.dataset.i;
    el.onclick = () => { sel = i; classHi = boxes[i][4]; renderBoxes(); renderClassList(); render(); };
    el.querySelector('.x').onclick = e => {
      e.stopPropagation();
      remember();
      boxes.splice(i, 1); sel = -1; dirty = true; renderBoxes(); render();
    };
  });
}

function renderCands() {
  $('#cands').innerHTML = cands.length ? cands.map((c, i) =>
    `<div class="cd${i === candHi ? ' hi' : ''}" data-i="${i}">
       <span class="tag ${c.src}">${c.src}</span>
       ${c.label ? `<b class="cl">${esc(c.label)}</b>` : ''}
       <span>${Math.round(c.box[2] - c.box[0])}×${Math.round(c.box[3] - c.box[1])}</span>
       <span class="at">${c.score}</span>
     </div>`).join('')
    : '<div class="empty">Nothing proposed on this frame.</div>';
  $('#cands').querySelectorAll('.cd').forEach(el => {
    const i = +el.dataset.i;
    el.onmouseenter = () => { candHi = i; render(); };
    el.onclick = () => {
      const c = cands[i], b = c.box;
      // a model's class goes to the class of the same name, if this dataset has one;
      // otherwise (and for motion/contrast finds) to the class you are drawing in
      const same = c.label && classes.find(k => k.name.toLowerCase() === c.label.toLowerCase());
      const cls = same ? same.id : activeClass;
      remember();
      boxes.push([b[0], b[1], b[2], b[3], cls]);
      sel = boxes.length - 1; dirty = true;
      renderBoxes(); render();
      toast(`Accepted as ${nameOf(cls)}`, 'ok');
    };
  });
}

async function save(quiet) {
  if (!cur) return;
  const payload = boxes.map(b => ({x0: b[0], y0: b[1], x1: b[2], y1: b[3], cls: b[4]}));
  try {
    const r = await post('/api/label', {folder, name: cur, boxes: payload});
    dirty = false;
    const f = frames.find(x => x.name === cur);
    if (f) {
      f.boxes = r.boxes;
      // a deleted frame stays deleted; what changes is the state a restore brings back
      f[f.status === 'deleted' ? 'was' : 'status'] = r.boxes ? 'annotated' : 'null';
      f.classes = Array.from(new Set(boxes.map(b => b[4]))).sort((a, b) => a - b);
      renderFrameList();
    }
    if (!quiet) toast(r.boxes ? `Saved ${r.boxes} box(es)` : 'Saved as null', 'ok');
  } catch (e) { toast('Save failed: ' + e.message, 'err'); }
}

/* ---- what AI assist runs: everything, your model, COCO, or imaging alone. An option that cannot
 * run says why, in its own label, and cannot be picked. The choice is remembered. */
let aiSources = {model: {ok: false, why: ''}, coco: {ok: false, why: ''}};
let aiUse = (() => { try { return localStorage.getItem('dl-aisrc') || 'all'; } catch { return 'all'; } })();

function renderAiSources() {
  const m = aiSources.model, c = aiSources.coco;
  const opts = [
    ['all', 'All sources', true, 'Your model, COCO, motion and contrast - model results listed first'],
    ['model', m.ok ? `Your model · ${m.name}` : `Your model — ${m.why}`, m.ok,
     m.ok ? `Only ${m.name} (${m.from === 'given' ? 'given with --model' : 'from model/'})` : m.why],
    ['coco', c.ok ? `COCO default${c.downloaded ? '' : ' · downloads ~40 MB once'}` : `COCO — ${c.why}`, c.ok,
     c.ok ? `Only COCO ${c.name}: 80 everyday classes (bird, airplane, kite…), no drone class` : c.why],
    ['imaging', 'Imaging only', true, 'No model: motion between frames and local contrast'],
  ];
  if (!opts.some(o => o[0] === aiUse && o[2])) aiUse = 'all';
  $('#aisrc').innerHTML = opts.map(([v, label, ok, tip]) =>
    `<option value="${v}"${ok ? '' : ' disabled'}${v === aiUse ? ' selected' : ''} title="${esc(tip)}">${esc(label)}</option>`).join('');
  const now = opts.find(o => o[0] === aiUse);
  $('#aisrc').title = `AI assist runs: ${now[3]}` +
    (m.ok ? '' : `\nYour model: unavailable - ${m.why}`);
}
$('#aisrc').onchange = e => {
  aiUse = e.target.value;
  try { localStorage.setItem('dl-aisrc', aiUse); } catch {}
  renderAiSources(); e.target.blur();
};

async function runAI() {
  if (!cur) return;
  const cocoFirst = (aiUse === 'all' || aiUse === 'coco') && aiSources.coco.ok && !aiSources.coco.downloaded;
  toast(cocoFirst ? 'Running the detectors… first COCO use downloads its weights (~40 MB, once)'
                  : 'Running the detectors…');
  try {
    const r = await post('/api/ai', {folder, name: cur, use: aiUse});
    if (cocoFirst) api('/api/state').then(s => { aiSources = s.sources || aiSources; renderAiSources(); }).catch(() => {});
    cands = r.candidates; candHi = -1;
    renderCands(); render();
    (r.notes || []).forEach(n => toast(n, 'err'));
    if (!cands.length) toast('Nothing proposed on this frame', 'err');
  } catch (e) { toast('AI assist failed: ' + e.message, 'err'); }
}

/* ---------------------------------------------------------------- buttons */
$('#back').onclick = async () => {
  if (dirty) await save(true);
  await refreshFolderCounts();
  await loadFrames();
  setView('gallery');
};
$('#prev').onclick = () => step(-1);
$('#next').onclick = () => step(1);
$('#save').onclick = () => save();
$('#ai').onclick = runAI;
$('#enh').onclick = () => {
  enhance = !enhance;
  $('#enh').classList.toggle('on', enhance);
  const next = new Image();
  next.onload = () => { img = next; render(); };
  next.src = imgUrl(cur);
};
$('#blank').onclick = async () => {
  remember(); boxes = []; sel = -1; dirty = true; renderBoxes(); render(); await save();
};
// The button says what the frame IS once rejected, not only by its colour.
function showRejected(on) {
  $('#reject').classList.toggle('on', on);
  $('#reject span').textContent = on ? 'Rejected' : 'Reject';
  $('#reject').title = on ? 'This frame is rejected - click to un-reject (R)' : 'Reject this frame (R)';
}
$('#reject').onclick = async () => {
  const on = $('#reject').classList.contains('on');
  await post('/api/status', {folder, names: [cur], status: on ? 'unrejected' : 'rejected'});
  showRejected(!on);
  const f = frames.find(x => x.name === cur);
  if (f) {
    f[f.status === 'deleted' ? 'was' : 'status'] =
      on ? (f.boxes ? 'annotated' : 'unassigned') : 'rejected';
    renderFrameList();
  }
  toast(on ? 'Un-rejected' : 'Rejected', 'ok');
};

// On a deleted frame the Delete button becomes Restore.
function showDeleted(on) {
  $('#del').classList.toggle('restore', on);
  $('#del span').textContent = on ? 'Restore' : 'Delete';
  $('#del').title = on ? 'This frame is deleted - click to restore it'
                       : 'Move this frame to the Deleted tab (Del) - it can be restored from there';
}

/* Delete (or restore) the open frame and carry on. No dialog: deleting only moves the frame to
 * the Deleted tab, where it can be restored or removed for good. Any boxes drawn on it are
 * saved first, so a restore brings them back too. */
async function moveCurrent(trash) {
  if (!cur || stepping) return;
  const name = cur;
  const f = frames.find(x => x.name === name);
  if (trash && f && f.status === 'deleted') {
    toast('Already deleted. Restore it, or delete it forever from the Deleted tab.');
    return;
  }
  stepping = true;
  let next = null;
  try {
    if (dirty) await save(true);
    const i = frames.findIndex(x => x.name === name);
    await post(trash ? '/api/trash' : '/api/untrash', {folder, names: [name]});
    toast(trash ? `${name} moved to Deleted` : `${name} restored`, 'ok');
    refreshFolderCounts().catch(() => {});
    if (status_ === 'all') {
      // the frame stays in this list, only its colour changes
      if (f) {
        if (trash) { f.was = f.status; f.status = 'deleted'; f.deleted_at = Date.now() / 1000; }
        else { f.status = f.was || (f.boxes ? 'annotated' : 'unassigned'); }
      }
      renderFrameList(); showDeleted(trash);
      if (trash) next = 'step';
    } else {
      // it has left this tab: reload the page, and the frame after it slides into its place
      const before = page;
      await loadFrames();
      if (!frames.length) { setView('gallery'); return; }
      const j = page !== before ? frames.length - 1 : Math.min(Math.max(i, 0), frames.length - 1);
      next = frames[j].name;
    }
  } catch (e) {
    toast((trash ? 'Delete' : 'Restore') + ' failed: ' + e.message, 'err');
  } finally { stepping = false; }
  if (next === 'step') step(1);
  else if (next) openFrame(next, true);
}
$('#del').onclick = () => moveCurrent(!$('#del').classList.contains('restore'));

/* ---------------------------------------------------------------- view bar */
function zoomTo(scale, ax, ay) {
  const [W, H] = stageSize();
  const cxp = ax === undefined ? W / 2 : ax;
  const cyp = ay === undefined ? H / 2 : ay;
  const [ix, iy] = toImg(cxp, cyp);
  viewport.s = Math.max(0.02, Math.min(120, scale));
  viewport.x = cxp - ix * viewport.s;
  viewport.y = cyp - iy * viewport.s;
  render(); syncZoom();
}

function syncZoom() {
  const el = $('#zoomval');
  if (el) el.textContent = Math.round(viewport.s * 100) + '%';
}

$('#z-in').onclick = () => zoomTo(viewport.s * 1.25);
$('#z-out').onclick = () => zoomTo(viewport.s / 1.25);
$('#z-fit').onclick = () => { fitView(); render(); syncZoom(); };
$('#zoomval').onclick = () => zoomTo(1);
$('#undo').onclick = undo;
$('#redo').onclick = redo;
$('#bright').oninput = e => { bright = +e.target.value; render(); };
$('#bright').ondblclick = e => { bright = 0; e.target.value = 0; render(); };
function setLoupe(on) {
  loupeOn = on;
  localStorage.setItem('dl-loupe', on ? 'on' : 'off');
  $('#loupebtn').classList.toggle('on', on);
  if (!on) $('#loupe').style.display = 'none';
}
$('#loupebtn').onclick = () => setLoupe(!loupeOn);

function setLoupePct(p) {
  loupePct = p;
  try { localStorage.setItem('dl-loupe-pct', p); } catch {}
  $('#loupepct').value = String(p);
  if (!loupeOn) setLoupe(true);           // picking a strength means you want to see it
}
$('#loupepct').value = String(loupePct);
$('#loupepct').onchange = e => { setLoupePct(+e.target.value); e.target.blur(); };

$('#keys').onclick = () => $('#keysveil').classList.add('on');
$('#keysclose').onclick = () => $('#keysveil').classList.remove('on');
$('#keysveil').onclick = e => { if (e.target.id === 'keysveil') e.currentTarget.classList.remove('on'); };

/* ---------------------------------------------------------------- keys */
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' ||
      e.target.tagName === 'SELECT') return;
  if (document.body.dataset.view !== 'annotate') {
    if (e.key === 'Escape') setView('gallery');
    return;
  }
  // The menu is modal: with it open, only its own keys work (it stops them reaching here
  // while it has focus). Anything else typed is ignored, except Esc, which closes it.
  if (!boxMenu.hidden) {
    if (e.key === 'Escape') { e.preventDefault(); closeBoxMenu(); }
    return;
  }
  if (e.code === 'Space') { space = true; idleCursor(e); e.preventDefault(); return; }
  if (e.key === 'Control' || e.key === 'Meta') { if (!draw_ && !grab) idleCursor(e); return; }
  const k = e.key;
  if ((e.ctrlKey || e.metaKey) && (k === 'z' || k === 'Z')) {
    e.preventDefault(); e.shiftKey ? redo() : undo(); return;
  }
  if ((e.ctrlKey || e.metaKey) && (k === 'y' || k === 'Y')) { e.preventDefault(); redo(); return; }
  if ((e.ctrlKey || e.metaKey) && (k === 's' || k === 'S')) { e.preventDefault(); save(); return; }
  // Every other shortcut is a bare key. With Ctrl, Alt or Cmd held the key belongs to the
  // browser: Ctrl+Shift+R (hard reload) used to reject the open frame on its way out.
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k === 'b' || k === 'B') { setTool('draw'); return; }
  if (k === 'c' || k === 'C') { e.preventDefault(); openClassPick(); return; }
  if (k === 'h' || k === 'H') { setTool('pan'); return; }

  // One press, one frame. A held key repeats ~30 times a second, and stepping on every repeat
  // ran 20 frames past in under a second - frames that were then never looked at.
  if (k === 'ArrowRight' || k === 'd' || k === 'D') { if (!e.repeat) step(1); return; }
  if (k === 'ArrowLeft') { if (!e.repeat) step(-1); return; }
  if (k === 'ArrowUp' || k === 'ArrowDown') {
    e.preventDefault();
    if (e.repeat) return;                        // holding it must not spin through classes
    const d = k === 'ArrowDown' ? 1 : -1;
    setActiveClass((activeClass + d + classes.length) % classes.length, 'key');
    return;
  }
  if (k >= '2' && k <= '9') {                    // 2..9 pick a class directly
    const i = +k - 2;
    if (i < classes.length) setActiveClass(i, 'key');
    return;
  }
  if (k === '0') { fitView(); render(); syncZoom(); return; }
  if (k === '1') { zoomTo(1); return; }
  if (k === '?') { $('#keysveil').classList.toggle('on'); return; }
  if (k === 's' || k === 'S') { save(); return; }
  if (k === 'n' || k === 'N') { $('#blank').click(); return; }
  if (k === 'r' || k === 'R') { $('#reject').click(); return; }
  if (k === 'a' || k === 'A') { runAI(); return; }
  if (k === 'e' || k === 'E') { $('#enh').click(); return; }
  if (k === 'l' || k === 'L') { setLoupe(!loupeOn); return; }
  if (k === '[' || k === ']') {                  // magnifier weaker / stronger
    const i = LOUPE_STEPS.indexOf(loupePct) + (k === ']' ? 1 : -1);
    if (i >= 0 && i < LOUPE_STEPS.length) {
      setLoupePct(LOUPE_STEPS[i]); toast(`Magnifier ${loupePct}%`);
    }
    return;
  }
  if (k === 'f' || k === 'F') { fitView(); render(); syncZoom(); return; }
  if (k === 'Escape') { sel = -1; cands = []; renderBoxes(); renderCands(); render(); return; }
  if ((k === 'Delete' || k === 'Backspace') && sel >= 0) {
    e.preventDefault();
    remember();
    boxes.splice(sel, 1); sel = -1; dirty = true; renderBoxes(); render();
    return;
  }
  // Del with no box selected deletes the frame itself: it goes to the Deleted tab, no dialog.
  // Only Del, not Backspace - Backspace is too easy to hit while meaning to remove a box.
  // Holding the key down must not run through a string of frames, so repeats are ignored.
  if (k === 'Delete' && sel < 0 && !draw_ && !grab) {
    e.preventDefault();
    if (!e.repeat) moveCurrent(true);
  }
});
document.addEventListener('keyup', e => {
  if (e.code === 'Space') space = false;
  if (e.code === 'Space' || e.key === 'Control' || e.key === 'Meta') { if (!pan) idleCursor(e); }
});
window.addEventListener('resize', () => {
  if (document.body.dataset.view === 'annotate') { render(); }
});
window.addEventListener('beforeunload', e => {
  if (dirty) { e.preventDefault(); e.returnValue = ''; }
});

/* ---------------------------------------------------------------- boot */
function prefetchNextPage() {
  if (page >= pages) return;
  api(framesUrl(page + 1))
    .then(r => r.items.slice(0, 24).forEach(f => { new Image().src = thumbUrl(f.name); }))
    .catch(() => {});
}

(async () => {
  setMode(gridMode);
  setLoupe(loupeOn);
  const s = await api('/api/state');
  base = s.base; root = s.root;
  // your own model for AI assist, or why there is none
  aiSources = s.sources || aiSources;
  renderAiSources();
  const sm = aiSources.model;
  $('#modelbadge').textContent = sm.ok ? `model: ${sm.name}` : 'no model in model/';
  $('#modelbadge').title = sm.ok
    ? `AI assist can use ${sm.name} (${sm.from === 'given' ? 'given with --model / LABELLER_MODEL' : 'from model/'})`
    : `Your model: ${sm.why}. COCO and imaging still work - pick them beside AI assist.`;
  $('#modelbadge').classList.toggle('ok', !!sm.ok);
  $('#modelbadge').classList.toggle('coco', false);
  if (root) {
    $('#rootpath').textContent = root;
    folders = await api('/api/folders');
    await loadClasses();
    renderFolders();
    if (folders.length) await restoreFromUrl();
    else { toast('No images under that folder — pick another', 'err'); openBrowse(root); }
  } else {
    openBrowse(base);
  }
})();
