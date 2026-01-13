# Widget conventions (HTML single-file)

This repo’s widgets follow a consistent layout + interaction convention so new demos feel familiar and are easy to maintain. This README documents the **patterns** and provides **copy/paste snippets**.

---

### High-level layout (3-column)

- **Left column**: small “metrics” card (plots / time-series).
- **Center**: the interactive disk/ball canvas (`#wrap` + `#c`) with **corner buttons**, **resizable handles**, and **draggable title**.
- **Right column**: controls in this order:
  - **Parameters**: sliders (top)
  - **Statistics (+ optional description)**: live stats + short model notes (middle)
  - **Switches**: toggle switches (bottom)

Minimal skeleton:

```html
<div class="app">
  <aside class="side metricsSide">
    <div class="stack">
      <div class="card">
        <div class="cardTitle">Metrics</div>
        <div class="metricsWrap" id="metricsWrap">
          <canvas id="metricsCanvas"></canvas>
        </div>
        <div class="metricsLegend">
          <div class="legendItem"><span class="legendSwatch a"></span>Series A</div>
          <div class="legendItem"><span class="legendSwatch b"></span>Series B</div>
        </div>
      </div>
    </div>
  </aside>

  <div class="left">
    <div class="wrap" id="wrap">
      <canvas id="c"></canvas>

      <!-- Corner buttons -->
      <button id="btnReset" class="cornerBtn cornerLeft" type="button">Reset</button>
      <button id="btnKick" class="cornerBtn cornerRight" type="button">Random kick</button>

      <!-- Resizable handles -->
      <div class="resizeHandle nw" data-handle="nw" aria-hidden="true"></div>
      <div class="resizeHandle ne" data-handle="ne" aria-hidden="true"></div>
      <div class="resizeHandle sw" data-handle="sw" aria-hidden="true"></div>
      <div class="resizeHandle se" data-handle="se" aria-hidden="true"></div>

      <!-- Overlay (draggable title) -->
      <div class="overlay">
        <div id="titleDrag" class="titleDrag" aria-label="Drag to reposition the title">
          <div class="title">Your title</div>
          <div class="subtitle">Your subtitle / short instructions</div>
        </div>
      </div>
    </div>
  </div>

  <aside class="side">
    <div class="stack">
      <div class="card">
        <div class="cardTitle">Parameters</div>
        <div class="controlGrid">
          <!-- sliders go here -->
        </div>
      </div>

      <div class="card">
        <div class="cardTitle">Statistics</div>
        <div class="statGrid" aria-live="polite">
          <!-- live stats go here -->
        </div>
        <div class="divider"></div>
        <div class="noteTitle">Model</div>
        <p class="noteText">Short description…</p>
      </div>

      <div class="card">
        <div class="cardTitle">Switches</div>
        <div class="controlGrid">
          <!-- toggle switches go here -->
        </div>
      </div>
    </div>
  </aside>
</div>
```

---

### Core CSS (wrap, draggable title, resizable canvas)

Key ideas:
- `#wrap` is `position: relative` and owns CSS vars:
  - `--uiScale` (typography scaling with wrap size)
  - `--titleX`, `--titleY` (draggable title offset)
- The overlay itself is `pointer-events:none`, but the draggable title is `pointer-events:auto`.

```css
.wrap{
  position:relative;
  width: min(var(--canvas-max), 100%);
  aspect-ratio:1/1;
  --uiScale: 1;
  --titleX: 0px;
  --titleY: 0px;
  user-select:none; -webkit-user-select:none; touch-action:none;
  background: rgba(255,255,255,0.72);
  border: 1px solid var(--border);
  border-radius: 18px;
  overflow:hidden;
}
canvas{ width:100%; height:100%; display:block; }

.overlay{ position:absolute; inset:0; pointer-events:none; }
.titleDrag{
  position:absolute;
  left: 50%;
  top: 6%;
  transform: translate(-50%, 0) translate(var(--titleX), var(--titleY));
  pointer-events:auto;
  cursor: grab;
  touch-action:none;
  z-index: 7;
  width: min(92%, 980px);
  display:flex;
  flex-direction:column;
  align-items:center;
  background: transparent;
}
.titleDrag:active{ cursor: grabbing; }
.title{
  text-align:center;
  font-size: calc(clamp(18px, 3.0vw, 34px) * var(--uiScale));
  line-height:1.15;
  margin:0;
  padding: 0 10%;
}
.subtitle{
  margin-top:10px;
  text-align:center;
  padding: 0 10%;
  font-size: calc(clamp(13px, 1.9vw, 19px) * var(--uiScale));
  opacity:0.90;
}

/* Resizable handles */
.resizeHandle{
  position:absolute;
  width: 14px;
  height: 14px;
  pointer-events:auto;
  z-index: 6;
  background: transparent;
}
.resizeHandle.nw{ left: 0; top: 0; cursor: nwse-resize; }
.resizeHandle.ne{ right: 0; top: 0; cursor: nesw-resize; }
.resizeHandle.sw{ left: 0; bottom: 0; cursor: nesw-resize; }
.resizeHandle.se{ right: 0; bottom: 0; cursor: nwse-resize; }
```

---

### Corner buttons convention

Place primary “utility” actions as corner buttons inside `#wrap` (not in the settings panel).

```css
button.cornerBtn{
  position:absolute;
  bottom: 12px;
  z-index: 5;
  pointer-events:auto;
  appearance:none;
  -webkit-appearance:none;
  border: 1px solid rgba(0,0,0,0.18);
  background: rgba(255,255,255,0.82);
  border-radius: 12px;
  padding: 10px 12px;
  font-family: system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
  font-size: 14px;
  cursor:pointer;
}
button.cornerBtn:hover{ background: rgba(255,255,255,0.95); }
.cornerLeft{ left: 12px; }
.cornerRight{ right: 12px; }
```

---

### Sliders + switches (right panel)

**Sliders** go in the **Parameters** card:

```html
<div class="sliderRow">
  <div class="sliderTop">
    <div class="sliderLabel">Coupling \(K\)</div>
    <div class="sliderVal" id="valK">—</div>
  </div>
  <input id="sliderK" type="range" min="0" max="6" step="0.05" value="1.0">
</div>
```

**Switches** go in the **Switches** card:

```html
<div class="toggleRow">
  <div class="toggleLabel">
    Phase + density heatmap
    <span class="toggleHint">Hue = phase, intensity = density</span>
  </div>
  <label class="switch" aria-label="Toggle heatmap">
    <input id="toggleDensity" type="checkbox" checked>
  </label>
</div>
```

Recommended styling:

```css
.controlGrid{ display:grid; gap: 12px; margin-top: 10px; }
.toggleRow{
  display:flex; align-items:center; justify-content:space-between;
  gap: 14px;
  padding: 10px 12px;
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 14px;
  background: rgba(255,255,255,0.55);
}
.sliderRow{
  padding: 10px 12px;
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 14px;
  background: rgba(255,255,255,0.55);
}
```

---

### JS wiring: resizing + UI scaling + draggable title

**Canvas resize (DPI aware) + typography scaling**

Keep the baseline typography unchanged at initial size, then scale proportionally with wrap width:

```js
const wrap = document.getElementById("wrap");
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const titleDrag = document.getElementById("titleDrag");

let uiBaseW = null;
let titleDragState = null;

function resize(){
  const rect = wrap.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width  = Math.max(1, Math.round(rect.width  * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0);

  if (uiBaseW === null) uiBaseW = Math.max(1, rect.width);
  const ui = Math.max(0.72, Math.min(1.65, rect.width / uiBaseW));
  wrap.style.setProperty("--uiScale", ui.toFixed(3));
}

window.addEventListener("resize", resize);
if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);
resize();
```

**Draggable title**

```js
function setTitleOffset(xPx, yPx){
  wrap.style.setProperty("--titleX", `${xPx.toFixed(1)}px`);
  wrap.style.setProperty("--titleY", `${yPx.toFixed(1)}px`);
}

if (titleDrag) {
  titleDrag.addEventListener("pointerdown", (ev) => {
    const cs = getComputedStyle(wrap);
    const baseX = parseFloat(cs.getPropertyValue("--titleX")) || 0;
    const baseY = parseFloat(cs.getPropertyValue("--titleY")) || 0;
    titleDragState = { startX: ev.clientX, startY: ev.clientY, baseX, baseY, pid: ev.pointerId };
    try { titleDrag.setPointerCapture(ev.pointerId); } catch {}
    ev.preventDefault();
    ev.stopPropagation();
  });
  window.addEventListener("pointermove", (ev) => {
    if (!titleDragState) return;
    const dx = ev.clientX - titleDragState.startX;
    const dy = ev.clientY - titleDragState.startY;
    setTitleOffset(titleDragState.baseX + dx, titleDragState.baseY + dy);
    ev.preventDefault();
  });
  window.addEventListener("pointerup", () => { titleDragState = null; });
  window.addEventListener("pointercancel", () => { titleDragState = null; });
}
```

**Resizable canvas (corner handles)**

```js
let resizing = null; // {handle,startX,startY,startW,pid}

function startResize(ev, handle){
  const rect = wrap.getBoundingClientRect();
  resizing = { handle, startX: ev.clientX, startY: ev.clientY, startW: rect.width, pid: ev.pointerId };
  try { wrap.setPointerCapture(ev.pointerId); } catch {}
  ev.preventDefault();
  ev.stopPropagation();
}

function onResizeMove(ev){
  if (!resizing) return;
  const dx = ev.clientX - resizing.startX;
  const dy = ev.clientY - resizing.startY;
  let delta = 0;
  if (resizing.handle === "se") delta = Math.max(dx, dy);
  if (resizing.handle === "nw") delta = -Math.max(dx, dy);
  if (resizing.handle === "ne") delta = Math.max(dx, -dy);
  if (resizing.handle === "sw") delta = Math.max(-dx, dy);

  const minSize = 320;
  const maxSize = Math.min(window.innerWidth * 0.98, window.innerHeight * 0.98, 2400);
  const next = Math.max(minSize, Math.min(maxSize, resizing.startW + delta));

  wrap.style.width = `${next}px`;
  wrap.style.height = `${next}px`;
  resize();

  ev.preventDefault();
  ev.stopPropagation();
}

function endResize(ev){
  if (!resizing) return;
  resizing = null;
  try { wrap.releasePointerCapture(ev.pointerId); } catch {}
  ev.preventDefault();
  ev.stopPropagation();
}

wrap.querySelectorAll?.(".resizeHandle").forEach((el) => {
  el.addEventListener("pointerdown", (ev) => startResize(ev, el.dataset.handle || "se"));
});
wrap.addEventListener("pointermove", onResizeMove);
wrap.addEventListener("pointerup", endResize);
wrap.addEventListener("pointercancel", endResize);
```

---

### Rendering convention: clip to the disk/circle

When drawing any raster content that could show edge artifacts (e.g., heatmaps), **clip to the circle** before `drawImage`:

```js
ctx.save();
ctx.beginPath();
ctx.arc(cx, cy, R - 0.5, 0, TAU);
ctx.clip();

// draw heatmap image here (it can even extend beyond the disk)
ctx.drawImage(heatCanvas, cx - R, cy - R, 2 * R, 2 * R);

ctx.restore();
```

This avoids “square boundary” bleed when the underlying offscreen canvas is scaled.

---

### Naming + IDs

- Use stable IDs for UI elements: `sliderX`, `valX`, `toggleX`, `btnReset`, `btnKick`.
- Keep **all** widget logic inside a single IIFE:

```html
<script>
(() => {
  // all widget code here
})();
</script>
```


