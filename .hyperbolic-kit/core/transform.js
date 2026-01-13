/**
 * Vector and Coordinate Transform Utilities
 * Pure math, no DOM dependencies (except for concept of "pixels").
 */

export const EPS = 1e-10;
export const TAU = Math.PI * 2;

// --- Basic Vector Ops ---
export const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
export const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
export const scale = (v, s) => ({ x: v.x * s, y: v.y * s });
export const dot = (a, b) => a.x * b.x + a.y * b.y;
export const cross = (a, b) => a.x * b.y - a.y * b.x;
export const norm2 = (v) => v.x * v.x + v.y * v.y;
export const norm = (v) => Math.hypot(v.x, v.y);
export const neg = (v) => ({ x: -v.x, y: -v.y });
export const perp = (v) => ({ x: -v.y, y: v.x });

export const unit = (v) => {
    const r = norm(v);
    if (r < EPS) return { x: 1, y: 0 };
    return scale(v, 1 / r);
};

export const clamp01 = (x) => Math.max(0, Math.min(1, x));

export const clampToDisk = (v, maxR = 0.999) => {
    const r = norm(v);
    if (r <= maxR) return v;
    return scale(v, maxR / r);
};

// --- Coordinate Transforms ---
// geom = { w, h, R, cx, cy }
export const diskToPx = (v, geom) => ({
    x: geom.cx + v.x * geom.R,
    y: geom.cy - v.y * geom.R,
});

export const pxToDisk = (p, geom) => ({
    x: (p.x - geom.cx) / geom.R,
    y: (geom.cy - p.y) / geom.R,
});
