/**
 * Geodesic Engine
 * Computes Euclidean circles/lines representing hyperbolic geodesics.
 */

import { add, sub, scale, cross, dot, norm, norm2, unit, perp, clampToDisk, EPS, TAU } from './transform.js';

export function mod2pi(x) {
    x = x % TAU;
    if (x < 0) x += TAU;
    return x;
}

// Compute geodesic passing through P and Q.
// Returns { type: "diameter", u } or { type: "circle", c, r }
export function geodesicFromTwoPoints(P, Q) {
    const det = cross(P, Q);
    if (Math.abs(det) < 1e-8) {
        // Collinear with origin -> diameter
        const u = unit(norm(P) > EPS ? P : (norm(Q) > EPS ? Q : { x: 1, y: 0 }));
        return { type: "diameter", u };
    }
    // Circle orthogonal to unit disk
    const k1 = (1 + norm2(P)) / 2;
    const k2 = (1 + norm2(Q)) / 2;
    const invDet = 1 / det;
    const cx = (k1 * Q.y - k2 * P.y) * invDet;
    const cy = (-k1 * Q.x + k2 * P.x) * invDet;
    const c = { x: cx, y: cy };
    const r2 = norm2(c) - 1;
    const r = Math.sqrt(Math.max(r2, 0));
    return { type: "circle", c, r };
}

// Compute geodesic through M with tangent direction tDir.
export function geodesicFromPointTangent(M, tDir) {
    const t = unit(tDir);
    if (norm(M) < EPS) {
        return { type: "diameter", u: t };
    }
    if (Math.abs(cross(M, t)) < 1e-8) {
        return { type: "diameter", u: unit(M) };
    }
    const n = unit(perp(t));
    const denom = 2 * dot(M, n);
    if (Math.abs(denom) < 1e-10) {
        return { type: "diameter", u: t };
    }
    const s = (1 - norm2(M)) / denom;
    const c = add(M, scale(n, s));
    const r2 = norm2(c) - 1;
    const r = Math.sqrt(Math.max(r2, 0));
    return { type: "circle", c, r };
}

// Find intersections of a circle (c,r) with the unit circle.
export function circleIntersectionsUnit(c, r) {
    const d = norm(c);
    // a is distance from c center to chord; h is half-length of chord
    const a = (1 - r * r + d * d) / (2 * d);
    const h2 = 1 - a * a;
    const h = Math.sqrt(Math.max(h2, 0));
    const p2 = scale(c, a / d);
    const v = scale(perp(c), h / d);
    return [add(p2, v), sub(p2, v)];
}

// Sample points along the geodesic arc inside the disk.
export function sampleGeodesic(g, N = 400) {
    if (g.type === "diameter") {
        const pts = [];
        for (let i = 0; i <= N; i++) {
            // sample from -0.999 to +0.999
            const s = -0.999 + (1.998) * i / N;
            pts.push(scale(g.u, s));
        }
        return pts;
    }

    // Circle case
    const [I1, I2] = circleIntersectionsUnit(g.c, g.r);
    const a1 = Math.atan2(I1.y - g.c.y, I1.x - g.c.x);
    const a2 = Math.atan2(I2.y - g.c.y, I2.x - g.c.x);

    const dccw = mod2pi(a2 - a1);
    const midCcw = a1 + dccw / 2;
    const midPt = add(g.c, { x: g.r * Math.cos(midCcw), y: g.r * Math.sin(midCcw) });

    // Decide which arc is inside the unit disk
    let delta = dccw;
    if (norm(midPt) >= 1 - 1e-6) {
        delta = dccw - TAU;
    }

    const pts = [];
    for (let i = 0; i <= N; i++) {
        const t = i / N;
        const ang = a1 + delta * t;
        const p = add(g.c, { x: g.r * Math.cos(ang), y: g.r * Math.sin(ang) });
        pts.push(clampToDisk(p, 0.999999));
    }
    return pts;
}
