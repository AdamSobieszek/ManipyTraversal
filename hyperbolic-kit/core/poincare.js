/**
 * Poincaré Disk Model Primitives
 * Möbius addition, scalar multiplication, and hyperbolic metrics.
 */

import { dot, norm, norm2, scale, EPS } from './transform.js';

export const MAX_ATANH = 0.999999999;

// Möbius addition: a ⊕ b
export function mobiusAdd(a, b) {
    const a2 = norm2(a);
    const b2 = norm2(b);
    const ab = dot(a, b);
    const denom = 1 + 2 * ab + a2 * b2;
    return {
        x: ((1 + 2 * ab + b2) * a.x + (1 - a2) * b.x) / denom,
        y: ((1 + 2 * ab + b2) * a.y + (1 - a2) * b.y) / denom
    };
}

// Möbius scalar multiplication: t ⊗ v
// r⊗v = tanh(r * atanh(|v|)) * (v/|v|)
export function mobiusScalar(t, v) {
    const r = norm(v);
    if (r < EPS) return { x: 0, y: 0 };
    const rr = Math.min(r, MAX_ATANH);
    const k = Math.tanh(t * Math.atanh(rr)) / r;
    return scale(v, k);
}

// Gyroline point: P(t) = A ⊕ (⊖A ⊕ B)⊗t
export function gyroPoint(A, B, t) {
    const negA = { x: -A.x, y: -A.y };
    const delta = mobiusAdd(negA, B);
    return mobiusAdd(A, mobiusScalar(t, delta));
}

// Hyperbolic distance: d_H(x,y)
export function hypDist(x, y) {
    const negX = { x: -x.x, y: -x.y };
    const d = mobiusAdd(negX, y);
    const r = Math.min(MAX_ATANH, norm(d));
    return 2 * Math.atanh(r);
}

// Hyperbolic radius from origin: ρ(v)
export function hypRadius(v) {
    const r = Math.min(MAX_ATANH, norm(v));
    return 2 * Math.atanh(r);
}
