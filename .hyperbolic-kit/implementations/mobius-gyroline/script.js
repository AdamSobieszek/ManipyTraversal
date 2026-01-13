import {
    add, sub, scale, norm, norm2, unit, perp,
    diskToPx, pxToDisk, clampToDisk, TAU
} from '../../core/transform.js';
import {
    gyroPoint, hypDist, hypRadius
} from '../../core/poincare.js';
import {
    geodesicFromTwoPoints, geodesicFromPointTangent, sampleGeodesic
} from '../../core/geodesic.js';
import { setupCanvas, getDiskGeometry } from '../../engine/canvas.js';
import { createInteraction } from '../../engine/interact.js';
import { createOverlay } from '../../ui/overlay.js';

// --- State ---
let A = clampToDisk({ x: 0.68, y: 0.10 });
let B = clampToDisk({ x: 0.30, y: 0.30 });

// --- UI Setup ---
const app = document.getElementById('app');
const canvas = document.getElementById('c');

const ui = createOverlay(app, {
    title: `The Möbius Gyroline <span style="font-style:italic">L<sub>AB</sub></span>`,
    formula: `A ⊕ (⊖A ⊕ B) ⊗ t`
});

// --- Drawing Loop ---
function draw({ ctx, width, height }) {
    const geom = getDiskGeometry(width, height);
    const { R, cx, cy } = geom;

    ctx.clearRect(0, 0, width, height);

    // 1. Boundary
    ctx.save();
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = "#000";
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, TAU);
    ctx.stroke();
    ctx.restore();

    // 2. Calculations
    const M = gyroPoint(A, B, 0.5);
    const dAB = hypDist(A, B);

    // Tangents & Orthogonal Curves (Logic ported exactly from gyro.html)
    const g1 = geodesicFromTwoPoints(A, B);
    let tangent1;
    if (g1.type === "diameter") {
        tangent1 = g1.u;
    } else {
        // tangent is perpendicular to radius at M
        tangent1 = unit(perp(sub(M, g1.c)));
    }
    const tangent2 = unit(perp(tangent1));
    const g2 = geodesicFromPointTangent(M, tangent2);

    // 3. Draw Curves (Clipped)
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R - 0.5, 0, TAU);
    ctx.clip();

    // Gyroline A->B (black thick)
    // Sampling: direct gyroPoint evaluation
    const N = 400;
    ctx.lineWidth = 6.0;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#000";
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
        const t = -3 + (6) * i / N; // t in [-3, 3]
        const P = gyroPoint(A, B, t);
        const pp = diskToPx(P, geom);
        if (i === 0) ctx.moveTo(pp.x, pp.y);
        else ctx.lineTo(pp.x, pp.y);
    }
    ctx.stroke();

    // Orthogonal Geodesic (red dotted)
    const pts2 = sampleGeodesic(g2, 400);
    ctx.lineWidth = 2.5;
    ctx.setLineDash([3, 7]);
    ctx.strokeStyle = "red";
    if (pts2.length) {
        const p0 = diskToPx(pts2[0], geom);
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        for (let i = 1; i < pts2.length; i++) {
            const p = diskToPx(pts2[i], geom);
            ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
    }

    ctx.restore(); // end clip

    // 4. Draw Points (A, B)
    const drawPt = (pt, label) => {
        const p = diskToPx(pt, geom);
        ctx.fillStyle = "#000";
        ctx.beginPath();
        ctx.arc(p.x, p.y, 9, 0, TAU);
        ctx.fill();

        // Simple label
        ctx.font = "italic 20px Georgia";
        ctx.fillText(label, p.x + 12, p.y - 12);
    };

    drawPt(A, "A");
    drawPt(B, "B");

    // 5. Update Stats
    ui.updateInfo([
        { label: "A: r²", val: norm2(A).toFixed(3) },
        { label: "B: r²", val: norm2(B).toFixed(3) },
        { label: "dₕ(A,B)", val: dAB.toFixed(3) },
    ]);
}

// --- Interaction wire-up ---
const { resize, ctx } = setupCanvas(canvas, draw);

createInteraction(canvas,
    () => getDiskGeometry(canvas.width, canvas.height), // geom provider
    {
        // Hit Test Logic
        hitTest: (ptPx) => {
            const geom = getDiskGeometry(canvas.width, canvas.height);
            const aPx = diskToPx(A, geom);
            const bPx = diskToPx(B, geom);
            const da = Math.hypot(ptPx.x - aPx.x, ptPx.y - aPx.y);
            const db = Math.hypot(ptPx.x - bPx.x, ptPx.y - bPx.y);
            if (da < 20) return "A";
            if (db < 20) return "B";
            return null;
        }
    },
    // Drag Callback
    (key, ptPx) => {
        const geom = getDiskGeometry(canvas.width, canvas.height);
        const v = clampToDisk(pxToDisk(ptPx, geom), 0.999);

        if (key === "A") A = v;
        if (key === "B") B = v;

        // Redraw
        draw({ ctx, width: canvas.width, height: canvas.height });
    }
);
