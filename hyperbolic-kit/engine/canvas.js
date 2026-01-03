/**
 * Canvas Engine
 * Handles high-DPR setup and resize loops.
 */

export function setupCanvas(canvas, onDraw) {
    const ctx = canvas.getContext("2d");

    const resize = () => {
        // Look up to the wrap? or just custom size?
        // Usually we want to fill the parent
        const parent = canvas.parentElement;
        if (!parent) return;

        const rect = parent.getBoundingClientRect();
        const dpr = Math.max(1, window.devicePixelRatio || 1);

        canvas.width = Math.round(rect.width * dpr);
        canvas.height = Math.round(rect.height * dpr);

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // normalize coordinate system

        if (onDraw) onDraw({ width: rect.width, height: rect.height, ctx });
    };

    window.addEventListener("resize", resize);

    // Initial size
    resize();

    return {
        resize,
        ctx
    };
}

export function getDiskGeometry(width, height) {
    // Standard margin
    const pad = 20;
    const R = Math.min(width, height) / 2 - pad;
    return {
        w: width,
        h: height,
        R,
        cx: width / 2,
        cy: height / 2
    };
}
