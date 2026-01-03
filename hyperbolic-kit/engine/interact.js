/**
 * Interaction Engine
 * standardized drag handlers for canvas items.
 */

import { pxToDisk, clampToDisk, norm } from '../core/transform.js';

export function createInteraction(canvas, geomFn, state, onDrag) {
    let dragging = null;

    const hitTest = (ptPx) => {
        // Check all draggable keys in state
        // We expect state keys that match "draggable" items
        // For now, simpler: user passes a custom hitTest function or we iterate keys
        // Let's iterate keys that look like points {x,y}
        const geom = geomFn();
        const hitR = 20; // px

        // We assume state has properties that are points.
        // But to be generic, let's reverse it: the caller provides a "hitTest" callback?
        // Or we provide a helper "checkHits"

        // Let's do: caller provides `getDraggables()` -> returns { key: {x,y}, ... }
    };

    const getPointerPos = (e) => {
        const r = canvas.getBoundingClientRect();
        const x = (e.clientX ?? e.touches?.[0]?.clientX) - r.left;
        const y = (e.clientY ?? e.touches?.[0]?.clientY) - r.top;
        return { x, y };
    };

    const handleStart = (e) => {
        const p = getPointerPos(e);
        // Ask caller what was hit
        if (state.hitTest) {
            const hit = state.hitTest(p);
            if (hit) {
                dragging = hit;
                e.preventDefault();
            }
        }
    };

    const handleMove = (e) => {
        if (!dragging) return;
        const p = getPointerPos(e);
        // Convert to disk coordinates for the callback
        // The callback receives (key, pxPos) and decides what to do
        onDrag(dragging, p);
        e.preventDefault();
    };

    const handleEnd = () => {
        dragging = null;
    };

    // Mouse
    canvas.addEventListener("mousedown", handleStart);
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleEnd);

    // Touch
    canvas.addEventListener("touchstart", handleStart, { passive: false });
    window.addEventListener("touchmove", handleMove, { passive: false });
    window.addEventListener("touchend", handleEnd);
}
