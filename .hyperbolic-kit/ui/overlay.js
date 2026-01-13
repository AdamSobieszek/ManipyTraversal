/**
 * UI Overlay Factory
 * Injects the standard HTML structure into a container.
 */

export function createOverlay(container, { title, subtitle, formula }) {
    const overlay = document.createElement('div');
    overlay.className = 'hk-overlay';

    let html = `
    <div class="hk-title">
      ${title}
    </div>
  `;

    if (subtitle) {
        html += `<div style="margin-top:8px; opacity:0.8;">${subtitle}</div>`;
    }

    if (formula) {
        html += `
      <div style="margin-top:12px; display:grid; justify-items:center; gap:8px;">
        <div style="
          border:2px solid #000; padding:8px 14px;
          font-size:clamp(18px,2.6vw,34px); line-height:1;
          background:rgba(255,255,255,0.92);
        ">${formula}</div>
      </div>
    `;
    }

    // Grid spacer
    html += `<div></div>`;

    overlay.innerHTML = html;
    container.appendChild(overlay);

    // Optional info box
    const info = document.createElement('div');
    info.className = 'hk-info';
    container.appendChild(info);

    return {
        updateInfo: (rows) => {
            // rows = [ { label, val }, ... ]
            info.innerHTML = rows.map(r =>
                `<div class="hk-row"><span class="hk-label">${r.label}</span><span class="hk-val">${r.val}</span></div>`
            ).join('');
        }
    };
}
