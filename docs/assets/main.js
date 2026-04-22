/* onnx2oracle — theme toggle + table sort (no framework) */
(function () {
  'use strict';

  const STORAGE_KEY = 'onnx2oracle:theme';
  const root = document.documentElement;

  function readTheme() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    const btn = document.querySelector('[data-theme-toggle]');
    if (btn) {
      btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
      btn.setAttribute('data-state', theme);
    }
  }

  // Apply immediately to avoid FOUC
  applyTheme(readTheme());

  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.querySelector('[data-theme-toggle]');
    if (btn) {
      applyTheme(readTheme());
      btn.addEventListener('click', () => {
        const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        localStorage.setItem(STORAGE_KEY, next);
        applyTheme(next);
      });
    }

    // Sortable tables
    document.querySelectorAll('table[data-sortable]').forEach((table) => {
      const headers = table.querySelectorAll('thead th');
      headers.forEach((th, colIdx) => {
        const mark = document.createElement('span');
        mark.className = 'sort-mark';
        mark.textContent = '↕';
        th.appendChild(mark);
        th.addEventListener('click', () => sortTable(table, colIdx, th));
      });
    });
  });

  function sortTable(table, colIdx, th) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const current = th.getAttribute('aria-sort');
    const dir = current === 'ascending' ? 'descending' : 'ascending';

    table.querySelectorAll('thead th').forEach((h) => {
      h.removeAttribute('aria-sort');
      const m = h.querySelector('.sort-mark');
      if (m) m.textContent = '↕';
    });
    th.setAttribute('aria-sort', dir);
    const mark = th.querySelector('.sort-mark');
    if (mark) mark.textContent = dir === 'ascending' ? '↑' : '↓';

    rows.sort((a, b) => {
      const ac = a.children[colIdx];
      const bc = b.children[colIdx];
      const av = (ac && (ac.dataset.sortValue || ac.textContent) || '').trim();
      const bv = (bc && (bc.dataset.sortValue || bc.textContent) || '').trim();
      const an = parseFloat(av.replace(/[^0-9.\-]/g, ''));
      const bn = parseFloat(bv.replace(/[^0-9.\-]/g, ''));
      let cmp;
      if (!isNaN(an) && !isNaN(bn) && /[0-9]/.test(av) && /[0-9]/.test(bv)) {
        cmp = an - bn;
      } else {
        cmp = av.localeCompare(bv, undefined, { sensitivity: 'base', numeric: true });
      }
      return dir === 'ascending' ? cmp : -cmp;
    });

    rows.forEach((r) => tbody.appendChild(r));
  }
})();
