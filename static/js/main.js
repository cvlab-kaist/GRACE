(function () {
  const D = window.GRACE;

  // Generation time in seconds, the same values as the Quantitative table below. The 'x faster'
  // figure is the ratio of two of them, so both numbers are shown together.
  const LAT = { '480': { T2V: { wan: 851.5, ours: 75.8, dc: 157.1, ltx: 99.6 }, I2V: { wan: 863.2, ours: 77.7, dc: 165.0, ltx: 104.1 } },
                '736': { T2V: { wan: 3361.3, ours: 215.6, dc: 456.4, ltx: 264.2 }, I2V: { wan: 3396.8, ours: 218.8, dc: 550.7, ltx: 274.6 } } };
  const TOK = { '480': { wan: '32.8k', ours: '4.3k', dc: '8.2k', ltx: '4.3k' }, '736': { wan: '77.3k', ours: '10.1k', dc: '19.3k', ltx: '10.1k' } };
  const secs = (v) => Math.round(v).toLocaleString() + ' s';
  // [10-08] the speedups are pinned to the paper's abstract (11.1x and 15.5x, measured on I2V).
  //   Dividing instead gives 11.2 for t2v (851.5/75.8) and 15.6 (3361.3/215.6), which put two
  //   different values on one page. gallery.js has the same constant as FASTG - change one and
  //   they diverge again.
  const FASTG = { '480': '11.1\u00d7', '736': '15.5\u00d7' };
  const ratio = (r, k) => FASTG[r];


  // Count-up: both models climb at the same seconds-per-millisecond rate, so the smaller number
  // stops first and the speed difference is felt rather than read.
  const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ROLLS = [2.6, 1.7, 1.1, 0.8];   // digits further right spin more, like a real odometer
  // Odometer: each digit wheel slides upward as the value grows, then settles on the final one.
  //   opt = { dec: decimal places, suf: suffix }  (defaults to seconds)
  function countUp(el, target, peer, opt) {
    if (!el) return;
    const o = opt || {}, dec = o.dec || 0, suf = (o.suf === undefined) ? ' s' : o.suf;
    const txt = (dec ? Number(target).toFixed(dec) : Math.round(target).toLocaleString()) + suf;
    el.textContent = ''; el.classList.remove('counting', 'landed', 'odo');
    if (REDUCED) { el.textContent = txt; return; }
    el.classList.add('odo');
    const rank = {}; let r = 0;
    for (let i = txt.length - 1; i >= 0; i--) if (txt[i] >= '0' && txt[i] <= '9') rank[i] = r++;
    const wheels = [];
    for (let i = 0; i < txt.length; i++) {
      const ch = txt[i];
      if (ch >= '0' && ch <= '9') {
        const d = +ch, R = ROLLS[Math.min(rank[i], ROLLS.length - 1)], end = d + 10 * Math.ceil(R);
        const w = document.createElement('span'); w.className = 'odo-w';
        const st = document.createElement('span'); st.className = 'odo-s';
        for (let k = 0; k <= end; k++) { const c = document.createElement('i'); c.textContent = String(k % 10); st.appendChild(c); }
        w.appendChild(st); el.appendChild(w);
        wheels.push({ st: st, end: end, span: 10 * R });
      } else {
        const c = document.createElement('span'); c.className = 'odo-c';
        c.textContent = (ch === ' ') ? '\u00A0' : ch; el.appendChild(c);
      }
    }
    const set = (w, pos) => { w.st.style.transform = 'translateY(' + (-pos).toFixed(3) + 'em)'; };
    const top = Math.max(target, peer || target), FULL = 4000;   // [10-02] 1500 -> 4000: it has to climb slowly for the difference to show
    const dur = Math.max(600, FULL * (target / top));   // same rate for both, so the smaller value stops first
    const t0 = performance.now();
    el.classList.add('counting');
    (function step(now) {
      const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      wheels.forEach((w) => set(w, w.end - w.span * (1 - e)));
      if (p < 1) requestAnimationFrame(step);
      else { wheels.forEach((w) => set(w, w.end)); el.classList.remove('counting'); el.classList.add('landed'); setTimeout(() => el.classList.remove('landed'), 700); }
    })(t0);
  }

  function makeVideo(src, opts) {
    const v = document.createElement('video');
    v.muted = true; v.loop = true; v.playsInline = true; v.preload = (opts && opts.preload) || 'metadata';
    if (opts && opts.autoplay) v.autoplay = true;
    v.addEventListener('error', () => { v.style.visibility = 'hidden'; });
    if (src) v.src = src;
    return v;
  }

  // Keep a group of videos on the same frame as the first one.
  function syncGroup(videos) {
    const lead = videos[0];
    let raf;
    function tick() {
      videos.slice(1).forEach((v) => {
        if (v.readyState >= 2 && Math.abs(v.currentTime - lead.currentTime) > 0.06) v.currentTime = lead.currentTime;
        if (lead.paused !== v.paused) (lead.paused ? v.pause() : v.play().catch(() => {}));
      });
      raf = requestAnimationFrame(tick);
    }
    tick();
    return () => cancelAnimationFrame(raf);
  }

  /* sidebar: highlight the section in view */
  const links = Array.from(document.querySelectorAll('.chapters a[href^="#"]'));
  const map = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting && map.has(e.target.id)) {
        links.forEach((a) => a.classList.remove('active'));
        map.get(e.target.id).classList.add('active');
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });
  map.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });

  // [10-03] 28 of VBench's official spatial-relationship captions end in ', front view'. That is
  //   benchmark notation, so it is dropped on screen only. The prompt in the data keeps it: the
  //   picks, the matching and the filenames all key on that exact string.
  const disp = (t) => String(t).replace(/,\s*front view\s*$/i, '');
  function hlStyle(text) {
    var esc = String(text).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; });
    var re = /(in the (?:iconic |distinctive |signature |classic )?style of [^,.;…”]+|Van Gogh[- ]?(?:style|inspired|esque)?[^,.;…”]*|Hokusai[^,.;…”]*|Ukiyo-e[^,.;…”]*|pixel art[^,.;…”]*|oil painting[^,.;…”]*|watercolor[^,.;…”]*|black and white[^,.;…”]*|cyberpunk[^,.;…”]*|surrealist[^,.;…”]*|surrealism[^,.;…”]*|impressionist[^,.;…”]*|anime[- ]style[^,.;…”]*|cartoon[- ]style[^,.;…”]*|Picasso[^,.;…”]*|Monet[^,.;…”]*|pencil sketch[^,.;…”]*|charcoal[^,.;…”]*|retro[- ]style[^,.;…”]*|vintage[- ]style[^,.;…”]*)/gi;
    return esc.replace(re, '<em class="style">$1</em>');
  }

  /* teaser */
  const stage = document.getElementById('stage');
  if (stage) {
    const leftBox = stage.querySelector('.slot-left'), rightBox = stage.querySelector('.slot-right');
    const promptEl = stage.querySelector('.prompt span');
    const playBtn = document.getElementById('play');
    const seek = document.getElementById('seek');
    const time = document.getElementById('time');
    const split = document.getElementById('split');
    let pair = [], stop = () => {};
    const fmt = (t) => { t = Math.max(0, t || 0); return '00:' + String(Math.floor(t)).padStart(2, '0'); };

    function load(i) {
      stop();
      const t = D.teasers[i];
      leftBox.innerHTML = ''; rightBox.innerHTML = '';
      const a = makeVideo(t.wan, { autoplay: true, preload: 'auto' });
      const b = makeVideo(t.grace, { autoplay: true, preload: 'auto' });
      leftBox.appendChild(a); rightBox.appendChild(b);
      pair = [a, b];
      stop = syncGroup(pair);
      // 'T2V 480x832 . caption' -> a task and resolution chip, then the input prompt
      const pm = t.prompt.match(/^(T2V|I2V)(?:\s+(\d+\s*[×x]\s*\d+))?\s*·\s*([\s\S]*)$/);
      promptEl.textContent = '';
      if (pm) {
        const meta = document.createElement('span'); meta.className = 'meta';
        meta.textContent = pm[1] + ' · ' + (pm[2] ? pm[2].replace(/\s/g, '') : '480×832') + '×81';
        const cap = document.createElement('span'); cap.className = 'cap'; cap.innerHTML = '“' + hlStyle(disp(pm[3])) + '”';
        promptEl.appendChild(meta); promptEl.appendChild(cap);
      } else promptEl.textContent = '“' + disp(t.prompt) + '”';
      const rs = /736/.test(t.prompt) ? '736' : '480', tk = pm && pm[1] === 'I2V' ? 'I2V' : 'T2V', L = LAT[rs][tk];
      const q = (sel) => stage.querySelector(sel);
      const put = (sel, txt) => { const e = q(sel); if (e) e.textContent = txt; };
      put('.tag-right .stat em', ratio(rs, tk));
      countUp(q('.tag-left .lat'), L.wan, L.wan);     // both pace off Wan's value, so ours stops first
      countUp(q('.tag-right .lat'), L.ours, L.wan);
      const TW = parseFloat(TOK[rs].wan), TO = parseFloat(TOK[rs].ours);   // '32.8k' → 32.8
      countUp(q('.tag-left .tok'), TW, TW, { dec: 1, suf: 'k' });
      countUp(q('.tag-right .tok'), TO, TW, { dec: 1, suf: 'k' });
      a.addEventListener('timeupdate', () => {
        if (a.duration) seek.value = String((a.currentTime / a.duration) * 1000);
        time.textContent = fmt(a.currentTime) + ' / ' + fmt(a.duration);
      });
      a.addEventListener('play', () => { playBtn.textContent = '❚❚'; playBtn.setAttribute('aria-label', 'Pause'); });
      a.addEventListener('pause', () => { playBtn.textContent = '▶'; playBtn.setAttribute('aria-label', 'Play'); });
      document.querySelectorAll('.thumb').forEach((el) => el.setAttribute('aria-current', Number(el.dataset.i) === i ? 'true' : 'false'));
    }

    playBtn.addEventListener('click', () => { const a = pair[0]; if (!a) return; a.paused ? a.play().catch(() => {}) : a.pause(); });
    seek.addEventListener('input', () => { const a = pair[0]; if (a && a.duration) a.currentTime = (seek.value / 1000) * a.duration; });
    if (split) { const setSplit = (pct) => { pct = Math.min(95, Math.max(5, pct)); stage.style.setProperty('--split', pct + '%'); split.value = String(Math.round(pct)); }; split.addEventListener('input', () => setSplit(Number(split.value))); }

    const thumbs = document.getElementById('thumbs');
    // [10-02] grouped as Image-to-video and Text-to-video so a reader knows what they are picking
    // before they pick it. I2V comes first.
    const grp = { I2V: [], T2V: [] };
    D.teasers.forEach((t, i) => { (/^I2V/.test(t.prompt) ? grp.I2V : grp.T2V).push(i); });
    [['T2V', 'Text-to-video'], ['I2V', 'Image-to-video']].forEach(([k, label]) => {
      if (!grp[k].length) return;
      const h = document.createElement('div'); h.className = 'thumb-head';
      h.textContent = label + ' · ' + grp[k].length;
      thumbs.appendChild(h);
      const row = document.createElement('div'); row.className = 'thumb-row';
      grp[k].forEach((i) => {
        const t = D.teasers[i];
        const btn = document.createElement('button');
        btn.type = 'button'; btn.className = 'thumb'; btn.dataset.i = String(i);
        btn.setAttribute('aria-label', label + ': ' + t.prompt);
        btn.appendChild(makeVideo(t.grace, { preload: 'metadata' }));
        btn.addEventListener('click', () => load(i));
        row.appendChild(btn);
      });
      thumbs.appendChild(row);
    });
    // open on 'A person is filling eyebrows', found by caption so a reordering does not move it
    const first = D.teasers.findIndex((t) => /filling eyebrows/i.test(t.prompt));
    load(first >= 0 ? first : 0);
  }

  /* comparison carousel */
  const car = document.getElementById('carousel');
  if (car) {
    const tabs = Array.from(document.querySelectorAll('#cmp-tabs .tab'));
    const grid = car.querySelector('.cmp-grid');
    const promptEl = car.querySelector('.prompt-text');
    const countEl = car.querySelector('.count');
    const dotsEl = car.querySelector('.dots');
    let tab = 't2v', idx = 0, stop = () => {};

    function cell(label, src, ours, hr, still) {
      const c = document.createElement('div');
      c.className = 'vcell' + (ours ? ' ours' : '');
      const box = document.createElement('div');
      box.className = 'vbox' + (hr ? ' hr' : '');
      if (still) { const img = new Image(); img.alt = 'Input frame'; img.src = src; img.onerror = () => { img.style.visibility = 'hidden'; }; box.appendChild(img); }
      else box.appendChild(makeVideo(src, { autoplay: true, preload: 'auto' }));
      const l = document.createElement('div'); l.className = 'label';
      const spec = typeof label === 'string' ? { name: label } : label;
      const tl = document.createElement('div'); tl.className = 'tline';
      const nb = document.createElement('span'); nb.className = 'name'; nb.textContent = spec.name; tl.appendChild(nb);
      // [10-03] the badge lives in a strip above the video. On the title line it would make only
      //   our column taller and break the row; over the video it would cover the picture. So every
      //   column gets a strip of the same height and only ours is filled.
      const strip = document.createElement('div'); strip.className = 'badge-row';
      if (spec.fast) {
        const st = document.createElement('span'); st.className = 'stat';
        const em = document.createElement('em'); em.textContent = spec.fast;
        st.appendChild(em); const lb = document.createElement('span'); lb.className = 'lbl'; lb.textContent = 'faster inference'; st.appendChild(lb);
        strip.appendChild(st);
      }
      l.appendChild(tl);
      if (spec.inf) {   // the figures go on one line, two label-value pairs, kept low so it does not read as a cell
        const mt = document.createElement('div'); mt.className = 'metrics';
        [['Inference time', spec.inf], ['Latent tokens #', spec.tok]].forEach(function (kv) {
          const pr = document.createElement('span'); pr.className = 'm';   // label and value stay one unit, so they never wrap apart
          const ke = document.createElement('span'); ke.className = 'k'; ke.textContent = kv[0];
          const ve = document.createElement('span'); ve.className = 'val'; ve.textContent = kv[1];
          if (kv[0] === 'Inference time' && spec.secs) { ve.textContent = ''; countUp(ve, spec.secs, spec.peer); }
          if (kv[0] === 'Latent tokens #' && spec.tnum) { ve.textContent = ''; countUp(ve, spec.tnum, spec.tpeer, { dec: 1, suf: 'k' }); }
          pr.appendChild(ke); pr.appendChild(ve); mt.appendChild(pr);
        });
        l.appendChild(mt);
      }
      c.appendChild(strip); c.appendChild(box); c.appendChild(l);
      return c;
    }
    function render() {
      stop();
      const list = D.comparisons[tab], item = list[idx], hr = tab === 'hr';
      grid.innerHTML = '';
      const cells = [];
      // [10-09] the item decides whether it is i2v, not the tab name. The 736 tab now carries i2v
      //   comparisons too, and keying off the tab would drop the input cell and label them with
      //   T2V latency. Existing items are unaffected: only the i2v ones carry an input.
      if (item.input) cells.push(cell('Input frame', item.input, false, hr, true));
      const rs = hr ? '736' : '480', tk = item.input ? 'I2V' : 'T2V', L = LAT[rs][tk];
      const TN = { wan: parseFloat(TOK[rs].wan), ours: parseFloat(TOK[rs].ours), dc: parseFloat(TOK[rs].dc) };
      cells.push(cell({ name: 'Wan2.1-14B', inf: secs(L.wan), tok: TOK[rs].wan, secs: L.wan, peer: L.wan, tnum: TN.wan, tpeer: TN.wan }, item.wan, false, hr));
      // [10-08] the LTX column appears only when the item carries ltx; other items keep three columns.
      if (item.ltx) cells.push(cell({ name: 'LTX-Video 0.9.7', inf: secs(L.ltx), tok: TOK[rs].ltx, secs: L.ltx, peer: L.wan, tnum: parseFloat(TOK[rs].ltx), tpeer: TN.wan }, item.ltx, false, hr));
      cells.push(cell({ name: 'GRACE (Ours)', inf: secs(L.ours), tok: TOK[rs].ours, fast: ratio(rs, tk), secs: L.ours, peer: L.wan, tnum: TN.ours, tpeer: TN.wan }, item.grace, true, hr));
      if (item.dcgen) cells.push(cell({ name: 'DC-Gen', inf: secs(L.dc), tok: TOK[rs].dc, secs: L.dc, peer: L.wan, tnum: TN.dc, tpeer: TN.wan }, item.dcgen, false, hr));
      grid.style.gridTemplateColumns = 'repeat(' + cells.length + ', minmax(0, 1fr))';
      cells.forEach((c) => grid.appendChild(c));
      stop = syncGroup(Array.from(grid.querySelectorAll('video')));
      promptEl.innerHTML = '“' + hlStyle(disp(item.excerpt || item.prompt)) + '”';
      countEl.textContent = (idx + 1) + ' / ' + list.length;
      dotsEl.innerHTML = list.map((_, i) => '<span class="' + (i === idx ? 'on' : '') + '"></span>').join('');
    }
    tabs.forEach((t) => t.addEventListener('click', () => {
      tab = t.dataset.tab; idx = 0;
      tabs.forEach((x) => x.setAttribute('aria-selected', x === t ? 'true' : 'false'));
      render();
    }));
    car.querySelector('.prev').addEventListener('click', () => { const n = D.comparisons[tab].length; idx = (idx + n - 1) % n; render(); });
    car.querySelector('.next').addEventListener('click', () => { const n = D.comparisons[tab].length; idx = (idx + 1) % n; render(); });
    render();
  }

  /* quantitative tabs */
  const qtabs = Array.from(document.querySelectorAll('#q-tabs .tab'));
  qtabs.forEach((t) => t.addEventListener('click', () => {
    qtabs.forEach((x) => x.setAttribute('aria-selected', x === t ? 'true' : 'false'));
    document.querySelectorAll('.q-panel').forEach((p) => { p.hidden = p.id !== t.getAttribute('aria-controls'); });
  }));

  /* [10-03] the opening film: muted autoplay plus an 'unmute' button.
     Browsers block autoplay with sound, so it starts muted and unmutes when the viewer asks.
     play() is called a second time because autoplay does not take if the tab was in the
     background or the poster loaded late. */
  const filmV = document.getElementById('film-v'), filmB = document.getElementById('film-unmute');
  if (filmV) {
    const kick = () => { const r = filmV.play(); if (r && r.catch) r.catch(() => {}); };
    if (filmV.readyState >= 2) kick(); else filmV.addEventListener('loadeddata', kick, { once: true });
    if (filmB) {
      filmB.addEventListener('click', () => { filmV.muted = false; filmV.volume = 1; kick(); filmB.hidden = true; });
      filmV.addEventListener('volumechange', () => { if (!filmV.muted) filmB.hidden = true; });
    }
  }

  /* bibtex copy */
  const copy = document.getElementById('copy');
  if (copy) copy.addEventListener('click', () => {
    navigator.clipboard.writeText(document.getElementById('bibtex').textContent).then(() => {
      copy.textContent = 'Copied'; setTimeout(() => { copy.textContent = 'Copy'; }, 1600);
    });
  });
})();
