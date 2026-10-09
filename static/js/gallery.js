(function () {
  const G = window.GRACE.gallery;
  // [10-03] 28 of VBench's official spatial-relationship captions end in ', front view'. That is
  //   benchmark notation, so it is dropped on screen only. The prompt in the data keeps it: the
  //   picks, the matching and the filenames all key on that exact string.
  const disp = (t) => String(t).replace(/,\s*front view\s*$/i, '');
  function hlStyle(text) {
    var esc = String(text).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; });
    var re = /(in the (?:iconic |distinctive |signature |classic )?style of [^,.;…”]+|Van Gogh[- ]?(?:style|inspired|esque)?[^,.;…”]*|Hokusai[^,.;…”]*|Ukiyo-e[^,.;…”]*|pixel art[^,.;…”]*|oil painting[^,.;…”]*|watercolor[^,.;…”]*|black and white[^,.;…”]*|cyberpunk[^,.;…”]*|surrealist[^,.;…”]*|surrealism[^,.;…”]*|impressionist[^,.;…”]*|anime[- ]style[^,.;…”]*|cartoon[- ]style[^,.;…”]*|Picasso[^,.;…”]*|Monet[^,.;…”]*|pencil sketch[^,.;…”]*|charcoal[^,.;…”]*|retro[- ]style[^,.;…”]*|vintage[- ]style[^,.;…”]*)/gi;
    esc = esc.replace(re, '<em class="style">$1</em>');
    // [10-07] highlight the camera phrase too. VBench i2v captions append ", camera <motion>",
    //   and seeing which part of the prompt that is makes the motion comparison readable.
    //   It gets its own class so it stays distinct from the style highlight, which is yellow.
    var cam = /(,\s*camera\s+(?:pans?|tilts?|zooms?|rotat\w*|track\w*|dolly|push\w*|pull\w*|orbit\w*|static|moves?|turns?)[^,.;…”]*)/gi;
    return esc.replace(cam, '<em class="cam">$1</em>');
  }

  let filter = 'All';
  let section = 'all';   // the section axis is independent of the tag filter; both can be active

  let open = null; // { wall, tile, els, stop }


  // Count-up: both climb at the same seconds-per-millisecond rate, so the smaller number stops
  // first and the speed difference is felt rather than read.
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

  function makeVideo(src, autoplay, poster) {
    const v = document.createElement('video');
    v.muted = true; v.loop = true; v.playsInline = true; v.preload = autoplay ? 'auto' : 'none';
    if (autoplay) v.autoplay = true;
    // [10-03] lay a poster (the first frame) underneath. On phones, data saving can ignore preload
    //   and iOS caps how many decoders run at once, and the tile was left as a grey rectangle.
    if (poster) v.poster = poster;
    v.addEventListener('error', () => { v.style.visibility = 'hidden'; });
    v.src = src;
    return v;
  }
  // [10-03] hovering a video that had not finished downloading made the picture judder. The cause
  //   was this loop rewriting currentTime every frame at 60Hz: with too little buffered, that
  //   assignment starts a seek, and the next frame starts another before the first finishes, so
  //   seeking restarts forever. Three fixes: sync at 4Hz only, skip while seeking, and touch only
  //   videos that are actually playable (readyState 3).
  function syncGroup(videos) {
    const lead = videos[0]; let raf, last = 0;
    (function tick(now) {
      if (now - last > 250) {
        last = now;
        if (lead && lead.readyState >= 3) {
          videos.slice(1).forEach((v) => {
            if (v.readyState >= 3 && !v.seeking && Math.abs(v.currentTime - lead.currentTime) > 0.12) {
              v.currentTime = lead.currentTime;
            }
          });
        }
      }
      raf = requestAnimationFrame(tick);
    })(0);
    return () => cancelAnimationFrame(raf);
  }
  const resText = (it) => (it.res === '736' ? '736×1280×81' : '480×832×81');

  // [10-02] thumbnails were slow to appear because hundreds of tiles all queued a preload request.
  //   Only tiles on screen, or about to be, fetch their first frame.
  const primed = new WeakSet();
  const io = ('IntersectionObserver' in window) ? new IntersectionObserver((es) => {
    es.forEach((e) => {
      if (!e.isIntersecting) return;
      const v = e.target.querySelector('video');
      if (v && !primed.has(v)) { primed.add(v); v.preload = 'metadata'; try { v.load(); } catch (_) {} }
      io.unobserve(e.target);
    });
  }, { rootMargin: '800px 0px' }) : null;

  // [10-03] thumbnails play continuously. Hundreds at once would overwhelm the decoder, so they
  //   play only while visible and pause when they leave: the last frame stays, so a paused tile
  //   never looks blank. preload is raised to 'auto' here; the observer above only goes as far as
  //   metadata, enough for the first frame.
  const ioPlay = ('IntersectionObserver' in window) ? new IntersectionObserver((es) => {
    es.forEach((e) => {
      const v = e.target.querySelector('video'); if (!v) return;
      if (e.isIntersecting) {
        if (v.preload !== 'auto') { v.preload = 'auto'; try { v.load(); } catch (_) {} }
        const r = v.play(); if (r && r.catch) r.catch(() => {});
      } else { v.pause(); }
    });
  }, { rootMargin: '150px 0px' }) : null;

  // [10-03] since thumbnails always play, leaving the tiles behind a popup running makes the popup
  //   stutter. They pause when it opens and resume when it closes, and only the tiles currently on
  //   screen resume.
  // No list of 'what was paused' is carried around. That approach had a bug (10-03): moving between
  //   tiles let a second pauseWall overwrite the list with an empty one, and nothing ever resumed.
  //   Instead, pausing takes the whole wall and resuming recomputes from whatever is visible now.
  //   ioPlay only fires when visibility *changes*, so a manually paused tile has to resume here.
  function pauseWall(wall) {
    wall.querySelectorAll('.tile video').forEach((v) => { if (!v.paused) v.pause(); });
  }
  function resumeWall() {
    document.querySelectorAll('.tile video').forEach((v) => {
      if (!v.paused) return;
      const r = v.getBoundingClientRect();
      if (r.bottom > -150 && r.top < window.innerHeight + 150) { const q = v.play(); if (q && q.catch) q.catch(() => {}); }
    });
  }

  function close(keepPaused) {
    if (!open) return;
    open.stop(); open.els.forEach((e) => e.remove());
    open.wall.classList.remove('dimmed'); open.tile.classList.remove('active');
    open = null;
    if (!keepPaused) resumeWall();
  }

  function show(kind, it, tile, wall) {
    if (open && open.tile === tile) return;
    close(true);
    const r = tile.getBoundingClientRect();
    const ox = ((r.left + r.width / 2) / window.innerWidth) * 100;
    const oy = ((r.top + r.height / 2) / window.innerHeight) * 100;
    const back = document.createElement('div'); back.className = 'backdrop';
    const pop = document.createElement('div'); pop.className = 'pop r' + (it.res === '736' ? '736' : '480');
    pop.style.transformOrigin = ox + '% ' + oy + '%';
    const hr = it.res === '736' ? ' hr' : '';
    const vids = [];

    if (kind === 'solo') {
      pop.innerHTML = '<div class="pop-solo"><div class="v' + hr + '"></div><div class="meta"><i></i><span></span></div>'
                    + '<p class="pop-note">81 frames · generated on one A100 80GB at 50 steps, CFG 5, batch 1, bf16.</p></div>';
      const v = makeVideo(it.grace, true, it.poster); vids.push(v);
      pop.querySelector('.v').appendChild(v);
      pop.querySelector('i').innerHTML = '“' + hlStyle(disp(it.excerpt || it.prompt)) + '”';
      pop.querySelector('.meta span').textContent = resText(it);
    } else {
      // token count = latent frames x (H/f/p) x (W/f/p): Wan f8t4p2, GRACE f16t8p2, DC-Gen f32t4p1, 81 frames
      // generation time in seconds and token count, the same values as the paper's Quantitative
      // table. Showing both makes clear what the 'x faster' figure is measured against.
      const LATG = { '480': { T2V: { wan: 851.5, ours: 75.8, dc: 157.1, ltx: 99.6 }, I2V: { wan: 863.2, ours: 77.7, dc: 165.0, ltx: 104.1 } },
                     '736': { T2V: { wan: 3361.3, ours: 215.6, dc: 456.4, ltx: 264.2 }, I2V: { wan: 3396.8, ours: 218.8, dc: 550.7, ltx: 274.6 } } };
      // [10-07] the speedups match the paper and the body text. Computing them gives 11.2 for t2v
      //   (851.5/75.8) and 11.1 for i2v (863.2/77.7), which put two values on one site. The paper's
      //   abstract uses 11.1x and 15.5x, measured on I2V, so these follow that - which leaves the
      //   t2v card 0.1 off its own displayed seconds.
      const FASTG = { '480': '11.1\u00d7', '736': '15.5\u00d7' };
      const TOKG = { '480': { wan: '32.8k', ours: '4.3k', dc: '8.2k', ltx: '4.3k' }, '736': { wan: '77.3k', ours: '10.1k', dc: '19.3k', ltx: '10.1k' } };
      const rs = it.res === '736' ? '736' : '480', tsk = it.tag === 'I2V' ? 'I2V' : 'T2V', L = LATG[rs][tsk], T = TOKG[rs];
      const sec = (v) => Math.round(v).toLocaleString() + ' s';
      const tn = (x) => parseFloat(x), TP = tn(T.wan);   // tokens pace off Wan too, so ours stops first
      const TK = { wan: { secs: L.wan, peer: L.wan, inf: sec(L.wan), tok: T.wan, tnum: tn(T.wan), tpeer: TP },
                   ours: { secs: L.ours, peer: L.wan, inf: sec(L.ours), tok: T.ours, tnum: tn(T.ours), tpeer: TP, fast: FASTG[rs] },
                   dc: { secs: L.dc, peer: L.wan, inf: sec(L.dc), tok: T.dc, tnum: tn(T.dc), tpeer: TP },
                   ltx: { secs: L.ltx, peer: L.wan, inf: sec(L.ltx), tok: T.ltx, tnum: tn(T.ltx), tpeer: TP } };
      const rows = [['Wan2.1-14B', TK.wan, it.wan, false], ['GRACE (Ours)', TK.ours, it.grace, true]];
      if (kind === 'three' && it.dcgen) rows.push(['DC-Gen', TK.dc, it.dcgen, false]);
      if (kind === 'three' && it.ltx) rows.push(['LTX-Video 0.9.7', TK.ltx, it.ltx, false]);
      pop.innerHTML = '<div class="pop-cmp"><div class="pop-input"><span class="h">Input</span></div><div class="pop-sep"></div>'
                    + '<div class="pop-right"><div class="pop-rows n' + rows.length + '"></div>'
                    + '<p class="pop-note"><b>Inference time</b> = how long it takes to make one 81-frame video, autoencoder included. '
                    + 'Measured on one A100 80GB at 50 steps, CFG 5, batch 1, bf16.</p></div></div>';
      const inp = pop.querySelector('.pop-input');
      if (it.input) {
        const f = document.createElement('div'); f.className = 'frame';
        const img = new Image(); img.alt = 'Input frame'; img.src = it.input; img.onerror = () => { img.style.visibility = 'hidden'; };
        f.appendChild(img); inp.appendChild(f);
      }
      const p = document.createElement('p'); p.className = 'ptext'; p.innerHTML = '“' + hlStyle(disp(it.excerpt || it.prompt)) + '”'; inp.appendChild(p);
      const res = document.createElement('span'); res.className = 'res'; res.textContent = resText(it); inp.appendChild(res);
      const holder = pop.querySelector('.pop-rows');
      rows.forEach(([name, cfg, src, ours]) => {
        const row = document.createElement('div'); row.className = 'pop-row' + (ours ? ' ours' : '');
        row.innerHTML = '<div class="who"><div class="tline"><b></b></div><div class="metrics"></div></div><div class="v' + hr + '"></div>';
        row.querySelector('b').textContent = name;
        if (cfg.fast) {
          const st = document.createElement('span'); st.className = 'stat';
          const em = document.createElement('em'); em.textContent = cfg.fast;
          st.appendChild(em); const lb = document.createElement('span'); lb.className = 'lbl'; lb.textContent = 'faster inference'; st.appendChild(lb);
          row.querySelector('.tline').appendChild(st);
        }
        const mt = row.querySelector('.metrics');
        [['Inference time', cfg.inf], ['Latent tokens #', cfg.tok]].forEach(function (kv) {
          const d = document.createElement('div'); d.className = 'm';
          const ke = document.createElement('span'); ke.className = 'k'; ke.textContent = kv[0];
          const ve = document.createElement('span'); ve.className = 'val'; ve.textContent = kv[1];   // never 'v': the popup video box is .v, and querySelector would pick this up instead
          if (kv[0] === 'Inference time' && cfg.secs) { ve.textContent = ''; countUp(ve, cfg.secs, cfg.peer); }
          if (kv[0] === 'Latent tokens #' && cfg.tnum) { ve.textContent = ''; countUp(ve, cfg.tnum, cfg.tpeer, { dec: 1, suf: 'k' }); }
          d.appendChild(ke); d.appendChild(ve); mt.appendChild(d);
        });
        const v = makeVideo(src, true); vids.push(v); row.querySelector(':scope > .v').appendChild(v);   // direct children only, as a second guard
        holder.appendChild(row);
      });
    }
    document.body.appendChild(back); document.body.appendChild(pop);
    pauseWall(wall);
    wall.classList.add('dimmed'); tile.classList.add('active');
    open = { wall, tile, els: [back, pop], stop: syncGroup(vids) };
  }

  function matches(it) {
    if (filter === 'All') return true;
    if (filter === '480×832') return it.res !== '736';
    if (filter === '736×1280') return it.res === '736';
    // [10-03] the tag axis mixes tasks (T2V/I2V) with other labels (736x1280, Styles). Everything
    //   under 736x1280 and Styles is T2V, so the T2V button has to count those as well for the
    //   number to be right - otherwise it shows 86 of the actual 196. The Styles button still
    //   selects styles alone.
    if (filter === 'T2V') return it.tag !== 'I2V';
    return it.tag === filter;
  }
  function buildWall(id, list, kind) {
    const wall = document.getElementById(id);
    wall.innerHTML = '';
    const items = list.filter(matches);
    let n = 0;
    [['480', '480×832×81 · 4.3k tokens'], ['736', '736×1280×81 · 10.1k tokens']].forEach(([res, label]) => {
      const group = items.filter((it) => (it.res === '736' ? '736' : '480') === res);
      if (!group.length) return;
      const head = document.createElement('div'); head.className = 'wall-label'; head.textContent = label; wall.appendChild(head);
      const grid = document.createElement('div'); grid.className = 'wall-grid r' + res; wall.appendChild(grid);
      group.forEach((it) => {
        n += 1;
        const t = document.createElement('button');
        t.type = 'button'; t.className = 'tile';
        t.setAttribute('aria-label', (kind === 'solo' ? 'Enlarge ' : 'Compare ') + n + ': ' + disp(it.prompt));
        const v = makeVideo(it.grace, false, it.poster); t.appendChild(v);
        t.insertAdjacentHTML('beforeend', '<span class="t-tag"></span>' + (kind === 'two' ? '<span class="t-vs">vs Wan2.1-14B</span>' : ''));
        t.querySelector('.t-tag').textContent = it.tag + ' · ' + (res === '736' ? '736p' : '480p');
        const enter = () => {
          if (v.readyState >= 3) v.play().catch(() => {});
          else {                                   // nothing buffered yet, so play once it arrives
            if (v.preload === 'none') { v.preload = 'metadata'; try { v.load(); } catch (_) {} }
            v.addEventListener('canplay', () => v.play().catch(() => {}), { once: true });
          }
          show(kind, it, t, wall);
        };
        t.addEventListener('mouseenter', enter);
        t.addEventListener('focus', enter);
        t.addEventListener('click', enter);
        // [10-03] the pause on mouseleave is gone: thumbnails keep playing, and ioPlay controls them by visibility alone.
        grid.appendChild(t);
        if (io) io.observe(t); else v.preload = 'metadata';
        if (ioPlay) ioPlay.observe(t); else { v.preload = 'auto'; const r0 = v.play(); if (r0 && r0.catch) r0.catch(() => {}); }
      });
    });
    wall.onmouseleave = () => close();   // passing close directly would hand the MouseEvent to the keepPaused argument
    const c = document.querySelector('[data-count="' + id + '"]'); if (c) c.textContent = items.length + ' videos';
    // the page is split by section, so an empty section hides its heading too; it used to read "0 videos"
    wall.dataset.empty = items.length === 0 ? '1' : '0';
    const sec = wall.closest('.g-section'); if (sec) sec.hidden = items.length === 0;
  }
  function applySection() {
    document.querySelectorAll('.g-section').forEach((sec) => {
      const w = sec.querySelector('.wall');
      if (!w) return;
      // leave a section that is already hidden for being empty; buildWall sets hidden
      if (w.dataset.empty === '1') { sec.hidden = true; return; }
      sec.hidden = !(section === 'all' || w.id === section);
    });
    // [10-03] combining a tag with a section can select nothing, for example Styles plus vs Wan.
    //   An unexplained empty page reads as a fault, so say what happened.
    const any = Array.from(document.querySelectorAll('.g-section')).some((x) => !x.hidden);
    const e = document.getElementById('g-empty');
    if (e) e.hidden = any;
  }
  function buildAll() {
    close();
    if (io) io.disconnect();   // the previous page's tiles are gone, so stop observing them
    if (ioPlay) ioPlay.disconnect();
    buildWall('wall-vs', G.vsWan, 'two');
    buildWall('wall-more', G.moreBaselines, 'three');
    buildWall('wall-solo', G.samples, 'solo');
    applySection();
  }

  document.querySelectorAll('.filter').forEach((b) => b.addEventListener('click', () => {
    filter = b.dataset.f;
    document.querySelectorAll('.filter').forEach((x) => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
    buildAll();
  }));
  document.querySelectorAll('.sfilter').forEach((b) => b.addEventListener('click', () => {
    const on = b.getAttribute('aria-pressed') === 'true';   // pressing the same button again goes back to everything
    section = on ? 'all' : b.dataset.s;
    document.querySelectorAll('.sfilter').forEach((x) => x.setAttribute('aria-pressed', (!on && x === b) ? 'true' : 'false'));
    applySection();
  }));
  // [10-03] a last safety net: every time scrolling settles, restart any tile that is visible but
  //   paused. It recovers from a play() that was silently refused (a mobile decoder limit) or from
  //   a pause that arrived by some other route.
  let sweepAt = 0;
  window.addEventListener('scroll', () => {
    const t = Date.now(); if (t - sweepAt < 600) return; sweepAt = t;
    if (!open) resumeWall();
  }, { passive: true });

  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  document.addEventListener('click', (e) => { if (!e.target.closest('.tile')) close(); });
  buildAll();
})();
