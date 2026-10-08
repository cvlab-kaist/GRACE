(function () {
  const D = window.GRACE;

  // 생성 시간(초) — 아래 Quantitative 표와 같은 값. '몇 배 빠름' 은 이 둘의 비율이라 두 숫자를 같이 보여준다.
  const LAT = { '480': { T2V: { wan: 851.5, ours: 75.8, dc: 157.1, ltx: 99.6 }, I2V: { wan: 863.2, ours: 77.7, dc: 165.0, ltx: 104.1 } },
                '736': { T2V: { wan: 3361.3, ours: 215.6, dc: 456.4, ltx: 264.2 }, I2V: { wan: 3396.8, ours: 218.8, dc: 550.7, ltx: 274.6 } } };
  const TOK = { '480': { wan: '32.8k', ours: '4.3k', dc: '8.2k', ltx: '4.3k' }, '736': { wan: '77.3k', ours: '10.1k', dc: '19.3k', ltx: '10.1k' } };
  const secs = (v) => Math.round(v).toLocaleString() + ' s';
  // [10-08] 배수는 논문 abstract(I2V 기준 11.1x / 15.5x)와 같은 값으로 고정한다.
  //   나누면 t2v 가 11.2(851.5/75.8) · 15.6(3361.3/215.6) 으로 갈려 같은 페이지에 두 값이 보였다.
  //   gallery.js 의 FASTG 와 같은 상수다 — 한쪽만 고치면 또 갈린다.
  const FASTG = { '480': '11.1\u00d7', '736': '15.5\u00d7' };
  const ratio = (r, k) => FASTG[r];


  // 숫자 카운트업: 두 모델을 같은 '초/ms' 속도로 올려서 적은 쪽이 먼저 탁 멈추게 (우리가 빠른 느낌)
  const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ROLLS = [2.6, 1.7, 1.1, 0.8];   // 오른쪽 자리일수록 많이 돈다 — 실제 적산계(odometer) 느낌
  // 숫자 오도미터: 자릿수 휠이 아래→위로 슬라이드되며 값이 커지고 최종값에서 멈춘다.
  //   opt = { dec: 소수점 자릿수, suf: 접미사 }  (기본 '초')
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
    const top = Math.max(target, peer || target), FULL = 4000;   // [10-02] 1500→4000: 천천히 올라야 속도 차이가 눈에 보인다
    const dur = Math.max(600, FULL * (target / top));   // 같은 '초/ms' 속도 → 적은 쪽이 먼저 멈춘다
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

  // [10-03] VBench 공식 캡션은 공간 관계 28편이 ', front view' 로 끝난다. 벤치마크 표기일 뿐이라
  //   화면에서는 떼고 보여준다. 데이터의 prompt 는 그대로 둔다 — 픽·매칭·파일명이 전부 그 문자열을 키로 쓴다.
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
      // 'T2V 480×832 · caption' → 태스크/해상도 칩 + 입력 프롬프트
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
      countUp(q('.tag-left .lat'), L.wan, L.wan);     // 둘 다 Wan 값을 기준 속도로 → ours 가 먼저 멈춘다
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
    // [10-02] 고르기 전에 무엇인지 알 수 있도록 Image-to-video / Text-to-video 로 묶어 보여준다. I2V 가 먼저.
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
    // 첫 화면은 'A person is filling eyebrows' (캡션으로 찾아 순서가 바뀌어도 유지)
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
      // [10-03] 배지 자리는 **영상 위쪽 띠**다. 제목 줄에 두면 우리 칸만 높아져 줄이 어긋나고,
      //   영상 위에 얹으면 화면을 가린다. 그래서 모든 칸에 같은 높이의 띠를 두고 우리 칸만 채운다.
      const strip = document.createElement('div'); strip.className = 'badge-row';
      if (spec.fast) {
        const st = document.createElement('span'); st.className = 'stat';
        const em = document.createElement('em'); em.textContent = spec.fast;
        st.appendChild(em); const lb = document.createElement('span'); lb.className = 'lbl'; lb.textContent = 'faster inference'; st.appendChild(lb);
        strip.appendChild(st);
      }
      l.appendChild(tl);
      if (spec.inf) {   // 수치는 한 줄(라벨+값 ×2) — 칸처럼 보이지 않게 낮게
        const mt = document.createElement('div'); mt.className = 'metrics';
        [['Inference time', spec.inf], ['Latent tokens #', spec.tok]].forEach(function (kv) {
          const pr = document.createElement('span'); pr.className = 'm';   // 라벨+값은 한 덩어리로 (중간 줄바꿈 금지)
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
      if (tab === 'i2v') cells.push(cell('Input frame', item.input, false, hr, true));
      const rs = hr ? '736' : '480', tk = tab === 'i2v' ? 'I2V' : 'T2V', L = LAT[rs][tk];
      const TN = { wan: parseFloat(TOK[rs].wan), ours: parseFloat(TOK[rs].ours), dc: parseFloat(TOK[rs].dc) };
      cells.push(cell({ name: 'Wan2.1-14B', inf: secs(L.wan), tok: TOK[rs].wan, secs: L.wan, peer: L.wan, tnum: TN.wan, tpeer: TN.wan }, item.wan, false, hr));
      // [10-08] LTX 열은 항목에 ltx 가 있을 때만 — 기존 탭은 3열 그대로다.
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

  /* [10-03] 오프닝 필름 — 소리 끈 자동재생 + '소리 켜기' 버튼.
     브라우저가 소리 켜진 자동재생을 막으므로 muted 로 시작하고, 사용자가 누르면 켠다.
     play() 를 한 번 더 부르는 이유: 탭이 백그라운드였거나 poster 로드가 늦으면 autoplay 가 걸리지 않는다. */
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
