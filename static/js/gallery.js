(function () {
  const G = window.GRACE.gallery;
  // [10-03] VBench 공식 캡션은 공간 관계 28편이 ', front view' 로 끝난다. 벤치마크 표기일 뿐이라
  //   화면에서는 떼고 보여준다. 데이터의 prompt 는 그대로 둔다 — 픽·매칭·파일명이 전부 그 문자열을 키로 쓴다.
  const disp = (t) => String(t).replace(/,\s*front view\s*$/i, '');
  function hlStyle(text) {
    var esc = String(text).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; });
    var re = /(in the (?:iconic |distinctive |signature |classic )?style of [^,.;…”]+|Van Gogh[- ]?(?:style|inspired|esque)?[^,.;…”]*|Hokusai[^,.;…”]*|Ukiyo-e[^,.;…”]*|pixel art[^,.;…”]*|oil painting[^,.;…”]*|watercolor[^,.;…”]*|black and white[^,.;…”]*|cyberpunk[^,.;…”]*|surrealist[^,.;…”]*|surrealism[^,.;…”]*|impressionist[^,.;…”]*|anime[- ]style[^,.;…”]*|cartoon[- ]style[^,.;…”]*|Picasso[^,.;…”]*|Monet[^,.;…”]*|pencil sketch[^,.;…”]*|charcoal[^,.;…”]*|retro[- ]style[^,.;…”]*|vintage[- ]style[^,.;…”]*)/gi;
    return esc.replace(re, '<em class="style">$1</em>');
  }

  let filter = 'All';
  let section = 'all';   // 섹션 축은 태그 필터와 독립이다 — 둘 다 걸 수 있다

  let open = null; // { wall, tile, els, stop }


  // 숫자 카운트업: 같은 '초/ms' 속도로 올라가 적은 쪽이 먼저 멈춘다 (우리가 빠른 느낌)
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

  function makeVideo(src, autoplay, poster) {
    const v = document.createElement('video');
    v.muted = true; v.loop = true; v.playsInline = true; v.preload = autoplay ? 'auto' : 'none';
    if (autoplay) v.autoplay = true;
    // [10-03] 포스터(첫 프레임)를 깔아둔다. 폰에서는 데이터 절약으로 preload 가 무시되거나
    //   iOS 가 동시 디코더 수를 제한해 영상이 안 뜨는데, 그때 타일이 회색 사각형으로 남았다.
    if (poster) v.poster = poster;
    v.addEventListener('error', () => { v.style.visibility = 'hidden'; });
    v.src = src;
    return v;
  }
  // [10-03] 영상이 아직 다 안 받아졌을 때 마우스를 올리면 화면이 '지지직' 떨리던 문제.
  //   원인은 이 루프가 매 프레임(60Hz) currentTime 을 다시 쓰던 것. 버퍼가 모자라면 그 대입이
  //   곧바로 seek 를 걸고, seek 가 끝나기 전에 다음 프레임이 또 걸어서 탐색이 끝없이 재시작한다.
  //   고친 점 셋: ① 4Hz 로만 맞춘다 ② seeking 중이면 건너뛴다 ③ 재생 가능한 수준(readyState 3)만 건드린다.
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

  // [10-02] 썸네일이 늦게 뜨던 원인: 타일 수백 개가 모두 preload 요청을 걸어 큐가 막혔다.
  //   화면에 보이는(또는 곧 보일) 타일만 첫 프레임을 받게 한다.
  const primed = new WeakSet();
  const io = ('IntersectionObserver' in window) ? new IntersectionObserver((es) => {
    es.forEach((e) => {
      if (!e.isIntersecting) return;
      const v = e.target.querySelector('video');
      if (v && !primed.has(v)) { primed.add(v); v.preload = 'metadata'; try { v.load(); } catch (_) {} }
      io.unobserve(e.target);
    });
  }, { rootMargin: '800px 0px' }) : null;

  // [10-03] 썸네일을 계속 재생한다. 수백 개를 다 틀면 디코더가 못 버티므로 '보이는 동안만' 튼다.
  //   벗어나면 pause — 마지막 프레임이 그대로 남아 멈춰도 빈칸으로 보이지 않는다.
  //   preload 는 여기서 'auto' 로 올린다(위 관찰자는 첫 프레임용 metadata 까지만).
  const ioPlay = ('IntersectionObserver' in window) ? new IntersectionObserver((es) => {
    es.forEach((e) => {
      const v = e.target.querySelector('video'); if (!v) return;
      if (e.isIntersecting) {
        if (v.preload !== 'auto') { v.preload = 'auto'; try { v.load(); } catch (_) {} }
        const r = v.play(); if (r && r.catch) r.catch(() => {});
      } else { v.pause(); }
    });
  }, { rootMargin: '150px 0px' }) : null;

  // [10-03] 썸네일이 상시 재생이라, 팝업이 떠 있는 동안 뒤 타일까지 돌면 팝업이 끊긴다.
  //   팝업을 열 때 멈추고 닫을 때 되살린다. 되살리는 대상은 지금 화면에 보이는 타일뿐.
  // '무엇을 멈췄는지' 목록을 들고 다니지 않는다. 목록 방식은 타일 사이를 옮겨 다닐 때
  //   두 번째 pauseWall 이 빈 목록으로 덮어써 복구가 영영 안 되는 버그를 냈다(10-03).
  //   대신 멈출 땐 그 벽 전부, 되살릴 땐 '지금 화면에 보이는 것 전부'로 상태를 다시 계산한다.
  //   ioPlay 는 가시성이 '바뀔 때'만 동작하므로, 수동으로 멈춘 타일은 여기서 되살려야 한다.
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
      // 토큰 수 = 잠재 프레임 × (H/f/p) × (W/f/p): Wan f8t4p2, GRACE f16t8p2, DC-Gen f32t4p1 · 81프레임
      // 생성 시간(초)·토큰 — 논문 Quantitative 표와 같은 값. 두 숫자를 같이 보여줘야 '몇 배 빠름' 이 무엇 대비인지 분명해진다.
      const LATG = { '480': { T2V: { wan: 851.5, ours: 75.8, dc: 157.1, ltx: 99.6 }, I2V: { wan: 863.2, ours: 77.7, dc: 165.0, ltx: 104.1 } },
                     '736': { T2V: { wan: 3361.3, ours: 215.6, dc: 456.4, ltx: 264.2 }, I2V: { wan: 3396.8, ours: 218.8, dc: 550.7, ltx: 274.6 } } };
      // [10-07] 배수는 논문·본문과 같은 수치를 쓴다. 계산하면 t2v 가 11.2(851.5/75.8), i2v 가
      //   11.1(863.2/77.7) 로 갈려 같은 사이트에서 두 값이 보였다. 논문 abstract 가 I2V 기준
      //   11.1x / 15.5x 를 쓰므로 거기에 맞춘다 (t2v 카드는 표시 초수와 0.1 차이가 난다).
      const FASTG = { '480': '11.1\u00d7', '736': '15.5\u00d7' };
      const TOKG = { '480': { wan: '32.8k', ours: '4.3k', dc: '8.2k', ltx: '4.3k' }, '736': { wan: '77.3k', ours: '10.1k', dc: '19.3k', ltx: '10.1k' } };
      const rs = it.res === '736' ? '736' : '480', tsk = it.tag === 'I2V' ? 'I2V' : 'T2V', L = LATG[rs][tsk], T = TOKG[rs];
      const sec = (v) => Math.round(v).toLocaleString() + ' s';
      const tn = (x) => parseFloat(x), TP = tn(T.wan);   // 토큰도 Wan 을 기준 속도로 → 우리가 먼저 멈춘다
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
          const ve = document.createElement('span'); ve.className = 'val'; ve.textContent = kv[1];   // ★ 'v' 금지 — 팝업 영상 박스가 .v 라서 querySelector 가 이걸 잡는다
          if (kv[0] === 'Inference time' && cfg.secs) { ve.textContent = ''; countUp(ve, cfg.secs, cfg.peer); }
          if (kv[0] === 'Latent tokens #' && cfg.tnum) { ve.textContent = ''; countUp(ve, cfg.tnum, cfg.tpeer, { dec: 1, suf: 'k' }); }
          d.appendChild(ke); d.appendChild(ve); mt.appendChild(d);
        });
        const v = makeVideo(src, true); vids.push(v); row.querySelector(':scope > .v').appendChild(v);   // 직계 자식만 (이중 안전장치)
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
    // [10-03] 태그가 태스크(T2V/I2V)와 그 외(736×1280·Styles)를 한 축에 섞어 담고 있다.
    //   736×1280 과 Styles 는 전부 T2V 라서, T2V 버튼이 그것들까지 세야 숫자가 맞는다
    //   (안 그러면 실제 196편 중 86편만 보인다). Styles 버튼은 그대로 스타일만 고른다.
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
          else {                                   // 아직 버퍼가 없으면 받아진 뒤에 튼다
            if (v.preload === 'none') { v.preload = 'metadata'; try { v.load(); } catch (_) {} }
            v.addEventListener('canplay', () => v.play().catch(() => {}), { once: true });
          }
          show(kind, it, t, wall);
        };
        t.addEventListener('mouseenter', enter);
        t.addEventListener('focus', enter);
        t.addEventListener('click', enter);
        // [10-03] mouseleave 에서 pause 하던 줄 삭제 — 썸네일은 계속 재생된다(ioPlay 가 가시성으로만 제어).
        grid.appendChild(t);
        if (io) io.observe(t); else v.preload = 'metadata';
        if (ioPlay) ioPlay.observe(t); else { v.preload = 'auto'; const r0 = v.play(); if (r0 && r0.catch) r0.catch(() => {}); }
      });
    });
    wall.onmouseleave = () => close();   // ★ close 를 그대로 넘기면 MouseEvent 가 keepPaused 인자로 들어간다
    const c = document.querySelector('[data-count="' + id + '"]'); if (c) c.textContent = items.length + ' videos';
    // 장을 섹션별로 나눴으므로 비어 있는 섹션은 제목째 숨긴다 (예전엔 "0 videos" 로 남았다)
    wall.dataset.empty = items.length === 0 ? '1' : '0';
    const sec = wall.closest('.g-section'); if (sec) sec.hidden = items.length === 0;
  }
  function applySection() {
    document.querySelectorAll('.g-section').forEach((sec) => {
      const w = sec.querySelector('.wall');
      if (!w) return;
      // 비어서 이미 숨긴 섹션은 그대로 둔다 (buildWall 이 hidden 을 세운다)
      if (w.dataset.empty === '1') { sec.hidden = true; return; }
      sec.hidden = !(section === 'all' || w.id === section);
    });
    // [10-03] 태그와 섹션을 같이 걸면 0 편인 조합이 생긴다(예: Styles + vs Wan).
    //   그때 아무 설명 없이 빈 화면이 되면 고장으로 보인다 — 안내를 띄운다.
    const any = Array.from(document.querySelectorAll('.g-section')).some((x) => !x.hidden);
    const e = document.getElementById('g-empty');
    if (e) e.hidden = any;
  }
  function buildAll() {
    close();
    if (io) io.disconnect();   // 이전 장의 타일은 버려졌으므로 관찰 해제
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
    const on = b.getAttribute('aria-pressed') === 'true';   // 같은 버튼을 다시 누르면 전체로 돌아간다
    section = on ? 'all' : b.dataset.s;
    document.querySelectorAll('.sfilter').forEach((x) => x.setAttribute('aria-pressed', (!on && x === b) ? 'true' : 'false'));
    applySection();
  }));
  // [10-03] 마지막 안전망: 스크롤이 멈출 때마다 '보이는데 멈춰 있는' 타일을 다시 튼다.
  //   play() 가 조용히 거부당했거나(모바일 디코더 한도) 어떤 경로로 멈춰 있어도 여기서 복구된다.
  let sweepAt = 0;
  window.addEventListener('scroll', () => {
    const t = Date.now(); if (t - sweepAt < 600) return; sweepAt = t;
    if (!open) resumeWall();
  }, { passive: true });

  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  document.addEventListener('click', (e) => { if (!e.target.closest('.tile')) close(); });
  buildAll();
})();
