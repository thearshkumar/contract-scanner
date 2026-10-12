/* A decorative glass contract page that turns slowly, with a second sheet behind it. The page's faces are cut into
   horizontal clause strips. While a scan runs a glow band sweeps down the strips and they drift apart a little; when
   it finishes, a share of the strips (in proportion to what was flagged) light up in the risk colour, then everything
   settles back. Ported from the search.arshkumar.com background (same raw WebGL, no dependencies); colours follow
   the theme tokens.

   Events (window):  ak:scan-start  ak:scan-progress {p, flagged}  ak:scan-done {flagged, total}  ak:error  ak:theme */
(function () {
  'use strict';
  var root = document.documentElement;
  var canvas = document.getElementById('bg');
  if (!canvas) return;
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  var gl = canvas.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: true });
  if (!gl) { canvas.style.display = 'none'; return; }

  var clamp = function (v, a, b) { return v < a ? a : v > b ? b : v; };
  var smooth = function (x) { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };

  /* ---------- colours (theme tokens) ---------- */
  function hex(v) {
    v = v.trim().replace('#', '');
    if (v.length === 3) v = v[0] + v[0] + v[1] + v[1] + v[2] + v[2];
    return [parseInt(v.slice(0, 2), 16) / 255, parseInt(v.slice(2, 4), 16) / 255, parseInt(v.slice(4, 6), 16) / 255];
  }
  var colOx, colCo, colRisk;
  function readColors() {
    var cs = getComputedStyle(root);
    colOx = hex(cs.getPropertyValue('--oxide') || '#3ddc84'); colCo = hex(cs.getPropertyValue('--cobalt') || '#6b8cff');
    colRisk = hex(cs.getPropertyValue('--risk') || '#e8705a');
  }
  readColors();

  /* ---------- matrices (column-major) ---------- */
  function mul(a, b) {
    var o = new Array(16);
    for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) {
      var s = 0; for (var k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  }
  function rotX(a) { var c = Math.cos(a), s = Math.sin(a); return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]; }
  function rotY(a) { var c = Math.cos(a), s = Math.sin(a); return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]; }
  var CAM = 9, FOV = 0.61;
  function persp(asp) {
    var f = 1 / Math.tan(FOV / 2), n = 1, fa = 40;
    return [f / asp, 0, 0, 0, 0, f, 0, 0, 0, 0, (fa + n) / (n - fa), -1, 0, 0, 2 * fa * n / (n - fa), 0];
  }

  /* ---------- geometry: a page slab whose faces are cut into G horizontal strips ---------- */
  function Builder() { this.t = []; this.l = []; }
  Builder.prototype.tri = function (a, b, c, id, n) {
    this.t.push(a[0], a[1], a[2], n[0], n[1], n[2], id, b[0], b[1], b[2], n[0], n[1], n[2], id, c[0], c[1], c[2], n[0], n[1], n[2], id);
  };
  Builder.prototype.line = function (a, b, id, n) {
    this.l.push(a[0], a[1], a[2], n[0], n[1], n[2], id, b[0], b[1], b[2], n[0], n[1], n[2], id);
  };
  Builder.prototype.quad = function (q, id, n, edges) {
    this.tri(q[0], q[1], q[2], id, n); this.tri(q[0], q[2], q[3], id, n);
    for (var k = 0; k < 4; k++) if (edges !== false || k % 2 === 0) this.line(q[k], q[(k + 1) % 4], id, n);
  };
  /* half-sizes X, Y, Z; o = offset baked into positions; ids run 0 (top) .. 1 (bottom) so the wave sweeps downwards.
     A clause strip's front, back and side facets share one id, so they glow together. */
  function buildPage(X, Y, Z, G, o) {
    var b = new Builder(), i;
    var P = function (x, y, z) { return [x + o[0], y + o[1], z + o[2]]; };
    for (i = 0; i < G; i++) {
      var y1 = Y - 2 * Y * i / G, y0 = Y - 2 * Y * (i + 1) / G, id = (i + 0.5) / G;
      var gx = G > 1 ? 0.012 : 0;                                       /* a hair of inset so the strips read as lines of text */
      b.quad([P(-X + gx, y0, Z), P(X - gx, y0, Z), P(X - gx, y1, Z), P(-X + gx, y1, Z)], id, [0, 0, 1]);
      b.quad([P(X - gx, y0, -Z), P(-X + gx, y0, -Z), P(-X + gx, y1, -Z), P(X - gx, y1, -Z)], id, [0, 0, -1]);
      b.quad([P(X, y0, Z), P(X, y0, -Z), P(X, y1, -Z), P(X, y1, Z)], id, [1, 0, 0], false);
      b.quad([P(-X, y0, -Z), P(-X, y0, Z), P(-X, y1, Z), P(-X, y1, -Z)], id, [-1, 0, 0], false);
    }
    b.quad([P(-X, Y, Z), P(X, Y, Z), P(X, Y, -Z), P(-X, Y, -Z)], 0, [0, 1, 0]);
    b.quad([P(-X, -Y, -Z), P(X, -Y, -Z), P(X, -Y, Z), P(-X, -Y, Z)], 0.999, [0, -1, 0]);
    return b;
  }

  /* ---------- shaders ---------- */
  var VS = 'attribute vec3 aPos;attribute vec3 aNrm;attribute float aId;' +
    'uniform mat4 uM;uniform mat4 uVP;uniform float uScale;uniform float uExplode;uniform vec2 uJ;' +
    'varying vec3 vN;varying float vId;' +
    'void main(){' +
    'float f=clamp(uExplode*1.7-aId*0.7,0.0,1.0);f=f*f*(3.0-2.0*f);' +      /* per-strip stagger: top clauses leave first */
    'vec3 p=aPos*uScale;' +
    'float side=fract(sin(aId*91.7)*437.5)*2.0-1.0;' +                      /* each strip slides out sideways a little... */
    'p.x+=side*f*0.35;p.z+=sign(aPos.z)*f*0.12;p.y+=(0.5-aId)*f*0.5;' +     /* ...and the strips spread apart vertically */
    'vec4 w=uM*vec4(p,1.0);vN=mat3(uM)*aNrm;vId=aId;' +
    'vec4 cl=uVP*w;cl.xy+=uJ*cl.w;gl_Position=cl;}';
  var FS = 'precision mediump float;varying vec3 vN;varying float vId;' +
    'uniform vec3 uCol;uniform vec3 uCol2;uniform vec3 uCol3;uniform float uA;uniform float uRim;uniform float uEnergy;uniform vec2 uWave;uniform float uDim;uniform float uLit;' +
    'void main(){vec3 n=normalize(vN);' +
    'float rim=pow(1.0-abs(n.z),2.0);' +
    'float light=0.5+0.5*abs(dot(n,normalize(vec3(-0.4,0.7,0.6))));' +
    'float d=(vId-uWave.x);d-=floor(d+0.5);' +
    'float wv=uWave.y>0.0?exp(-d*d/(uWave.y*uWave.y)):0.0;' +                /* the scan band sweeping down the strips */
    'float lit=uLit>0.0?step(fract(sin(vId*127.1+3.1)*311.7),uLit):0.0;' +                   /* flagged clauses: a scattered share of strips */
    'float g=clamp(uEnergy+wv*0.8,0.0,1.0);' +
    'float a=uA*(0.55+0.45*light)+rim*uRim+g*0.45+lit*0.4;' +
    'a=clamp(a,0.0,1.0)*uDim;vec3 col=mix(mix(uCol,uCol2,g),uCol3,lit);' +
    'gl_FragColor=vec4(col*a,a);}';

  function sh(type, src) {
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { console.error(gl.getShaderInfoLog(s)); }
    return s;
  }
  var prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { canvas.style.display = 'none'; return; }
  gl.useProgram(prog);
  var U = {};
  ['uM', 'uVP', 'uScale', 'uExplode', 'uJ', 'uCol', 'uCol2', 'uCol3', 'uA', 'uRim', 'uEnergy', 'uWave', 'uDim', 'uLit'].forEach(function (n) { U[n] = gl.getUniformLocation(prog, n); });
  var A = { pos: gl.getAttribLocation(prog, 'aPos'), nrm: gl.getAttribLocation(prog, 'aNrm'), id: gl.getAttribLocation(prog, 'aId') };
  gl.enableVertexAttribArray(A.pos); gl.enableVertexAttribArray(A.nrm); gl.enableVertexAttribArray(A.id);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.disable(gl.DEPTH_TEST);

  function upload(b) {
    var mk = function (arr) {
      var buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW);
      return { buf: buf, n: arr.length / 7 };
    };
    return { t: mk(b.t), l: mk(b.l) };
  }
  var page = upload(buildPage(0.52, 0.7, 0.045, 12, [0, 0, 0]));
  var sheet = upload(buildPage(0.52, 0.7, 0.02, 1, [0.14, -0.12, -0.22]));

  /* ---------- state ---------- */
  var st = { active: false, tStart: 0, res: null, workAmt: 0, prog: 0, flagged: 0, pulse: 0 };
  var mx = 0, my = 0, tx = 0, ty = 0;
  var HOLD = 3.5, BACK = 1.2;                                           /* seconds the result stays lit, then eases back */

  window.addEventListener('ak:scan-start', function () { st.tStart = performance.now(); st.res = null; st.active = true; st.prog = 0; st.flagged = 0; });
  window.addEventListener('ak:scan-progress', function (e) {
    var d = e.detail || {};
    st.prog = clamp(d.p || 0, 0, 1);
    if ((d.flagged || 0) > st.flagged) st.pulse = 1;                    /* a soft pulse each time new clauses are found */
    st.flagged = d.flagged || 0;
  });
  window.addEventListener('ak:scan-done', function (e) {
    if (!st.active) return;
    var d = e.detail || {}, f = d.flagged || 0, n = Math.max(1, d.total || 1);
    st.res = { start: Math.max(performance.now(), st.tStart + 1500), lit: f ? clamp(f / (n * 3), 0.15, 0.6) : 0, error: false };
  });
  window.addEventListener('ak:error', function () { if (st.active) st.res = { start: performance.now(), lit: 0, error: true }; });
  window.addEventListener('ak:theme', function () { readColors(); if (reduce) drawStill(); });
  window.addEventListener('pointermove', function (e) { tx = e.clientX / innerWidth - 0.5; ty = e.clientY / innerHeight - 0.5; }, { passive: true });

  /* ---------- sizing ---------- */
  var W = 1, H = 1, asp = 1;
  function resize() {
    var dpr = Math.min(devicePixelRatio || 1, 2);
    W = canvas.clientWidth || innerWidth; H = canvas.clientHeight || innerHeight; asp = W / H;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    if (reduce) drawStill();
  }
  if ('ResizeObserver' in window) new ResizeObserver(resize).observe(canvas); else addEventListener('resize', resize);

  /* ---------- drawing ---------- */
  function bindMesh(m) {
    gl.bindBuffer(gl.ARRAY_BUFFER, m.buf);
    gl.vertexAttribPointer(A.pos, 3, gl.FLOAT, false, 28, 0);
    gl.vertexAttribPointer(A.nrm, 3, gl.FLOAT, false, 28, 12);
    gl.vertexAttribPointer(A.id, 1, gl.FLOAT, false, 28, 24);
  }
  var jit = 0.6 * 2 / 1000;
  function drawObj(mesh, M, o) {
    gl.uniformMatrix4fv(U.uM, false, new Float32Array(M));
    gl.uniform1f(U.uScale, o.scale); gl.uniform1f(U.uExplode, o.explode || 0);
    gl.uniform3fv(U.uCol, colOx); gl.uniform3fv(U.uCol2, colCo); gl.uniform3fv(U.uCol3, colRisk);
    gl.uniform1f(U.uEnergy, o.energy); gl.uniform2f(U.uWave, o.wave[0], o.wave[1]);
    gl.uniform1f(U.uLit, o.lit || 0);
    bindMesh(mesh.t);
    gl.uniform1f(U.uA, o.fa); gl.uniform1f(U.uRim, 0.22); gl.uniform2f(U.uJ, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, mesh.t.n);
    bindMesh(mesh.l);
    gl.uniform1f(U.uA, o.la); gl.uniform1f(U.uRim, 0.2);
    gl.uniform2f(U.uJ, 0, 0); gl.drawArrays(gl.LINES, 0, mesh.l.n);
    gl.uniform2f(U.uJ, jit / asp, 0); gl.drawArrays(gl.LINES, 0, mesh.l.n);
  }

  var lastT = performance.now(), tAbs = 0, raf = 0, ang = 0.5, sweep = 0;

  function render(dt, now, still) {
    var narrow = W < 900 || asp < 1.1;
    var res = st.res, r = (res && now >= res.start) ? (now - res.start) / 1000 : -1;
    var relT = res && r > HOLD ? (r - HOLD) / BACK : 0;                  /* 0..1 while the page settles back together */
    var tw = (now - st.tStart) / 1000, err = res && res.error;

    if (!still) {
      if (st.active) {
        if (r < 0) st.workAmt += (1 - st.workAmt) * Math.min(1, dt * 5);
        else st.workAmt = 1 - smooth(err ? r / BACK : relT);
        if (r >= 0 && (err ? r >= BACK : relT >= 1)) { st.active = false; st.res = null; st.workAmt = 0; }
      } else st.workAmt = 0;
    }
    var wa = still ? 0 : st.workAmt;
    var scanning = !still && st.active && r < 0;
    var ex = smooth(tw / 1.2) * wa * (scanning ? 0.55 : 0.35);          /* strips drift apart while scanning, close up on resolve */
    var lit = (!still && st.active && r >= 0 && !err) ? res.lit * (1 - smooth(relT)) * smooth(r / 0.4) : 0;
    var flash = (!still && st.active && r >= 0 && !err) ? Math.exp(-r * 3) : 0;

    // camera / rig (same framing as the search background: object on the right on wide screens)
    mx += (tx - mx) * Math.min(1, dt * 3); my += (ty - my) * Math.min(1, dt * 3);
    var halfH = CAM * Math.tan(FOV / 2), halfW = halfH * asp;
    var shift = narrow ? 0 : halfW * 0.5;
    var rigS = narrow ? Math.min(0.9, halfW / 2.3) : Math.min(0.95, halfW * 0.5 / 1.8);
    var VP = mul(persp(asp), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, shift, narrow ? halfH * 0.18 : 0, -CAM, 1]);
    var rig = mul(mul(rotX(0.18 + my * 0.4), rotY(0.35 * mx)), [rigS, 0, 0, 0, 0, rigS, 0, 0, 0, 0, rigS, 0, 0, 0, 0, 1]);

    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniformMatrix4fv(U.uVP, false, new Float32Array(VP));
    gl.uniform1f(U.uDim, narrow ? 0.55 : 1);

    // slow turn that speeds up smoothly while scanning (integrated, so speed changes never jump)
    if (!still) { ang += dt * (0.2 + 0.5 * ex); sweep += dt * (scanning ? 0.5 + 0.4 * st.prog : 0.12); }
    var jitter = err && r >= 0 ? 0.08 * Math.sin(40 * tAbs) * Math.exp(-4 * r) : 0;
    var M = mul(rig, mul(rotX(0.12 + jitter), rotY(ang)));

    st.pulse *= Math.exp(-dt * 2.5);
    var e = clamp(0.2 * ex + 0.6 * flash + 0.35 * st.pulse, 0, 1);

    drawObj(sheet, M, { scale: 1, explode: 0, energy: clamp(0.15 + 0.4 * st.prog * wa + 0.3 * flash, 0, 1),
      wave: [0, 0], fa: 0.03, la: 0.22 });
    drawObj(page, M, { scale: 1 + 0.03 * flash, explode: ex, energy: e,
      wave: [sweep % 1, scanning ? 0.06 : 0.035], lit: lit, fa: 0.045, la: 0.38 });
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    var dt = Math.min(0.05, (now - lastT) / 1000); lastT = now; tAbs += dt;
    render(dt, now, false);
  }
  function drawStill() { render(0.016, performance.now(), true); }

  resize();
  if (reduce) { drawStill(); return; }
  raf = requestAnimationFrame(frame);
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) { cancelAnimationFrame(raf); raf = 0; } else if (!raf) { lastT = performance.now(); raf = requestAnimationFrame(frame); }
  });
})();
