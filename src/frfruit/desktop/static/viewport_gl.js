/* Pure WebGL2 Z-Up Interactive 3D Viewport Engine (Offline, Zero External Deps) */

(function (global) {
  "use strict";

  function normalize(v) {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  }

  function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  function cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function mat4Perspective(fovyRad, aspect, near, far) {
    const f = 1.0 / Math.tan(fovyRad * 0.5);
    const nf = 1.0 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / Math.max(0.01, aspect);
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1.0;
    out[14] = 2.0 * far * near * nf;
    return out;
  }

  function mat4LookAt(eye, target, up) {
    const f = normalize(sub(target, eye));
    let r = cross(f, up);
    if (Math.hypot(r[0], r[1], r[2]) < 1e-5) {
      r = cross(f, [0, 1, 0]);
    }
    r = normalize(r);
    const u = cross(r, f);

    const out = new Float32Array(16);
    out[0] = r[0];  out[1] = u[0];  out[2] = -f[0];  out[3] = 0;
    out[4] = r[1];  out[5] = u[1];  out[6] = -f[1];  out[7] = 0;
    out[8] = r[2];  out[9] = u[2];  out[10] = -f[2]; out[11] = 0;
    out[12] = -dot(r, eye);
    out[13] = -dot(u, eye);
    out[14] = dot(f, eye);
    out[15] = 1;
    return out;
  }

  function mat4Mul(a, b) {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        out[col * 4 + row] =
          a[0 * 4 + row] * b[col * 4 + 0] +
          a[1 * 4 + row] * b[col * 4 + 1] +
          a[2 * 4 + row] * b[col * 4 + 2] +
          a[3 * 4 + row] * b[col * 4 + 3];
      }
    }
    return out;
  }

  const MESH_VS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aNorm;
    layout(location = 2) in vec2 aUv;
    uniform mat4 uVP;
    uniform vec2 uUvScale;
    out vec3 vWorldPos;
    out vec3 vNormal;
    out vec2 vUv;
    void main() {
      vWorldPos = aPos;
      vNormal = aNorm;
      vUv = aUv * uUvScale;
      gl_Position = uVP * vec4(aPos, 1.0);
    }
  `;

  const MESH_FS = `#version 300 es
    precision highp float;
    in vec3 vWorldPos;
    in vec3 vNormal;
    in vec2 vUv;

    uniform vec3 uCamPos;
    uniform vec4 uBaseColor;
    uniform float uMetallic;
    uniform float uRoughness;
    uniform vec3 uEmission;
    uniform int uMode; // 0=PBR, 1=Clay, 2=Normals, 3=WireTint
    uniform int uSelected;
    uniform int uHasTex;
    uniform sampler2D uAlbedoTex;
    uniform vec3 uKeyDir;
    uniform vec3 uKeyColor;
    uniform vec3 uSkyColor;

    out vec4 fragColor;

    void main() {
      vec3 N = normalize(gl_FrontFacing ? vNormal : -vNormal);
      if (uMode == 2) {
        fragColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
      }
      vec3 albedo = uBaseColor.rgb;
      if (uHasTex == 1) {
        albedo = texture(uAlbedoTex, vUv).rgb * albedo;
      }
      if (uMode == 1) {
        albedo = vec3(0.68, 0.70, 0.73);
      } else if (uMode == 3) {
        albedo = vec3(0.11, 0.13, 0.16);
      }

      vec3 V = normalize(uCamPos - vWorldPos);
      vec3 L1 = normalize(-uKeyDir);
      vec3 L2 = normalize(vec3(-0.6, 0.5, 0.5));
      float r = clamp(uMode == 1 ? 0.55 : uRoughness, 0.06, 0.99);
      float m = uMode == 1 ? 0.0 : uMetallic;

      // Key & fill diffuse + GGX-like specular highlight
      float ndl1 = max(dot(N, L1), 0.0);
      float ndl2 = max(dot(N, L2), 0.0);
      vec3 H1 = normalize(L1 + V);
      float shin = pow(2.0 / (r * r + 0.001) - 1.0, 1.0);
      float specPow = clamp(shin, 4.0, 256.0);
      float spec1 = pow(max(dot(N, H1), 0.0), specPow) * (0.15 + 0.85 * (1.0 - r));

      vec3 F0 = mix(vec3(0.04), albedo, m);
      vec3 fresnel = F0 + (1.0 - F0) * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);

      // Sky-ground Z-up hemisphere ambient
      float hemi = 0.5 + 0.5 * N.z;
      vec3 ambient = mix(vec3(0.10, 0.11, 0.13), uSkyColor * 0.42, hemi) * albedo * (1.0 - 0.55 * m);

      vec3 diffuse = albedo * (1.0 - m * 0.88) * (uKeyColor * ndl1 * 0.92 + vec3(0.42, 0.52, 0.68) * ndl2 * 0.45);
      vec3 specular = fresnel * uKeyColor * spec1 * (1.2 + 1.5 * m);
      vec3 col = ambient + diffuse + specular + uEmission;

      if (uSelected == 1 && uMode != 3) {
        float rim = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 3.0);
        col = mix(col, vec3(0.24, 0.55, 0.98), rim * 0.42);
      }

      // ACES-like tone curve + sRGB gamma
      col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
      col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
      fragColor = vec4(col, uMode == 0 ? uBaseColor.a : 1.0);
    }
  `;

  const LINE_VS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aColor;
    uniform mat4 uVP;
    out vec3 vColor;
    void main() {
      vColor = aColor;
      gl_PointSize = 7.0;
      gl_Position = uVP * vec4(aPos, 1.0);
    }
  `;

  const LINE_FS = `#version 300 es
    precision highp float;
    in vec3 vColor;
    uniform float uAlpha;
    out vec4 fragColor;
    void main() {
      fragColor = vec4(vColor, uAlpha);
    }
  `;

  function compileShader(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s) || "Shader compile failed");
    }
    return s;
  }

  function createProgram(gl, vsSrc, fsSrc) {
    const p = gl.createProgram();
    gl.attachShader(p, compileShader(gl, gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, compileShader(gl, gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p) || "Program link failed");
    }
    return p;
  }

  class ViewportGL {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
      this.fallback2d = !this.gl;

      this.mode = "pbr";
      this.showGrid = true;
      this.showAxes = true;
      this.showBounds = false;
      this.showBones = true;
      this.selectedId = null;

      // Edit Mode state & element overlays
      this.editMode = false;
      this.editElementMode = "face"; // "face" | "edge" | "vertex"
      this.editData = null;
      this.selectedFaces = new Set();
      this.selectedEdges = new Set();
      this.selectedVertices = new Set();
      this.onViewportClick = null;

      this.target = [0, 0, 0.4];
      this.yaw = Math.PI * 0.28;
      this.pitch = 0.42;
      this.distance = 6.0;
      this.fovDeg = 45;
      this.sceneBounds = { min: [-1, -1, 0], max: [1, 1, 1], center: [0, 0, 0.5], size: [2, 2, 1] };

      this.meshes = [];
      this.armatures = [];
      this.lights = [];
      this.world = { background_color: [0.07, 0.08, 0.10], ambient_strength: 0.35 };
      this.texCache = new Map();

      if (this.gl) {
        this._initGL();
      }
      this._bindPointer();
      this._resize();
      window.addEventListener("resize", () => this._resize());
      this.render();
    }

    _initGL() {
      const gl = this.gl;
      this.meshProg = createProgram(gl, MESH_VS, MESH_FS);
      this.lineProg = createProgram(gl, LINE_VS, LINE_FS);

      // Build persistent 1m XY Grid + Origin Axis lines
      const gridVerts = [];
      const half = 12;
      for (let i = -half; i <= half; i++) {
        if (i === 0) continue;
        const c = (i % 5 === 0) ? 0.25 : 0.16;
        gridVerts.push(i, -half, 0, c, c, c + 0.02, i, half, 0, c, c, c + 0.02);
        gridVerts.push(-half, i, 0, c, c, c + 0.02, half, i, 0, c, c, c + 0.02);
      }
      this.gridCount = gridVerts.length / 6;
      this.gridVao = this._makeLineVao(new Float32Array(gridVerts));

      const axisVerts = new Float32Array([
        -half, 0, 0.001, 0.85, 0.24, 0.24,  half, 0, 0.001, 0.85, 0.24, 0.24,
        0, -half, 0.001, 0.22, 0.76, 0.32,  0, half, 0.001, 0.22, 0.76, 0.32,
        0, 0, 0,         0.25, 0.52, 0.96,  0, 0, 3.0,   0.25, 0.52, 0.96,
      ]);
      this.axisCount = 6;
      this.axisVao = this._makeLineVao(axisVerts);

      this.dynLineVao = gl.createVertexArray();
      this.dynLineBuf = gl.createBuffer();
      gl.bindVertexArray(this.dynLineVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.dynLineBuf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
      gl.bindVertexArray(null);
    }

    _makeLineVao(arr) {
      const gl = this.gl;
      const vao = gl.createVertexArray();
      const buf = gl.createBuffer();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
      gl.bindVertexArray(null);
      return vao;
    }

    _bindPointer() {
      let dragging = false;
      let downOnCanvas = false;
      let downBtn = 0;
      let panMode = false;
      let lx = 0;
      let ly = 0;
      let movedDist = 0;

      this.canvas.addEventListener("mousedown", (e) => {
        dragging = true;
        downOnCanvas = true;
        downBtn = e.button;
        panMode = e.button === 1 || e.button === 2 || e.shiftKey;
        lx = e.clientX;
        ly = e.clientY;
        movedDist = 0;
      });
      this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
      window.addEventListener("mouseup", (e) => {
        if (downOnCanvas && downBtn === 0 && movedDist < 5 && this.onViewportClick) {
          const ray = this.screenPointToRay(e.clientX, e.clientY);
          this.onViewportClick(ray.origin, ray.direction, e);
        }
        dragging = false;
        downOnCanvas = false;
      });
      window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - lx;
        const dy = e.clientY - ly;
        movedDist += Math.hypot(dx, dy);
        lx = e.clientX;
        ly = e.clientY;
        if (panMode) {
          const cy = Math.cos(this.yaw);
          const sy = Math.sin(this.yaw);
          const right = [-sy, cy, 0];
          const up = [0, 0, 1];
          const scale = this.distance * 0.0014;
          this.target[0] -= (right[0] * dx - up[0] * dy) * scale;
          this.target[1] -= (right[1] * dx - up[1] * dy) * scale;
          this.target[2] -= (right[2] * dx - up[2] * dy) * scale;
        } else {
          this.yaw -= dx * 0.0075;
          this.pitch = Math.max(-1.52, Math.min(1.52, this.pitch + dy * 0.0075));
        }
        this.render();
      });
      this.canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const factor = e.deltaY > 0 ? 1.12 : 0.89;
        this.distance = Math.max(0.25, Math.min(250.0, this.distance * factor));
        this.render();
      }, { passive: false });
    }

    screenPointToRay(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      const ndcX = ((clientX - rect.left) / Math.max(1, rect.width)) * 2.0 - 1.0;
      const ndcY = 1.0 - ((clientY - rect.top) / Math.max(1, rect.height)) * 2.0;
      const eye = this._eyePos();
      const f = normalize(sub(this.target, eye));
      let r = cross(f, [0, 0, 1]);
      if (Math.hypot(r[0], r[1], r[2]) < 1e-5) r = cross(f, [0, 1, 0]);
      r = normalize(r);
      const u = cross(r, f);
      const tanHalf = Math.tan((this.fovDeg * Math.PI) / 360.0);
      const aspect = this.canvas.width / Math.max(1, this.canvas.height);
      const dir = normalize([
        f[0] + ndcX * aspect * tanHalf * r[0] + ndcY * tanHalf * u[0],
        f[1] + ndcX * aspect * tanHalf * r[1] + ndcY * tanHalf * u[1],
        f[2] + ndcX * aspect * tanHalf * r[2] + ndcY * tanHalf * u[2],
      ]);
      return { origin: eye, direction: dir };
    }

    _resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(64, Math.floor(rect.width * dpr));
      const h = Math.max(64, Math.floor(rect.height * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      if (this.gl) {
        this.gl.viewport(0, 0, w, h);
      }
      this.render();
    }

    _ensureTexture(mat) {
      if (!this.gl || !mat.has_albedo_map) return null;
      const key = `${mat.id}:${mat.name}`;
      if (this.texCache.has(key)) return this.texCache.get(key);

      const gl = this.gl;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        new Uint8Array([255, 255, 255, 255])
      );
      const rec = { tex, ready: false };
      this.texCache.set(key, rec);

      const img = new Image();
      img.onload = () => {
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        rec.ready = true;
        this.render();
      };
      img.src = `/api/texture_preview?material_id=${encodeURIComponent(mat.id)}&channel=albedo`;
      return rec;
    }

    setGeometry(payload, autoFit = false) {
      this.sceneBounds = payload.scene_bounds || this.sceneBounds;
      this.armatures = payload.armatures || [];
      this.lights = payload.lights || [];
      this.world = payload.world || this.world;

      if (!this.gl) {
        this.render();
        return;
      }
      const gl = this.gl;
      // Dispose old buffers cleanly
      for (const m of this.meshes) {
        gl.deleteVertexArray(m.vao);
        gl.deleteBuffer(m.posBuf);
        gl.deleteBuffer(m.normBuf);
        gl.deleteBuffer(m.uvBuf);
        gl.deleteBuffer(m.idxBuf);
        if (m.wireVao) {
          gl.deleteVertexArray(m.wireVao);
          gl.deleteBuffer(m.wireBuf);
        }
      }
      this.meshes = [];

      for (const md of payload.meshes || []) {
        const pos = new Float32Array(md.positions);
        const norm = new Float32Array(md.normals);
        const uv = new Float32Array(md.uvs);
        const idx = new Uint32Array(md.indices);

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const posBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        const normBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
        gl.bufferData(gl.ARRAY_BUFFER, norm, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

        const uvBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
        gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);

        const idxBuf = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
        gl.bindVertexArray(null);

        // Build wireframe edge positions (stride 6 floats: x,y,z, r,g,b)
        const wireArr = new Float32Array(idx.length * 2 * 6);
        for (let t = 0; t < idx.length; t += 3) {
          const pairs = [[idx[t], idx[t + 1]], [idx[t + 1], idx[t + 2]], [idx[t + 2], idx[t]]];
          for (let p = 0; p < 3; p++) {
            const base = (t * 2 + p * 2) * 6;
            const a = pairs[p][0] * 3;
            const b = pairs[p][1] * 3;
            wireArr[base] = pos[a]; wireArr[base + 1] = pos[a + 1]; wireArr[base + 2] = pos[a + 2];
            wireArr[base + 3] = 0.36; wireArr[base + 4] = 0.65; wireArr[base + 5] = 0.96;
            wireArr[base + 6] = pos[b]; wireArr[base + 7] = pos[b + 1]; wireArr[base + 8] = pos[b + 2];
            wireArr[base + 9] = 0.36; wireArr[base + 10] = 0.65; wireArr[base + 11] = 0.96;
          }
        }
        const wireVao = gl.createVertexArray();
        const wireBuf = gl.createBuffer();
        gl.bindVertexArray(wireVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, wireBuf);
        gl.bufferData(gl.ARRAY_BUFFER, wireArr, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
        gl.bindVertexArray(null);

        this.meshes.push({
          id: md.id,
          name: md.name,
          vao, posBuf, normBuf, uvBuf, idxBuf,
          count: idx.length,
          wireVao, wireBuf,
          wireVertCount: idx.length * 2,
          bounds: md.bounds,
          material: md.material,
        });
      }

      if (autoFit) {
        this.fitScene();
      } else {
        this.render();
      }
    }

    fitScene(targetObjId = null) {
      let b = this.sceneBounds;
      if (targetObjId) {
        const hit = this.meshes.find((m) => m.id === targetObjId);
        if (hit && hit.bounds) b = hit.bounds;
      }
      const c = b.center || [0, 0, 0.5];
      const mn = b.min || [-1, -1, 0];
      const mx = b.max || [1, 1, 1];
      const diag = Math.max(1.2, Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]));
      this.target = [c[0], c[1], c[2]];
      this.distance = diag * 1.45;
      this.render();
    }

    setPresetView(dir) {
      if (dir === "fit") {
        this.fitScene(this.selectedId);
        return;
      }
      if (dir === "front") { this.yaw = -Math.PI * 0.5; this.pitch = 0.04; }
      else if (dir === "right") { this.yaw = 0.0; this.pitch = 0.04; }
      else if (dir === "top") { this.yaw = -Math.PI * 0.5; this.pitch = 1.50; }
      else { this.yaw = Math.PI * 0.28; this.pitch = 0.42; }
      this.fitScene();
    }

    _eyePos() {
      const cp = Math.cos(this.pitch);
      const sp = Math.sin(this.pitch);
      const cy = Math.cos(this.yaw);
      const sy = Math.sin(this.yaw);
      return [
        this.target[0] + this.distance * cp * cy,
        this.target[1] + this.distance * cp * sy,
        this.target[2] + this.distance * sp,
      ];
    }

    render() {
      if (!this.gl) return;
      const gl = this.gl;
      const bg = this.world.background_color || [0.06, 0.07, 0.09];
      gl.clearColor(bg[0] * 0.65, bg[1] * 0.65, bg[2] * 0.65, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      const aspect = this.canvas.width / Math.max(1, this.canvas.height);
      const near = Math.max(0.02, this.distance * 0.005);
      const far = Math.max(120.0, this.distance * 25.0);
      const proj = mat4Perspective((this.fovDeg * Math.PI) / 180, aspect, near, far);
      const eye = this._eyePos();
      const view = mat4LookAt(eye, this.target, [0, 0, 1]);
      const vp = mat4Mul(proj, view);

      // 1. Grid & Axes
      gl.useProgram(this.lineProg);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, "uVP"), false, vp);
      if (this.showGrid) {
        gl.uniform1f(gl.getUniformLocation(this.lineProg, "uAlpha"), 0.7);
        gl.bindVertexArray(this.gridVao);
        gl.drawArrays(gl.LINES, 0, this.gridCount);
      }
      if (this.showAxes) {
        gl.uniform1f(gl.getUniformLocation(this.lineProg, "uAlpha"), 0.9);
        gl.bindVertexArray(this.axisVao);
        gl.drawArrays(gl.LINES, 0, this.axisCount);
      }

      // 2. Meshes
      const modeInt =
        this.mode === "clay" ? 1 :
        this.mode === "normals" ? 2 :
        this.mode === "wireframe" ? 3 : 0;
      const keyLight = this.lights.find((l) => l.kind === "sun") || this.lights[0];
      const keyDir = (keyLight && keyLight.direction) ? keyLight.direction : [-0.55, 0.45, -0.7];
      const keyColor = (keyLight && keyLight.color) ? keyLight.color : [1.0, 0.96, 0.9];

      gl.useProgram(this.meshProg);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.meshProg, "uVP"), false, vp);
      gl.uniform3fv(gl.getUniformLocation(this.meshProg, "uCamPos"), eye);
      gl.uniform3fv(gl.getUniformLocation(this.meshProg, "uKeyDir"), keyDir);
      gl.uniform3fv(gl.getUniformLocation(this.meshProg, "uKeyColor"), keyColor);
      gl.uniform3fv(gl.getUniformLocation(this.meshProg, "uSkyColor"), [0.70, 0.78, 0.92]);
      gl.uniform1i(gl.getUniformLocation(this.meshProg, "uMode"), modeInt);

      if (this.mode === "wireframe") {
        gl.enable(gl.POLYGON_OFFSET_FILL);
        gl.polygonOffset(1.0, 1.0);
      }

      const opaqueFirst = [...this.meshes].sort((a, b) => (b.material.alpha || 1) - (a.material.alpha || 1));
      for (const m of opaqueFirst) {
        const mat = m.material;
        const texRec = this._ensureTexture(mat);
        const hasTex = texRec && texRec.ready ? 1 : 0;
        if (hasTex) {
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, texRec.tex);
          gl.uniform1i(gl.getUniformLocation(this.meshProg, "uAlbedoTex"), 0);
        }
        gl.uniform1i(gl.getUniformLocation(this.meshProg, "uHasTex"), hasTex);
        gl.uniform2fv(gl.getUniformLocation(this.meshProg, "uUvScale"), mat.uv_scale || [1, 1]);
        gl.uniform4fv(gl.getUniformLocation(this.meshProg, "uBaseColor"), [
          mat.base_color[0], mat.base_color[1], mat.base_color[2], mat.alpha,
        ]);
        gl.uniform1f(gl.getUniformLocation(this.meshProg, "uMetallic"), mat.metallic);
        gl.uniform1f(gl.getUniformLocation(this.meshProg, "uRoughness"), mat.roughness);
        gl.uniform3fv(gl.getUniformLocation(this.meshProg, "uEmission"), mat.emission || [0, 0, 0]);
        gl.uniform1i(gl.getUniformLocation(this.meshProg, "uSelected"), m.id === this.selectedId ? 1 : 0);

        gl.bindVertexArray(m.vao);
        gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
      }
      if (this.mode === "wireframe") {
        gl.disable(gl.POLYGON_OFFSET_FILL);
      }

      // 3. Wireframe overlay
      if (this.mode === "wireframe") {
        gl.useProgram(this.lineProg);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, "uVP"), false, vp);
        gl.uniform1f(gl.getUniformLocation(this.lineProg, "uAlpha"), 0.85);
        for (const m of this.meshes) {
          gl.bindVertexArray(m.wireVao);
          gl.drawArrays(gl.LINES, 0, m.wireVertCount);
        }
      }

      // 4. Bounds & Skeletal Armature overlays
      const dyn = [];
      for (const m of this.meshes) {
        if ((this.showBounds || m.id === this.selectedId) && m.bounds) {
          const mn = m.bounds.min;
          const mx = m.bounds.max;
          const col = m.id === this.selectedId ? [0.24, 0.55, 0.98] : [0.45, 0.48, 0.55];
          const edges = [
            [mn[0], mn[1], mn[2], mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2], mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mn[2], mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2], mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2], mx[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2], mx[0], mx[1], mx[2]],
            [mx[0], mx[1], mx[2], mn[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2], mn[0], mn[1], mx[2]],
            [mn[0], mn[1], mn[2], mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mn[2], mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2], mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mn[2], mn[0], mx[1], mx[2]],
          ];
          for (const e of edges) {
            dyn.push(e[0], e[1], e[2], col[0], col[1], col[2], e[3], e[4], e[5], col[0], col[1], col[2]);
          }
        }
      }

      if (this.showBones) {
        gl.disable(gl.DEPTH_TEST);
        for (const arm of this.armatures) {
          for (const b of arm.bones) {
            dyn.push(
              b.head[0], b.head[1], b.head[2], 0.96, 0.62, 0.04,
              b.tail[0], b.tail[1], b.tail[2], 0.99, 0.82, 0.25
            );
          }
        }
      }

      if (dyn.length > 0) {
        gl.useProgram(this.lineProg);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, "uVP"), false, vp);
        gl.uniform1f(gl.getUniformLocation(this.lineProg, "uAlpha"), 0.95);
        gl.bindVertexArray(this.dynLineVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.dynLineBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(dyn), gl.DYNAMIC_DRAW);
        gl.drawArrays(gl.LINES, 0, dyn.length / 6);
        gl.enable(gl.DEPTH_TEST);
      }

      // 5. Interactive Edit Mode Element Overlays (Faces, Edges, Vertices)
      if (this.editMode && this.editData && this.editData.world_vertices && this.editData.world_vertices.length) {
        const wv = this.editData.world_vertices;
        const faces = this.editData.faces || [];
        const edges = this.editData.edges || [];

        gl.useProgram(this.lineProg);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, "uVP"), false, vp);
        gl.bindVertexArray(this.dynLineVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.dynLineBuf);

        // 5a. Selected faces highlight fill
        if (this.selectedFaces.size > 0 && faces.length > 0) {
          const triOverlay = [];
          for (const fi of this.selectedFaces) {
            const tri = faces[fi];
            if (!tri) continue;
            for (const vi of tri) {
              const p = wv[vi];
              if (p) triOverlay.push(p[0], p[1], p[2], 0.96, 0.58, 0.12);
            }
          }
          if (triOverlay.length > 0) {
            gl.enable(gl.POLYGON_OFFSET_FILL);
            gl.polygonOffset(-1.5, -1.5);
            gl.uniform1f(gl.getUniformLocation(this.lineProg, "uAlpha"), 0.42);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(triOverlay), gl.DYNAMIC_DRAW);
            gl.drawArrays(gl.TRIANGLES, 0, triOverlay.length / 6);
            gl.disable(gl.POLYGON_OFFSET_FILL);
          }
        }

        // 5b. Edit cage wireframe + highlighted edges / face borders
        const edgeLines = [];
        for (const e of edges) {
          const a = wv[e[0]];
          const b = wv[e[1]];
          if (!a || !b) continue;
          const ek = `${Math.min(e[0], e[1])}-${Math.max(e[0], e[1])}`;
          const sel = this.selectedEdges.has(ek);
          const col = sel ? [1.0, 0.65, 0.12] : [0.22, 0.28, 0.36];
          edgeLines.push(a[0], a[1], a[2], col[0], col[1], col[2], b[0], b[1], b[2], col[0], col[1], col[2]);
        }
        if (this.selectedFaces.size > 0) {
          for (const fi of this.selectedFaces) {
            const tri = faces[fi];
            if (!tri) continue;
            const pairs = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
            for (const [u, v] of pairs) {
              const a = wv[u];
              const b = wv[v];
              if (!a || !b) continue;
              edgeLines.push(a[0], a[1], a[2], 1.0, 0.72, 0.18, b[0], b[1], b[2], 1.0, 0.72, 0.18);
            }
          }
        }
        if (edgeLines.length > 0) {
          gl.depthFunc(gl.LEQUAL);
          gl.uniform1f(gl.getUniformLocation(this.lineProg, "uAlpha"), 0.92);
          gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(edgeLines), gl.DYNAMIC_DRAW);
          gl.drawArrays(gl.LINES, 0, edgeLines.length / 6);
          gl.depthFunc(gl.LESS);
        }

        // 5c. Vertex dots in Vertex mode or when vertices selected
        if (this.editElementMode === "vertex" || this.selectedVertices.size > 0) {
          const pts = [];
          for (let vi = 0; vi < wv.length; vi++) {
            const p = wv[vi];
            const sel = this.selectedVertices.has(vi);
            if (this.editElementMode !== "vertex" && !sel) continue;
            const col = sel ? [1.0, 0.65, 0.12] : [0.42, 0.68, 0.98];
            pts.push(p[0], p[1], p[2], col[0], col[1], col[2]);
          }
          if (pts.length > 0) {
            gl.disable(gl.DEPTH_TEST);
            gl.uniform1f(gl.getUniformLocation(this.lineProg, "uAlpha"), 1.0);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.DYNAMIC_DRAW);
            gl.drawArrays(gl.POINTS, 0, pts.length / 6);
            gl.enable(gl.DEPTH_TEST);
          }
        }
      }

      gl.bindVertexArray(null);
    }
  }

  global.ViewportGL = ViewportGL;
})(window);
