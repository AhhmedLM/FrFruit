/* FrFruit Studio Workstation Controller — REST + WebSocket + Live MCP Bridge */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const state = {
    scene: null,
    selectedId: null,
    viewSource: "gl", // "gl" | "cpu"
    editMode: false,
    editElementMode: "face",
    lastSelectionRule: "top",
    mcpInfo: null,
    playing: false,
    playTimer: null,
    ws: null,
    firstLoad: true,
  };

  const vp = new window.ViewportGL($("gl-canvas"));

  function toast(msg, kind = "info") {
    const bar = $("toast-bar");
    const div = document.createElement("div");
    div.className = `toast ${kind}`;
    div.textContent = msg;
    bar.appendChild(div);
    setTimeout(() => div.remove(), 3800);
  }

  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
      ...opts,
    });
    const data = await res.json();
    if (!res.ok || data.status === "error") {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  function updateEditSelectionUI() {
    const f = vp.selectedFaces.size;
    const e = vp.selectedEdges.size;
    const v = vp.selectedVertices.size;
    if ($("edit-sel-summary")) {
      $("edit-sel-summary").textContent = `${f}F · ${e}E · ${v}V`;
    }
  }

  async function syncEditElements() {
    if (!state.editMode || !state.selectedId) {
      vp.editData = null;
      vp.render();
      return;
    }
    try {
      const data = await api(`/api/objects/${encodeURIComponent(state.selectedId)}/elements`);
      vp.editData = data;
      // Prune stale indices if mesh topology shrunk
      for (const fi of Array.from(vp.selectedFaces)) {
        if (fi >= data.face_count) vp.selectedFaces.delete(fi);
      }
      for (const vi of Array.from(vp.selectedVertices)) {
        if (vi >= data.vertex_count) vp.selectedVertices.delete(vi);
      }
      const fmatN = Object.keys(data.face_material_ids || {}).length;
      if ($("edit-fmat-count")) {
        $("edit-fmat-count").textContent = `${fmatN} override${fmatN === 1 ? "" : "s"}`;
      }
      updateEditSelectionUI();
      vp.render();
    } catch (_) {
      vp.editData = null;
    }
  }

  async function setEditMode(enabled) {
    if (enabled && !state.selectedId) {
      toast("Select a mesh object first to enter Edit Mode", "warn");
      return;
    }
    state.editMode = !!enabled;
    vp.editMode = state.editMode;
    $("btn-mode-object").classList.toggle("active", !state.editMode);
    $("btn-mode-edit").classList.toggle("active", state.editMode);
    $("edit-elem-switch").style.display = state.editMode ? "inline-flex" : "none";
    if (state.editMode) {
      const editTabBtn = document.querySelector('#inspector-tabs .tab-btn[data-tab="edit"]');
      if (editTabBtn) editTabBtn.click();
      await syncEditElements();
    } else {
      vp.editData = null;
      vp.render();
    }
  }

  async function refreshAll(autoFit = false) {
    try {
      const [st, geo] = await Promise.all([api("/api/state"), api("/api/geometry")]);
      state.scene = st;
      renderStateUI(st);
      vp.selectedId = state.selectedId;
      vp.setGeometry(geo, autoFit || state.firstLoad);
      state.firstLoad = false;
      if (state.editMode) {
        await syncEditElements();
      }
      if (state.viewSource === "cpu") {
        await refreshCpuRender();
      }
    } catch (err) {
      toast(err.message, "err");
    }
  }

  function renderStateUI(st) {
    $("hdr-scene-name").textContent = st.scene_name;
    $("hdr-engine").value = st.render_engine || "software";
    $("pp-engine").value = st.render_engine || "software";

    $("outliner-count").textContent = String(st.objects.length);
    $("stat-verts").textContent = st.stats.total_vertices.toLocaleString();
    $("stat-faces").textContent = st.stats.total_faces.toLocaleString();
    $("stat-mats").textContent = String(st.stats.material_count);

    $("sb-workspace").textContent = `Workspace: ${st.workspace_root}`;
    $("sb-telemetry").textContent =
      `Engine: ${st.render_engine} | Meshes: ${st.stats.mesh_count} | Verts: ${st.stats.total_vertices} | Tools: ${st.mcp_tools_count}`;

    // Timeline
    const tl = st.timeline;
    $("tl-slider").min = String(tl.frame_start);
    $("tl-slider").max = String(tl.frame_end);
    $("tl-slider").value = String(tl.current_frame);
    $("tl-frame-num").value = String(tl.current_frame);
    $("tl-range-lbl").textContent = `${tl.frame_start}..${tl.frame_end}`;
    $("tl-fps-lbl").textContent = String(tl.fps);
    $("phys-fs").value = String(tl.frame_start);
    $("phys-fe").value = String(tl.frame_end);
    $("vid-fs").value = String(tl.frame_start);
    $("vid-fe").value = String(tl.frame_end);

    // Undo/Redo
    $("btn-undo").disabled = !st.history.can_undo;
    $("btn-redo").disabled = !st.history.can_redo;

    // Outliner
    renderOutliner(st);

    // Materials dropdowns
    const matSel = $("prop-material");
    const faceMatSel = $("edit-face-mat");
    matSel.innerHTML = '<option value="">(None / Default)</option>';
    if (faceMatSel) faceMatSel.innerHTML = "";
    for (const [mid, m] of Object.entries(st.materials)) {
      const opt = document.createElement("option");
      opt.value = mid;
      opt.textContent = m.name;
      matSel.appendChild(opt);
      if (faceMatSel) {
        const opt2 = document.createElement("option");
        opt2.value = mid;
        opt2.textContent = m.name;
        faceMatSel.appendChild(opt2);
      }
    }
    if (faceMatSel && !faceMatSel.options.length) {
      faceMatSel.innerHTML = '<option value="">(Create a material first)</option>';
    }

    // Populate selected object inspector
    if (state.selectedId && !st.objects.find((o) => o.id === state.selectedId)) {
      state.selectedId = null;
    }
    if (!state.selectedId && st.objects.length > 0) {
      const firstMesh = st.objects.find((o) => o.kind === "mesh" || o.kind === "procedural");
      if (firstMesh) state.selectedId = firstMesh.id;
    }
    renderSelectedInspector(st);

    // Post-process controls
    const pp = st.post_process;
    $("pp-enabled").checked = !!pp.enabled;
    $("pp-tonemap").value = pp.tone_mapper || "aces";
    $("pp-exposure").value = String(pp.exposure ?? 0);
    $("pp-exposure-v").textContent = Number(pp.exposure ?? 0).toFixed(2);
    $("pp-bloom-on").checked = !!pp.bloom_enabled;
    $("pp-bloom-int").value = String(pp.bloom_intensity ?? 0.22);
    $("pp-bloom-int-v").textContent = Number(pp.bloom_intensity ?? 0.22).toFixed(2);
    $("pp-ssao-on").checked = !!pp.ssao_enabled;
    $("pp-ssao-str").value = String(pp.ssao_strength ?? 0.32);
    $("pp-ssao-str-v").textContent = Number(pp.ssao_strength ?? 0.32).toFixed(2);
    $("pp-dof-on").checked = !!pp.dof_enabled;
    $("pp-vig").value = String(pp.vignette_intensity ?? 0.12);
    $("pp-vig-v").textContent = Number(pp.vignette_intensity ?? 0.12).toFixed(2);
    $("pp-ca").value = String(pp.chromatic_aberration ?? 0);
    $("pp-ca-v").textContent = Number(pp.chromatic_aberration ?? 0).toFixed(1);
    $("pp-sat").value = String(pp.saturation ?? 1.05);
    $("pp-sat-v").textContent = Number(pp.saturation ?? 1.05).toFixed(2);
    $("pp-fxaa").checked = !!pp.fxaa_enabled;

    if (st.audit_log && st.audit_log.length) {
      renderAuditLog(st.audit_log);
    }
  }

  function renderOutliner(st) {
    const filter = ($("outliner-filter").value || "").toLowerCase();
    const list = $("outliner-list");
    list.innerHTML = "";
    for (const obj of st.objects) {
      if (filter && !obj.name.toLowerCase().includes(filter) && !obj.kind.includes(filter)) {
        continue;
      }
      const row = document.createElement("div");
      row.className = "outliner-item" + (obj.id === state.selectedId ? " selected" : "");
      row.onclick = () => {
        state.selectedId = obj.id;
        vp.selectedId = obj.id;
        vp.render();
        renderOutliner(state.scene);
        renderSelectedInspector(state.scene);
      };

      const main = document.createElement("div");
      main.className = "outliner-main";

      const badge = document.createElement("span");
      badge.className = "tag-mini";
      badge.textContent = obj.kind.slice(0, 3).toUpperCase();
      main.appendChild(badge);

      const nm = document.createElement("span");
      nm.className = "outliner-name";
      nm.textContent = obj.name;
      main.appendChild(nm);

      if (obj.cloth && obj.cloth.enabled) {
        const t = document.createElement("span");
        t.className = "tag-mini phys";
        t.textContent = "CL";
        main.appendChild(t);
      } else if (obj.rigid_body && obj.rigid_body.enabled) {
        const t = document.createElement("span");
        t.className = "tag-mini phys";
        t.textContent = "RB";
        main.appendChild(t);
      }
      if (obj.skin_armature_id) {
        const t = document.createElement("span");
        t.className = "tag-mini rig";
        t.textContent = "RIG";
        main.appendChild(t);
      }

      const acts = document.createElement("div");
      acts.className = "outliner-actions";

      const eye = document.createElement("button");
      eye.className = "mini-icon-act";
      eye.title = obj.visible ? "Hide object" : "Show object";
      eye.textContent = obj.visible ? "V" : "-";
      eye.onclick = async (e) => {
        e.stopPropagation();
        await api(`/api/objects/${encodeURIComponent(obj.id)}/transform`, {
          method: "POST",
          body: JSON.stringify({ visible: !obj.visible }),
        });
        await refreshAll();
      };
      acts.appendChild(eye);

      row.appendChild(main);
      row.appendChild(acts);
      list.appendChild(row);
    }
  }

  function renderSelectedInspector(st) {
    const obj = st.objects.find((o) => o.id === state.selectedId);
    if (!obj) {
      $("no-selection-msg").style.display = "block";
      $("prop-editor").style.display = "none";
      $("hud-sel-label").textContent = "Selection: None";
      return;
    }
    $("no-selection-msg").style.display = "none";
    $("prop-editor").style.display = "block";
    $("hud-sel-label").textContent = `Selection: ${obj.name} (${obj.vertex_count}v / ${obj.face_count}f)`;

    $("prop-kind-badge").textContent = obj.kind.toUpperCase();
    $("prop-name").value = obj.name;
    $("loc-x").value = obj.location[0];
    $("loc-y").value = obj.location[1];
    $("loc-z").value = obj.location[2];
    $("rot-x").value = obj.rotation_deg[0];
    $("rot-y").value = obj.rotation_deg[1];
    $("rot-z").value = obj.rotation_deg[2];
    $("scl-x").value = obj.scale[0];
    $("scl-y").value = obj.scale[1];
    $("scl-z").value = obj.scale[2];
    $("prop-material").value = obj.material_id || "";
    $("prop-topo-lbl").textContent = `${obj.vertex_count} verts / ${obj.face_count} tris`;
    if (obj.bounds) {
      const sz = obj.bounds.size;
      $("prop-dims-lbl").textContent = `${sz[0]} x ${sz[1]} x ${sz[2]} m`;
    } else {
      $("prop-dims-lbl").textContent = "-";
    }
  }

  async function refreshCpuRender() {
    const stage = $("viewport-stage").getBoundingClientRect();
    const w = Math.max(320, Math.min(1080, Math.floor(stage.width)));
    const h = Math.max(240, Math.min(720, Math.floor(stage.height)));
    const eng = $("pp-engine").value;
    const samples = $("pp-samples").value;
    const mode = vp.mode;
    const imgEl = $("cpu-render-img");
    $("btn-render-frame").disabled = true;
    $("btn-render-frame").textContent = "Rendering...";
    const t0 = performance.now();
    try {
      const url = `/api/render?width=${w}&height=${h}&mode=${mode}&engine=${eng}&samples=${samples}&t=${Date.now()}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Render failed (${res.status})`);
      const blob = await res.blob();
      const objUrl = URL.createObjectURL(blob);
      imgEl.onload = () => URL.revokeObjectURL(objUrl);
      imgEl.src = objUrl;
      const ms = Math.round(performance.now() - t0);
      toast(`CPU ${eng.toUpperCase()} frame rendered in ${ms}ms`, "ok");
    } catch (err) {
      toast(err.message, "err");
    } finally {
      $("btn-render-frame").disabled = false;
      $("btn-render-frame").textContent = "Render Frame";
    }
  }

  function setViewSource(src) {
    state.viewSource = src;
    $("btn-src-gl").classList.toggle("active", src === "gl");
    $("btn-src-cpu").classList.toggle("active", src === "cpu");
    $("cpu-render-overlay").classList.toggle("visible", src === "cpu");
    if (src === "cpu") {
      refreshCpuRender();
    }
  }

  function renderAuditLog(entries) {
    const feed = $("mcp-audit-feed");
    feed.innerHTML = "";
    for (let i = entries.length - 1; i >= 0; i--) {
      const e = entries[i];
      const div = document.createElement("div");
      div.className = "audit-entry";
      const t = new Date((e.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString();
      div.innerHTML = `<span style="color:var(--text-muted)">[${t}]</span> <span class="audit-evt">${e.event}</span> · ${JSON.stringify(e.data || {})}`;
      feed.appendChild(div);
    }
  }

  function appendAuditEntry(e) {
    const feed = $("mcp-audit-feed");
    if (feed.textContent.includes("Waiting for telemetry")) feed.innerHTML = "";
    const div = document.createElement("div");
    div.className = "audit-entry";
    const t = new Date((e.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString();
    div.innerHTML = `<span style="color:var(--text-muted)">[${t}]</span> <span class="audit-evt">${e.event}</span> · ${JSON.stringify(e.data || {})}`;
    feed.prepend(div);
    while (feed.children.length > 60) feed.lastChild.remove();
  }

  function connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/events`);
    state.ws = ws;

    ws.onopen = () => {
      $("mcp-led").className = "led online";
      $("mcp-pill-text").textContent = "MCP LIVE";
    };
    ws.onclose = () => {
      $("mcp-led").className = "led offline";
      $("mcp-pill-text").textContent = "MCP RECONNECTING";
      setTimeout(connectWebSocket, 2000);
    };
    ws.onmessage = async (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        appendAuditEntry(msg);
        if (msg.event && msg.event !== "connected") {
          await refreshAll();
        }
      } catch (_) {}
    };
  }

  async function loadMcpInfo() {
    try {
      const info = await api("/api/mcp/info");
      state.mcpInfo = info;
      $("mcp-tools-badge").textContent = `${info.tools_count} TOOLS`;
      $("mcp-url-http").value = info.streamable_http_url;
      $("mcp-url-sse").value = info.sse_url;
      $("mcp-url-ws").value = info.ws_url;
      $("mcp-config-preview").textContent = JSON.stringify(info.configs.streamable_http_client, null, 2);
    } catch (_) {}
  }

  async function searchTextures() {
    const q = $("tex-query").value.trim();
    const src = $("tex-source").value;
    const container = $("tex-results");
    container.innerHTML = '<div class="tex-item"><span class="tex-item-meta">Searching catalogs...</span></div>';
    try {
      const data = await api(`/api/textures/search?query=${encodeURIComponent(q)}&source=${encodeURIComponent(src)}&limit=15`);
      container.innerHTML = "";
      if (!data.results.length) {
        container.innerHTML = '<div class="tex-item"><span class="tex-item-meta">No textures matched.</span></div>';
        return;
      }
      for (const hit of data.results) {
        const card = document.createElement("div");
        card.className = "tex-item";
        const info = document.createElement("div");
        info.className = "tex-item-info";
        info.innerHTML = `<div class="tex-item-name">${hit.name || hit.asset_id}</div>
          <div class="tex-item-meta">${hit.source.toUpperCase()} · ${hit.asset_id}</div>`;
        const btn = document.createElement("button");
        btn.textContent = "Apply";
        btn.onclick = async () => {
          btn.disabled = true;
          btn.textContent = "...";
          try {
            const res = await api("/api/textures/apply", {
              method: "POST",
              body: JSON.stringify({
                asset_id: hit.asset_id,
                source: hit.source,
                resolution: $("tex-res").value,
                uv_scale: [parseFloat($("tex-uv-u").value) || 1, parseFloat($("tex-uv-v").value) || 1],
                object_id: state.selectedId,
              }),
            });
            toast(`Applied PBR maps (${res.manifest.source}: ${hit.asset_id})`, "ok");
            await refreshAll();
          } catch (err) {
            toast(err.message, "err");
          } finally {
            btn.disabled = false;
            btn.textContent = "Apply";
          }
        };
        card.appendChild(info);
        card.appendChild(btn);
        container.appendChild(card);
      }
    } catch (err) {
      toast(err.message, "err");
    }
  }

  // Event Wiring
  document.querySelectorAll("#inspector-tabs .tab-btn").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll("#inspector-tabs .tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`tab-${btn.dataset.tab}`).classList.add("active");
    };
  });

  $("btn-mcp-pill").onclick = () => {
    document.querySelector('#inspector-tabs .tab-btn[data-tab="mcp"]').click();
  };

  document.querySelectorAll("#mode-tabs button").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll("#mode-tabs button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      vp.mode = btn.dataset.mode;
      $("hud-mode-label").textContent = vp.mode.toUpperCase();
      vp.render();
      if (state.viewSource === "cpu") refreshCpuRender();
    };
  });

  document.querySelectorAll("#cam-presets button").forEach((btn) => {
    btn.onclick = () => vp.setPresetView(btn.dataset.cam);
  });

  $("tgl-grid").onclick = (e) => { vp.showGrid = !vp.showGrid; e.target.classList.toggle("active", vp.showGrid); vp.render(); };
  $("tgl-axes").onclick = (e) => { vp.showAxes = !vp.showAxes; e.target.classList.toggle("active", vp.showAxes); vp.render(); };
  $("tgl-bounds").onclick = (e) => { vp.showBounds = !vp.showBounds; e.target.classList.toggle("active", vp.showBounds); vp.render(); };
  $("tgl-bones").onclick = (e) => { vp.showBones = !vp.showBones; e.target.classList.toggle("active", vp.showBones); vp.render(); };

  $("btn-src-gl").onclick = () => setViewSource("gl");
  $("btn-src-cpu").onclick = () => setViewSource("cpu");
  $("btn-render-frame").onclick = () => setViewSource("cpu");

  $("outliner-filter").oninput = () => state.scene && renderOutliner(state.scene);

  $("btn-new").onclick = async () => {
    await api("/api/scene/new", { method: "POST", body: JSON.stringify({ name: "NewScene" }) });
    state.selectedId = null;
    await refreshAll(true);
    toast("Created new empty scene", "ok");
  };

  $("sel-demo").onchange = async (e) => {
    const val = e.target.value;
    if (!val) return;
    e.target.value = "";
    await api("/api/scene/demo", { method: "POST", body: JSON.stringify({ preset: val }) });
    state.selectedId = null;
    await refreshAll(true);
    toast(`Loaded showcase preset: ${val}`, "ok");
  };

  $("btn-open").onclick = async () => {
    const path = prompt("Relative path inside workspace (.frf.json):", "scene.frf.json");
    if (!path) return;
    await api("/api/scene/open", { method: "POST", body: JSON.stringify({ path }) });
    state.selectedId = null;
    await refreshAll(true);
    toast(`Opened ${path}`, "ok");
  };

  $("btn-save").onclick = async () => {
    const path = prompt("Save .frf.json path in workspace:", `${(state.scene && state.scene.scene_name) || "scene"}.frf.json`);
    if (!path) return;
    const res = await api("/api/scene/save", { method: "POST", body: JSON.stringify({ path }) });
    toast(`Saved ${res.path}`, "ok");
  };

  $("btn-undo").onclick = async () => { await api("/api/undo", { method: "POST" }); await refreshAll(); };
  $("btn-redo").onclick = async () => { await api("/api/redo", { method: "POST" }); await refreshAll(); };

  $("sel-add-prim").onchange = async (e) => {
    const prim = e.target.value;
    if (!prim) return;
    e.target.value = "";
    const res = await api("/api/objects/primitive", {
      method: "POST",
      body: JSON.stringify({ primitive: prim }),
    });
    state.selectedId = res.object_id;
    await refreshAll();
    toast(`Added ${res.name}`, "ok");
  };

  $("btn-apply-transform").onclick = async () => {
    if (!state.selectedId) return;
    await api(`/api/objects/${encodeURIComponent(state.selectedId)}/transform`, {
      method: "POST",
      body: JSON.stringify({
        name: $("prop-name").value,
        location: [parseFloat($("loc-x").value), parseFloat($("loc-y").value), parseFloat($("loc-z").value)],
        rotation_deg: [parseFloat($("rot-x").value), parseFloat($("rot-y").value), parseFloat($("rot-z").value)],
        scale: [parseFloat($("scl-x").value), parseFloat($("scl-y").value), parseFloat($("scl-z").value)],
        material_id: $("prop-material").value || null,
      }),
    });
    await refreshAll();
    toast("Updated object transform & material", "ok");
  };

  $("btn-delete-obj").onclick = async () => {
    if (!state.selectedId) return;
    await api(`/api/objects/${encodeURIComponent(state.selectedId)}`, { method: "DELETE" });
    state.selectedId = null;
    await refreshAll();
    toast("Deleted object", "warn");
  };

  $("btn-tex-search").onclick = searchTextures;

  $("cloth-stiff").oninput = (e) => { $("cloth-stiff-v").textContent = Number(e.target.value).toFixed(2); };
  $("btn-set-cloth").onclick = async () => {
    if (!state.selectedId) return toast("Select an object first", "warn");
    await api("/api/physics/setup", {
      method: "POST",
      body: JSON.stringify({
        object_id: state.selectedId,
        kind: "cloth",
        pin_mode: $("cloth-pin").value,
        stiffness: parseFloat($("cloth-stiff").value),
        wind: [parseFloat($("wind-x").value), parseFloat($("wind-y").value), parseFloat($("wind-z").value)],
      }),
    });
    await refreshAll();
    toast("Cloth physics enabled on selected object", "ok");
  };

  $("btn-set-rb").onclick = async () => {
    if (!state.selectedId) return toast("Select an object first", "warn");
    await api("/api/physics/setup", {
      method: "POST",
      body: JSON.stringify({
        object_id: state.selectedId,
        kind: "rigid_body",
        body_type: $("rb-type").value,
        collider_shape: $("rb-shape").value,
        restitution: parseFloat($("rb-rest").value),
        friction: parseFloat($("rb-fric").value),
      }),
    });
    await refreshAll();
    toast("Rigid body assigned to selected object", "ok");
  };

  $("btn-phys-bake").onclick = async () => {
    $("btn-phys-bake").disabled = true;
    try {
      const res = await api("/api/physics/bake", {
        method: "POST",
        body: JSON.stringify({
          frame_start: parseInt($("phys-fs").value, 10),
          frame_end: parseInt($("phys-fe").value, 10),
          ground_z: parseFloat($("phys-gz").value),
        }),
      });
      await refreshAll();
      toast(`Physics baked: ${res.cloth_objects} cloth, ${res.rigid_bodies} rigid bodies`, "ok");
    } finally {
      $("btn-phys-bake").disabled = false;
    }
  };

  $("btn-phys-clear").onclick = async () => {
    await api("/api/physics/clear", { method: "POST" });
    await refreshAll();
    toast("Physics caches cleared", "warn");
  };

  // Post-FX & Engine wiring
  ["pp-samples", "pp-exposure", "pp-bloom-int", "pp-ssao-str", "pp-vig", "pp-ca", "pp-sat"].forEach((id) => {
    $(id).oninput = (e) => {
      const v = Number(e.target.value);
      $(`${id}-v`).textContent = id === "pp-samples" ? String(v) : id === "pp-ca" ? v.toFixed(1) : v.toFixed(2);
    };
  });

  $("hdr-engine").onchange = async (e) => {
    await api("/api/postprocess", { method: "POST", body: JSON.stringify({ render_engine: e.target.value }) });
    await refreshAll();
  };

  $("pp-preset").onchange = async (e) => {
    await api("/api/postprocess", { method: "POST", body: JSON.stringify({ preset: e.target.value }) });
    await refreshAll();
    toast(`Applied Post-FX preset: ${e.target.value}`, "ok");
  };

  $("btn-pp-apply").onclick = async () => {
    await api("/api/postprocess", {
      method: "POST",
      body: JSON.stringify({
        render_engine: $("pp-engine").value,
        post_process: {
          enabled: $("pp-enabled").checked,
          tone_mapper: $("pp-tonemap").value,
          exposure: parseFloat($("pp-exposure").value),
          bloom_enabled: $("pp-bloom-on").checked,
          bloom_intensity: parseFloat($("pp-bloom-int").value),
          ssao_enabled: $("pp-ssao-on").checked,
          ssao_strength: parseFloat($("pp-ssao-str").value),
          dof_enabled: $("pp-dof-on").checked,
          vignette_intensity: parseFloat($("pp-vig").value),
          chromatic_aberration: parseFloat($("pp-ca").value),
          saturation: parseFloat($("pp-sat").value),
          fxaa_enabled: $("pp-fxaa").checked,
        },
      }),
    });
    await refreshAll();
    setViewSource("cpu");
  };

  // Timeline scrubbing & playback
  async function setFrame(fr) {
    await api("/api/timeline/frame", { method: "POST", body: JSON.stringify({ frame: fr }) });
    await refreshAll();
  }

  $("tl-slider").oninput = (e) => {
    $("tl-frame-num").value = e.target.value;
    setFrame(parseInt(e.target.value, 10));
  };
  $("tl-frame-num").onchange = (e) => setFrame(parseInt(e.target.value, 10));
  $("tl-start").onclick = () => setFrame(parseInt($("tl-slider").min, 10));
  $("tl-end").onclick = () => setFrame(parseInt($("tl-slider").max, 10));
  $("tl-play").onclick = () => {
    state.playing = !state.playing;
    $("tl-play").textContent = state.playing ? "Pause" : "Play";
    $("tl-play").classList.toggle("active", state.playing);
    if (state.playing) {
      state.playTimer = setInterval(async () => {
        if (!state.playing) return;
        const mn = parseInt($("tl-slider").min, 10);
        const mx = parseInt($("tl-slider").max, 10);
        let next = parseInt($("tl-frame-num").value, 10) + 1;
        if (next > mx) next = mn;
        await setFrame(next);
      }, 100);
    } else {
      clearInterval(state.playTimer);
    }
  };

  // Exports & Video
  $("btn-export-run").onclick = async () => {
    const outBox = $("exp-result");
    outBox.style.display = "block";
    outBox.textContent = "Exporting...";
    try {
      const res = await api("/api/export", {
        method: "POST",
        body: JSON.stringify({
          path: $("exp-path").value,
          format: $("exp-format").value,
          target_engine: $("exp-engine").value,
        }),
      });
      const dl = `/api/files/download?path=${encodeURIComponent(res.path)}`;
      outBox.innerHTML = `Exported ${res.format.toUpperCase()} (${res.bytes.toLocaleString()} B)<br><a href="${dl}" download>Download ${res.path}</a>`;
      toast(`Exported ${res.format}`, "ok");
    } catch (err) {
      outBox.textContent = `Error: ${err.message}`;
      toast(err.message, "err");
    }
  };

  $("btn-vid-run").onclick = async () => {
    const outBox = $("vid-result");
    outBox.style.display = "block";
    outBox.textContent = "Encoding video frames via FFmpeg...";
    $("btn-vid-run").disabled = true;
    try {
      const res = await api("/api/video/render", {
        method: "POST",
        body: JSON.stringify({
          path: $("vid-path").value,
          kind: $("vid-kind").value,
          width: parseInt($("vid-w").value, 10),
          height: parseInt($("vid-h").value, 10),
          fps: parseInt($("vid-fps").value, 10),
          frame_start: parseInt($("vid-fs").value, 10),
          frame_end: parseInt($("vid-fe").value, 10),
          mode: vp.mode,
          engine: $("pp-engine").value,
          samples: parseInt($("pp-samples").value, 10),
        }),
      });
      const dl = `/api/files/download?path=${encodeURIComponent(res.path)}`;
      outBox.innerHTML = `Rendered ${res.frames_rendered} frames (${res.format.toUpperCase()}, ${res.bytes.toLocaleString()} B)<br><a href="${dl}" download>Download Video</a>`;
      toast(`Video rendered (${res.frames_rendered} frames)`, "ok");
    } catch (err) {
      outBox.textContent = `Error: ${err.message}`;
      toast(err.message, "err");
    } finally {
      $("btn-vid-run").disabled = false;
    }
  };

  $("btn-copy-http").onclick = () => {
    if (!state.mcpInfo) return;
    const txt = JSON.stringify(state.mcpInfo.configs.streamable_http_client, null, 2);
    $("mcp-config-preview").textContent = txt;
    navigator.clipboard && navigator.clipboard.writeText(txt);
    toast("Copied Streamable HTTP MCP config", "ok");
  };

  $("btn-copy-stdio").onclick = () => {
    if (!state.mcpInfo) return;
    const txt = JSON.stringify(state.mcpInfo.configs.stdio_client, null, 2);
    $("mcp-config-preview").textContent = txt;
    navigator.clipboard && navigator.clipboard.writeText(txt);
    toast("Copied Stdio MCP config", "ok");
  };

  $("btn-clear-audit").onclick = () => { $("mcp-audit-feed").innerHTML = ""; };

  // Edit Mode UI & Viewport Raycast Picking
  $("btn-mode-object").onclick = () => setEditMode(false);
  $("btn-mode-edit").onclick = () => setEditMode(true);

  document.querySelectorAll("#edit-elem-switch button").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll("#edit-elem-switch button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.editElementMode = btn.dataset.elem;
      vp.editElementMode = btn.dataset.elem;
      vp.render();
    };
  });

  vp.onViewportClick = async (origin, direction, evt) => {
    if (!state.editMode || !state.selectedId) return;
    try {
      const hit = await api(`/api/objects/${encodeURIComponent(state.selectedId)}/raycast_face`, {
        method: "POST",
        body: JSON.stringify({ origin, direction }),
      });
      if (!hit.hit) {
        if (!evt.shiftKey) {
          vp.selectedFaces.clear();
          vp.selectedEdges.clear();
          vp.selectedVertices.clear();
          updateEditSelectionUI();
          vp.render();
        }
        return;
      }

      if (state.editElementMode === "face") {
        if (evt.ctrlKey || evt.metaKey) {
          const elData = await api(
            `/api/objects/${encodeURIComponent(state.selectedId)}/elements?selection_mode=coplanar&seed_face=${hit.face_index}`
          );
          if (!evt.shiftKey) vp.selectedFaces.clear();
          for (const fi of elData.selected_faces || []) vp.selectedFaces.add(fi);
        } else {
          if (!evt.shiftKey) vp.selectedFaces.clear();
          if (evt.shiftKey && vp.selectedFaces.has(hit.face_index)) {
            vp.selectedFaces.delete(hit.face_index);
          } else {
            vp.selectedFaces.add(hit.face_index);
          }
        }
        state.lastSelectionRule = "indices";
      } else if (state.editElementMode === "edge") {
        const ek = `${hit.closest_edge[0]}-${hit.closest_edge[1]}`;
        if (!evt.shiftKey) vp.selectedEdges.clear();
        if (evt.shiftKey && vp.selectedEdges.has(ek)) vp.selectedEdges.delete(ek);
        else vp.selectedEdges.add(ek);
        state.lastSelectionRule = "edge_indices";
      } else if (state.editElementMode === "vertex") {
        if (!evt.shiftKey) vp.selectedVertices.clear();
        if (evt.shiftKey && vp.selectedVertices.has(hit.closest_vertex)) {
          vp.selectedVertices.delete(hit.closest_vertex);
        } else {
          vp.selectedVertices.add(hit.closest_vertex);
        }
        state.lastSelectionRule = "vertex_indices";
      }
      updateEditSelectionUI();
      vp.render();
    } catch (err) {
      toast(err.message, "err");
    }
  };

  async function applyQuickSelection(rule, extraParams = "") {
    if (!state.selectedId) return toast("Select an object first", "warn");
    if (!state.editMode) await setEditMode(true);
    const data = await api(
      `/api/objects/${encodeURIComponent(state.selectedId)}/elements?selection_mode=${encodeURIComponent(rule)}${extraParams}`
    );
    vp.editData = data;
    vp.selectedFaces = new Set(data.selected_faces || []);
    vp.selectedEdges = new Set((data.selected_edges || []).map((e) => `${e[0]}-${e[1]}`));
    vp.selectedVertices = new Set(data.selected_vertices || []);
    state.lastSelectionRule = rule;
    updateEditSelectionUI();
    vp.render();
  }

  document.querySelectorAll("#edit-quick-sel button[data-qsel]").forEach((btn) => {
    btn.onclick = () => applyQuickSelection(btn.dataset.qsel);
  });

  $("btn-sel-coplanar").onclick = () => {
    const seed = vp.selectedFaces.size ? Array.from(vp.selectedFaces)[0] : 0;
    applyQuickSelection("coplanar", `&seed_face=${seed}`);
  };

  $("btn-sel-invert").onclick = () => {
    if (!vp.editData) return;
    if (state.editElementMode === "vertex") {
      const next = new Set();
      for (let i = 0; i < vp.editData.vertex_count; i++) {
        if (!vp.selectedVertices.has(i)) next.add(i);
      }
      vp.selectedVertices = next;
    } else {
      const next = new Set();
      for (let i = 0; i < vp.editData.face_count; i++) {
        if (!vp.selectedFaces.has(i)) next.add(i);
      }
      vp.selectedFaces = next;
    }
    updateEditSelectionUI();
    vp.render();
  };

  $("btn-sel-clear").onclick = () => {
    vp.selectedFaces.clear();
    vp.selectedEdges.clear();
    vp.selectedVertices.clear();
    updateEditSelectionUI();
    vp.render();
  };

  function buildSelectionPayload() {
    if (state.editElementMode === "vertex" && vp.selectedVertices.size > 0) {
      return { selection_mode: "vertex_indices", vertex_indices: Array.from(vp.selectedVertices) };
    }
    if (state.editElementMode === "edge" && vp.selectedEdges.size > 0) {
      return {
        selection_mode: "edge_indices",
        edge_indices: Array.from(vp.selectedEdges).map((k) => k.split("-").map((n) => parseInt(n, 10))),
      };
    }
    if (vp.selectedFaces.size > 0) {
      return { selection_mode: "indices", indices: Array.from(vp.selectedFaces) };
    }
    return { selection_mode: state.lastSelectionRule || "top" };
  }

  $("btn-assign-face-mat").onclick = async () => {
    if (!state.selectedId) return toast("Select an object first", "warn");
    const mid = $("edit-face-mat").value;
    if (!mid) return toast("Create or choose a material first", "warn");
    const sel = buildSelectionPayload();
    const payload = {
      object_id: state.selectedId,
      material_id: mid,
      ...(sel.indices ? { face_indices: sel.indices } : { selection_mode: sel.selection_mode }),
    };
    const res = await api("/api/materials/assign_faces", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await refreshAll();
    toast(`Assigned material to ${res.count} face(s)`, "ok");
  };

  $("btn-clear-face-mat").onclick = async () => {
    if (!state.selectedId) return toast("Select an object first", "warn");
    const payload = {
      object_id: state.selectedId,
      clear: true,
      ...(vp.selectedFaces.size > 0 ? { face_indices: Array.from(vp.selectedFaces) } : {}),
    };
    const res = await api("/api/materials/assign_faces", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await refreshAll();
    toast(`Cleared ${res.count} face material override(s)`, "warn");
  };

  document.querySelectorAll("#tab-edit button[data-edop]").forEach((btn) => {
    btn.onclick = async () => {
      if (!state.selectedId) return toast("Select a mesh object first", "warn");
      const rawOp = btn.dataset.edop;
      const op = rawOp === "inset_individual" ? "inset_faces" : rawOp;
      const sel = buildSelectionPayload();
      const axis = $("ed-axis").value;
      const axisNormal = axis === "x" ? [1, 0, 0] : axis === "y" ? [0, 1, 0] : [0, 0, 1];
      const symMap = { x: "-x_to_+x", y: "-y_to_+y", z: "-z_to_+z" };
      const body = {
        operation: op,
        ...sel,
        distance: parseFloat($("ed-dist").value) || 0.25,
        thickness: Math.max(0.002, parseFloat($("ed-width").value) || 0.05),
        width: Math.max(0.002, parseFloat($("ed-width").value) || 0.05),
        segments: Math.max(1, parseInt($("ed-segs").value, 10) || 1),
        cuts: Math.max(1, parseInt($("ed-cuts").value, 10) || 1),
        slide: parseFloat($("ed-slide").value) || 0.0,
        axis,
        plane_normal: axisNormal,
        clear_inner: op === "bisect_plane",
        individual: rawOp === "inset_individual" || op === "extrude_individual",
        factor: parseFloat($("ed-factor").value) || 0.5,
        projection: $("ed-uv-proj").value,
        symmetrize_direction: symMap[axis] || "-x_to_+x",
      };
      try {
        const res = await api(`/api/objects/${encodeURIComponent(state.selectedId)}/edit`, {
          method: "POST",
          body: JSON.stringify(body),
        });
        vp.selectedFaces.clear();
        vp.selectedEdges.clear();
        vp.selectedVertices.clear();
        await refreshAll();
        toast(`Applied ${rawOp}: ${res.stats.vertex_count}v / ${res.stats.face_count}f`, "ok");
      } catch (err) {
        toast(err.message, "err");
      }
    };
  });

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "Tab") { e.preventDefault(); setEditMode(!state.editMode); }
    else if (e.key === "f" || e.key === "F") vp.fitScene(state.selectedId);
    else if (e.key === "1") vp.setPresetView("front");
    else if (e.key === "3") vp.setPresetView("right");
    else if (e.key === "7") vp.setPresetView("top");
    else if (e.key === " ") { e.preventDefault(); $("tl-play").click(); }
  });

  // Boot
  refreshAll(true);
  loadMcpInfo();
  connectWebSocket();
  searchTextures();
})();
