// Three.js rendering of the volatility surface. Accessible fallback table.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { announceAssertive, announcePolite } from "./live-region.js";

let renderer;
let scene;
let camera;
let controls;
let surfaceMesh;
let violationPoints;
let autoRotate = false;
let animationId = null;
const canvasEl = document.getElementById("surface-canvas");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function buildSceneIfNeeded() {
  if (renderer) return true;
  if (!canvasEl) return false;
  try {
    const width = canvasEl.clientWidth;
    const height = canvasEl.clientHeight || Math.round(width * 9 / 16);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(width, height, false);
    canvasEl.innerHTML = "";
    canvasEl.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(3.2, 2.4, 3.2);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = !reducedMotion;
    controls.dampingFactor = 0.1;
    controls.enablePan = false;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 0.6;

    const ambient = new THREE.AmbientLight(0x8899bb, 0.6);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(5, 10, 7);
    scene.add(dir);

    const grid = new THREE.GridHelper(3, 12, 0x334155, 0x1f2937);
    grid.position.y = -0.01;
    scene.add(grid);

    window.addEventListener("resize", handleResize);
    return true;
  } catch (err) {
    console.error(err);
    announceAssertive("Error, 3D surface failed to load. See data table below.");
    showFallbackTable();
    return false;
  }
}

function handleResize() {
  if (!renderer || !camera) return;
  const width = canvasEl.clientWidth;
  const height = canvasEl.clientHeight || Math.round(width * 9 / 16);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function makeSurfaceGeometry(strikes, tenors, iv) {
  const nStrike = strikes.length;
  const nTenor = tenors.length;
  const geom = new THREE.PlaneGeometry(2.2, 2.2, nStrike - 1, nTenor - 1);
  const pos = geom.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  const sMin = Math.min(...strikes);
  const sMax = Math.max(...strikes);
  const tMin = Math.min(...tenors);
  const tMax = Math.max(...tenors);
  let vmin = Infinity, vmax = -Infinity;
  for (let i = 0; i < iv.length; i++) for (let j = 0; j < iv[i].length; j++) {
    vmin = Math.min(vmin, iv[i][j]);
    vmax = Math.max(vmax, iv[i][j]);
  }
  for (let i = 0; i < pos.count; i++) {
    const col = i % nStrike;
    const row = Math.floor(i / nStrike);
    const s = strikes[col];
    const t = tenors[row];
    const v = iv[row][col];
    const nx = (s - sMin) / (sMax - sMin) * 2.2 - 1.1;
    const nz = (t - tMin) / (tMax - tMin) * 2.2 - 1.1;
    const h = ((v - vmin) / Math.max(1e-6, vmax - vmin)) * 1.2;
    pos.setXYZ(i, nx, h, nz);
    const c = ivToColour(v, vmin, vmax);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geom.computeVertexNormals();
  return geom;
}

function ivToColour(v, vmin, vmax) {
  const t = Math.max(0, Math.min(1, (v - vmin) / Math.max(1e-6, vmax - vmin)));
  // Accessible gradient from teal-blue to amber-pink, avoids red-green.
  const a = new THREE.Color(0x38BDF8);
  const b = new THREE.Color(0xF472B6);
  return a.clone().lerp(b, t);
}

function updateSummary(bundle) {
  const summary = document.getElementById("surface-summary");
  if (!summary) return;
  const m = bundle.meta;
  const metrics = bundle.metrics;
  const atmAvg =
    bundle.surface.iv.map((row) => row[Math.floor(row.length / 2)])
      .reduce((p, c) => p + c, 0) / bundle.surface.iv.length;
  summary.textContent =
    `The surface covers ${m.n_strikes} strikes and ${m.n_tenors} tenors, ` +
    `average at the money implied volatility is ${(atmAvg * 100).toFixed(2)} percent, ` +
    `SVI root mean squared error is ${metrics.svi_rmse.toFixed(4)} and ` +
    `combined RMSE after the neural residual is ${metrics.combined_rmse.toFixed(4)}. ` +
    `${metrics.total_violations} static arbitrage violations were flagged.`;
}

function populateDataTable(bundle) {
  const thead = document.getElementById("surface-thead-row");
  const tbody = document.getElementById("surface-tbody");
  if (!thead || !tbody) return;
  thead.innerHTML = "";
  tbody.innerHTML = "";
  const strikes = bundle.surface.strikes;
  const tenors = bundle.surface.tenors;
  const iv = bundle.surface.iv;

  const tenorHeader = document.createElement("th");
  tenorHeader.scope = "col";
  tenorHeader.textContent = "Tenor (years)";
  thead.appendChild(tenorHeader);
  for (const s of strikes) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = `K ${s.toFixed(2)}`;
    thead.appendChild(th);
  }
  for (let i = 0; i < tenors.length; i++) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = tenors[i].toFixed(3);
    tr.appendChild(th);
    for (let j = 0; j < strikes.length; j++) {
      const td = document.createElement("td");
      td.textContent = (iv[i][j] * 100).toFixed(2) + "%";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function wireToggleTable() {
  const btn = document.getElementById("toggle-data-table");
  const table = document.getElementById("surface-data-table");
  if (!btn || !table) return;
  btn.addEventListener("click", () => {
    const expanded = btn.getAttribute("aria-expanded") === "true";
    btn.setAttribute("aria-expanded", String(!expanded));
    table.hidden = expanded;
    const label = btn.querySelector(".btn-label");
    if (label) label.textContent = expanded ? "Show data table" : "Hide data table";
  });
}

function wireCanvasKeyboard() {
  if (!canvasEl) return;
  canvasEl.addEventListener("keydown", (event) => {
    if (!camera) return;
    const step = 5 * Math.PI / 180;
    let handled = true;
    if (event.key === "ArrowLeft") camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), step);
    else if (event.key === "ArrowRight") camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), -step);
    else if (event.key === "ArrowUp") camera.position.y += 0.2;
    else if (event.key === "ArrowDown") camera.position.y = Math.max(0.2, camera.position.y - 0.2);
    else if (event.key === "+" || event.key === "=") camera.position.multiplyScalar(0.9);
    else if (event.key === "-" || event.key === "_") camera.position.multiplyScalar(1.1);
    else if (event.key === "Home") {
      camera.position.set(3.2, 2.4, 3.2);
      announcePolite("Camera view reset.");
    } else handled = false;
    if (handled) {
      camera.lookAt(0, 0.3, 0);
      event.preventDefault();
    }
  });
}

function wirePlayToggle() {
  const btn = document.getElementById("play-toggle");
  if (!btn || !controls) return;
  btn.addEventListener("click", () => {
    autoRotate = !autoRotate;
    if (reducedMotion) autoRotate = false;
    controls.autoRotate = autoRotate;
    btn.setAttribute("aria-pressed", String(autoRotate));
    const label = btn.querySelector(".btn-label");
    if (label) label.textContent = autoRotate ? "Pause auto rotate" : "Play auto rotate";
    announcePolite(autoRotate ? "Auto rotation on." : "Auto rotation off.");
  });
}

function startLoop() {
  if (animationId) cancelAnimationFrame(animationId);
  function tick() {
    animationId = requestAnimationFrame(tick);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
  }
  tick();
}

function addViolationMarkers(bundle) {
  if (!scene) return;
  if (violationPoints) {
    scene.remove(violationPoints);
    violationPoints.geometry.dispose();
  }
  const strikes = bundle.surface.strikes;
  const tenors = bundle.surface.tenors;
  const sMin = Math.min(...strikes);
  const sMax = Math.max(...strikes);
  const tMin = Math.min(...tenors);
  const tMax = Math.max(...tenors);
  const group = new THREE.Group();
  for (const v of bundle.violations.slice(0, 80)) {
    const nx = (v.strike - sMin) / Math.max(1e-6, sMax - sMin) * 2.2 - 1.1;
    const nz = (v.tenor - tMin) / Math.max(1e-6, tMax - tMin) * 2.2 - 1.1;
    const colour = v.severity === "high" ? 0xF87171 : v.severity === "medium" ? 0xF59E0B : 0x60A5FA;
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.035, 12, 12),
      new THREE.MeshStandardMaterial({ color: colour, emissive: colour, emissiveIntensity: 0.3 })
    );
    sphere.position.set(nx, 1.25, nz);
    group.add(sphere);
  }
  violationPoints = group;
  scene.add(group);
}

export function showFallbackTable() {
  const table = document.getElementById("surface-data-table");
  const btn = document.getElementById("toggle-data-table");
  if (table) table.hidden = false;
  if (btn) btn.setAttribute("aria-expanded", "true");
}

export function renderSurface(bundle) {
  const ok = buildSceneIfNeeded();
  populateDataTable(bundle);
  updateSummary(bundle);
  wireToggleTable();
  wireCanvasKeyboard();
  if (!ok) return;
  if (surfaceMesh) {
    scene.remove(surfaceMesh);
    surfaceMesh.geometry.dispose();
  }
  const geom = makeSurfaceGeometry(bundle.surface.strikes, bundle.surface.tenors, bundle.surface.iv);
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    flatShading: false,
    roughness: 0.6,
    metalness: 0.1,
  });
  surfaceMesh = new THREE.Mesh(geom, mat);
  surfaceMesh.rotation.x = -Math.PI / 2;
  scene.add(surfaceMesh);
  addViolationMarkers(bundle);
  wirePlayToggle();
  startLoop();
  const loading = document.getElementById("surface-loading");
  if (loading) loading.remove();
}
