import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { sunAtLocalMean, sunDirection } from "./solar.js";
import { decodeHeights, buildBuildingGeometry } from "./terrain.js";
import { initQuiz } from "./quiz.js";

const CITIES = window.SC_CITIES || [];
const NAMES = { newyork: "New York", buenosaires: "Buenos Aires", capetown: "Cape Town" };
const nice = (id) => NAMES[id] || id.charAt(0).toUpperCase() + id.slice(1);
const $ = (id) => document.getElementById(id);
const FIELD = 256; // metres: 512 px x 0.5 m
const DOME = 300;

const state = { city: null, date: "2027-06-21", hour: 12, view: "orbit", playing: false, speed: 1 };
const cache = new Map();
let current = null; // {city, mesh}
let needsRender = true;

// ---------------------------------------------------------------- renderer, scene, lights
const canvas = $("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;

const scene = new THREE.Scene();
scene.background = new THREE.Color("#9fc4ea");
const hemi = new THREE.HemisphereLight(0xcfe0ff, 0x6f7f5f, 0.6);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 2.5);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
const sc = sun.shadow.camera;
sc.left = -DOME; sc.right = DOME; sc.top = DOME; sc.bottom = -DOME; sc.near = 10; sc.far = 1400;
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.25;
scene.add(sun, sun.target);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(FIELD * 3, FIELD * 3),
  new THREE.MeshStandardMaterial({ color: "#86a174", roughness: 1, metalness: 0 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
const frame = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.PlaneGeometry(FIELD, FIELD)),
  new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 })
);
frame.rotation.x = -Math.PI / 2;
frame.position.y = 0.05;
scene.add(frame);

const sunBall = new THREE.Mesh(new THREE.SphereGeometry(7, 20, 12), new THREE.MeshBasicMaterial({ color: 0xffd35a }));
scene.add(sunBall);
const pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd35a, transparent: true, opacity: 0.75 }));
scene.add(pathLine);

// ---------------------------------------------------------------- cameras
const persp = new THREE.PerspectiveCamera(40, 1, 5, 4000);
persp.position.set(210, 190, 260);
const ortho = new THREE.OrthographicCamera(-FIELD / 2, FIELD / 2, FIELD / 2, -FIELD / 2, 1, 2000);
ortho.position.set(0, 800, 0);
ortho.up.set(0, 0, -1); // north (-z) is up on screen, east (+x) right: the benchmark image convention
ortho.lookAt(0, 0, 0);
const controls = new OrbitControls(persp, canvas);
controls.target.set(0, 20, 0);
controls.maxPolarAngle = Math.PI / 2 - 0.02;
controls.minDistance = 40;
controls.maxDistance = 1200;
controls.addEventListener("change", () => (needsRender = true));
controls.update();
const activeCamera = () => (state.view === "orbit" ? persp : ortho);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  persp.aspect = w / h;
  persp.updateProjectionMatrix();
  const a = w / h, s = FIELD / 2;
  ortho.left = -s * Math.max(a, 1); ortho.right = s * Math.max(a, 1);
  ortho.top = s / Math.min(a, 1); ortho.bottom = -s / Math.min(a, 1);
  ortho.updateProjectionMatrix();
  needsRender = true;
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------- sun, sky, path
const lerpHex = (a, b, t) => new THREE.Color(a).lerp(new THREE.Color(b), Math.min(1, Math.max(0, t)));
const cityById = (id) => CITIES.find((c) => c.id === id);
const sunAt = (city, date, hour) => sunAtLocalMean(city.lat, city.lon, date, hour);

// flat = benchmark lighting: sun strength E0 / sin(elevation) keeps horizontal-surface brightness constant, so image
// brightness is NOT a time cue (this is the dataset's default; see generate_dataset.py --physical-brightness).
function applySun(date, hour, flat = state.view === "nadir") {
  const city = cityById(state.city);
  const { az, el } = sunAt(city, date, hour);
  const d = sunDirection(az, el);
  sun.position.set(d[0] * 600, Math.max(d[1], 0.02) * 600, d[2] * 600);
  sun.target.position.set(0, 0, 0);
  const up = el > 0;
  const s = Math.sin((Math.max(el, 0) * Math.PI) / 180);
  if (flat && up) {
    sun.intensity = 2.2 / Math.max(s, 0.2);
    sun.color.set("#fff4e6");
    hemi.intensity = 0.55;
  } else {
    sun.intensity = up ? Math.min(3.0, 0.35 + s * 3.0) : 0;
    sun.color.copy(lerpHex("#ffb070", "#fff4e6", el / 18));
    hemi.intensity = up ? 0.35 + 0.35 * s : 0.12;
  }
  let sky = lerpHex("#101a2c", "#e8a47c", (el + 8) / 8);
  if (el > 0) sky = lerpHex("#e8a47c", "#9fc4ea", el / 14);
  scene.background = sky;
  sunBall.position.set(d[0] * DOME, d[1] * DOME, d[2] * DOME);
  sunBall.visible = up && state.view === "orbit";
  return { az, el };
}

function buildPath() {
  const city = cityById(state.city);
  const pts = [];
  for (let h = 0; h <= 24; h += 0.1) {
    const { az, el } = sunAt(city, state.date, h);
    if (el < 0) continue;
    const d = sunDirection(az, el);
    pts.push(new THREE.Vector3(d[0] * DOME, d[1] * DOME, d[2] * DOME));
  }
  pathLine.geometry.dispose();
  pathLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  pathLine.visible = state.view === "orbit" && pts.length > 1;
}

// ---------------------------------------------------------------- cities
async function ensureCity(id) {
  if (cache.has(id)) return cache.get(id);
  const city = cityById(id);
  const hdm = await decodeHeights(city.h);
  const geo = buildBuildingGeometry(hdm, city.size, city.gsd);
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  const rec = { mesh, tris: geo.getAttribute("position").count / 3 };
  cache.set(id, rec);
  return rec;
}

async function setCity(id) {
  const rec = await ensureCity(id);
  if (current) scene.remove(current.mesh);
  current = rec;
  scene.add(rec.mesh);
  state.city = id;
  const city = cityById(id);
  frameCamera(city);
  if (!city.dates.includes(state.date) && !/^\d{4}-\d{2}-\d{2}$/.test(state.date)) state.date = city.dates[0];
  $("city").value = id;
  updateCityInfo();
  buildPath();
  update();
}

function frameCamera(city) {
  const dist = Math.max(340, city.maxHeight * 1.7);
  persp.position.set(dist * 0.62, dist * 0.56, dist * 0.78);
  controls.target.set(0, Math.min(city.maxHeight * 0.12, 40), 0);
  controls.update();
  needsRender = true;
}

function updateCityInfo() {
  const c = cityById(state.city);
  const hemiName = c.lat >= 0 ? "northern" : "southern";
  const rows = [
    ["Location", `${c.lat.toFixed(3)}, ${c.lon.toFixed(3)} (${hemiName} hemisphere)`],
    ["Buildings in view", `${c.nBuildings}, ${(100 * c.builtFraction).toFixed(0)}% of the 256 m field built up`],
    ["Heights from OSM", `${(100 * c.explicitHeightFraction).toFixed(0)}% (the rest are defaults), tallest ${c.maxHeight.toFixed(0)} m`],
  ];
  if (c.baselineR1 != null) {
    rows.push(["Pixel baseline, R1", `${c.baselineR1.toFixed(0)}% [${c.baselineR1Lo.toFixed(0)}-${c.baselineR1Hi.toFixed(0)}], ${c.nPairs} pairs`]);
    rows.push(["Pixel baseline, R2", `${c.baselineR2.toFixed(0)}%`]);
  }
  $("cityinfo").innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

// ---------------------------------------------------------------- UI sync, compass, hash
const pad = (n) => String(n).padStart(2, "0");
const clock = (h) => `${pad(Math.floor(h) % 24)}:${pad(Math.floor((h % 1) * 60))}`;

function updateCompass(az, el) {
  const cam = activeCamera();
  cam.updateMatrixWorld();
  const inv = cam.matrixWorldInverse;
  const toAngle = (x, z) => {
    const v = new THREE.Vector3(x, 0, z).transformDirection(inv);
    return (Math.atan2(v.x, v.y) * 180) / Math.PI;
  };
  const a = (az * Math.PI) / 180;
  const sunA = toAngle(Math.sin(a), -Math.cos(a)); // direction toward the sun on screen
  const nAng = toAngle(0, -1);
  $("cp-n").setAttribute("transform", `rotate(${nAng} 50 50)`);
  $("cp-ntxt").setAttribute("transform", `rotate(${-nAng} 50 16)`);
  const sa = $("cp-sun"), sh = $("cp-shadow");
  sa.setAttribute("transform", `rotate(${sunA} 50 50)`);
  sh.setAttribute("transform", `rotate(${sunA + 180} 50 50)`);
  const vis = el > 0 ? "visible" : "hidden";
  sa.style.visibility = vis;
  sh.style.visibility = vis;
}

function update() {
  if (!state.city) return;
  const { az, el } = applySun(state.date, state.hour);
  $("time").value = state.hour;
  $("timeout").textContent = clock(state.hour);
  $("az").textContent = `${az.toFixed(1)}°`;
  $("el").textContent = `${el.toFixed(1)}°`;
  $("shbear").textContent = el > 0 ? `${((az + 180) % 360).toFixed(1)}°` : "none (sun below horizon)";
  const inWindow = el >= 20 && el <= 70;
  $("window").textContent = el <= 0 ? "night" : inWindow ? "inside the benchmark's 20-70° window" : el < 20 ? "below the benchmark's 20° floor" : "above the 70° cap (almost no shadow)";
  $("window").className = inWindow ? "tag good" : "tag";
  updateCompass(az, el);
  try {
    history.replaceState(null, "", `#city=${state.city}&date=${state.date}&t=${state.hour.toFixed(2)}&view=${state.view}`);
  } catch (e) { /* file:// in some browsers */ }
  needsRender = true;
}

function setView(v) {
  state.view = v;
  $("view-orbit").classList.toggle("on", v === "orbit");
  $("view-nadir").classList.toggle("on", v === "nadir");
  controls.enabled = v === "orbit";
  pathLine.visible = v === "orbit";
  update();
}

function parseHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get("date") && /^\d{4}-\d{2}-\d{2}$/.test(p.get("date"))) state.date = p.get("date");
  if (p.get("t") && !Number.isNaN(parseFloat(p.get("t")))) state.hour = Math.min(24, Math.max(0, parseFloat(p.get("t"))));
  const c = p.get("city");
  state.view = p.get("view") === "nadir" ? "nadir" : "orbit";
  return CITIES.some((x) => x.id === c) ? c : "delhi";
}

// ---------------------------------------------------------------- offscreen nadir render for the quiz
const rt = new THREE.WebGLRenderTarget(512, 512, { samples: 4 });
rt.texture.colorSpace = THREE.SRGBColorSpace;
const rtBuf = new Uint8Array(512 * 512 * 4);
const nadirCam = new THREE.OrthographicCamera(-FIELD / 2, FIELD / 2, FIELD / 2, -FIELD / 2, 1, 2000);
nadirCam.position.set(0, 800, 0);
nadirCam.up.set(0, 0, -1);
nadirCam.lookAt(0, 0, 0);

function renderNadir(target, date, hour) {
  const keepFrame = frame.visible, keepBall = sunBall.visible, keepPath = pathLine.visible;
  frame.visible = sunBall.visible = pathLine.visible = false;
  applySun(date, hour, true);
  sunBall.visible = false;
  renderer.setRenderTarget(rt);
  renderer.render(scene, nadirCam);
  renderer.readRenderTargetPixels(rt, 0, 0, 512, 512, rtBuf);
  renderer.setRenderTarget(null);
  const img = new ImageData(512, 512);
  for (let y = 0; y < 512; y++) {
    const src = (511 - y) * 512 * 4;
    img.data.set(rtBuf.subarray(src, src + 512 * 4), y * 512 * 4);
  }
  target.width = 512;
  target.height = 512;
  target.getContext("2d").putImageData(img, 0, 0);
  frame.visible = keepFrame; sunBall.visible = keepBall; pathLine.visible = keepPath;
  applySun(state.date, state.hour);
  needsRender = true;
}

// ---------------------------------------------------------------- loop
let last = performance.now();
function loop(now) {
  const dt = (now - last) / 1000;
  last = now;
  if (state.playing) {
    state.hour = (state.hour + dt * state.speed) % 24;
    update();
  }
  if (needsRender) {
    renderer.render(scene, activeCamera());
    if (state.view === "orbit") updateCompass(sunAt(cityById(state.city), state.date, state.hour).az, sunAt(cityById(state.city), state.date, state.hour).el);
    needsRender = false;
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- wire up
async function main() {
  if (!CITIES.length) { $("status").textContent = "cities.js missing: run scripts/build_world_data.py"; return; }
  const sel = $("city");
  sel.innerHTML = CITIES.map((c) => `<option value="${c.id}">${nice(c.id)}</option>`).join("");
  const startCity = parseHash();
  sel.addEventListener("change", () => setCity(sel.value));
  $("time").addEventListener("input", (e) => { state.hour = parseFloat(e.target.value); update(); });
  $("date").value = state.date;
  $("date").addEventListener("change", (e) => { if (e.target.value) { state.date = e.target.value; buildPath(); update(); } });
  document.querySelectorAll("[data-date]").forEach((b) => b.addEventListener("click", () => {
    state.date = b.dataset.date; $("date").value = state.date; buildPath(); update();
  }));
  $("play").addEventListener("click", () => { state.playing = !state.playing; $("play").textContent = state.playing ? "Pause" : "Play day"; });
  $("speed").addEventListener("change", (e) => (state.speed = parseFloat(e.target.value)));
  $("view-orbit").addEventListener("click", () => setView("orbit"));
  $("view-nadir").addEventListener("click", () => setView("nadir"));
  $("open-quiz").addEventListener("click", () => $("quiz").classList.toggle("hidden"));
  $("close-quiz").addEventListener("click", () => $("quiz").classList.add("hidden"));
  resize();
  $("status").textContent = "building scene...";
  await setCity(startCity);
  setView(state.view);
  $("status").textContent = "";
  const quiz = initQuiz({
    cities: CITIES,
    sunAt: (city, date, hour) => sunAt(city, date, hour),
    setCity,
    renderNadir,
  });
  $("open-quiz").addEventListener("click", () => { if (!$("quiz").classList.contains("hidden") && !$("q-hint").textContent) quiz.newPair(); });
  window.__shadowclock = { state, setCity, setView, update, renderNadir, renderer, scene, sun, redraw: () => (needsRender = true), stats: () => ({ tris: current.tris }) };
  requestAnimationFrame(loop);
}

main().catch((e) => { console.error(e); $("status").textContent = "error: " + e.message; });
