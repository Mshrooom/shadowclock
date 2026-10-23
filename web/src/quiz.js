// "Play the model": the benchmark question, asked of a person. Two north-up nadir renders of the same city and date at
// different local mean solar times; which is earlier? Same sampling rules as the dataset: sun elevation 20-70 deg in
// both images, gaps of 1, 2 or 4 hours, hours on the hour. Answers stay in this browser (localStorage) and can be exported.

const KEY = "shadowclock_quiz_v1";
const GAPS = [1, 2, 4];

const store = {
  mem: [],
  load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (e) { return this.mem; }
  },
  save(rows) {
    this.mem = rows;
    try { localStorage.setItem(KEY, JSON.stringify(rows)); } catch (e) { /* private mode: keep in memory only */ }
  },
};

const wilson = (k, n, z = 1.96) => {
  if (!n) return [0, 0];
  const p = k / n, d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
};

const hhmm = (h) => `${String(Math.floor(h)).padStart(2, "0")}:${String(Math.round((h % 1) * 60)).padStart(2, "0")}`;

export function initQuiz(api) {
  const $ = (id) => document.getElementById(id);
  const cvA = $("q-a"), cvB = $("q-b"), hint = $("q-hint"), fb = $("q-feedback"), stats = $("q-stats");
  let cur = null;
  let rows = store.load();

  function renderStats() {
    const n = rows.length, k = rows.filter((r) => r.correct).length;
    if (!n) { stats.textContent = "No answers yet."; return; }
    const [lo, hi] = wilson(k, n);
    const byGap = GAPS.map((g) => {
      const rr = rows.filter((r) => r.dt === g);
      return rr.length ? `${g} h: ${Math.round((100 * rr.filter((r) => r.correct).length) / rr.length)}% (${rr.length})` : null;
    }).filter(Boolean).join(" · ");
    stats.textContent = `${k}/${n} correct = ${((100 * k) / n).toFixed(0)}% (95% CI ${(100 * lo).toFixed(0)}-${(100 * hi).toFixed(0)}%). ${byGap}`;
  }

  function candidates(city, date) {
    const ok = [];
    for (let h = 0; h <= 23; h++) {
      const el = api.sunAt(city, date, h).el;
      ok.push(el >= 20 && el <= 70);
    }
    const pairs = [];
    for (const dt of GAPS) for (let h = 0; h + dt <= 23; h++) if (ok[h] && ok[h + dt]) pairs.push({ h, dt });
    return pairs;
  }

  async function newPair() {
    fb.textContent = "";
    fb.className = "q-feedback";
    $("q-answers").classList.remove("hidden");
    $("q-next").classList.add("hidden");
    let city, date, cand;
    for (let tries = 0; tries < 50; tries++) {
      city = api.cities[Math.floor(Math.random() * api.cities.length)];
      date = city.dates[Math.floor(Math.random() * city.dates.length)];
      cand = candidates(city, date);
      if (cand.length) break;
    }
    if (!cand || !cand.length) { fb.textContent = "No valid pair found, try again."; return; }
    const { h, dt } = cand[Math.floor(Math.random() * cand.length)];
    const aEarlier = Math.random() < 0.5;
    const ta = aEarlier ? h : h + dt, tb = aEarlier ? h + dt : h;
    await api.setCity(city.id);
    api.renderNadir(cvA, date, ta);
    api.renderNadir(cvB, date, tb);
    const regime = $("q-r2").checked ? "R2" : "R1";
    const hemi = city.lat >= 0 ? "northern" : "southern";
    hint.textContent = regime === "R1"
      ? `The location is in the ${hemi} hemisphere. North is up.`
      : `Latitude ${city.lat.toFixed(1)}, longitude ${city.lon.toFixed(1)}, date ${date}. Times are local mean solar time. North is up.`;
    cur = { city: city.id, date, hemisphere: hemi, dt, regime, ta, tb, aEarlier };
  }

  function answer(choice) {
    if (!cur) return;
    const correct = (choice === "A") === cur.aEarlier;
    rows.push({ ts: new Date().toISOString(), city: cur.city, date: cur.date, hemisphere: cur.hemisphere, regime: cur.regime,
      dt: cur.dt, a_time: cur.ta, b_time: cur.tb, answer: choice, correct });
    store.save(rows);
    fb.className = "q-feedback " + (correct ? "ok" : "bad");
    fb.textContent = `${correct ? "Correct" : "Not quite"}: A was ${hhmm(cur.ta)}, B was ${hhmm(cur.tb)} (gap ${cur.dt} h, local mean solar time).`;
    $("q-answers").classList.add("hidden");
    $("q-next").classList.remove("hidden");
    cur = null;
    renderStats();
  }

  $("q-new").addEventListener("click", newPair);
  $("q-next").addEventListener("click", newPair);
  $("q-pickA").addEventListener("click", () => answer("A"));
  $("q-pickB").addEventListener("click", () => answer("B"));
  $("q-reset").addEventListener("click", () => {
    if (rows.length && !confirm("Delete all stored answers in this browser?")) return;
    rows = [];
    store.save(rows);
    renderStats();
  });
  $("q-export").addEventListener("click", () => {
    const head = "ts,city,date,hemisphere,regime,dt_hours,a_time,b_time,answer,correct\n";
    const body = rows.map((r) => [r.ts, r.city, r.date, r.hemisphere, r.regime, r.dt, r.a_time, r.b_time, r.answer, r.correct ? 1 : 0].join(",")).join("\n");
    const url = URL.createObjectURL(new Blob([head + body + "\n"], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "shadowclock_human_answers.csv";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  renderStats();
  return { newPair };
}
