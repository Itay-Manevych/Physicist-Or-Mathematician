/* Guess the department: a separate 7-way mode (its own data in dept/data.json). Uses the fight engine in app.js.
 * Department keys are prefixed ("d:math") so they never clash with the main game's "phys"/"math". */
const DEPT_COLORS = { physics: "#22D3EE", math: "#FF5CB0", chem: "#3DDC84", bio: "#A3E635", cs: "#FB923C", phil: "#B7A8F0", econ: "#FFD93D" };
const DEPT_SHORT = { physics: "PHYS", math: "MATH", chem: "CHEM", bio: "BIO", cs: "CS", phil: "PHIL", econ: "ECON" };
let deptPool = null;

async function loadDept() {
  if (deptPool) return deptPool;
  const d = await (await fetch("dept/data.json", { cache: "no-store" })).json();
  const keys = Object.keys(d.departments);
  for (const k of keys) {  // register labels/colours with the shared fight engine
    LABEL[`d:${k}`] = d.departments[k];
    SHORT[`d:${k}`] = DEPT_SHORT[k] || k.toUpperCase();
    COLOR[`d:${k}`] = DEPT_COLORS[k] || "var(--lav-2)";
  }
  deptPool = d.people.map((p) => ({
    ...p, answer: `d:${p.answer}`, p: Object.fromEntries(Object.entries(p.p).map(([k, v]) => [`d:${k}`, v])),
  }));
  $("deptChoices").innerHTML = keys.map((k, i) =>
    `<button type="button" class="dept-btn" data-key="d:${k}" style="background: ${COLOR[`d:${k}`]}"><span class="dept-num">${i + 1}</span>${d.departments[k]}</button>`).join("");
  for (const b of $("deptChoices").querySelectorAll("button")) {
    b.onclick = () => {
      if (!game || game.mode !== "dept" || busy || game.log[game.i]) return;
      b.classList.add("picked");
      for (const x of $("deptChoices").querySelectorAll("button")) x.disabled = true;
      guess(b.dataset.key);
    };
  }
  return deptPool;
}

function pickDept() {
  let seen = new Set(store.get("pom-dept-seen", []));
  let fresh = deptPool.filter((p) => !seen.has(p.id));
  if (fresh.length < ROUNDS) { seen = new Set(); fresh = deptPool.slice(); }
  const chosen = shuffle(fresh).slice(0, ROUNDS);
  chosen.forEach((p) => seen.add(p.id));
  store.set("pom-dept-seen", [...seen]);
  return chosen;
}

async function startDept() {
  Sfx.start();
  try { await loadDept(); } catch {
    $("deptBtn").textContent = "DEPARTMENTS: COMING SOON";
    return;
  }
  newGame("dept", pickDept(), MACHINE);
  show("fight");
  renderRound();
}

$("deptBtn").onclick = startDept;
// the mode only appears once its data has been built (game/dept/data.json)
fetch("dept/data.json", { method: "HEAD", cache: "no-store" }).then((r) => { if (r.ok) $("deptRow").hidden = false; }).catch(() => {});
