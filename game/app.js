/* Physicist or Mathematician? — game logic.
 * Modes: "machine" (vs the model), "daily" (same 10 for everyone today, vs the model), "duel" (live vs a friend). */
const ROUNDS = 10, HIT = 100 / ROUNDS;
const $ = (id) => document.getElementById(id);
const LABEL = { phys: "PHYSICIST", math: "MATHEMATICIAN" };
const SHORT = { phys: "PHYS", math: "MATH" };
const COLOR = { phys: "var(--phys)", math: "var(--math)" };
const RANKS = ["1ST", "2ND", "3RD", "4TH", "5TH", "6TH", "7TH", "8TH", "9TH", "10TH"];
const SITE = location.origin + location.pathname;
const MACHINE = { name: "MODEL", isModel: true };
let pool = [], byId = new Map(), game = null, board = null, busy = false, countdown = null;

/* ---------- helpers ---------- */
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
const session = {
  get(k) { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const capital = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");
const aiPick = (p) => typeof p.p === "object"
  ? Object.keys(p.p).reduce((a, b) => (p.p[a] >= p.p[b] ? a : b))
  : (p.p >= 0.5 ? "phys" : "math");
const aiConf = (p) => Math.round(100 * (typeof p.p === "object" ? Math.max(...Object.values(p.p)) : Math.max(p.p, 1 - p.p)));
const pct = (a, b) => (b ? Math.round((100 * a) / b) : 0);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const rankLabel = (n) => RANKS[n - 1] || `#${n}`;
const cleanName = (s) => s.normalize("NFKC").replace(/[^\p{L}\p{N} ._-]/gu, "").replace(/\s+/g, " ").trim().slice(0, 12);

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function show(id) {
  for (const s of document.querySelectorAll(".screen")) s.hidden = s.id !== id;
  window.scrollTo(0, 0);
}
function setBar(el, hp) { el.style.width = hp + "%"; el.style.background = hp > 40 ? "var(--hp-ok)" : "var(--hp-low)"; }
function replay(el, cls) { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); }
function damage(box) {
  const d = document.createElement("span");
  d.className = "dmg";
  d.textContent = `-${HIT}`;
  d.style[box.id === "hpBoxYou" ? "left" : "right"] = "8px";
  box.appendChild(d);
  setTimeout(() => d.remove(), 1000);
  replay(box.querySelector(".bar"), "flash");
}

async function api(url, body, token) {
  const headers = token ? { "x-pom-token": token } : {}; // session tokens go in a header, never in the URL
  const opts = body ? { method: "POST", headers: { ...headers, "Content-Type": "application/json" }, body: JSON.stringify(body) } : { headers };
  const r = await fetch(url, { cache: "no-store", ...opts });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error || r.status), { status: r.status });
  return data;
}

function renderList(el, rows, highlightId, empty = "NO SCORES YET. BE THE FIRST!") {
  if (!rows.length) { el.innerHTML = `<li><span>${empty}</span></li>`; return; }
  el.innerHTML = rows.map((r, k) => `<li class="${r.id && r.id === highlightId ? "me" : ""}">
      <span class="who">${rankLabel(r.rank || k + 1)}&nbsp; ${esc(r.name)}</span>
      <span>${r.you}/${ROUNDS}${r.ai !== undefined ? ` <span style="opacity:.6">vs ${r.ai}</span>` : ""}</span></li>`).join("");
}

/* ---------- photos: the main portrait plus up to two extra photos verified to be the same person ---------- */
const photosOf = (p) => [p.img, ...(p.more || []).map((m) => m.img)];
function renderStrip(strip, main, p, onPick = () => {}) {
  const photos = photosOf(p);
  main.src = photos[0];
  strip.hidden = photos.length < 2;
  strip.innerHTML = photos.length < 2 ? "" : photos.map((src, k) =>
    `<button type="button" class="${k ? "" : "on"}" aria-label="Photo ${k + 1} of ${photos.length}"><img class="photo" src="${src}" alt=""></button>`).join("");
  strip.querySelectorAll("button").forEach((b, k) => {
    b.onclick = () => {
      main.src = photos[k];
      strip.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      onPick(k);
    };
  });
}

/* ---------- game state ---------- */
function newGame(mode, fighters, opp, extra = {}) {
  game = { mode, fighters, opp, i: 0, log: [], savedId: null, ...extra };
  fighters.forEach((p) => photosOf(p).forEach((src) => { new Image().src = src; }));
}
function totals() {
  const done = game.log.filter(Boolean);
  return { you: done.filter((r) => r.youRight).length, ai: done.filter((r) => r.oppRight).length };
}
const oppName = () => game.opp.name.toUpperCase();

/* ---------- daily ---------- */
const today = () => Daily.today();
function dailyDone() { const d = store.get("pom-daily", null); return d && d.day === today() ? d : null; }
function streak() {
  const hist = store.get("pom-daily-history", {});
  let n = 0, t = Date.now();
  if (!hist[Daily.today(t)]) t -= 86400000; // today not played yet: the streak can still be alive from yesterday
  while (hist[Daily.today(t)]) { n++; t -= 86400000; }
  return n;
}

/* ---------- title ---------- */
async function renderTitle() {
  clearInterval(countdown);
  stopDuel();
  if (typeof stopParty === "function") stopParty();
  show("title");
  $("oneUp").textContent = String(store.get("pom-last", 0) * 100).padStart(6, "0");
  const d = dailyDone(), n = Daily.number(today()), st = streak();
  const prog = store.get("pom-daily-progress", null);
  const started = prog && prog.day === today() && prog.log.length;
  const badge = d ? `✓ ${d.you}/${ROUNDS}` : started ? `<span class="badge">RESUME ${started}/${ROUNDS}</span>` : '<span class="badge">NEW</span>';
  $("dailyBtn").innerHTML = `DAILY #${n} ${badge}${st > 1 ? ` 🔥${st}` : ""}`;
  try { board = await api(`/api/leaderboard?day=${today()}`); } catch { board = null; }
  const top = board ? board.top : store.get("pom-scores-v2", []).slice(0, 10);
  const best = top[0];
  $("hiScore").textContent = best ? `[${best.name.slice(0, 3).toUpperCase()}] ${best.you}/${ROUNDS}` : `[---] -/${ROUNDS}`;
  renderList($("fameList"), top.slice(0, 10));
  $("dailyTitle").textContent = board ? `DAILY #${n} · ${board.daily.count} PLAYED` : `DAILY #${n}`;
  renderList($("dailyList"), board ? board.daily.top.slice(0, 10) : [], null, board ? "NOBODY YET TODAY. GO FIRST!" : "OFFLINE");
  const s = board && board.stats;
  $("globalStats").textContent = s && s.rounds
    ? `HUMANS ${pct(s.human, s.rounds)}% · MACHINE ${pct(s.ai, s.rounds)}% · ${s.rounds.toLocaleString()} ROUNDS FOUGHT`
    : "CREDIT 01";
}

/* ---------- vs machine / daily ---------- */
function pickFighters() {
  let seen = new Set(store.get("pom-seen", []));
  let fresh = pool.filter((p) => !seen.has(p.id));
  if (fresh.length < ROUNDS) { seen = new Set(); fresh = pool.slice(); }
  const chosen = shuffle(fresh).slice(0, ROUNDS);
  chosen.forEach((p) => seen.add(p.id));
  store.set("pom-seen", [...seen]);
  return chosen;
}

function startMachine() {
  if (!pool.length) return;
  Sfx.start();
  newGame("machine", pickFighters(), MACHINE);
  show("fight");
  renderRound();
}

function startDaily() {
  if (!pool.length) return;
  const done = dailyDone();
  if (done) return showDailyDone(done);
  Sfx.start();
  const day = today();
  newGame("daily", Daily.fighters(day, pool.map((p) => p.id), ROUNDS).map((id) => byId.get(id)), MACHINE, { day });
  // quitting mid-daily doesn't give a fresh start: earlier answers are kept and you resume where you left off
  const progress = store.get("pom-daily-progress", null);
  if (progress && progress.day === day) {
    progress.log.forEach((x, i) => {
      const p = game.fighters[i], opp = aiPick(p);
      if (p && p.id === x.id) game.log[i] = { p, guess: x.g, oppPick: opp, youRight: x.g === p.answer, oppRight: opp === p.answer };
    });
    game.i = game.log.filter(Boolean).length;
    if (game.i >= ROUNDS) return renderResults();
  }
  show("fight");
  renderRound();
}

function showDailyDone(d) {
  newGame("daily", d.log.map((x) => byId.get(x.id)), MACHINE, { day: d.day, savedId: d.savedId, restored: true });
  d.log.forEach((x, i) => {
    const p = game.fighters[i], opp = aiPick(p);
    game.log[i] = { p, guess: x.g, oppPick: opp, youRight: x.g === p.answer, oppRight: opp === p.answer };
  });
  renderResults();
}

function hudLabels() {
  $("oppLabel").textContent = `${oppName()} · P2`;
  $("oppLineLabel").textContent = `P2 ${oppName()}`;
  $("roundLabel").textContent = game.mode === "daily" ? `DAILY #${Daily.number(game.day)}` : "ROUND";
}

function showFighter(p, i) {
  $("roundNo").textContent = `${i + 1}/${ROUNDS}`;
  renderStrip($("fStrip"), $("fPhoto"), p);
  const dept = game.mode === "dept";
  $("fClass").textContent = dept ? "DEPT: ???" : "CLASS: ???";
  $("fClass").style.background = "var(--white)";
  $("fMeta").textContent = `FIGHTER ${String(i + 1).padStart(2, "0")} · BORN ${p.born} · ${dept ? "DEPT" : "CLASS"}: CLASSIFIED`;
  $("fName").textContent = "MYSTERY SCIENTIST";
  $("fBio").textContent = dept ? "Which department do they work in? Seven doors, one face." : `${p.more ? `${photosOf(p).length} photos. ` : "Study the face. "}Chalkboard or particle accelerator?`;
  $("abChoices").hidden = dept;
  $("deptChoices").hidden = !dept;
  for (const b of $("deptChoices").querySelectorAll("button")) { b.disabled = false; b.classList.remove("picked"); }
  for (const b of [$("pickPhys"), $("pickMath")]) { b.disabled = false; b.classList.remove("picked"); }
  $("guessArea").hidden = false;
  $("revealArea").hidden = true;
  $("timer").hidden = true;
  updateHud();
  window.scrollTo(0, 0);
}

async function playIntro(i) {
  $("introRound").textContent = i === ROUNDS - 1 ? "FINAL ROUND" : `ROUND ${i + 1}`;
  $("intro").hidden = false;
  replay($("intro"), "intro");
  Sfx.round();
  await sleep(reducedMotion() ? 50 : 650);
  $("intro").hidden = true;
}

async function renderRound() {
  busy = true;
  hudLabels();
  showFighter(game.fighters[game.i], game.i);
  $("prompt").textContent = "THE MODEL HAS LOCKED IN. YOUR MOVE.";
  await playIntro(game.i);
  busy = false;
}

function updateHud() {
  const t = totals();
  const hpYou = 100 - HIT * t.ai, hpAi = 100 - HIT * t.you;
  $("hpYou").textContent = hpYou; $("hpAi").textContent = hpAi;
  setBar($("barYou"), hpYou); setBar($("barAi"), hpAi);
}

function guess(f) {
  if (!game || $("fight").hidden || busy) return;
  resetQuit();
  if (game.mode === "duel") return duelAnswer(f);
  if (game.log[game.i]) return;
  const p = game.fighters[game.i];
  showReveal(game.i, f, aiPick(p));
  if (game.mode === "daily") {
    store.set("pom-daily-progress", { day: game.day, log: game.log.map((r) => ({ id: r.p.id, g: r.guess })) });
  }
}

/** Reveal round i: record it, play hit effects, fill the reveal panel. myPick/oppPick may be null (ran out of time). */
function showReveal(i, myPick, oppPick) {
  const p = game.fighters[i];
  const youRight = myPick === p.answer, oppRight = oppPick === p.answer;
  game.log[i] = { p, guess: myPick, oppPick, youRight, oppRight };
  updateHud();
  if (youRight) damage($("hpBoxAi"));
  if (oppRight) { damage($("hpBoxYou")); if (!reducedMotion()) replay($("arena"), "shake"); }
  if (youRight && !oppRight) Sfx.hit(); else if (oppRight) Sfx.hurt(); else Sfx.miss();

  const [text, bg] = youRight && !oppRight ? ["CRITICAL HIT!", "var(--yellow)"]
    : youRight && oppRight ? ["TRADE!", "var(--hp-ok)"]
    : !youRight && oppRight ? ["OUCH!", "var(--hp-low)"]
    : ["DOUBLE MISS", "var(--lav-4)"];
  $("bannerText").textContent = text;
  $("banner").style.background = bg;
  $("answerLabel").textContent = LABEL[p.answer];
  $("fClass").textContent = "CLASS: " + LABEL[p.answer];
  $("fClass").style.background = COLOR[p.answer];
  $("fName").textContent = p.name.toUpperCase();
  $("fMeta").textContent = `FIGHTER ${String(i + 1).padStart(2, "0")} · BORN ${p.born}`;
  $("fBio").textContent = capital(p.bio) || LABEL[p.answer].toLowerCase();

  const line = (pick, right, target) => pick
    ? `Picked ${LABEL[pick]} · ${right ? `HIT −${HIT} to ${target}` : "MISSED"}`
    : "Ran out of time · MISSED";
  $("youLine").textContent = line(myPick, youRight, oppName());
  $("aiLine").textContent = game.opp.isModel
    ? `Picked ${LABEL[oppPick]} (${aiConf(p)}% sure) · ${oppRight ? `HIT −${HIT} to YOU` : "MISSED"}`
    : line(oppPick, oppRight, "YOU");

  $("cluesRow").hidden = !Array.isArray(p.clues);
  const seen = (p.clues || []).join(" · ") || "nothing obvious — pure gut feeling";
  const verdict = `${LABEL[aiPick(p)]} (${aiConf(p)}%)`;
  $("cluesRow").firstElementChild.textContent = game.opp.isModel ? "THE MACHINE NOTICED ▸ " : "THE MACHINE (NOT PLAYING) SAW ▸ ";
  $("clues").textContent = game.opp.isModel ? `${seen} → ${LABEL[aiPick(p)]}` : `${seen} → ${verdict}`;
  const links = [p.wiki && `<a href="${esc(p.wiki)}" target="_blank" rel="noopener">Read about them</a>`,
    `<a href="${esc(p.source)}" target="_blank" rel="noopener">Photo: Wikimedia Commons</a>`].filter(Boolean).join(" · ");
  $("lore").innerHTML = `${esc(loreLine(p, youRight, aiPick(p) === p.answer))} ${links}`;
  $("nextBtn").textContent = i === ROUNDS - 1 ? "FINAL SCORE ▶" : "NEXT ROUND ▶";
  $("nextBtn").disabled = false;
  $("guessArea").hidden = true;
  $("revealArea").hidden = false;
  $("nextBtn").focus({ preventScroll: true });
}

function loreLine(p, youRight, aiRight) {
  const c = aiConf(p);
  if (!aiRight && c >= 75) return `The machine was ${c}% sure — and dead wrong.`;
  if (aiRight && c >= 85) return `The machine called it at ${c}%. Textbook ${LABEL[p.answer].toLowerCase()}.`;
  if (c < 58) return `Even the machine was guessing (${c}%).`;
  return youRight ? "Your instincts held." : "Faces can lie.";
}

function nextRound() {
  if (!game) return;
  if (game.mode === "duel") return duelNext();
  if (!game.log[game.i]) return;
  Sfx.select();
  if (++game.i < ROUNDS) return renderRound();
  renderResults();
}

/* ---------- results ---------- */
function outcome() {
  const t = totals();
  return t.you > t.ai ? "win" : t.you < t.ai ? "lose" : "draw";
}

function renderResults() {
  clearInterval(countdown);
  const t = totals(), out = (game.outcome = outcome()), opp = oppName();
  const hpYou = 100 - HIT * t.ai, hpAi = 100 - HIT * t.you;
  if (game.mode === "machine") store.set("pom-last", t.you);
  if (game.mode === "daily" && !game.restored) {
    store.set("pom-daily", { day: game.day, you: t.you, ai: t.ai, log: game.log.map((r) => ({ id: r.p.id, g: r.guess })) });
    const hist = store.get("pom-daily-history", {});
    hist[game.day] = { you: t.you, ai: t.ai };
    store.set("pom-daily-history", hist);
    store.set("pom-daily-progress", null);
  }

  $("rKicker").textContent = { win: `${opp} · K.O.`, lose: "YOU · K.O.", draw: "DOUBLE K.O." }[out];
  $("rHeadline").textContent = { win: "YOU WIN!", lose: game.mode === "duel" ? `${opp} WINS` : "GAME OVER", draw: "DRAW!" }[out];
  $("rHeadline").style.color = { win: "var(--yellow)", lose: "var(--lose)", draw: "var(--white)" }[out];
  $("rYouName").textContent = (store.get("pom-name", "") || "YOU").toUpperCase();
  $("rOppName").textContent = opp;
  $("rYou").textContent = `${t.you}/${ROUNDS} · ${hpYou} HP`;
  $("rAi").textContent = `${t.ai}/${ROUNDS} · ${hpAi} HP`;
  setBar($("rBarYou"), hpYou); setBar($("rBarAi"), hpAi);
  const fooled = game.log.filter((r) => r.youRight && !r.oppRight).length;
  const daily = game.mode === "daily" ? `Daily #${Daily.number(game.day)}: ` : "";
  $("rTagline").textContent = out === "win"
    ? `${daily}You out-read ${game.opp.isModel ? "the machine" : opp} on ${fooled} fighter${fooled === 1 ? "" : "s"}.`
    : out === "lose" ? `${daily}${game.opp.isModel ? "The machine" : opp} read them better this time. ${game.mode === "daily" ? "Come back tomorrow!" : "Rematch?"}`
    : `${daily}Dead even. Best of three?`;

  $("log").innerHTML = "";
  game.log.forEach((r) => {
    const row = document.createElement("div");
    row.className = "log-row";
    const oppText = r.oppPick ? `${r.oppRight ? "✓" : "✗"} ${SHORT[r.oppPick]}${game.opp.isModel ? " " + aiConf(r.p) + "%" : ""}` : "✗ TIME";
    row.innerHTML = `
      <button type="button" class="thumb" style="box-shadow: 4px 4px 0 ${COLOR[r.p.answer]}" aria-label="Show photo of ${esc(r.p.name)}"><img class="photo" src="${r.p.img}" alt=""></button>
      <span class="px" style="flex: 1 1 220px; font-size: 13px; line-height: 1.6">${esc(r.p.name.toUpperCase())}</span>
      <span class="tag" style="flex: 0 0 auto; width: 200px; border-width: 3px; background: ${COLOR[r.p.answer]}">${LABEL[r.p.answer]}</span>
      <span style="flex: 0 0 150px; font-size: 28px"><span class="label-sm" style="font-size: 9px">YOU </span>${r.guess ? (r.youRight ? "✓ HIT" : "✗ MISS") : "✗ TIME"}</span>
      <span style="flex: 0 0 210px; font-size: 28px"><span class="label-sm" style="font-size: 9px">${esc(opp.slice(0, 8))} </span>${oppText}</span>`;
    row.querySelector("button").onclick = () => openModal(r.p);
    $("log").appendChild(row);
  });

  const ranked = game.mode === "machine" || game.mode === "daily";
  $("nameForm").hidden = !ranked || Boolean(game.savedId);
  $("nameLabel").textContent = game.mode === "daily" ? `ENTER YOUR NAME FOR DAILY #${Daily.number(game.day)}` : "ENTER YOUR NAME FOR THE LEADERBOARD";
  $("playerName").value = store.get("pom-name", "");
  $("saveBtn").disabled = false;
  $("nameNote").textContent = "";
  $("boardSection").hidden = !ranked;
  $("rematchBtn").hidden = game.mode !== "duel";
  $("rematchBtn").textContent = "REMATCH ⚔";
  $("againBtn").textContent = game.mode === "duel" ? "TITLE SCREEN" : "PLAY AGAIN";
  $("sharePanel").hidden = true;
  $("rankNote").textContent = "";
  if (ranked) renderResultBoard(game.rank, game.total);
  startCountdown();
  show("results");
  if (!game.restored) {
    replay($("rHeadline"), "ko-pop");
    (out === "win" ? Sfx.win : Sfx.lose)();
  }
}

async function renderResultBoard(rank, total) {
  if (game.mode === "daily" && !board) { try { board = await api(`/api/leaderboard?day=${game.day}`); } catch {} }
  $("boardTitle").textContent = game.mode === "daily" ? `DAILY #${Daily.number(game.day)} LEADERBOARD` : "LEADERBOARD";
  if (board) {
    const rows = (game.mode === "daily" ? board.daily.top : board.top).slice(0, 10);
    if (game.savedId && rank && !rows.some((r) => r.id === game.savedId)) {
      const t = totals();
      rows.push({ id: game.savedId, name: store.get("pom-name", "YOU"), you: t.you, ai: t.ai, rank });
    }
    renderList($("scoreList"), rows.map((r, k) => ({ ...r, rank: r.rank || k + 1 })), game.savedId);
    $("rankNote").textContent = rank ? `You placed #${rank} of ${total}.` : game.savedId ? "" : "Save your name to join the leaderboard.";
  } else {
    renderList($("scoreList"), game.mode === "daily" ? [] : store.get("pom-scores-v2", []).slice(0, 10), game.savedId, "OFFLINE");
    $("rankNote").textContent = "Online leaderboard unavailable right now.";
  }
}

async function saveScore(e) {
  e.preventDefault();
  const name = cleanName($("playerName").value);
  if (!name) { $("nameNote").textContent = "Pick a name with letters or numbers."; return; }
  store.set("pom-name", name);
  $("rYouName").textContent = name.toUpperCase();
  $("saveBtn").disabled = true;
  $("nameNote").textContent = "Saving…";
  const t = totals();
  try {
    const body = { name, guesses: game.log.map((r) => ({ id: r.p.id, g: r.guess })) };
    if (game.mode === "daily") body.daily = game.day;
    const res = await api("/api/leaderboard", body);
    board = { top: res.top, stats: res.stats, daily: res.daily };
    Object.assign(game, { savedId: res.entry.id, rank: res.rank, total: res.total });
    if (game.mode === "daily") store.set("pom-daily", { ...dailyDone(), savedId: res.entry.id });
    $("nameForm").hidden = true;
    renderResultBoard(res.rank, res.total);
  } catch (err) {
    if (err.status === 429) { $("nameNote").textContent = "Too many saves — wait a minute."; $("saveBtn").disabled = false; return; }
    if (game.mode === "machine") {
      const scores = store.get("pom-scores-v2", []);
      const entry = { id: `local-${Date.now()}`, name, you: t.you, ai: t.ai, t: Date.now() };
      scores.push(entry);
      scores.sort((a, b) => b.you - a.you || (b.you - b.ai) - (a.you - a.ai) || a.t - b.t);
      store.set("pom-scores-v2", scores.slice(0, 10));
      game.savedId = entry.id;
      board = null;
      $("nameForm").hidden = true;
      renderResultBoard();
    } else {
      $("nameNote").textContent = "Couldn't reach the leaderboard. Try again in a moment.";
      $("saveBtn").disabled = false;
    }
  }
}

/* ---------- sharing ---------- */
function shareText() {
  const t = totals(), name = store.get("pom-name", "");
  const grid = game.log.map((r) => (r.youRight ? "🟩" : "🟥")).join("");
  if (game.mode === "daily") {
    const st = streak();
    return `Physicist or Mathematician? Daily #${Daily.number(game.day)}\n${grid}\n${t.you}/${ROUNDS} vs the machine's ${t.ai}/${ROUNDS}${st > 1 ? ` · 🔥${st} day streak` : ""}\n${SITE}`;
  }
  if (game.mode === "dept") {
    return `Guess the Department: I scored ${t.you}/${ROUNDS} vs the machine's ${t.ai}/${ROUNDS} in Physicist or Mathematician?
${grid}
Seven departments, one face: ${SITE}`;
  }
  if (game.mode === "duel") {
    const verb = { win: "beat", lose: "lost to", draw: "tied with" }[game.outcome];
    return `I ${verb} ${game.opp.name} ${t.you}-${t.ai} in a Physicist or Mathematician? duel!\n${grid}\nYour turn: ${SITE}`;
  }
  const lead = { win: "I beat the AI!", lose: "The AI got me this time.", draw: "I tied the AI!" }[game.outcome];
  return `${lead} ${name ? name + " scored" : "I scored"} ${t.you}/${ROUNDS} vs the machine's ${t.ai}/${ROUNDS} in Physicist or Mathematician?\n${grid}\nCan you beat it? ${SITE}`;
}

async function shareNative(text, file) {
  if (!navigator.share || !matchMedia("(pointer: coarse)").matches) return false;
  const data = { text, title: "Physicist or Mathematician?" };
  if (file && navigator.canShare && navigator.canShare({ files: [file] })) data.files = [file];
  try { await navigator.share(data); return true; } catch (e) { return e.name === "AbortError"; }
}

async function share() {
  Sfx.select();
  const t = totals(), text = shareText();
  const card = await drawScorecard({
    name: store.get("pom-name", "") || "YOU", you: t.you, ai: t.ai, rounds: ROUNDS, outcome: game.outcome, log: game.log, url: SITE,
    oppName: game.opp.isModel ? "THE MACHINE" : oppName(),
    subtitle: game.mode === "daily" ? `DAILY #${Daily.number(game.day)}` : game.mode === "duel" ? "DUEL" : game.mode === "dept" ? "DEPARTMENTS" : "",
  });
  const file = card.blob && new File([card.blob], "physicist-or-mathematician.png", { type: "image/png" });
  if (await shareNative(text, file)) return;
  $("scorecardImg").src = card.url;
  $("downloadLink").href = card.url;
  $("shareText").textContent = text;
  $("xLink").href = "https://twitter.com/intent/tweet?text=" + encodeURIComponent(text);
  $("waLink").href = "https://wa.me/?text=" + encodeURIComponent(text);
  $("copyBtn").textContent = "COPY TEXT";
  $("sharePanel").hidden = false;
  $("sharePanel").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "nearest" });
}

/* ---------- live duel ---------- */
const duel = { code: null, token: null, state: null, offset: 0, poll: null, tick: null, phase: null, autoNext: null };
const serverNow = () => Date.now() + duel.offset;
const duelLink = (code) => `${SITE}?duel=${code}`;

function stopDuel() {
  clearInterval(duel.poll); clearInterval(duel.tick); clearTimeout(duel.autoNext);
  Object.assign(duel, { code: null, token: null, state: null, poll: null, tick: null, phase: null });
  if (new URLSearchParams(location.search).has("duel")) history.replaceState(null, "", SITE);
}

function openLobby() {
  stopDuel();
  show("lobby");
  $("lKicker").textContent = "VS A FRIEND · LIVE";
  $("lTitle").textContent = "DUEL";
  $("lText").textContent = "Create a room, send the link, and fight your friend live on the same 10 scientists.";
  $("lForm").hidden = false;
  $("lForm").dataset.join = "";
  $("lGo").textContent = "CREATE ROOM ▶";
  $("lName").value = store.get("pom-name", "");
  $("lNote").textContent = "";
  $("lInvite").hidden = true;
  $("lStatus").textContent = "";
}

async function openInvite(code) {
  show("lobby");
  $("lInvite").hidden = true;
  $("lStatus").textContent = "";
  const saved = session.get("pom-duel");
  if (saved && saved.code === code) return enterRoom(code, saved.token); // reloaded mid-duel: resume
  try {
    const s = await api(`/api/room?code=${code}`);
    if (s.players.guest) throw Object.assign(new Error("full"), { status: 409 });
    $("lKicker").textContent = "A NEW CHALLENGER APPROACHES!";
    $("lTitle").textContent = `${s.players.host.name.toUpperCase()} WANTS TO DUEL`;
    $("lText").textContent = "Same 10 scientists, live, 15 seconds a round. Pick a fighter name.";
    $("lForm").hidden = false;
    $("lForm").dataset.join = code;
    $("lGo").textContent = "JOIN DUEL ▶";
    $("lName").value = store.get("pom-name", "");
    $("lNote").textContent = "";
  } catch (err) {
    $("lKicker").textContent = "DUEL";
    $("lTitle").textContent = err.status === 409 ? "ROOM IS FULL" : "DUEL NOT FOUND";
    $("lText").textContent = err.status === 409 ? "Someone already joined this duel." : "This link expired or is wrong. Start your own!";
    $("lForm").hidden = true;
  }
}

async function lobbySubmit(e) {
  e.preventDefault();
  const name = cleanName($("lName").value);
  if (!name) { $("lNote").textContent = "Pick a name with letters or numbers."; return; }
  store.set("pom-name", name);
  $("lGo").disabled = true;
  $("lNote").textContent = "Connecting…";
  const code = $("lForm").dataset.join;
  try {
    const res = code ? await api(`/api/room?code=${code}`, { action: "join", name })
                     : await api("/api/room", { action: "create", name });
    Sfx.select();
    await enterRoom(res.code, res.token);
  } catch (err) {
    $("lNote").textContent = err.status === 409 ? "Someone else already joined this duel."
      : err.status === 503 ? "Duels need the online database — it isn't set up yet." : "Couldn't connect. Try again.";
  } finally {
    $("lGo").disabled = false;
  }
}

async function enterRoom(code, token) {
  stopDuel();
  Object.assign(duel, { code, token });
  session.set("pom-duel", { code, token });
  history.replaceState(null, "", `?duel=${code}`);
  await pollDuel();
  duel.poll = setInterval(pollDuel, 900);
  duel.tick = setInterval(tickDuel, 200);
}

async function pollDuel() {
  if (!duel.code) return;
  try {
    const s = await api(`/api/room?code=${duel.code}`, null, duel.token);
    if (!duel.code || s.code !== duel.code) return;
    duel.offset = s.now - Date.now();
    duel.state = s;
    renderDuel();
  } catch (err) {
    if (err.status === 404) { stopDuel(); openInvite(""); }
  }
}

function showInvite(code) {
  $("lForm").hidden = true;
  $("lInvite").hidden = false;
  $("lLink").textContent = duelLink(code);
  $("lWa").href = "https://wa.me/?text=" + encodeURIComponent(`Duel me in Physicist or Mathematician? ${duelLink(code)}`);
  $("lKicker").textContent = "ROOM " + code;
  $("lTitle").textContent = "WAITING FOR A CHALLENGER";
  $("lText").textContent = "Keep this page open. The fight starts as soon as your friend joins.";
  $("lStatus").textContent = "WAITING FOR YOUR FRIEND…";
}

function renderDuel() {
  const s = duel.state;
  const me = s.you, opp = me === "host" ? "guest" : "host";
  const other = s.players[opp];
  if (!other) { if ($("lobby").hidden) show("lobby"); showInvite(s.code); return; }

  if (!game || game.mode !== "duel" || game.code !== s.code) {
    newGame("duel", s.fighters.map((id) => byId.get(id)), { name: other.name, isModel: false }, { code: s.code, me, oppRole: opp });
    hudLabels();
  }
  if (game.ended) return checkRematch(s);
  if (s.left && s.left[opp]) return opponentLeft();
  const r = s.rounds.length - 1;
  if (r < 0) return;
  const round = s.rounds[r];
  // rounds revealed before (e.g. after a page reload) are recorded silently
  for (let k = 0; k < r; k++) {
    if (!game.log[k] && s.rounds[k].revealed) {
      const p = game.fighters[k], a = s.rounds[k].answers;
      game.log[k] = { p, guess: a[me], oppPick: a[opp], youRight: a[me] === p.answer, oppRight: a[opp] === p.answer };
    }
  }
  const now = serverNow();
  const phase = now < round.start ? `cd${r}` : !round.revealed ? `q${r}` : `r${r}`;
  const dropped = other.seen && now - other.seen > 20000;

  if (phase !== duel.phase) {
    duel.phase = phase;
    clearTimeout(duel.autoNext);
    if ($("fight").hidden) show("fight");
    game.i = r;
    if (phase.startsWith("cd")) {
      showFighter(game.fighters[r], r);
      game.shown = r;
      $("introRound").textContent = r === ROUNDS - 1 ? "FINAL ROUND" : `ROUND ${r + 1}`;
      $("intro").classList.add("hold");  // stays up until the round starts
      $("intro").hidden = false;
      Sfx.round();
    } else if (phase.startsWith("q")) {
      $("intro").hidden = true;
      $("intro").classList.remove("hold");
      if (game.shown !== r) { showFighter(game.fighters[r], r); game.shown = r; }  // e.g. after a reload
      $("timer").hidden = false;
    } else {
      $("intro").hidden = true;
      $("intro").classList.remove("hold");
      $("timer").hidden = true;
      if (game.shown !== r) { showFighter(game.fighters[r], r); game.shown = r; }
      showReveal(r, round.answers[me], round.answers[opp]);
      if (r < ROUNDS - 1) duel.autoNext = setTimeout(duelNext, 10000);
    }
  }

  if (phase.startsWith("q")) {
    const mine = round.answers[me];
    for (const [id, f] of [["pickPhys", "phys"], ["pickMath", "math"]]) {
      $(id).disabled = Boolean(mine);
      $(id).classList.toggle("picked", mine === f);
    }
    $("prompt").textContent = dropped ? `${oppName()} DISCONNECTED…`
      : mine ? `LOCKED IN. WAITING FOR ${oppName()}…`
      : round.answered[opp] ? `${oppName()} HAS LOCKED IN. YOUR MOVE.` : `${oppName()} IS THINKING…`;
  } else if (phase.startsWith("r") && r < ROUNDS - 1) {
    const ready = round.ready[me];
    $("nextBtn").disabled = ready;
    $("nextBtn").textContent = dropped ? `${oppName()} DISCONNECTED` : ready ? `WAITING FOR ${oppName()}…` : "NEXT ROUND ▶";
  }
}

function tickDuel() {
  if (!duel.state || !game || game.mode !== "duel" || game.ended) return;
  const round = duel.state.rounds[duel.state.rounds.length - 1];
  if (!round) return;
  const now = serverNow();
  if (duel.phase && duel.phase.startsWith("cd")) {
    const left = Math.max(1, Math.ceil((round.start - now) / 1000));
    $("introRound").textContent = `${game.i === ROUNDS - 1 ? "FINAL ROUND" : "ROUND " + (game.i + 1)} · ${left}`;
    if (now >= round.start) renderDuel();
  } else if (duel.phase && duel.phase.startsWith("q")) {
    const left = Math.max(0, round.deadline - now);
    $("timerBar").style.width = (100 * left) / duel.state.answerMs + "%";
    if (left <= 0) { $("pickPhys").disabled = $("pickMath").disabled = true; $("prompt").textContent = "TIME'S UP!"; }
  }
}

async function duelAnswer(f) {
  const s = duel.state;
  if (!s || !duel.phase || !duel.phase.startsWith("q")) return;
  const r = s.rounds.length - 1;
  if (s.rounds[r].answers[s.you]) return;
  Sfx.select();
  $("pickPhys").disabled = $("pickMath").disabled = true;
  $(f === "phys" ? "pickPhys" : "pickMath").classList.add("picked");
  $("prompt").textContent = `LOCKED IN. WAITING FOR ${oppName()}…`;
  try {
    const state = await api(`/api/room?code=${duel.code}`, { action: "answer", token: duel.token, round: r, g: f });
    if (duel.code !== state.code) return; // quit while the answer was in flight
    duel.state = state;
    renderDuel();
  } catch { /* the next poll shows the real state */ }
}

async function duelNext() {
  const s = duel.state;
  if (!s) return;
  clearTimeout(duel.autoNext);
  const r = s.rounds.length - 1;
  if (r === ROUNDS - 1) {
    game.ended = true;
    clearInterval(duel.tick);
    clearInterval(duel.poll);
    duel.poll = setInterval(pollDuel, 2500); // keep listening for a rematch
    return renderResults();
  }
  if (s.rounds[r].ready[s.you]) return;
  Sfx.select();
  $("nextBtn").disabled = true;
  $("nextBtn").textContent = `WAITING FOR ${oppName()}…`;
  try {
    const state = await api(`/api/room?code=${duel.code}`, { action: "next", token: duel.token, round: r });
    if (duel.code !== state.code) return;
    duel.state = state;
    renderDuel();
  } catch {}
}

function opponentLeft() {
  clearInterval(duel.poll); clearInterval(duel.tick); clearTimeout(duel.autoNext);
  if ($("fight").hidden) show("fight");
  $("intro").hidden = true;
  $("guessArea").hidden = false;
  $("revealArea").hidden = true;
  $("timer").hidden = true;
  $("pickPhys").disabled = $("pickMath").disabled = true;
  $("prompt").textContent = `${oppName()} LEFT THE DUEL. PRESS ◀ QUIT TO GO BACK.`;
}

function checkRematch(s) {
  if (s.rematch && s.rematch !== game.rematchCode) {
    $("rematchBtn").textContent = `${oppName()} WANTS A REMATCH! JOIN ⚔`;
    replay($("rematchBtn"), "ko-pop");
  }
}

async function rematch() {
  Sfx.select();
  const name = store.get("pom-name", "") || "PLAYER";
  $("rematchBtn").disabled = true;
  try {
    const res = await api(`/api/room?code=${duel.code}`, { action: "rematch", token: duel.token });
    game.rematchCode = res.code;
    if (res.token) await enterRoom(res.code, res.token);  // we created it: wait for the friend
    else {                                                 // the friend created it: join
      const joined = await api(`/api/room?code=${res.code}`, { action: "join", name });
      await enterRoom(joined.code, joined.token);
    }
  } catch {
    $("rematchBtn").textContent = "REMATCH FAILED — START A NEW DUEL";
  } finally {
    $("rematchBtn").disabled = false;
  }
}

async function shareInvite() {
  const link = $("lLink").textContent;
  if (await shareNative(`Duel me in Physicist or Mathematician? ${link}`)) return;
  try { await navigator.clipboard.writeText(link); $("lCopy").textContent = "COPIED!"; } catch {}
}

/* ---------- quitting mid-game ---------- */
let quitTimer = null;
function resetQuit() {
  clearTimeout(quitTimer);
  $("quitBtn").classList.remove("sure");
  $("quitBtn").textContent = "◀ QUIT";
}
function quit() {
  const b = $("quitBtn");
  if (!b.classList.contains("sure")) { // first tap asks for confirmation
    b.classList.add("sure");
    b.textContent = "SURE? TAP TO QUIT";
    clearTimeout(quitTimer);
    quitTimer = setTimeout(resetQuit, 3000);
    return;
  }
  resetQuit();
  Sfx.select();
  if (game && game.mode === "duel") {
    if (duel.code && !game.ended) api(`/api/room?code=${duel.code}`, { action: "leave", token: duel.token }).catch(() => {});
    session.set("pom-duel", null);
  }
  game = null;
  busy = false;
  $("intro").hidden = true;
  renderTitle();
}

/* ---------- misc ---------- */
function startCountdown() {
  clearInterval(countdown);
  let n = 9;
  $("continueText").textContent = `CONTINUE? ${n}`;
  countdown = setInterval(() => {
    n -= 1;
    $("continueText").textContent = n > 0 ? `CONTINUE? ${n}` : "INSERT COIN";
    if (n <= 0) clearInterval(countdown);
  }, 1000);
}

function openModal(p) {
  $("mClass").textContent = LABEL[p.answer];
  $("mHead").style.background = COLOR[p.answer];
  const sources = [p.source, ...(p.more || []).map((m) => m.source)];
  renderStrip($("mStrip"), $("mPhoto"), p, (k) => { $("mSource").href = sources[k]; });
  $("mPhoto").alt = `Photo of ${p.name}`;
  $("mName").textContent = p.name.toUpperCase();
  $("mMeta").textContent = `Born ${p.born}${p.bio ? " · " + capital(p.bio) : ""}`;
  $("mWiki").hidden = !p.wiki; $("mWiki").href = p.wiki || "#";
  $("mSource").href = p.source;
  $("modal").hidden = false;
  $("mClose").focus();
}
const closeModal = () => { $("modal").hidden = true; };

function renderMute() {
  $("muteBtn").textContent = Sfx.muted ? "SOUND: OFF" : "SOUND: ON";
  $("muteBtn").setAttribute("aria-pressed", String(Sfx.muted));
}

/* ---------- wiring ---------- */
$("startBtn").onclick = startMachine;
$("dailyBtn").onclick = startDaily;
$("friendBtn").onclick = () => { Sfx.select(); openLobby(); };
$("againBtn").onclick = () => { Sfx.select(); renderTitle(); };
$("titleBtn").onclick = () => { Sfx.select(); renderTitle(); };
$("lCancel").onclick = () => { Sfx.select(); renderTitle(); };
$("pickPhys").onclick = () => guess("phys");
$("pickMath").onclick = () => guess("math");
$("nextBtn").onclick = nextRound;
$("quitBtn").onclick = quit;
$("shareBtn").onclick = share;
$("rematchBtn").onclick = rematch;
$("shareClose").onclick = () => { $("sharePanel").hidden = true; };
$("copyBtn").onclick = async () => {
  try { await navigator.clipboard.writeText($("shareText").textContent); $("copyBtn").textContent = "COPIED!"; }
  catch { $("copyBtn").textContent = "SELECT & COPY"; }
};
$("lForm").onsubmit = lobbySubmit;
$("lShare").onclick = shareInvite;
$("lCopy").onclick = async () => { try { await navigator.clipboard.writeText($("lLink").textContent); $("lCopy").textContent = "COPIED!"; } catch {} };
$("nameForm").onsubmit = saveScore;
$("mClose").onclick = closeModal;
$("modal").onclick = (e) => { if (e.target.id === "modal") closeModal(); };
$("muteBtn").onclick = () => { Sfx.setMuted(!Sfx.muted); renderMute(); if (!Sfx.muted) Sfx.select(); };

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (!$("modal").hidden) { if (e.key === "Escape") closeModal(); return; }
  const k = e.key.toLowerCase();
  if (!$("title").hidden && (k === "enter" || k === " ") && e.target.tagName !== "A" && e.target.tagName !== "BUTTON") {
    e.preventDefault(); startMachine();
  } else if (!$("fight").hidden && $("revealArea").hidden && game && game.mode === "dept") {
    const b = $("deptChoices").querySelectorAll("button")[Number(k) - 1];
    if (b) b.click();
  } else if (!$("fight").hidden && $("revealArea").hidden && (k === "a" || k === "arrowleft")) guess("phys");
  else if (!$("fight").hidden && $("revealArea").hidden && (k === "b" || k === "arrowright")) guess("math");
});

renderMute();
fetch("data.json").then((r) => r.json()).then((d) => {
  pool = d.people;
  byId = new Map(pool.map((p) => [p.id, p]));
  $("modelAcc").textContent = Math.round(100 * d.model_accuracy) + "%";
  const q = new URLSearchParams(location.search);
  const hostSaved = session.get("pom-party-host");
  if (q.get("duel")) openInvite(q.get("duel").toUpperCase());
  else if (q.get("party")) openPartyJoin(q.get("party").toUpperCase());
  else if (q.get("tv") && hostSaved && hostSaved.code === q.get("tv").toUpperCase()) enterHost(hostSaved.code, hostSaved.hostToken);
  else renderTitle();
}).catch(() => {
  $("startBtn").textContent = "LOADING FAILED";
  $("startBtn").disabled = true;
  renderTitle();
});
