const ROUNDS = 10, HIT = 100 / ROUNDS;
const $ = (id) => document.getElementById(id);
const LABEL = { phys: "PHYSICIST", math: "MATHEMATICIAN" };
const SHORT = { phys: "PHYS", math: "MATH" };
const COLOR = { phys: "var(--phys)", math: "var(--math)" };
const RANKS = ["1ST", "2ND", "3RD", "4TH", "5TH", "6TH", "7TH", "8TH", "9TH", "10TH"];
const SITE = location.origin + location.pathname;
let pool = [], game = null, countdown = null, board = null, busy = false;

/* ---------- helpers ---------- */
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const capital = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");
const aiPick = (p) => (p.p >= 0.5 ? "phys" : "math");
const aiConf = (p) => Math.round(100 * Math.max(p.p, 1 - p.p));
const pct = (a, b) => (b ? Math.round((100 * a) / b) : 0);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

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

/* ---------- online leaderboard (falls back to scores on this device) ---------- */
const api = {
  async get() {
    const r = await fetch("/api/leaderboard", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    return r.json();
  },
  async post(body) {
    const r = await fetch("/api/leaderboard", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(data.error || r.status), { status: r.status });
    return data;
  },
};
async function loadBoard() {
  try { board = await api.get(); } catch { board = null; }
  return board;
}
function localScores() { return store.get("pom-scores-v2", []); }
const rankLabel = (n) => RANKS[n - 1] || `#${n}`;

function renderList(el, rows, highlightId) {
  if (!rows.length) { el.innerHTML = `<li><span>NO SCORES YET. BE THE FIRST!</span></li>`; return; }
  el.innerHTML = rows.map((r, k) => `<li class="${r.id && r.id === highlightId ? "me" : ""}">
      <span class="who">${rankLabel(r.rank || k + 1)}&nbsp; ${esc(r.name)}</span>
      <span>${r.you}/${ROUNDS}${r.ai !== undefined ? ` <span style="opacity:.6">vs ${r.ai}</span>` : ""}</span></li>`).join("");
}

/* ---------- title ---------- */
async function renderTitle() {
  clearInterval(countdown);
  show("title");
  $("oneUp").textContent = String(store.get("pom-last", 0) * 100).padStart(6, "0");
  await loadBoard();
  const rows = board ? board.top : localScores().slice(0, 10);
  const best = rows[0];
  $("hiScore").textContent = best ? `[${best.name.slice(0, 3).toUpperCase()}] ${best.you}/${ROUNDS}` : `[---] -/${ROUNDS}`;
  renderList($("fameList"), rows.slice(0, 10));
  const s = board && board.stats;
  $("globalStats").textContent = s && s.rounds
    ? `HUMANS ${pct(s.human, s.rounds)}% · MACHINE ${pct(s.ai, s.rounds)}% · ${s.rounds.toLocaleString()} ROUNDS FOUGHT`
    : "CREDIT 01";
}

/* ---------- fight ---------- */
function pickFighters() {
  let seen = new Set(store.get("pom-seen", []));
  let fresh = pool.filter((p) => !seen.has(p.id));
  if (fresh.length < ROUNDS) { seen = new Set(); fresh = pool.slice(); }
  const chosen = shuffle(fresh).slice(0, ROUNDS);
  chosen.forEach((p) => seen.add(p.id));
  store.set("pom-seen", [...seen]);
  chosen.forEach((p) => { new Image().src = p.img; });
  return chosen;
}

function startGame() {
  if (!pool.length) return;
  Sfx.start();
  game = { fighters: pickFighters(), i: 0, you: 0, ai: 0, log: [] };
  show("fight");
  renderRound();
}

async function renderRound() {
  const p = game.fighters[game.i];
  game.guess = null;
  busy = true;
  $("roundNo").textContent = `${game.i + 1}/${ROUNDS}`;
  updateHud();
  $("fPhoto").src = p.img;
  $("fClass").textContent = "CLASS: ???";
  $("fClass").style.background = "var(--white)";
  $("fMeta").textContent = `FIGHTER ${String(game.i + 1).padStart(2, "0")} · BORN ${p.born} · CLASS: CLASSIFIED`;
  $("fName").textContent = "MYSTERY SCIENTIST";
  $("fBio").textContent = "Study the face. Chalkboard or particle accelerator? The machine has already made up its mind.";
  $("guessArea").hidden = false;
  $("revealArea").hidden = true;
  window.scrollTo(0, 0);

  $("introRound").textContent = game.i === ROUNDS - 1 ? "FINAL ROUND" : `ROUND ${game.i + 1}`;
  $("intro").hidden = false;
  replay($("intro"), "intro");
  Sfx.round();
  await sleep(reducedMotion() ? 50 : 650);
  $("intro").hidden = true;
  busy = false;
}

function updateHud() {
  const hpYou = 100 - HIT * game.ai, hpAi = 100 - HIT * game.you;
  $("hpYou").textContent = hpYou; $("hpAi").textContent = hpAi;
  setBar($("barYou"), hpYou); setBar($("barAi"), hpAi);
}

function guess(f) {
  if (!game || game.guess || busy || $("fight").hidden) return;
  const p = game.fighters[game.i], ai = aiPick(p);
  const youRight = f === p.answer, aiRight = ai === p.answer;
  game.guess = f;
  game.you += youRight; game.ai += aiRight;
  game.log.push({ p, guess: f, youRight, aiRight, ai });
  updateHud();

  if (youRight) damage($("hpBoxAi"));
  if (aiRight) { damage($("hpBoxYou")); if (!reducedMotion()) replay($("arena"), "shake"); }
  if (youRight && !aiRight) Sfx.hit(); else if (aiRight) Sfx.hurt(); else Sfx.miss();

  const [text, bg] = youRight && !aiRight ? ["CRITICAL HIT!", "var(--yellow)"]
    : youRight && aiRight ? ["TRADE!", "var(--hp-ok)"]
    : !youRight && aiRight ? ["OUCH!", "var(--hp-low)"]
    : ["DOUBLE MISS", "var(--lav-4)"];
  $("bannerText").textContent = text;
  $("banner").style.background = bg;
  $("answerLabel").textContent = LABEL[p.answer];
  $("fClass").textContent = "CLASS: " + LABEL[p.answer];
  $("fClass").style.background = COLOR[p.answer];
  $("fName").textContent = p.name.toUpperCase();
  $("fMeta").textContent = `FIGHTER ${String(game.i + 1).padStart(2, "0")} · BORN ${p.born}`;
  $("fBio").textContent = capital(p.bio) || LABEL[p.answer].toLowerCase();
  $("youLine").textContent = `Picked ${LABEL[f]} · ${youRight ? `HIT −${HIT} to MODEL` : "MISSED"}`;
  $("aiLine").textContent = `Picked ${LABEL[ai]} (${aiConf(p)}% sure) · ${aiRight ? `HIT −${HIT} to YOU` : "MISSED"}`;
  const links = [p.wiki && `<a href="${esc(p.wiki)}" target="_blank" rel="noopener">Read about them</a>`,
    `<a href="${esc(p.source)}" target="_blank" rel="noopener">Photo: Wikimedia Commons</a>`].filter(Boolean).join(" · ");
  $("lore").innerHTML = `${esc(loreLine(p, youRight, aiRight))} ${links}`;
  $("nextBtn").textContent = game.i === ROUNDS - 1 ? "FINAL SCORE ▶" : "NEXT ROUND ▶";
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
  if (!game || !game.guess) return;
  Sfx.select();
  if (++game.i < ROUNDS) return renderRound();
  renderResults();
}

/* ---------- results ---------- */
function outcomeOf(g) {
  const hpYou = 100 - HIT * g.ai, hpAi = 100 - HIT * g.you;
  return hpYou > hpAi ? "win" : hpYou < hpAi ? "lose" : "draw";
}

function renderResults() {
  const hpYou = 100 - HIT * game.ai, hpAi = 100 - HIT * game.you;
  const outcome = (game.outcome = outcomeOf(game));
  store.set("pom-last", game.you);
  $("rKicker").textContent = { win: "MODEL · K.O.", lose: "YOU · K.O.", draw: "DOUBLE K.O." }[outcome];
  $("rHeadline").textContent = { win: "YOU WIN!", lose: "GAME OVER", draw: "DRAW!" }[outcome];
  $("rHeadline").style.color = { win: "var(--yellow)", lose: "var(--lose)", draw: "var(--white)" }[outcome];
  $("rYouName").textContent = "YOU";
  $("rYou").textContent = `${game.you}/${ROUNDS} · ${hpYou} HP`;
  $("rAi").textContent = `${game.ai}/${ROUNDS} · ${hpAi} HP`;
  setBar($("rBarYou"), hpYou); setBar($("rBarAi"), hpAi);
  const fooled = game.log.filter((r) => r.youRight && !r.aiRight).length;
  $("rTagline").textContent = outcome === "win"
    ? `You out-read the machine on ${fooled} fighter${fooled === 1 ? "" : "s"}.`
    : outcome === "lose" ? "The machine read them better this time. Rematch?" : "Dead even. Best of three?";

  $("log").innerHTML = "";
  game.log.forEach((r) => {
    const row = document.createElement("div");
    row.className = "log-row";
    row.innerHTML = `
      <button type="button" class="thumb" style="box-shadow: 4px 4px 0 ${COLOR[r.p.answer]}" aria-label="Show photo of ${esc(r.p.name)}"><img class="photo" src="${r.p.img}" alt=""></button>
      <span class="px" style="flex: 1 1 220px; font-size: 13px; line-height: 1.6">${esc(r.p.name.toUpperCase())}</span>
      <span class="tag" style="flex: 0 0 auto; width: 200px; border-width: 3px; background: ${COLOR[r.p.answer]}">${LABEL[r.p.answer]}</span>
      <span style="flex: 0 0 150px; font-size: 28px"><span class="label-sm" style="font-size: 9px">YOU </span>${r.youRight ? "✓ HIT" : "✗ MISS"}</span>
      <span style="flex: 0 0 210px; font-size: 28px"><span class="label-sm" style="font-size: 9px">MODEL </span>${r.aiRight ? "✓" : "✗"} ${SHORT[r.ai]} ${aiConf(r.p)}%</span>`;
    row.querySelector("button").onclick = () => openModal(r.p);
    $("log").appendChild(row);
  });

  game.savedId = null;
  $("nameForm").hidden = false;
  $("playerName").value = store.get("pom-name", "");
  $("saveBtn").disabled = false;
  $("nameNote").textContent = "";
  $("sharePanel").hidden = true;
  $("rankNote").textContent = "";
  renderResultBoard();
  startCountdown();
  show("results");
  replay($("rHeadline"), "ko-pop");
  (outcome === "win" ? Sfx.win : Sfx.lose)();
}

function renderResultBoard(rank, total) {
  if (board) {
    const rows = board.top.slice(0, 10);
    if (game.savedId && !rows.some((r) => r.id === game.savedId) && rank) {
      rows.push({ id: game.savedId, name: $("playerName").value.trim() || "YOU", you: game.you, ai: game.ai, rank });
    }
    renderList($("scoreList"), rows.map((r, k) => ({ ...r, rank: r.rank || k + 1 })), game.savedId);
    $("rankNote").textContent = rank ? `You placed #${rank} of ${total}.` : "Save your name to join the leaderboard.";
  } else {
    const rows = localScores().slice(0, 10);
    renderList($("scoreList"), rows, game.savedId);
    $("rankNote").textContent = "Online leaderboard unavailable: scores are saved on this device.";
  }
}

async function saveScore(e) {
  e.preventDefault();
  const name = $("playerName").value.normalize("NFKC").replace(/[^\p{L}\p{N} ._-]/gu, "").replace(/\s+/g, " ").trim().slice(0, 12);
  if (!name) { $("nameNote").textContent = "Pick a name with letters or numbers."; return; }
  store.set("pom-name", name);
  $("saveBtn").disabled = true;
  $("nameNote").textContent = "Saving…";
  try {
    const res = await api.post({ name, guesses: game.log.map((r) => ({ id: r.p.id, g: r.guess })) });
    board = { top: res.top, stats: res.stats };
    game.savedId = res.entry.id;
    $("nameForm").hidden = true;
    renderResultBoard(res.rank, res.total);
  } catch (err) {
    if (err.status === 429) { $("nameNote").textContent = "Too many saves — wait a minute."; $("saveBtn").disabled = false; return; }
    const scores = localScores();
    const entry = { id: `local-${Date.now()}`, name, you: game.you, ai: game.ai, t: Date.now() };
    scores.push(entry);
    scores.sort((a, b) => b.you - a.you || (b.you - b.ai) - (a.you - a.ai) || a.t - b.t);
    store.set("pom-scores-v2", scores.slice(0, 10));
    game.savedId = entry.id;
    board = null;
    $("nameForm").hidden = true;
    renderResultBoard();
  }
  $("rYouName").textContent = name.toUpperCase();
}

/* ---------- sharing ---------- */
function shareText() {
  const name = store.get("pom-name", "");
  const lead = { win: "I beat the AI!", lose: "The AI got me this time.", draw: "I tied the AI!" }[game.outcome];
  const grid = game.log.map((r) => (r.youRight ? "🟩" : "🟥")).join("");
  return `${lead} ${name ? name + " scored" : "I scored"} ${game.you}/${ROUNDS} vs the machine's ${game.ai}/${ROUNDS} in Physicist or Mathematician?\n${grid}\nCan you beat it? ${SITE}`;
}

async function share() {
  Sfx.select();
  const text = shareText();
  const card = await drawScorecard({ name: store.get("pom-name", "") || "YOU", you: game.you, ai: game.ai, rounds: ROUNDS,
    outcome: game.outcome, log: game.log, url: SITE });
  const file = card.blob && new File([card.blob], "physicist-or-mathematician.png", { type: "image/png" });
  if (file && navigator.canShare && navigator.canShare({ files: [file] }) && matchMedia("(pointer: coarse)").matches) {
    try { await navigator.share({ files: [file], text, title: "Physicist or Mathematician?" }); return; } catch (e) { if (e.name === "AbortError") return; }
  }
  $("scorecardImg").src = card.url;
  $("downloadLink").href = card.url;
  $("shareText").textContent = text;
  $("xLink").href = "https://twitter.com/intent/tweet?text=" + encodeURIComponent(text);
  $("waLink").href = "https://wa.me/?text=" + encodeURIComponent(text);
  $("copyBtn").textContent = "COPY TEXT";
  $("sharePanel").hidden = false;
  $("sharePanel").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "nearest" });
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
  $("mPhoto").src = p.img; $("mPhoto").alt = `Photo of ${p.name}`;
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
$("startBtn").onclick = startGame;
$("againBtn").onclick = () => { Sfx.select(); renderTitle(); };
$("titleBtn").onclick = () => { Sfx.select(); renderTitle(); };
$("pickPhys").onclick = () => guess("phys");
$("pickMath").onclick = () => guess("math");
$("nextBtn").onclick = nextRound;
$("shareBtn").onclick = share;
$("shareClose").onclick = () => { $("sharePanel").hidden = true; };
$("copyBtn").onclick = async () => {
  try { await navigator.clipboard.writeText($("shareText").textContent); $("copyBtn").textContent = "COPIED!"; }
  catch { $("copyBtn").textContent = "SELECT & COPY"; }
};
$("nameForm").onsubmit = saveScore;
$("mClose").onclick = closeModal;
$("modal").onclick = (e) => { if (e.target.id === "modal") closeModal(); };
$("muteBtn").onclick = () => { Sfx.setMuted(!Sfx.muted); renderMute(); if (!Sfx.muted) Sfx.select(); };

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (!$("modal").hidden) { if (e.key === "Escape") closeModal(); return; }
  const k = e.key.toLowerCase();
  if (!$("title").hidden && (k === "enter" || k === " ") && e.target.tagName !== "A") { e.preventDefault(); startGame(); }
  else if (!$("fight").hidden && game && !game.guess && (k === "a" || k === "arrowleft")) guess("phys");
  else if (!$("fight").hidden && game && !game.guess && (k === "b" || k === "arrowright")) guess("math");
});

renderMute();
fetch("data.json").then((r) => r.json()).then((d) => {
  pool = d.people;
  $("modelAcc").textContent = Math.round(100 * d.model_accuracy) + "%";
  renderTitle();
}).catch(() => {
  $("startBtn").textContent = "LOADING FAILED";
  $("startBtn").disabled = true;
  renderTitle();
});
