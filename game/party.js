/* Party mode: one host screen (TV) + everyone answering on their phones. Uses helpers from app.js.
 * Scoring: a right answer is worth 100 + up to 50 for speed. The machine plays too (100 per right answer). */
const PARTY_REVEAL_MS = 8000;
const party = { role: null, code: null, hostToken: null, token: null, playerId: null, state: null, offset: 0,
  poll: null, tick: null, timer: null, game: -1, shown: {}, requested: {}, qrFor: null };
const partyNow = () => Date.now() + party.offset;
const partyLink = (code) => `${SITE}?party=${code}`;

function stopParty() {
  clearInterval(party.poll); clearInterval(party.tick); clearTimeout(party.timer);
  Object.assign(party, { role: null, code: null, hostToken: null, token: null, playerId: null, state: null,
    poll: null, tick: null, timer: null, game: -1, shown: {}, requested: {} });
  $("phIntro").hidden = true;
  const q = new URLSearchParams(location.search);
  if (q.has("party") || q.has("tv")) history.replaceState(null, "", SITE);
}

function startPolling(ms) {
  clearInterval(party.poll); clearInterval(party.tick);
  pollParty();
  party.poll = setInterval(pollParty, ms);
  party.tick = setInterval(party.role === "host" ? hostTick : playerTick, 200);
}

async function pollParty() {
  if (!party.code) return;
  const code = party.code;
  try {
    const s = await api(`/api/party?code=${code}`, null, party.role === "host" ? party.hostToken : party.token);
    if (party.code !== code) return;
    party.offset = s.now - Date.now();
    party.state = s;
    (party.role === "host" ? hostRender : playerRender)();
  } catch (err) {
    if (err.status === 404 && party.code === code) { stopParty(); show("partyPlayer"); ppView("ppResult"); partyGone(); }
  }
}

function partyGone() {
  $("ppResKicker").textContent = "PARTY OVER";
  $("ppResBig").textContent = "BYE!";
  $("ppResText").innerHTML = `This party ended or expired. <a href="${esc(SITE)}">Back to the title screen</a>`;
}

/** Points per player (and the machine) over all revealed rounds. */
function partyScores(s) {
  const total = Object.fromEntries(s.players.map((p) => [p.id, 0]));
  const rounds = [];
  let machine = 0;
  s.rounds.forEach((rd, r) => {
    if (!rd.revealedAt) return;
    const f = byId.get(s.fighters[r]);
    const gains = {};
    for (const p of s.players) {
      const a = rd.answers[p.id];
      gains[p.id] = a && a.g === f.answer ? 100 + Math.round(50 * (1 - a.ms / s.answerMs)) : 0;
      total[p.id] += gains[p.id];
    }
    const machineGain = aiPick(f) === f.answer ? 100 : 0;
    machine += machineGain;
    rounds[r] = { gains, machineGain };
  });
  const ranking = s.players.map((p) => ({ id: p.id, name: p.name, pts: total[p.id] }))
    .concat([{ id: "machine", name: "THE MACHINE", pts: machine, machine: true }])
    .sort((a, b) => b.pts - a.pts);
  return { total, machine, rounds, ranking };
}

function scoreRows(ranking, highlight) {
  return ranking.map((x, k) => `<li class="${x.id === highlight ? "me" : ""}"><span class="who">${rankLabel(k + 1)}&nbsp; ${x.machine ? "🤖 " : ""}${esc(x.name)}</span><span>${x.pts}</span></li>`).join("");
}

/* ================= host (TV) ================= */
function hostView(id) { for (const v of ["phLobby", "phRound", "phReveal", "phFinal"]) $(v).hidden = v !== id; }

async function openPartyHost() {
  stopDuel(); stopParty();
  show("partyHost");
  hostView("phLobby");
  $("phCode").textContent = "····";
  $("phPlayers").innerHTML = "";
  $("phStart").disabled = true;
  try {
    const res = await api("/api/party", { action: "create" });
    await enterHost(res.code, res.hostToken);
  } catch (err) {
    $("phSite").textContent = err.status === 503 ? "Party mode needs the online database." : "Couldn't create a party. Try again.";
  }
}

async function enterHost(code, hostToken) {
  Object.assign(party, { role: "host", code, hostToken });
  session.set("pom-party-host", { code, hostToken });
  history.replaceState(null, "", `?tv=${code}`);
  show("partyHost");
  startPolling(800);
}

function hostLobby(s) {
  hostView("phLobby");
  $("phIntro").hidden = true;
  $("phCode").textContent = s.code;
  $("phSite").textContent = SITE.replace(/^https?:\/\//, "").replace(/\/$/, "");
  if (party.qrFor !== s.code) {
    party.qrFor = s.code;
    $("phQr").innerHTML = "";
    if (window.QRCode) new QRCode($("phQr"), { text: partyLink(s.code), width: 220, height: 220, colorDark: "#1A1033", colorLight: "#FFFFFF" });
    else $("phQr").hidden = true;
  }
  const known = new Set([...$("phPlayers").querySelectorAll(".chip")].map((c) => c.dataset.id));
  $("phPlayers").innerHTML = s.players.map((p) => `<span class="chip${known.has(p.id) ? "" : " new"}" data-id="${p.id}">${esc(p.name)}</span>`).join("")
    || `<span class="ph-note">Waiting for players…</span>`;
  if (s.players.length > known.size) Sfx.select(); // someone new joined
  $("phCount").textContent = `PLAYERS (${s.players.length})`;
  $("phStart").disabled = !s.players.length;
}

function hostRender() {
  const s = party.state;
  if (s.game !== party.game) { Object.assign(party, { game: s.game, shown: {}, requested: {} }); clearTimeout(party.timer); }
  if (s.phase === "lobby") return hostLobby(s);
  const r = s.rounds.length - 1, rd = s.rounds[r], now = partyNow();
  const f = byId.get(s.fighters[r]);

  if (now < rd.start || !rd.revealedAt) {
    if (party.shown.round !== r) { // new round: put the photo up
      party.shown.round = r;
      hostView("phRound");
      $("phPhoto").src = f.img;
      $("phRoundNo").textContent = `ROUND ${r + 1}/${ROUNDS}`;
      const next = byId.get(s.fighters[r + 1]);
      if (next) new Image().src = next.img;
    }
    const locked = new Set(rd.answered);
    $("phChips").innerHTML = s.players.map((p) => `<span class="chip${locked.has(p.id) ? " locked" : ""}">${locked.has(p.id) ? "✓ " : ""}${esc(p.name)}</span>`).join("")
      + `<span class="chip machine">✓ 🤖 MACHINE</span>`;
    $("phAnswered").textContent = `${rd.answered.length}/${s.players.length} LOCKED IN`;
    if (now < rd.start) {
      if ($("phIntro").hidden) {
        $("phIntroText").textContent = r === ROUNDS - 1 ? "FINAL ROUND" : `ROUND ${r + 1}`;
        $("phIntro").classList.add("hold");
        $("phIntro").hidden = false;
        Sfx.round();
      }
    } else {
      $("phIntro").hidden = true;
      if ((rd.answered.length >= s.players.length || now > rd.deadline) && !party.requested[`rv${r}`]) {
        party.requested[`rv${r}`] = true;
        api(`/api/party?code=${s.code}`, { action: "reveal", hostToken: party.hostToken, round: r })
          .then((st) => { party.state = st; hostRender(); })
          .catch(() => { party.requested[`rv${r}`] = false; });
      }
    }
    return;
  }

  if (party.shown[`rv${r}`]) return; // reveal already on screen; the timer moves on
  party.shown[`rv${r}`] = true;
  hostReveal(s, r, f);
  const revealEnds = partyNow() + PARTY_REVEAL_MS;
  party.shown.revealEnds = revealEnds;
  clearTimeout(party.timer);
  party.timer = setTimeout(() => {
    if (r === ROUNDS - 1) return hostFinal(party.state);
    if (party.requested[`n${r}`]) return;
    party.requested[`n${r}`] = true;
    api(`/api/party?code=${s.code}`, { action: "next", hostToken: party.hostToken, round: r })
      .then((st) => { party.state = st; hostRender(); }).catch(() => { party.requested[`n${r}`] = false; });
  }, PARTY_REVEAL_MS);
}

function hostReveal(s, r, f) {
  hostView("phReveal");
  $("phIntro").hidden = true;
  const sc = partyScores(s), rd = s.rounds[r], gains = sc.rounds[r].gains;
  $("phRPhoto").src = f.img;
  $("phRClass").textContent = LABEL[f.answer];
  $("phRClass").style.color = f.answer === "phys" ? "#0E9FB5" : "#D63A8A";
  $("phRName").textContent = f.name.toUpperCase();
  $("phRBio").textContent = `${capital(f.bio) || ""}${f.born ? ` · born ${f.born}` : ""}`;
  $("phRClues").textContent = `🤖 The machine saw ${(f.clues || []).join(", ") || "nothing obvious"} → ${LABEL[aiPick(f)]} (${aiConf(f)}%)`;
  const chip = (p) => {
    const a = rd.answers[p.id];
    return `<span class="chip${gains[p.id] ? " locked" : ""}">${esc(p.name)} <span class="pts">${gains[p.id] ? "+" + gains[p.id] : ""}</span>${a ? ` <span style="opacity:.6">${(a.ms / 1000).toFixed(1)}s</span>` : ""}</span>`;
  };
  for (const [g, col] of [["phys", "phColPhys"], ["math", "phColMath"]]) {
    const who = s.players.filter((p) => rd.answers[p.id] && rd.answers[p.id].g === g).map(chip);
    if (aiPick(f) === g) who.push(`<span class="chip machine">🤖 MACHINE${sc.rounds[r].machineGain ? ` <span style="color: var(--hp-ok)">+100</span>` : ""}</span>`);
    $(col).querySelector(".ph-col-list").innerHTML = who.join("") || `<span class="ph-note">nobody</span>`;
    $(col).classList.toggle("correct", f.answer === g);
  }
  const silent = s.players.filter((p) => !rd.answers[p.id]).map((p) => esc(p.name));
  if (silent.length) $("phRClues").textContent += `  ·  ⏱ too slow: ${silent.join(", ")}`;
  $("phBoard").innerHTML = scoreRows(sc.ranking.slice(0, 8));
  if (s.players.some((p) => gains[p.id])) Sfx.hit(); else Sfx.miss();
}

function hostFinal(s) {
  hostView("phFinal");
  const { ranking } = partyScores(s);
  const top = ranking.slice(0, 3);
  const order = [top[1], top[0], top[2]]; // 2nd, 1st, 3rd on the podium
  $("phPodium").innerHTML = order.map((x, i) => x ? `<div class="place p${[2, 1, 3][i]}">
      <span class="who">${x.machine ? "🤖 " : ""}${esc(x.name)}</span><span class="pts">${x.pts} PTS</span>
      <div class="block">${[2, 1, 3][i]}</div></div>` : "").join("");
  $("phFinalList").innerHTML = scoreRows(ranking);
  (ranking[0].machine ? Sfx.lose : Sfx.win)();
}

function hostTick() {
  const s = party.state;
  if (!s || s.phase === "lobby" || !party.code) return;
  const r = s.rounds.length - 1, rd = s.rounds[r], now = partyNow();
  if (now < rd.start) {
    $("phIntroText").textContent = `${r === ROUNDS - 1 ? "FINAL ROUND" : "ROUND " + (r + 1)} · ${Math.max(1, Math.ceil((rd.start - now) / 1000))}`;
  } else if (!rd.revealedAt) {
    if (!$("phIntro").hidden) hostRender();
    $("phTimerBar").style.width = (100 * Math.max(0, rd.deadline - now)) / s.answerMs + "%";
    if (now > rd.deadline && !party.requested[`rv${r}`]) hostRender();
  } else if (!$("phReveal").hidden) {
    const left = Math.max(0, Math.ceil((party.shown.revealEnds - now) / 1000));
    $("phNext").textContent = r === ROUNDS - 1 ? `FINAL RESULTS IN ${left}` : `NEXT ROUND IN ${left}`;
  }
}

async function hostStart() {
  Sfx.start();
  $("phStart").disabled = true;
  try { party.state = await api(`/api/party?code=${party.code}`, { action: "start", hostToken: party.hostToken }); hostRender(); }
  catch { $("phStart").disabled = false; }
}

async function hostAgain() {
  Sfx.select();
  try { party.state = await api(`/api/party?code=${party.code}`, { action: "again", hostToken: party.hostToken }); hostRender(); } catch {}
}

function hostEnd() {
  Sfx.select();
  session.set("pom-party-host", null);
  stopParty();
  renderTitle();
}

/* ================= player (phone) ================= */
function ppView(id) { for (const v of ["ppJoin", "ppWait", "ppPlay", "ppResult"]) $(v).hidden = v !== id; }

function openPartyJoin(code = "") {
  stopDuel(); stopParty();
  const saved = session.get("pom-party");
  if (code && saved && saved.code === code) return enterPlayer(saved.code, saved.token, saved.playerId); // reload: resume
  show("partyPlayer");
  ppView("ppJoin");
  $("ppCode").value = code;
  $("ppName").value = store.get("pom-name", "");
  $("ppNote").textContent = "";
  (code ? $("ppName") : $("ppCode")).focus();
}

async function joinSubmit(e) {
  e.preventDefault();
  const code = $("ppCode").value.toUpperCase().replace(/[^A-Z]/g, "");
  const name = cleanName($("ppName").value);
  if (code.length !== 4) { $("ppNote").textContent = "The code is the 4 letters on the TV."; return; }
  if (!name) { $("ppNote").textContent = "Pick a name with letters or numbers."; return; }
  store.set("pom-name", name);
  $("ppGo").disabled = true;
  $("ppNote").textContent = "Joining…";
  try {
    const res = await api(`/api/party?code=${code}`, { action: "join", name });
    Sfx.select();
    enterPlayer(code, res.token, res.playerId);
  } catch (err) {
    $("ppNote").textContent = err.status === 404 ? "No party with that code. Check the TV."
      : err.status === 409 ? "That party is full." : "Couldn't join. Try again.";
  } finally {
    $("ppGo").disabled = false;
  }
}

function enterPlayer(code, token, playerId) {
  Object.assign(party, { role: "player", code, token, playerId });
  session.set("pom-party", { code, token, playerId });
  history.replaceState(null, "", `?party=${code}`);
  show("partyPlayer");
  ppView("ppWait");
  startPolling(1000);
}

function playerRender() {
  const s = party.state, me = party.playerId;
  const self = s.players.find((p) => p.id === me);
  if (!self) { stopParty(); ppView("ppResult"); partyGone(); return; }
  if (s.game !== party.game) { Object.assign(party, { game: s.game, shown: {} }); }
  const sc = partyScores(s);
  const rank = sc.ranking.findIndex((x) => x.id === me) + 1;
  $("ppScore").textContent = `${sc.total[me]} PTS`;

  if (s.phase === "lobby") {
    ppView("ppWait");
    $("ppWaitKicker").textContent = s.game > 0 ? "NEW GAME" : "YOU'RE IN!";
    $("ppWaitName").textContent = self.name.toUpperCase();
    $("ppWaitText").textContent = `${s.players.length} player${s.players.length === 1 ? "" : "s"} in the party. Look at the TV — the game starts when the host presses START.`;
    return;
  }
  const r = s.rounds.length - 1, rd = s.rounds[r], now = partyNow();
  if (now < rd.start) {
    ppView("ppWait");
    $("ppWaitKicker").textContent = r === ROUNDS - 1 ? "FINAL ROUND" : `ROUND ${r + 1}`;
    $("ppWaitName").textContent = "GET READY!";
    $("ppWaitText").textContent = "Eyes on the TV.";
    return;
  }
  if (!rd.revealedAt) {
    const mine = rd.answers[me];
    if (party.shown.round !== r) {
      party.shown.round = r;
      ppView("ppPlay");
      $("ppPhoto").src = byId.get(s.fighters[r]).img;
      $("ppRoundNo").textContent = `ROUND ${r + 1}/${ROUNDS}`;
      for (const b of [$("ppPhys"), $("ppMath")]) b.classList.remove("picked");
    }
    $("ppPhys").disabled = $("ppMath").disabled = Boolean(mine) || now > rd.deadline;
    if (mine) $(mine.g === "phys" ? "ppPhys" : "ppMath").classList.add("picked");
    $("ppStatus").textContent = mine ? `LOCKED IN · ${(mine.ms / 1000).toFixed(1)}s` : now > rd.deadline ? "TIME'S UP!" : "PICK ONE!";
    return;
  }
  if (party.shown.revealed === r) return;
  party.shown.revealed = r;
  const gain = sc.rounds[r].gains[me], f = byId.get(s.fighters[r]);
  ppView("ppResult");
  const last = r === ROUNDS - 1;
  $("ppResKicker").textContent = last ? "FINAL RESULTS" : `${LABEL[f.answer]} · ${f.name.toUpperCase()}`;
  $("ppResBig").textContent = last ? `#${rank}` : gain ? `+${gain}` : rd.answers[me] ? "WRONG" : "TOO SLOW";
  $("ppResBig").style.color = last ? "var(--yellow)" : gain ? "var(--hp-ok)" : "var(--lose)";
  $("ppResText").textContent = last
    ? `You placed #${rank} of ${sc.ranking.length} with ${sc.total[me]} points${rank < sc.ranking.findIndex((x) => x.machine) + 1 ? " — you beat the machine!" : "."}`
    : `You're #${rank} of ${sc.ranking.length} · ${sc.total[me]} pts`;
  replay($("ppResBig"), "ko-pop");
  (gain ? Sfx.hit : Sfx.miss)();
}

function playerTick() {
  const s = party.state;
  if (!s || s.phase === "lobby" || !party.code) return;
  const rd = s.rounds[s.rounds.length - 1], now = partyNow();
  if (!rd.revealedAt && now >= rd.start) {
    if ($("ppPlay").hidden) playerRender();
    $("ppTimerBar").style.width = (100 * Math.max(0, rd.deadline - now)) / s.answerMs + "%";
  }
}

async function playerAnswer(g) {
  const s = party.state;
  if (!s || s.phase === "lobby") return;
  const r = s.rounds.length - 1;
  if (s.rounds[r].answers[party.playerId] || partyNow() > s.rounds[r].deadline) return;
  Sfx.select();
  $("ppPhys").disabled = $("ppMath").disabled = true;
  $(g === "phys" ? "ppPhys" : "ppMath").classList.add("picked");
  $("ppStatus").textContent = "LOCKED IN";
  try {
    const st = await api(`/api/party?code=${party.code}`, { action: "answer", token: party.token, round: r, g });
    if (party.code === st.code) { party.state = st; playerRender(); }
  } catch {}
}

/* ================= wiring ================= */
$("partyHostBtn").onclick = () => { Sfx.select(); openPartyHost(); };
$("partyJoinBtn").onclick = () => { Sfx.select(); openPartyJoin(); };
$("phStart").onclick = hostStart;
$("phAgain").onclick = hostAgain;
$("phEnd").onclick = hostEnd;
$("phEnd2").onclick = hostEnd;
$("ppForm").onsubmit = joinSubmit;
$("ppCancel").onclick = () => { Sfx.select(); renderTitle(); };
$("ppPhys").onclick = () => playerAnswer("phys");
$("ppMath").onclick = () => playerAnswer("math");
