// Party mode (Jackbox-style): a host screen (TV) runs the game, players answer on their phones.
//
// POST /api/party {action:"create"}                              -> { code, hostToken }
// POST /api/party?code=C {action:"join", name}                   -> { playerId, token }
// POST /api/party?code=C {action:"start", hostToken}             -> state   (host: lobby -> round 1)
// POST /api/party?code=C {action:"answer", token, round, g}      -> state   (player)
// POST /api/party?code=C {action:"reveal", hostToken, round}     -> state   (host: everyone answered or time's up)
// POST /api/party?code=C {action:"next", hostToken, round}       -> state   (host: start the next round)
// POST /api/party?code=C {action:"again", hostToken}             -> state   (host: same players, new fighters)
// GET  /api/party?code=C   (header x-pom-token: T)                 -> state   (host or player token)
//
// Answers stay hidden until the host reveals the round. Each answer stores how fast it came (speed bonus).
const crypto = require("crypto");
const G = require("./_game");

const ANSWER_MS = 15000, COUNTDOWN_MS = 4000, NEXT_DELAY_MS = 1200, MAX_PLAYERS = 16, TTL = 60 * 60 * 3;
const ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ";
const K = { meta: (c) => `pom:party:${c}`, players: (c) => `pom:party:${c}:p`, ans: (c) => `pom:party:${c}:a` };
const validCode = (c) => typeof c === "string" && /^[A-Z]{4}$/.test(c);
const token = () => crypto.randomBytes(12).toString("hex");
const newCode = () => Array.from(crypto.randomBytes(4), (b) => ALPHABET[b % ALPHABET.length]).join("");
const pickFighters = () => [...G.IDS].sort(() => Math.random() - 0.5).slice(0, G.ROUNDS);

function hash(arr) {
  const o = {};
  for (let i = 0; i < (arr || []).length; i += 2) o[arr[i]] = arr[i + 1];
  return o;
}

async function load(c) {
  const [meta, players, ans] = await G.redis.pipeline([["HGETALL", K.meta(c)], ["HGETALL", K.players(c)], ["HGETALL", K.ans(c)]]);
  const m = hash(meta);
  if (!m.hostTok) return null;
  const p = Object.entries(hash(players)).map(([id, json]) => ({ id, ...JSON.parse(json) }));
  return { m, players: p, a: hash(ans) };
}

function whoIs(room, tok) {
  if (!tok) return null;
  if (tok === room.m.hostTok) return { host: true };
  const p = room.players.find((x) => x.token === tok);
  return p ? { playerId: p.id } : null;
}

function view(c, room, me, now) {
  const { m, a } = room;
  const game = Number(m.game) || 0;
  const rounds = [];
  for (let r = 0; r < G.ROUNDS; r++) {
    const start = Number(a[`s${r}`]) || null;
    if (!start) break;
    const revealedAt = Number(a[`rv${r}`]) || null;
    const answers = {};
    for (const p of room.players) {
      const v = a[`a${r}:${p.id}`];
      if (v) answers[p.id] = v;
    }
    const parsed = Object.fromEntries(Object.entries(answers).map(([id, v]) => {
      const [g, ms] = v.split("|");
      return [id, { g, ms: Number(ms) }];
    }));
    const mine = me && me.playerId && parsed[me.playerId];
    rounds.push({
      start, deadline: start + ANSWER_MS, revealedAt,
      answered: Object.keys(parsed),
      answers: revealedAt ? parsed : mine ? { [me.playerId]: mine } : {},
    });
  }
  return {
    code: c, now, game, answerMs: ANSWER_MS, rounds,
    fighters: m.fighters ? JSON.parse(m.fighters) : [],
    phase: rounds.length ? "playing" : "lobby",
    players: room.players.map((p) => ({ id: p.id, name: p.name, joined: p.joined })).sort((x, y) => x.joined - y.joined),
    you: me ? (me.host ? "host" : me.playerId) : null,
  };
}

async function act(req, c) {
  const b = G.body(req);
  const now = Date.now();

  if (b.action === "create") {
    if (await G.rateLimited(req, "party", 10)) return [429, { error: "slow down" }];
    for (let attempt = 0; attempt < 6; attempt++) {
      const code = newCode(), hostTok = token();
      const [fresh] = await G.redis.pipeline([["HSETNX", K.meta(code), "hostTok", hostTok]]);
      if (!fresh) continue;
      await G.redis.pipeline([["HSET", K.meta(code), "created", now, "game", 0], ["EXPIRE", K.meta(code), TTL]]);
      return [200, { code, hostToken: hostTok }];
    }
    throw new Error("no free party code");
  }

  if (!validCode(c)) return [400, { error: "bad party code" }];
  const room = await load(c);
  if (!room) return [404, { error: "party not found or expired" }];
  const expire = [["EXPIRE", K.meta(c), TTL], ["EXPIRE", K.players(c), TTL], ["EXPIRE", K.ans(c), TTL]];

  if (b.action === "join") {
    const me = whoIs(room, b.token);
    if (me && me.playerId) return [200, { playerId: me.playerId, token: b.token }]; // already in (page reload)
    let name = G.cleanName(b.name);
    if (!name) return [400, { error: "name required" }];
    if (await G.rateLimited(req, "join", 30)) return [429, { error: "slow down" }]; // slows down code guessing
    if (room.players.length >= MAX_PLAYERS) return [409, { error: "party is full" }];
    const taken = new Set(room.players.map((p) => p.name.toLowerCase()));
    for (let n = 2; taken.has(name.toLowerCase()); n++) name = `${G.cleanName(b.name).slice(0, 10)} ${n}`;
    const id = crypto.randomBytes(4).toString("hex"), tok = token();
    await G.redis.pipeline([["HSET", K.players(c), id, JSON.stringify({ name, token: tok, joined: now })], ...expire]);
    return [200, { playerId: id, token: tok, name }];
  }

  const me = whoIs(room, b.token || b.hostToken);
  if (!me) return [403, { error: "not in this party" }];
  const r = Number(b.round);
  const validRound = Number.isInteger(r) && r >= 0 && r < G.ROUNDS;

  if (b.action === "answer") {
    if (!me.playerId || !validRound || (b.g !== "phys" && b.g !== "math")) return [400, { error: "bad answer" }];
    const start = Number(room.a[`s${r}`]);
    if (!start || now < start || now > start + ANSWER_MS + 1000 || room.a[`rv${r}`]) return [409, { error: "round not open" }];
    await G.redis.pipeline([["HSETNX", K.ans(c), `a${r}:${me.playerId}`, `${b.g}|${Math.min(now - start, ANSWER_MS)}`]]);
  } else if (!me.host) {
    return [403, { error: "only the host can do that" }];
  } else if (b.action === "start") {
    if (!room.players.length) return [409, { error: "nobody has joined yet" }];
    await G.redis.pipeline([
      ["HSET", K.meta(c), "fighters", JSON.stringify(pickFighters())],
      ["HSETNX", K.ans(c), "s0", now + COUNTDOWN_MS], ...expire]);
  } else if (b.action === "reveal") {
    const start = Number(room.a[`s${r}`]);
    const answered = room.players.filter((p) => room.a[`a${r}:${p.id}`]).length;
    if (!validRound || !start || (now < start + ANSWER_MS && answered < room.players.length)) return [409, { error: "not yet" }];
    await G.redis.pipeline([["HSETNX", K.ans(c), `rv${r}`, now]]);
  } else if (b.action === "next") {
    if (!validRound || r >= G.ROUNDS - 1 || !room.a[`rv${r}`]) return [409, { error: "not yet" }];
    await G.redis.pipeline([["HSETNX", K.ans(c), `s${r + 1}`, now + NEXT_DELAY_MS]]);
  } else if (b.action === "again") {
    await G.redis.pipeline([["DEL", K.ans(c)], ["HINCRBY", K.meta(c), "game", 1], ["HDEL", K.meta(c), "fighters"], ...expire]);
  } else {
    return [400, { error: "unknown action" }];
  }
  return [200, view(c, await load(c), me, Date.now())];
}

module.exports = G.handler(async (req) => {
  const q = G.query(req);
  const c = (q.get("code") || "").toUpperCase();
  if (req.method === "GET") {
    if (!validCode(c)) return [400, { error: "bad party code" }];
    const room = await load(c);
    if (!room) return [404, { error: "party not found or expired" }];
    return [200, view(c, room, whoIs(room, G.tokenOf(req)), Date.now())];
  }
  if (req.method === "POST") return act(req, c);
  return [405, { error: "method not allowed" }];
});
