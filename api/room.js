// Real-time duels (two players, same 10 fighters, live). Clients poll ~1x/second; state lives in Redis.
//
// POST /api/room {action:"create", name}                     -> { code, token }
// POST /api/room?code=C {action:"join", name}                -> { code, token }
// POST /api/room?code=C {action:"answer", token, round, g}   -> state
// POST /api/room?code=C {action:"next", token, round}        -> state   (ready for the round after `round`)
// POST /api/room?code=C {action:"rematch", token}            -> { code, token }   new room, opponent invited
// GET  /api/room?code=C&token=T                              -> state
//
// Round r starts at s{r}; answers lock at s{r} + ANSWER_MS (a missing answer counts as a miss).
// An answer is only revealed to the other player once both have answered or time is up.
const crypto = require("crypto");
const G = require("./_game");

const ANSWER_MS = 15000, COUNTDOWN_MS = 3500, NEXT_DELAY_MS = 900, TTL = 60 * 60 * 2;
const ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
const K = { meta: (c) => `pom:room:${c}`, ans: (c) => `pom:room:${c}:a` };
const validCode = (c) => typeof c === "string" && /^[A-Z2-9]{5}$/.test(c);
const token = () => crypto.randomBytes(12).toString("hex");
const code = () => Array.from(crypto.randomBytes(5), (b) => ALPHABET[b % ALPHABET.length]).join("");

function hash(arr) {
  const o = {};
  for (let i = 0; i < (arr || []).length; i += 2) o[arr[i]] = arr[i + 1];
  return o;
}

async function load(c) {
  const [meta, ans] = await G.redis.pipeline([["HGETALL", K.meta(c)], ["HGETALL", K.ans(c)]]);
  const m = hash(meta);
  return m.host ? { m, a: hash(ans) } : null;
}

function roleOf(m, tok) {
  if (tok && tok === m.hostTok) return "host";
  if (tok && tok === m.guestTok) return "guest";
  return null;
}

/** Public view of the room for one player (the other player's pending answer stays hidden). */
function view(c, { m, a }, role, now) {
  const fighters = JSON.parse(m.fighters);
  const rounds = [];
  for (let r = 0; r < G.ROUNDS; r++) {
    const start = Number(a[`s${r}`]) || null;
    if (!start) break;
    const deadline = start + ANSWER_MS;
    const ans = { host: a[`a${r}:host`] || null, guest: a[`a${r}:guest`] || null };
    const revealed = (ans.host && ans.guest) || now > deadline;
    rounds.push({
      start, deadline, revealed: Boolean(revealed),
      answered: { host: Boolean(ans.host), guest: Boolean(ans.guest) },
      answers: revealed ? ans : { host: role === "host" ? ans.host : null, guest: role === "guest" ? ans.guest : null },
      ready: { host: Boolean(a[`n${r}:host`]), guest: Boolean(a[`n${r}:guest`]) },
    });
  }
  return {
    code: c, you: role, now, fighters, answerMs: ANSWER_MS, rematch: m.rematch || null,
    players: {
      host: { name: m.host, seen: Number(m.seenHost) || 0 },
      guest: m.guest ? { name: m.guest, seen: Number(m.seenGuest) || 0 } : null,
    },
    rounds,
  };
}

async function create(name, extra = {}) {
  const fighters = [...G.IDS].sort(() => Math.random() - 0.5).slice(0, G.ROUNDS);
  for (let attempt = 0; attempt < 5; attempt++) {
    const c = code(), tok = token(), now = Date.now();
    const [fresh] = await G.redis.pipeline([["HSETNX", K.meta(c), "host", name]]);
    if (!fresh) continue; // code already in use
    await G.redis.pipeline([
      ["HSET", K.meta(c), "hostTok", tok, "fighters", JSON.stringify(fighters), "created", now, "seenHost", now,
        ...Object.entries(extra).flat()],
      ["EXPIRE", K.meta(c), TTL],
    ]);
    return { code: c, token: tok };
  }
  throw new Error("no free room code");
}

async function act(req, c) {
  const b = G.body(req);
  const now = Date.now();

  if (b.action === "create") {
    const name = G.cleanName(b.name);
    if (!name) return [400, { error: "name required" }];
    if (await G.rateLimited(req, "room", 20)) return [429, { error: "slow down" }];
    return [200, await create(name)];
  }

  if (!validCode(c)) return [400, { error: "bad room code" }];
  const room = await load(c);
  if (!room) return [404, { error: "room not found or expired" }];
  const { m } = room;

  if (b.action === "join") {
    const name = G.cleanName(b.name);
    if (!name) return [400, { error: "name required" }];
    const role = roleOf(m, b.token);
    if (role) return [200, { code: c, token: b.token }]; // already in this room
    const tok = token();
    const [won] = await G.redis.pipeline([["HSETNX", K.meta(c), "guestTok", tok]]);
    if (!won) return [409, { error: "room is full" }];
    await G.redis.pipeline([
      ["HSET", K.meta(c), "guest", name, "seenGuest", now],
      ["HSETNX", K.ans(c), "s0", now + COUNTDOWN_MS], // both players are in: round 1 starts after a countdown
      ["EXPIRE", K.meta(c), TTL], ["EXPIRE", K.ans(c), TTL],
    ]);
    return [200, { code: c, token: tok }];
  }

  const role = roleOf(m, b.token);
  if (!role) return [403, { error: "not a player in this room" }];
  const r = Number(b.round);
  const validRound = Number.isInteger(r) && r >= 0 && r < G.ROUNDS;

  if (b.action === "answer") {
    if (!validRound || (b.g !== "phys" && b.g !== "math")) return [400, { error: "bad answer" }];
    const start = Number(room.a[`s${r}`]);
    if (!start || now < start || now > start + ANSWER_MS + 1500) return [409, { error: "round not open" }];
    await G.redis.pipeline([["HSETNX", K.ans(c), `a${r}:${role}`, b.g]]);
  } else if (b.action === "next") {
    if (!validRound || r >= G.ROUNDS - 1) return [400, { error: "bad round" }];
    const [, ready] = await G.redis.pipeline([
      ["HSETNX", K.ans(c), `n${r}:${role}`, now],
      ["HMGET", K.ans(c), `n${r}:host`, `n${r}:guest`],
    ]);
    if (ready[0] && ready[1]) await G.redis.pipeline([["HSETNX", K.ans(c), `s${r + 1}`, now + NEXT_DELAY_MS]]);
  } else if (b.action === "rematch") {
    if (m.rematch) return [200, { code: m.rematch, token: null }]; // the other player already asked: join theirs
    const name = role === "host" ? m.host : m.guest;
    const fresh = await create(name);
    const [won] = await G.redis.pipeline([["HSETNX", K.meta(c), "rematch", fresh.code]]);
    if (!won) { // the other player asked at the same moment: everyone goes to their room
      const [existing] = await G.redis.pipeline([["HGET", K.meta(c), "rematch"]]);
      return [200, { code: existing, token: null }];
    }
    return [200, fresh];
  } else {
    return [400, { error: "unknown action" }];
  }
  return [200, view(c, (await load(c)), role, Date.now())];
}

module.exports = G.handler(async (req) => {
  const q = G.query(req);
  const c = (q.get("code") || "").toUpperCase();
  if (req.method === "GET") {
    if (!validCode(c)) return [400, { error: "bad room code" }];
    const room = await load(c);
    if (!room) return [404, { error: "room not found or expired" }];
    const role = roleOf(room.m, q.get("token"));
    const now = Date.now();
    if (role) {
      await G.redis.pipeline([["HSET", K.meta(c), role === "host" ? "seenHost" : "seenGuest", now]]);
      room.m[role === "host" ? "seenHost" : "seenGuest"] = String(now);
    }
    return [200, view(c, room, role, now)];
  }
  if (req.method === "POST") return act(req, c);
  return [405, { error: "method not allowed" }];
});
