// GET  /api/leaderboard          -> { top: [...], stats: {...} }
// POST /api/leaderboard {name, guesses:[{id, g}]}  -> { rank, total, entry, top, stats }
//
// Scores are recomputed on the server from the guesses and the game's answer key, so a submitted score
// always matches real fighters. (It's a game among friends; this is a sanity check, not anti-cheat.)
const redis = require("./_redis");
const DATA = require("../game/data.json");

const ROUNDS = 10;
const FIGHTERS = new Map(DATA.people.map((p) => [p.id, p]));
const K = { board: "pom:lb", entries: "pom:entries", stats: "pom:stats" };

function cleanName(name) {
  return String(name || "").normalize("NFKC").replace(/[^\p{L}\p{N} ._-]/gu, "").replace(/\s+/g, " ").trim().slice(0, 12);
}

async function top(n = 10) {
  const [ids, stats] = await redis.pipeline([["ZREVRANGE", K.board, 0, n - 1], ["HGETALL", K.stats]]);
  const rows = ids.length ? (await redis.pipeline([["HMGET", K.entries, ...ids]]))[0] : [];
  const s = {};
  for (let i = 0; i < (stats || []).length; i += 2) s[stats[i]] = Number(stats[i + 1]);
  return {
    top: rows.filter(Boolean).map((r) => JSON.parse(r)),
    stats: { games: s.games || 0, rounds: s.rounds || 0, human: s.human || 0, ai: s.ai || 0 },
  };
}

async function submit(body, ip) {
  const name = cleanName(body.name);
  const guesses = Array.isArray(body.guesses) ? body.guesses : [];
  const ids = new Set(guesses.map((x) => x && x.id));
  if (!name) return [400, { error: "name required" }];
  if (guesses.length !== ROUNDS || ids.size !== ROUNDS || ![...ids].every((id) => FIGHTERS.has(id))
      || !guesses.every((x) => x.g === "phys" || x.g === "math")) return [400, { error: "invalid game" }];

  const [count] = await redis.pipeline([["INCR", `pom:rl:${ip}`], ["EXPIRE", `pom:rl:${ip}`, 60]]);
  if (count > 10) return [429, { error: "slow down" }];

  let you = 0, ai = 0;
  for (const { id, g } of guesses) {
    const p = FIGHTERS.get(id);
    you += g === p.answer;
    ai += (p.p >= 0.5 ? "phys" : "math") === p.answer;
  }
  const t = Date.now();
  const id = `${t.toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  const entry = { id, name, you, ai, t };
  // rank by hits, then by margin over the machine, then earlier games first
  const score = you * 1000 + (you - ai + ROUNDS) * 10 + (1 - t / 1e13);
  const [, , , , , , rank, total] = await redis.pipeline([
    ["ZADD", K.board, score, id],
    ["HSET", K.entries, id, JSON.stringify(entry)],
    ["HINCRBY", K.stats, "games", 1],
    ["HINCRBY", K.stats, "rounds", ROUNDS],
    ["HINCRBY", K.stats, "human", you],
    ["HINCRBY", K.stats, "ai", ai],
    ["ZREVRANK", K.board, id],
    ["ZCARD", K.board],
  ]);
  return [200, { rank: rank + 1, total, entry, ...(await top()) }];
}

module.exports = async (req, res) => {
  res.setHeader("Cache-Control", "no-store");
  if (!redis.configured()) return res.status(503).json({ error: "leaderboard not configured" });
  try {
    if (req.method === "GET") return res.status(200).json(await top());
    if (req.method === "POST") {
      const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
      const ip = String(req.headers["x-forwarded-for"] || "local").split(",")[0].trim();
      const [status, payload] = await submit(body, ip);
      return res.status(status).json(payload);
    }
    return res.status(405).json({ error: "method not allowed" });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "server error" });
  }
};
