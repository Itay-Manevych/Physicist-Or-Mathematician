// Shared game logic for the API: answer key, scoring, validation, rate limiting.
const redis = require("./_redis");
const DATA = require("../game/data.json");
const Daily = require("../game/daily.js");

const ROUNDS = 10;
const FIGHTERS = new Map(DATA.people.map((p) => [p.id, p]));
const IDS = DATA.people.map((p) => p.id);

function cleanName(name) {
  return String(name || "").normalize("NFKC").replace(/[^\p{L}\p{N} ._-]/gu, "").replace(/\s+/g, " ").trim().slice(0, 12);
}

const aiPick = (p) => (p.p >= 0.5 ? "phys" : "math");

/** Validate a finished game's guesses: exactly ROUNDS distinct known fighters, each guess phys/math. */
function parseGuesses(guesses) {
  if (!Array.isArray(guesses) || guesses.length !== ROUNDS) return null;
  const ids = guesses.map((x) => x && x.id);
  if (new Set(ids).size !== ROUNDS || !ids.every((id) => FIGHTERS.has(id))) return null;
  if (!guesses.every((x) => x.g === "phys" || x.g === "math")) return null;
  return guesses.map(({ id, g }) => ({ id, g }));
}

/** Hits for the player and for the machine on the same fighters. */
function score(guesses) {
  let you = 0, ai = 0;
  for (const { id, g } of guesses) {
    const p = FIGHTERS.get(id);
    you += g === p.answer;
    ai += aiPick(p) === p.answer;
  }
  return { you, ai };
}

/** Rank by hits, then margin over the opponent, then earlier first. */
const rankScore = (you, opp, t) => you * 1000 + (you - opp + ROUNDS) * 10 + (1 - t / 1e13);

const newId = (t = Date.now()) => `${t.toString(36)}${Math.random().toString(36).slice(2, 6)}`;

async function rateLimited(req, bucket, max = 10) {
  const ip = String(req.headers["x-forwarded-for"] || "local").split(",")[0].trim();
  const key = `pom:rl:${bucket}:${ip}`;
  const [count] = await redis.pipeline([["INCR", key], ["EXPIRE", key, 60]]);
  return count > max;
}

function body(req) {
  // Vercel's req.body getter throws on malformed JSON; treat that (and non-JSON strings) as an empty request
  try {
    const b = req.body;
    if (typeof b !== "string") return b && typeof b === "object" ? b : {};
    return JSON.parse(b || "{}");
  } catch {
    return {};
  }
}

/** Session token: sent in a header (kept out of URLs and logs); the ?token= query is still accepted. */
function tokenOf(req) {
  return String(req.headers["x-pom-token"] || query(req).get("token") || "") || null;
}

function query(req) {
  return new URL(req.url, "http://localhost").searchParams;
}

/** Wrap a handler: no caching, storage check, JSON errors. */
function handler(fn) {
  return async (req, res) => {
    res.setHeader("Cache-Control", "no-store");
    if (!redis.configured()) return res.status(503).json({ error: "storage not configured" });
    try {
      const [status, payload] = await fn(req);
      return res.status(status).json(payload);
    } catch (e) {
      console.error(e);
      return res.status(500).json({ error: "server error" });
    }
  };
}

module.exports = { redis, Daily, ROUNDS, FIGHTERS, IDS, cleanName, parseGuesses, score, rankScore, newId,
  rateLimited, body, query, tokenOf, handler };
