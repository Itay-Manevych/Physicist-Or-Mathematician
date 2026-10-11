// Leaderboards for "vs machine" games: all-time, and one per daily challenge.
//
// GET  /api/leaderboard                  -> { top, stats, daily: { day, number, top, count } }
// GET  /api/leaderboard?day=YYYY-MM-DD   -> same, with that day's daily board
// POST /api/leaderboard {name, guesses:[{id, g}], daily?: "YYYY-MM-DD"}
//      -> { rank, total, entry, ...board }
//
// Scores are recomputed from the guesses and the answer key. Daily entries must use that day's fighters.
const G = require("./_game");

const K = { board: "pom:lb", entries: "pom:entries", stats: "pom:stats",
  daily: (day) => `pom:daily:${day}`, dailyEntries: (day) => `pom:daily:${day}:entries` };
const DAILY_TTL = 60 * 60 * 24 * 8;
const isDay = (d) => typeof d === "string" && /^\d{4}-\d{2}-\d{2}$/.test(d) && !isNaN(Date.parse(d));

async function topOf(boardKey, entriesKey, n = 10) {
  const [ids, count] = await G.redis.pipeline([["ZREVRANGE", boardKey, 0, n - 1], ["ZCARD", boardKey]]);
  const rows = ids.length ? (await G.redis.pipeline([["HMGET", entriesKey, ...ids]]))[0] : [];
  return { top: rows.filter(Boolean).map((r) => JSON.parse(r)), count };
}

async function boards(day) {
  const [all, daily, [stats]] = await Promise.all([
    topOf(K.board, K.entries), topOf(K.daily(day), K.dailyEntries(day)), G.redis.pipeline([["HGETALL", K.stats]])]);
  const s = {};
  for (let i = 0; i < (stats || []).length; i += 2) s[stats[i]] = Number(stats[i + 1]);
  return {
    top: all.top,
    stats: { games: s.games || 0, rounds: s.rounds || 0, human: s.human || 0, ai: s.ai || 0 },
    daily: { day, number: G.Daily.number(day), top: daily.top, count: daily.count },
  };
}

async function submit(req) {
  const b = G.body(req);
  const name = G.cleanName(b.name);
  const guesses = G.parseGuesses(b.guesses);
  if (!name) return [400, { error: "name required" }];
  if (!guesses) return [400, { error: "invalid game" }];
  const day = b.daily;
  if (day !== undefined) {
    // only today's (UTC) daily, allowing a day either side for time zones
    const ok = [-1, 0, 1].map((d) => G.Daily.today(Date.now() + d * 86400000)).includes(day);
    const expected = isDay(day) && G.Daily.fighters(day, G.IDS, G.ROUNDS);
    if (!ok || !expected || expected.some((id, i) => guesses[i].id !== id)) return [400, { error: "not today's daily" }];
  }
  if (await G.rateLimited(req, "lb")) return [429, { error: "slow down" }];

  const { you, ai } = G.score(guesses);
  const t = Date.now();
  const entry = { id: G.newId(t), name, you, ai, t };
  const [boardKey, entriesKey] = day ? [K.daily(day), K.dailyEntries(day)] : [K.board, K.entries];
  const cmds = [
    ["ZADD", boardKey, G.rankScore(you, ai, t), entry.id],
    ["HSET", entriesKey, entry.id, JSON.stringify(entry)],
    ["HINCRBY", K.stats, "games", 1],
    ["HINCRBY", K.stats, "rounds", G.ROUNDS],
    ["HINCRBY", K.stats, "human", you],
    ["HINCRBY", K.stats, "ai", ai],
    ["ZREVRANK", boardKey, entry.id],
    ["ZCARD", boardKey],
  ];
  if (day) cmds.push(["EXPIRE", boardKey, DAILY_TTL], ["EXPIRE", entriesKey, DAILY_TTL]);
  const out = await G.redis.pipeline(cmds);
  return [200, { rank: out[6] + 1, total: out[7], entry, ...(await boards(day || G.Daily.today())) }];
}

module.exports = G.handler(async (req) => {
  if (req.method === "GET") {
    const day = G.query(req).get("day");
    return [200, await boards(isDay(day) ? day : G.Daily.today())];
  }
  if (req.method === "POST") return submit(req);
  return [405, { error: "method not allowed" }];
});
