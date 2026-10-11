// Local dev server: serves game/ and the /api routes, with an in-memory Redis instead of Upstash.
//   node dev/server.js        -> http://localhost:8137
const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const PORT = Number(process.env.PORT || 8137);
const TYPES = { ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json", ".jpg": "image/jpeg",
  ".png": "image/png", ".js": "text/javascript", ".md": "text/markdown; charset=utf-8", ".svg": "image/svg+xml" };

// --- tiny in-memory Redis covering the commands the API uses ---
const kv = new Map();
const get = (k, make) => { if (!kv.has(k)) kv.set(k, make()); return kv.get(k); };
const zsorted = (k) => [...get(k, () => new Map()).entries()].sort((a, b) => b[1] - a[1]).map((e) => e[0]);
const commands = {
  GET: (k) => kv.get(k) ?? null,
  HGET: (k, f) => get(k, () => new Map()).get(f) ?? null,
  SET: (k, v) => { kv.set(k, v); return "OK"; },
  INCR: (k) => { const v = Number(kv.get(k) || 0) + 1; kv.set(k, v); return v; },
  EXPIRE: (k, s) => { setTimeout(() => kv.delete(k), Number(s) * 1000).unref(); return 1; },
  ZADD: (k, score, m) => { get(k, () => new Map()).set(m, Number(score)); return 1; },
  ZREVRANGE: (k, a, b) => zsorted(k).slice(Number(a), Number(b) + 1),
  ZREVRANK: (k, m) => zsorted(k).indexOf(m),
  ZCARD: (k) => get(k, () => new Map()).size,
  HSET: (k, ...fv) => { const h = get(k, () => new Map()); for (let i = 0; i < fv.length; i += 2) h.set(fv[i], String(fv[i + 1])); return fv.length / 2; },
  HSETNX: (k, f, v) => { const h = get(k, () => new Map()); if (h.has(f)) return 0; h.set(f, String(v)); return 1; },
  HMGET: (k, ...fs) => fs.map((f) => get(k, () => new Map()).get(f) ?? null),
  HINCRBY: (k, f, n) => { const h = get(k, () => new Map()); const v = Number(h.get(f) || 0) + Number(n); h.set(f, String(v)); return v; },
  HGETALL: (k) => [...get(k, () => new Map()).entries()].flat(),
};
require("../api/_redis").setClient({ pipeline: async (cmds) => cmds.map(([c, ...a]) => commands[c](...a)) });

const api = { "/api/leaderboard": require("../api/leaderboard"), "/api/room": require("../api/room") };

http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (api[url.pathname]) {
    let body = "";
    for await (const chunk of req) body += chunk;
    req.body = body ? JSON.parse(body) : {};
    res.status = (s) => { res.statusCode = s; return res; };
    res.json = (o) => { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(o)); };
    return api[url.pathname](req, res);
  }
  // dev only: save a PNG rendered in the browser (used to make game/og.png)
  if (req.method === "POST" && url.pathname === "/__save" && /^[a-z0-9_-]+\.png$/.test(url.searchParams.get("file") || "")) {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    fs.writeFileSync(path.join(ROOT, "game", url.searchParams.get("file")), Buffer.concat(chunks));
    return res.end("saved");
  }
  const file = path.join(ROOT, "game", decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname));
  if (!file.startsWith(path.join(ROOT, "game")) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.statusCode = 404;
    return res.end("not found");
  }
  res.setHeader("Content-Type", TYPES[path.extname(file)] || "application/octet-stream");
  fs.createReadStream(file).pipe(res);
}).listen(PORT, () => console.log(`game + api on http://localhost:${PORT}`));
