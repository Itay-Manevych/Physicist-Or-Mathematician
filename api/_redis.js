// Minimal Upstash Redis REST client (no dependencies). Files starting with "_" are not routes on Vercel.
// Env vars come from Vercel's Upstash integration (either naming scheme works).
const URL_ = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
const TOKEN = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;

let client = null; // the local dev server injects an in-memory client here

async function pipeline(commands) {
  if (client) return client.pipeline(commands);
  const r = await fetch(`${URL_}/pipeline`, {
    method: "POST",
    headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify(commands),
  });
  if (!r.ok) throw new Error(`redis ${r.status}`);
  return (await r.json()).map((x) => {
    if (x.error) throw new Error(x.error);
    return x.result;
  });
}

module.exports = {
  configured: () => Boolean(client || (URL_ && TOKEN)),
  pipeline,
  setClient: (c) => { client = c; },
};
