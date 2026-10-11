// Daily challenge: everyone gets the same 10 fighters on the same (UTC) day.
// Shared by the browser (window.Daily) and the API (require), so both agree on the day's fighters.
(function (root) {
  const LAUNCH = Date.UTC(2026, 9, 11); // daily #1 = 2026-10-11

  function today(now = Date.now()) {
    return new Date(now).toISOString().slice(0, 10);
  }

  function number(day) {
    return Math.floor((Date.parse(day + "T00:00:00Z") - LAUNCH) / 86400000) + 1;
  }

  // small deterministic PRNG seeded from the date string
  function rng(seed) {
    let h = 1779033703 ^ seed.length;
    for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 3432918353), h = (h << 13) | (h >>> 19);
    let a = h >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ids of the day's fighters, from the full pool of fighter ids
  function fighters(day, ids, rounds = 10) {
    const pool = [...ids].sort((a, b) => a - b);
    const rand = rng("pom-daily-" + day);
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, rounds);
  }

  const api = { today, number, fighters };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Daily = api;
})(this);
