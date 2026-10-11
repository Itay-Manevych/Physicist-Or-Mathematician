// Draws a shareable 1200x630 scorecard image of a finished game.
async function drawScorecard({ name, you, ai, rounds, outcome, log, url, oppName = "THE MACHINE", subtitle = "" }) {
  try { await Promise.all([document.fonts.load("40px 'Press Start 2P'"), document.fonts.load("40px 'VT323'")]); } catch {}
  const W = 1200, H = 630;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  const C = { purple: "#5B3FE0", ink: "#1A1033", yellow: "#FFD93D", white: "#FFFFFF", phys: "#22D3EE",
    math: "#FF5CB0", ok: "#3DDC84", bad: "#FF5A5A", lose: "#FF7A7A" };
  const px = (size) => `${size}px 'Press Start 2P', monospace`;
  const text = (s, x, y, size, color, align = "left", shadow = 0) => {
    g.font = px(size); g.textAlign = align; g.textBaseline = "alphabetic";
    if (shadow) { g.fillStyle = C.ink; g.fillText(s, x + shadow, y + shadow); }
    g.fillStyle = color; g.fillText(s, x, y);
  };
  const box = (x, y, w, h, fill, shadow = 8) => {
    g.fillStyle = C.ink; g.fillRect(x + shadow, y + shadow, w, h);
    g.fillRect(x - 4, y - 4, w + 8, h + 8);
    g.fillStyle = fill; g.fillRect(x, y, w, h);
  };

  g.fillStyle = C.purple; g.fillRect(0, 0, W, H);
  g.fillStyle = "rgba(26,16,51,0.08)";
  for (let y = 0; y < H; y += 3) g.fillRect(0, y, W, 1);

  text("PHYSICIST", 60, 80, 34, C.phys, "left", 4);
  text("OR", 418, 80, 22, C.yellow, "left", 3);
  text("MATHEMATICIAN?", 500, 80, 34, C.math, "left", 4);

  const head = { win: "YOU WIN!", lose: "GAME OVER", draw: "DRAW!" }[outcome];
  text(head, W / 2, 200, 64, { win: C.yellow, lose: C.lose, draw: C.white }[outcome], "center", 8);

  // score boxes
  const who = (name || "YOU").toUpperCase().slice(0, 12);
  box(60, 250, 500, 150, C.white);
  box(640, 250, 500, 150, C.white);
  text(who, 90, 300, 22, C.ink);
  text(`${you}/${rounds}`, 90, 375, 52, C.purple);
  text(oppName.slice(0, 12), 670, 300, 22, C.ink);
  if (subtitle) text(subtitle, W - 60, 128, 18, C.yellow, "right", 3);
  text(`${ai}/${rounds}`, 670, 375, 52, C.purple);

  // round-by-round hits
  const sq = 36, gap = 10, x0 = (W - (rounds * sq + (rounds - 1) * gap)) / 2;
  text("YOU", x0 - 20, 468, 14, C.yellow, "right");
  text(oppName === "THE MACHINE" ? "AI" : oppName.slice(0, 6), x0 - 20, 518, 14, C.yellow, "right");
  log.forEach((r, i) => {
    const x = x0 + i * (sq + gap);
    g.fillStyle = C.ink; g.fillRect(x - 3, 440, sq + 6, sq + 6); g.fillRect(x - 3, 490, sq + 6, sq + 6);
    g.fillStyle = r.youRight ? C.ok : C.bad; g.fillRect(x, 443, sq, sq);
    g.fillStyle = (r.oppRight ?? r.aiRight) ? C.ok : C.bad; g.fillRect(x, 493, sq, sq);
  });

  g.fillStyle = C.ink; g.fillRect(0, H - 70, W, 70);
  text(oppName === "THE MACHINE" ? "CAN YOU BEAT THE MACHINE?" : "CHALLENGE YOUR FRIENDS!", 60, H - 25, 16, C.yellow);
  g.font = "34px 'VT323', monospace"; g.textAlign = "right"; g.fillStyle = C.white;
  g.fillText(url.replace(/^https?:\/\//, "").replace(/\/$/, ""), W - 60, H - 25);

  return new Promise((resolve) => c.toBlob((blob) => resolve({ blob, url: c.toDataURL("image/png") }), "image/png"));
}
