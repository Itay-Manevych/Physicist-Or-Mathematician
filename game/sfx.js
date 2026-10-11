// 8-bit sound effects synthesised with Web Audio (no audio files).
const Sfx = (() => {
  let ctx = null;
  let muted = false;
  try { muted = localStorage.getItem("pom-muted") === "1"; } catch {}

  function audio() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  // square-wave notes: [frequency Hz, duration s], played back to back
  function notes(seq, { type = "square", volume = 0.08 } = {}) {
    const a = !muted && audio();
    if (!a) return;
    let t = a.currentTime;
    for (const [freq, dur] of seq) {
      const osc = a.createOscillator(), gain = a.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, t);
      gain.gain.setValueAtTime(volume, t);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      osc.connect(gain).connect(a.destination);
      osc.start(t);
      osc.stop(t + dur);
      t += dur;
    }
  }

  function noise(dur = 0.18, volume = 0.12) {
    const a = !muted && audio();
    if (!a) return;
    const buf = a.createBuffer(1, Math.floor(a.sampleRate * dur), a.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / data.length);
    const src = a.createBufferSource(), gain = a.createGain();
    gain.gain.value = volume;
    src.buffer = buf;
    src.connect(gain).connect(a.destination);
    src.start();
  }

  return {
    get muted() { return muted; },
    setMuted(m) { muted = m; try { localStorage.setItem("pom-muted", m ? "1" : "0"); } catch {} },
    start: () => notes([[523, 0.08], [659, 0.08], [784, 0.08], [1047, 0.16]]),
    select: () => notes([[880, 0.05]]),
    round: () => notes([[392, 0.1], [392, 0.1], [587, 0.2]]),
    hit: () => { noise(0.15, 0.14); notes([[220, 0.06], [110, 0.12]], { volume: 0.06 }); },
    hurt: () => { noise(0.2, 0.1); notes([[160, 0.1], [90, 0.2]], { type: "sawtooth", volume: 0.05 }); },
    miss: () => notes([[196, 0.12], [147, 0.18]], { type: "triangle", volume: 0.1 }),
    win: () => notes([[523, 0.1], [659, 0.1], [784, 0.1], [1047, 0.12], [784, 0.1], [1047, 0.3]]),
    lose: () => notes([[392, 0.18], [349, 0.18], [311, 0.18], [262, 0.4]], { type: "triangle", volume: 0.1 }),
  };
})();
