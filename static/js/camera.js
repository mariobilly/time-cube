// How the cube turns over time, and how it morphs between the looks.
// A pose is { yaw, pitch, roll } in degrees plus morph: 0 = Wire, 1 = Slices, 2 = Solid, 3 = Points.
// The Tour and Tumble choreographies only use Wire to Solid; Points is yours to choose.

// "Tour" (mode "reel") replays the moves of the nayana reel as a seamless 30 s loop: a flat front view of the video, the stack
// turning into wireframe and swinging edge-on, a dense dark pass, the solid block tumbling upside down, back to the
// wire, and home to the front. The last key is the first one turned twice round (yaw) and once over (roll).
export const REEL_PERIOD = 30;
export const REEL_KEYS = [
  // t     yaw  pitch  roll  morph
  [0.0,     0,    0,    0,  1.7],   // the video, flat, wire showing through the dark parts
  [1.6,    10,    5,   -2,  0.9],
  [3.2,    50,   16,   -6,  0.15],  // the wire stack: long lines through time
  [5.0,   110,   10,   -4,  0.0],
  [6.8,   180,    0,    0,  0.0],   // seen from behind, edge-on: the column planes fan out
  [8.2,   215,  -12,    6,  0.9],   // dense and dark
  [9.6,   250,  -28,   30,  1.9],   // the solid block, slit-scan sides
  [11.4,  290,  -18,   95,  2.0],
  [13.2,  330,   -6,  160,  2.0],
  [15.0,  360,    0,  180,  2.0],   // upside-down front
  [16.8,  395,   20,  200,  2.0],
  [18.4,  430,   12,  190,  1.6],
  [19.6,  470,    0,  180,  0.7],
  [21.0,  540,    0,  180,  0.0],   // the fan again
  [22.6,  580,   14,  230,  0.0],
  [24.2,  640,    8,  320,  0.4],
  [25.6,  720,    0,  360,  1.5],   // home: the video again
  [27.2,  760,  -18,  380,  2.0],
  [28.8,  740,  -10,  370,  2.0],
  [30.0,  720,    0,  360,  1.7],
];

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const mod = (a, n) => ((a % n) + n) % n;

// Smooth (C1) interpolation through looping keys: Hermite segments with finite-difference tangents.
export function loopSpline(keys, period, t) {
  const n = keys.length - 1;                  // the last key closes the loop
  const offset = keys[n].map((v, i) => v - keys[0][i]);
  offset[0] = period;
  const key = (i) => {                        // key i, extended periodically in both directions
    const wrap = Math.floor(i / n);
    const k = keys[mod(i, n)];
    return k.map((v, j) => v + wrap * offset[j]);
  };
  const tt = mod(t, period);
  let i = 0;
  while (i < n - 1 && keys[i + 1][0] <= tt) i++;
  const p0 = key(i), p1 = key(i + 1), pm = key(i - 1), p2 = key(i + 2);
  const dt = p1[0] - p0[0];
  const s = dt > 0 ? (tt - p0[0]) / dt : 0;
  const h00 = 2 * s ** 3 - 3 * s ** 2 + 1, h10 = s ** 3 - 2 * s ** 2 + s;
  const h01 = -2 * s ** 3 + 3 * s ** 2, h11 = s ** 3 - s ** 2;
  const out = [];
  for (let j = 1; j < p0.length; j++) {
    const m0 = (p1[j] - pm[j]) / (p1[0] - pm[0]);
    const m1 = (p2[j] - p0[j]) / (p2[0] - p0[0]);
    out.push(h00 * p0[j] + h10 * dt * m0 + h01 * p1[j] + h11 * dt * m1);
  }
  return out;
}

export function reelPose(t) {
  const [yaw, pitch, roll, morph] = loopSpline(REEL_KEYS, REEL_PERIOD, t);
  return { yaw, pitch, roll, morph: clamp(morph, 0, 2) };
}

// "Tumble": an endless slow tumble on three unrelated rhythms; the look swings Solid -> Wire -> Solid every 16 s
// and rests a little at each end.
export function tumblePose(t) {
  const yaw = 22 * t + 25 * Math.sin(t * 0.31);
  const pitch = 26 * Math.sin(t * 0.23 + 1.0) + 9 * Math.sin(t * 0.61);
  const roll = 38 * Math.sin(t * 0.17) + 6 * t;
  const morph = 1 + clamp(1.5 * Math.cos((2 * Math.PI * t) / 16), -1, 1);
  return { yaw, pitch, roll, morph };
}

// "Sway": swings left and right (and a little up and down) around the front view, so the relief keeps showing.
export function swayPose(t) {
  return { yaw: 38 * Math.sin((2 * Math.PI * t) / 12), pitch: 9 * Math.sin((2 * Math.PI * t) / 17), roll: 0, morph: null };
}

export const MODES = ["reel", "tumble", "sway", "spin", "still"];

// mode: reel | tumble | sway | spin | still. p: { speed, spinYaw, spinPitch, spinRoll } (degrees per second for spin).
// The morph of spin/still is null: the Look slider decides.
export function pose(mode, t, p = {}) {
  const ts = t * (p.speed ?? 1);
  if (mode === "reel") return reelPose(ts);
  if (mode === "tumble") return tumblePose(ts);
  if (mode === "sway") return swayPose(ts);
  if (mode === "spin") {
    return { yaw: (p.spinYaw ?? 20) * ts, pitch: (p.spinPitch ?? 0) * ts, roll: (p.spinRoll ?? 0) * ts, morph: null };
  }
  return { yaw: 0, pitch: 0, roll: 0, morph: null };
}

export const LOOKS = ["Wire", "Slices", "Solid", "Points"];

// Morph value -> how much of each look shows. They always add up to 1.
export function lookWeights(m) {
  m = clamp(m, 0, LOOKS.length - 1);
  const w = (i) => clamp(1 - Math.abs(m - i), 0, 1);
  return { wire: w(0), slices: w(1), solid: w(2), points: w(3) };
}
