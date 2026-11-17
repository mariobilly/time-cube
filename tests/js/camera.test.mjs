// node tests/js/camera.test.mjs - the choreography loops seamlessly and the look weights add up.
import assert from "node:assert/strict";
import { lookWeights, pose, REEL_KEYS, REEL_PERIOD, reelPose, swayPose, tumblePose } from "../../static/js/camera.js";

const near = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg}: ${a} vs ${b}`);
const wrap = (d) => ((((d + 180) % 360) + 360) % 360) - 180;

// The keys are hit exactly.
for (const [t, yaw, pitch, roll, morph] of REEL_KEYS.slice(0, -1)) {
  const p = reelPose(t);
  near(p.yaw, yaw, 1e-6, `yaw at ${t}`);
  near(p.pitch, pitch, 1e-6, `pitch at ${t}`);
  near(p.roll, roll, 1e-6, `roll at ${t}`);
  near(p.morph, Math.min(2, Math.max(0, morph)), 1e-6, `morph at ${t}`);
}
// Seamless loop: just before the end equals just after the start (angles modulo 360), no jump in speed.
const e = 1e-3;
const a = reelPose(REEL_PERIOD - e), b = reelPose(e), c = reelPose(0);
for (const k of ["yaw", "pitch", "roll"]) {
  near(wrap(a[k] - b[k]), wrap((reelPose(REEL_PERIOD - 2 * e)[k] - reelPose(2 * e)[k])) , 0.05, `${k} loop`);
  near(wrap(reelPose(REEL_PERIOD)[k] - c[k]), 0, 1e-6, `${k} period`);
}
// Smooth: no step bigger than the speed allows anywhere in the loop.
let prev = reelPose(0);
for (let t = 0.01; t <= REEL_PERIOD * 2; t += 0.01) {
  const p = reelPose(t);
  for (const k of ["yaw", "pitch", "roll"]) assert.ok(Math.abs(wrap(p[k] - prev[k])) < 2, `${k} jumps at ${t}`);
  assert.ok(p.morph >= 0 && p.morph <= 2);
  prev = p;
}
// Weights always add up to one, and only neighbouring looks blend.
for (let m = -0.5; m <= 3.5; m += 0.05) {
  const w = lookWeights(m);
  near(w.wire + w.slices + w.solid + w.points, 1, 1e-9, `weights at ${m}`);
  assert.ok(Object.values(w).filter((x) => x > 0).length <= 2, `at most two looks at ${m}`);
}
assert.deepEqual(lookWeights(0), { wire: 1, slices: 0, solid: 0, points: 0 });
assert.deepEqual(lookWeights(2), { wire: 0, slices: 0, solid: 1, points: 0 });
assert.deepEqual(lookWeights(3), { wire: 0, slices: 0, solid: 0, points: 1 });
assert.deepEqual(lookWeights(2.5), { wire: 0, slices: 0, solid: 0.5, points: 0.5 });
// The choreographies stay within Wire..Solid.
for (let t = 0; t < 60; t += 0.25) assert.ok(tumblePose(t).morph <= 2 && reelPose(t).morph <= 2);
// Modes.
assert.equal(pose("spin", 2, { spinYaw: 10, spinPitch: 0, spinRoll: 0, speed: 1 }).yaw, 20);
assert.equal(pose("still", 5).morph, null);
// Sway stays around the front view and loops.
for (let t = 0; t < 40; t += 0.1) assert.ok(Math.abs(swayPose(t).yaw) <= 38 && Math.abs(swayPose(t).pitch) <= 9);
near(swayPose(0).yaw, 0, 1e-9, "sway starts at the front");
assert.equal(pose("sway", 3).morph, null);
assert.ok(Number.isFinite(tumblePose(123.4).yaw));
assert.equal(pose("reel", 3, { speed: 2 }).yaw, reelPose(6).yaw);
console.log("camera tests passed");
