// node tests/js/shape.test.mjs - the shapes of time put points where they should go.
import assert from "node:assert/strict";
import { SHAPES, shapePoint } from "../../static/js/renderer.js";

const near = (a, b, msg) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-9, `${msg}: ${a} vs ${b}`));
const half = [0.5, 0.375, 0.5];   // a 4:3 cube, depth 1
const p = [0.5, 0.375, -0.5];

assert.deepEqual(SHAPES, ["cube", "stack", "echo"]);
near(shapePoint({ shape: "cube" }, half, p, 1), p, "the cube leaves points alone");
near(shapePoint({ shape: "stack" }, half, p, 0), p, "stack: the front frame stays put");
near(shapePoint({ shape: "stack" }, half, p, 1), [0.5, 0.375 + 0.75, -0.5], "stack: the back is lifted by a frame height");
near(shapePoint({ shape: "echo" }, half, p, 1), [0.15, 0.1125, -0.5], "echo: the back shrinks to 30 %");
near(shapePoint({ shape: "echo" }, half, p, 0), p, "echo: the front keeps its size");
near(shapePoint({ shape: "cube", spiral: 180 }, half, p, 0.5), [-0.375, 0.5, -0.5], "spiral: half way turns half the angle");
near(shapePoint({ shape: "cube", spiral: 360 }, half, p, 1), p, "spiral 360: the back comes all the way round");
console.log("shape tests passed");
