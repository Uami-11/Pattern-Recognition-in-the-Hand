// Verifies that the browser inference matches scikit-learn's predictions.
// Run with: node web/test_model.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

export function classify(model, features) {
  const sums = new Float64Array(model.classes.length);

  for (const tree of model.trees) {
    let node = 0;
    while (tree.l[node] !== -1) {
      node = features[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    }
    const probabilities = tree.p[node];
    for (let index = 0; index < probabilities.length; index += 1) {
      sums[index] += probabilities[index];
    }
  }

  let best = 0;
  for (let index = 1; index < sums.length; index += 1) {
    if (sums[index] > sums[best]) best = index;
  }

  const probabilities = Array.from(sums, (value) => value / model.trees.length);
  return { className: model.classes[best], probabilities };
}

const model = JSON.parse(
  fs.readFileSync(path.join(here, "model", "gesture_model.json"), "utf8"),
);
const vectors = JSON.parse(
  fs.readFileSync(path.join(here, "model", "test_vectors.json"), "utf8"),
);

let failures = 0;

for (const [index, testCase] of vectors.cases.entries()) {
  const result = classify(model, testCase.features);
  const maxDelta = Math.max(
    ...result.probabilities.map((value, classIndex) =>
      Math.abs(value - testCase.probabilities[classIndex]),
    ),
  );

  if (result.className !== testCase.expected || maxDelta > 1e-5) {
    failures += 1;
    console.error(
      `FAIL case ${index} (${testCase.label}): expected ${testCase.expected}, ` +
        `got ${result.className}, max probability delta ${maxDelta}`,
    );
  }
}

if (failures > 0) {
  console.error(`\n${failures}/${vectors.cases.length} cases failed.`);
  process.exit(1);
}

console.log(`All ${vectors.cases.length} model cases match scikit-learn.`);
