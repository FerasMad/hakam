import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const sourceRoots = ["app", "components", "lib"].map((directory) => join(root, directory));

function collect(directory) {
  return readdirSync(directory).flatMap((entry) => {
    const path = join(directory, entry);
    return statSync(path).isDirectory() ? collect(path) : [path];
  });
}

const source = sourceRoots
  .flatMap(collect)
  .filter((path) => /\.(?:ts|tsx|css)$/.test(path))
  .map((path) => readFileSync(path, "utf8"))
  .join("\n");

if (/card_colour\s*:\s*\{/.test(source)) {
  throw new Error("card_colour must remain null in the production frontend contract");
}

if (/label\s*:\s*[\"'](?:yellow|red)[\"']/.test(source)) {
  throw new Error("Found a prohibited card-colour prediction label");
}

if (!source.includes("الثقة منخفضة — هذه الحالة تحتاج مراجعة بشرية.")) {
  throw new Error("The repository abstention message is missing");
}

if (!source.includes("num_views: 1")) {
  throw new Error("The single-video contract invariant is missing");
}

const requiredUiInvariants = [
  'body.append("video", file, file.name)',
  "/api/analyze",
  "analysisErrorMessage",
  'isOffence && contract.card',
  'dir="rtl"',
  'dir="ltr"',
  'object-contain',
  'prefers-reduced-motion',
  'className="skip-link"',
  "ReviewIcon",
  'max-w-[52rem]',
  "processingStatus",
  'Action family not confidently determined',
  'Colour is not determined by this model.',
];

for (const invariant of requiredUiInvariants) {
  if (!source.includes(invariant)) {
    throw new Error(`Missing UI invariant: ${invariant}`);
  }
}

const forbiddenProductionMarkers = [
  "MockAnalysisService",
  "demo-scenario",
  "NEXT_PUBLIC_SHOW_DEMO_CONTROLS",
  "getMockAnalysis",
  "processingStages",
  "Retrieving relevant law",
  "Preparing ruling",
  "1 VIEW",
];

for (const marker of forbiddenProductionMarkers) {
  if (source.includes(marker)) {
    throw new Error(`Prototype marker leaked into the production flow: ${marker}`);
  }
}

if (existsSync(join(root, "lib", "mock"))) {
  throw new Error("The Phase 3 mock fixture directory must not ship in the production frontend");
}

if (/\b(?:yellow|red)[-_ ]card\b/i.test(source)) {
  throw new Error("Found prohibited yellow/red card-colour UI copy");
}

console.log(
  "Frontend invariants verified: real one-video API, cascade, honest processing, localized errors, RTL/LTR accessibility, and no card-colour prediction.",
);
