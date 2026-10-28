#!/usr/bin/env node

const fs = require("fs");
const solc = require("solc");

try {
  const input = fs.readFileSync(0, "utf8");
  const output = solc.compile(input);
  // Synchronous output matters here: the solcjs CLI can truncate large JSON
  // responses when Node exits before a buffered stdout write is complete.
  fs.writeSync(1, output);
} catch (error) {
  fs.writeSync(2, `${error.stack || error}\n`);
  process.exitCode = 1;
}
