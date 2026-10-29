import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";

const source = readFileSync(
  new URL("../../docs/site/javascripts/presentation.js", import.meta.url),
  "utf8",
);

function setup(missing) {
  let active;
  const trigger = new EventTarget();
  trigger.hidden = true;
  trigger.focus = () => { active = trigger; };
  const query = { focus: () => { active = query; } };
  const toggle = new EventTarget();
  toggle.checked = false;
  runInNewContext(source, {
    Event,
    document: {
      querySelector: (selector) => {
        const element = selector === ".capicove-hero__search" ? trigger : query;
        return element === missing?.(trigger, query, toggle) ? null : element;
      },
      getElementById: () => toggle === missing?.(trigger, query, toggle) ? null : toggle,
    },
  });
  return { trigger, query, toggle, active: () => active };
}

test("launches the existing search and restores focus when it closes", () => {
  const { trigger, query, toggle, active } = setup();
  assert.equal(trigger.hidden, false);
  trigger.dispatchEvent(new Event("click"));
  assert.equal(toggle.checked, true);
  assert.equal(active(), query);
  toggle.checked = false;
  toggle.dispatchEvent(new Event("change"));
  assert.equal(active(), trigger);
});

test("does not take focus when search was opened from the original header", () => {
  const { query, toggle, active } = setup();
  query.focus();
  toggle.checked = false;
  toggle.dispatchEvent(new Event("change"));
  assert.equal(active(), query);
});

for (const [name, missing] of [
  ["hero", (trigger) => trigger],
  ["query", (_trigger, query) => query],
  ["toggle", (_trigger, _query, toggle) => toggle],
]) {
  test(`leaves the launcher hidden when ${name} is unavailable`, () => {
    assert.equal(setup(missing).trigger.hidden, true);
  });
}
