import test from "node:test";
import assert from "node:assert/strict";
import { tourPreference } from "./onboarding.mjs";

test("tour preference reads and writes only a versioned UI marker, not other browser data", () => {
  const values = new Map([["unrelated", "keep me"]]), calls = [];
  const storage = { getItem: key => { calls.push(["get", key]); return values.get(key); },
    setItem: (key, value) => { calls.push(["set", key, value]); values.set(key, value); } };
  const pref = tourPreference(() => storage);
  assert.equal(pref.seen(), false);
  assert.equal(pref.remember("dismissed"), true); assert.equal(pref.seen(), true);
  pref.remember("completed"); assert.equal(pref.seen(), true);
  assert.equal(values.get("unrelated"), "keep me");
  assert.ok(calls.every(([, key]) => key === "kg.ui.onboarding.v1"));
  assert.throws(() => pref.remember("a record or token"), /INVALID_TOUR_PREFERENCE/);
});

test("blocked, absent and quota-limited storage do not throw into the application", () => {
  for (const factory of [() => { throw new Error("storage blocked"); }, () => null,
    () => ({ getItem() { throw new Error("read blocked"); }, setItem() { throw new Error("quota"); } })]) {
    const pref = tourPreference(factory);
    assert.equal(pref.seen(), false);
    assert.equal(pref.remember("dismissed"), false);
  }
});

test("only recognized outcomes suppress first-visit help", () => {
  for (const value of [undefined, null, "", "false", "true", "v0", "{not-json}"]) {
    assert.equal(tourPreference(() => ({ getItem: () => value })).seen(), false);
  }
  for (const value of ["dismissed", "completed"]) {
    assert.equal(tourPreference(() => ({ getItem: () => value })).seen(), true);
  }
});
