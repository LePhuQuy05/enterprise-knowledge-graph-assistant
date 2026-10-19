// Static product help. No API, provider, record, question or credential access.
const PREFERENCE_KEY = "kg.ui.onboarding.v1";

export const TOUR_STEPS = Object.freeze([
  { page: "overview", target: "overview-title", section: "Welcome to Evidence Lab", title: "Follow a question to its sources.",
    description: "Investigate AWS workflows in Vietnamese or English using official live search, or explore the recorded pilot.",
    points: ["Choose official live search or the frozen pilot when asking.", "Read the saved passages cited by an answer.", "Keep a brief and inspect a recorded retrieval experiment."],
    tip: "You can skip this tour and replay it from Quick tour at any time." },
  { page: "knowledge", target: "catalog-search", section: "Knowledge Explorer", title: "Explore the recorded pilot.",
    description: "Search a title or topic, such as key policies, and narrow the collection by AWS service.",
    points: ["Choose a document to inspect its sections and official link.", "Draft a question opens an editable question; it does not send it.", "This catalog covers the pilot. Official live search can find sources beyond it; neither mode inspects your AWS account."],
    tip: "Browsing the collection does not generate an answer." },
  { page: "investigate", target: "scenario-library", section: "Investigate · guided starting points", title: "Start with what you know.",
    description: "Open Use a guided scenario when you need help framing a workflow question. You can also write a question directly.",
    points: ["Choose an example and fill in the facts you actually know.", "Keep unknown conditions marked as unknown.", "Build an editable question prepares a draft for you to review."],
    tip: "Choosing a scenario or building a draft does not submit a question." },
  { page: "investigate", target: "question-form", section: "Investigate · your question", title: "You decide when to ask.",
    description: "Ask in Vietnamese or English. Choose AWS official live search for broad coverage, or the pilot snapshot for a fixed corpus. Select your answer language.",
    points: ["Ask a question sends the request and can use the demo allowance.", "Each question is independent; history is not added as model context.", "An answer may be partial or withheld when the sources are insufficient."],
    tip: "This tour never clicks Ask or edits your question." },
  { page: "investigate", target: "source-panel", section: "Investigate · source inspector", title: "Read beyond the answer.",
    description: "When an answer is available, select a statement and then a cited source or Read source passage.",
    points: ["The source inspector shows the stored passage and snapshot context.", "An exact quote is highlighted when it can be located uniquely.", "Check conditions and scope: a citation is not automatic proof that a statement is correct."],
    tip: "No example answer is generated or opened during this tour." },
  { page: "investigate", target: "history-title", section: "Investigate · history and briefs", title: "Return to your work and keep a brief.",
    description: "Your questions contains your own saved investigations. Open a completed record to revisit its answer and sources.",
    points: ["Opening saved history does not generate a new answer.", "Add personal notes and Export brief .md from a completed answer.", "Notes last for this session; export checks access to the saved answer again."],
    tip: "The tour preserves your current draft, notes and selected answer." },
  { page: "lab", target: "lab-question", section: "Recorded Retrieval Lab", title: "Compare retrieval, with the limits in view.",
    description: "Choose a development question to inspect frozen rankings, context and evidence groups. Try dev-11 after the tour.",
    points: ["Evidence Radar shows where required evidence was found or missed.", "The results are AI_DRAFT retrieval experiments, not measured answer accuracy.", "Graph answering is not enabled; the lab does not run live inference."],
    tip: "Finish to return to where you started. Quick tour is always available while signed in." },
].map(step => Object.freeze({ ...step, points: Object.freeze(step.points) })));

export function tourPreference(storage = () => globalThis.localStorage) {
  return {
    seen() {
      try { return ["dismissed", "completed"].includes(storage()?.getItem(PREFERENCE_KEY)); }
      catch { return false; }
    },
    remember(value) {
      if (!["dismissed", "completed"].includes(value)) throw new TypeError("INVALID_TOUR_PREFERENCE");
      try { const target = storage(); if (!target) return false; target.setItem(PREFERENCE_KEY, value); return true; }
      catch { return false; }
    },
  };
}

export function createOnboarding({ document, navigate, getPage, canStart, storage }) {
  const get = id => document.getElementById(id), dialog = get("workspace-tour"), replay = get("tour-replay");
  const preference = tourPreference(storage), win = document.defaultView;
  let active = false, autoAttempted = false, index = 0, highlighted = null, origin = null;
  function clearHighlight() { highlighted?.classList.remove("tour-target"); highlighted = null; }
  function render() {
    const step = TOUR_STEPS[index];
    clearHighlight(); navigate(step.page);
    get("tour-section").textContent = step.section;
    get("tour-title").textContent = step.title;
    get("tour-description").textContent = step.description;
    get("tour-points").replaceChildren(...step.points.map(text => { const li = document.createElement("li"); li.textContent = text; return li; }));
    get("tour-tip").textContent = step.tip;
    get("tour-count").textContent = `Step ${index + 1} of ${TOUR_STEPS.length}`;
    get("tour-progress").max = TOUR_STEPS.length; get("tour-progress").value = index + 1;
    get("tour-back").disabled = index === 0;
    get("tour-next").textContent = index === 0 ? "Start tour" : index === TOUR_STEPS.length - 1 ? "Finish tour" : "Next";
    get("tour-skip").textContent = index === 0 ? "Maybe later" : "Skip tour";
    dialog.dataset.welcome = String(index === 0);
    highlighted = get(step.target);
    if (highlighted?.getClientRects().length) {
      highlighted.classList.add("tour-target");
      highlighted.scrollIntoView({ behavior: "instant", block: "start" });
    }
    dialog.scrollTop = 0;
    get("tour-title").focus({ preventScroll: true });
  }
  function close({ remember = "dismissed", restore = true } = {}) {
    if (!dialog.open) return;
    if (remember) preference.remember(remember);
    dialog.close(); clearHighlight(); document.body.classList.remove("tour-open");
    if (restore && active && origin) {
      navigate(origin.page);
      // Keep the form/record state and return to the original location.
      win.scrollTo({ top: origin.scrollY, left: origin.scrollX, behavior: "instant" });
      replay.focus({ preventScroll: true });
    }
    origin = null;
  }
  function open() {
    if (!active || !canStart() || dialog.open || typeof dialog.showModal !== "function") return false;
    origin = { page: getPage(), scrollX: win.scrollX, scrollY: win.scrollY };
    index = 0;
    dialog.showModal(); document.body.classList.add("tour-open"); render();
    return true;
  }
  replay.addEventListener("click", open);
  get("tour-close").addEventListener("click", () => close());
  get("tour-skip").addEventListener("click", () => close());
  get("tour-back").addEventListener("click", () => { if (dialog.open && index > 0) { index--; render(); } });
  get("tour-next").addEventListener("click", () => {
    if (!dialog.open) return;
    if (index === TOUR_STEPS.length - 1) close({ remember: "completed" });
    else { index++; render(); }
  });
  dialog.addEventListener("cancel", event => { event.preventDefault(); close(); });
  function syncAvailability() { replay.disabled = !active || !canStart(); }
  return {
    open, syncAvailability,
    activate() {
      active = true; replay.hidden = typeof dialog.showModal !== "function"; syncAvailability();
      if (!autoAttempted && !preference.seen() && canStart()) { autoAttempted = true; open(); }
    },
    deactivate() {
      active = false; close({ remember: null, restore: false }); clearHighlight();
      replay.hidden = true; replay.disabled = true;
    },
  };
}
