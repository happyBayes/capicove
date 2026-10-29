/* Reuse Material's search dialog, index, keyboard handling and clear action. */
(() => {
  const trigger = document.querySelector(".capicove-hero__search");
  const query = document.querySelector('[data-md-component="search-query"]');
  const toggle = document.getElementById("__search");
  if (!trigger || !query || !toggle) return;
  let launchedFromHero = false;
  trigger.hidden = false;
  trigger.addEventListener("click", () => {
    launchedFromHero = true;
    toggle.checked = true;
    toggle.dispatchEvent(new Event("change", { bubbles: true }));
    query.focus();
  });
  toggle.addEventListener("change", () => {
    if (!toggle.checked && launchedFromHero) {
      launchedFromHero = false;
      trigger.focus({ preventScroll: true });
    }
  });
})();
