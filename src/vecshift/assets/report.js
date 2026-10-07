// Tooltips and the severity filter. The report is fully readable without this script.
(() => {
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.setAttribute("role", "tooltip");
  document.body.append(tip);

  const show = (el, x, y) => {
    tip.replaceChildren();
    const value = document.createElement("strong");
    value.textContent = el.dataset.tipValue || "";
    const label = document.createElement("span");
    label.textContent = el.dataset.tipLabel || "";
    tip.append(value, label);
    tip.style.display = "block";
    const pad = 12;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let left = x - w / 2;
    left = Math.max(pad, Math.min(left, window.innerWidth - w - pad));
    let top = y - h - 14;
    if (top < pad) top = y + 18;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  };
  const hide = () => { tip.style.display = "none"; };

  document.querySelectorAll("[data-tip-value]").forEach((el) => {
    el.addEventListener("pointermove", (e) => show(el, e.clientX, e.clientY));
    el.addEventListener("pointerleave", hide);
    el.addEventListener("focus", () => {
      const r = el.getBoundingClientRect();
      show(el, r.left + r.width / 2, r.top);
    });
    el.addEventListener("blur", hide);
  });
  window.addEventListener("scroll", hide, { passive: true });

  const buttons = document.querySelectorAll(".filters button");
  const findings = document.querySelectorAll(".finding");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const severity = button.dataset.filter;
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      findings.forEach((f) => {
        f.hidden = severity !== "all" && f.dataset.severity !== severity;
      });
    });
  });
})();
