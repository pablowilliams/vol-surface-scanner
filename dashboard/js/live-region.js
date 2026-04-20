// Live region helpers. Polite and assertive announcers for screen readers.

const polite = document.getElementById("status-polite");
const assertive = document.getElementById("status-assertive");

function announce(target, message) {
  if (!target) return;
  target.textContent = "";
  // Re-set on next animation frame so repeats announce.
  requestAnimationFrame(() => {
    target.textContent = message;
  });
}

export function announcePolite(message) {
  announce(polite, message);
}

export function announceAssertive(message) {
  announce(assertive, message);
}
