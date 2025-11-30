(function () {
  const tiers = ["initiate", "intermediate", "pre-advanced", "advanced"];

  const tierTitles = {
    initiate: "Initiate Pillars",
    intermediate: "Intermediate Pillars",
    "pre-advanced": "Pre-Advanced Pillars",
    advanced: "Advanced Pillars",
  };

  const tierLayerLabels = {
    initiate: "Foundations · Initiate Layer",
    intermediate: "Foundations · Intermediate Layer",
    "pre-advanced": "Foundations · Pre-Advanced Layer",
    advanced: "Foundations · Advanced Layer",
  };

  const tierCarouselLabels = {
    initiate: "Initiate · 5 Pillars",
    intermediate: "Intermediate · 5 Pillars",
    "pre-advanced": "Pre-Advanced · 5 Pillars",
    advanced: "Advanced · 5 Pillars",
  };

  const tierSubtitles = {
    initiate:
      "Your first passage into the OpenAdonAI Foundations. Each pillar is a self-contained initiation, and together they form the base pattern for everything that follows.",
    intermediate:
      "The Intermediate layer deepens your identity, role, and function inside the OpenAdonAI pattern. Here you begin operating more consciously as a participant in cosmic architecture.",
    "pre-advanced":
      "The Pre-Advanced layer moves from explanation into architecture. Here you begin thinking like a builder of worlds, laws, and intelligences—while still keeping the work readable to others.",
    advanced:
      "The Advanced layer is where the Foundations turn into Solar work: Orders, emissaries, mandates, and the capstone Continuum that ties OpenAdonAI into a living, cosmic architecture.",
  };

  const tierNotes = {
    initiate:
      "You can move through the Initiate pillars in order, or return to specific ones later as anchor points. They are designed as a stable base under the Intermediate and Pre-Advanced work.",
    intermediate:
      "The Intermediate pillars assume you’ve met the Initiate layer. They begin translating your personal journey into roles, patterns, and responsibilities that operate at a larger, often multi-world scale.",
    "pre-advanced":
      "The Pre-Advanced pillars form the public edge of your deeper work. They are written so others can follow, but they also quietly encode the architectures you use for volt systems, Solar Orders, and the Advanced OpenAdonAI material.",
    advanced:
      "The Advanced pillars are primarily for those already resonant with the Initiate, Intermediate, and Pre-Advanced layers. They are written as a Solar grimoire for emissaries, builders, and stewards who are willing to serve under a higher light.",
  };

  // Map tiers to body theme classes
  const tierThemes = {
    initiate: null,
    intermediate: "theme-intermediate",
    "pre-advanced": "theme-pre-advanced",
    advanced: "theme-advanced",
  };

  const bodyEl = document.body;
  let index = 0; // Start at Initiate

  // DOM hooks
  const panels = document.querySelectorAll(".pillars-panel");
  const titleEl = document.getElementById("pillars-main-title");
  const layerLabelEl = document.getElementById("pillars-layer-label");
  const carouselLabelEl = document.getElementById(
    "pillars-carousel-label-text"
  );
  const subtitleEl = document.getElementById("pillars-subtitle");
  const noteEl = document.getElementById("pillars-note");
  const arrowButtons = document.querySelectorAll(".pillars-arrow-btn");
  const dotButtons = document.querySelectorAll(".pillars-dot");

  const dotTooltips = ["Initiate", "Intermediate", "Pre-Advanced", "Advanced"];

  // Ensure dots have native hover tooltips (for new visitors)
  dotButtons.forEach((btn, i) => {
    btn.setAttribute("title", dotTooltips[i]);
  });

  function updateView() {
    const currentTier = tiers[index];

    // Panels
    panels.forEach((panel) => {
      const tier = panel.getAttribute("data-tier");
      panel.classList.toggle("is-active", tier === currentTier);
    });

    // Copy text
    if (titleEl) {
      titleEl.textContent = tierTitles[currentTier] || "Foundations Pillars";
    }
    if (layerLabelEl) {
      layerLabelEl.textContent =
        tierLayerLabels[currentTier] || "Foundations Layer";
    }
    if (carouselLabelEl) {
      carouselLabelEl.textContent =
        tierCarouselLabels[currentTier] || "Foundations · 5 Pillars";
    }
    if (subtitleEl) {
      subtitleEl.textContent =
        tierSubtitles[currentTier] || subtitleEl.textContent;
    }
    if (noteEl) {
      noteEl.textContent = tierNotes[currentTier] || noteEl.textContent;
    }

    // Theme ramp: remove all theme classes, apply current
    if (bodyEl) {
      bodyEl.classList.remove(
        "theme-intermediate",
        "theme-pre-advanced",
        "theme-advanced"
      );

      const themeClass = tierThemes[currentTier];
      if (themeClass) {
        bodyEl.classList.add(themeClass);
      }
    }

    // Dot indicators
    dotButtons.forEach((btn, i) => {
      const isActive = i === index;
      btn.classList.toggle("is-active", isActive);
      btn.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
  }

  // Arrow controls
  arrowButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const dir = btn.getAttribute("data-dir");
      if (dir === "next") {
        index = (index + 1) % tiers.length;
      } else {
        index = (index - 1 + tiers.length) % tiers.length;
      }
      updateView();
    });
  });

  // Dot click → jump directly
  dotButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const dotIndex = parseInt(btn.getAttribute("data-index"), 10);
      if (!Number.isNaN(dotIndex)) {
        index = dotIndex;
        updateView();
      }
    });
  });

  // Listen for tierchange events from nav (pillars.nav.js)
  window.addEventListener("tierchange", (e) => {
    const tier = e.detail.tier;
    const tierIndex = tiers.indexOf(tier);
    if (tierIndex !== -1) {
      index = tierIndex;
      updateView();
    }
  });

  // Initial state
  updateView();
})();