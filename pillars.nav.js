/* Nav behavior for Foundations tiers
   - Detects active tier from body `data-tier-active`, URL path, or hash
   - Updates tab active state and visible chapter rows
   - Applies theme classes to <body> (matching `pillars.css` themes)
   - Emits a `tierchange` event on window for other scripts to listen
*/
(function () {
  const tabs = Array.from(document.querySelectorAll(".tier-tab"));
  const rows = Array.from(document.querySelectorAll(".chapters-row"));

  const THEMES = {
    initiate: null,
    intermediate: "theme-intermediate",
    "pre-advanced": "theme-pre-advanced",
    advanced: "theme-advanced",
  };

  function normalizeTier(t) {
    if (!t) return t;
    t = String(t).trim();
    if (t === "pre-advanced" || t === "preadvanced" || t === "pre_adv") return "pre-advanced";
    if (t === "intermediate") return "intermediate";
    if (t === "advanced") return "advanced";
    return "initiate";
  }

  function setBodyTheme(tier) {
    document.body.classList.remove("theme-intermediate", "theme-pre-advanced", "theme-advanced");
    const cls = THEMES[tier];
    if (cls) document.body.classList.add(cls);
  }

  function setActiveTier(tier, pushState = true) {
    tier = normalizeTier(tier) || "initiate";

    tabs.forEach((t) => t.classList.toggle("active", t.getAttribute("data-tier") === tier));
    rows.forEach((r) => r.classList.toggle("active", r.getAttribute("data-tier") === tier));

    document.body.setAttribute("data-tier-active", tier);
    setBodyTheme(tier);

    if (pushState) {
      try {
        history.pushState({ tier }, "", `#tier=${tier}`);
      } catch (e) {
        /* ignore */
      }
    }

    window.dispatchEvent(new CustomEvent("tierchange", { detail: { tier } }));
  }

  function detectTierFromUrl() {
    // Priority: body attr > hash (#tier=...) > path contains segment
    const attr = document.body.getAttribute("data-tier-active");
    if (attr) return normalizeTier(attr);

    const hash = (location.hash || "").replace(/^#/, "");
    if (hash.startsWith("tier=")) return normalizeTier(hash.split("=")[1]);

    const path = (location.pathname || "").toLowerCase();
    if (path.includes("/pre-advanced") || path.includes("/pre_advanced") || path.includes("/preadvanced")) return "pre-advanced";
    if (path.includes("/intermediate")) return "intermediate";
    if (path.includes("/advanced")) return "advanced";
    return "initiate";
  }

  // Tab clicks
  tabs.forEach((tab) => {
    tab.addEventListener("click", (e) => {
      e.preventDefault();
      const tier = tab.getAttribute("data-tier");
      setActiveTier(tier, true);
    });
  });

  // React to back/forward and hash changes
  window.addEventListener("popstate", (e) => {
    const tier = (e.state && e.state.tier) || detectTierFromUrl();
    setActiveTier(tier, false);
  });

  window.addEventListener("hashchange", () => {
    const tier = detectTierFromUrl();
    setActiveTier(tier, false);
  });

  // Initial activation (don't pushState on first paint)
  setActiveTier(detectTierFromUrl(), false);

  // Expose a tiny API for other scripts
  window.FoundationsNav = {
    setActiveTier,
    detectTierFromUrl,
  };
})();
 (function () {
    const tabs = document.querySelectorAll(".tier-tab");
    const rows = document.querySelectorAll(".chapters-row");

    const activeTier = document.body.getAttribute("data-tier-active");

    if (activeTier) {
      tabs.forEach((t) => {
        t.classList.toggle("active", t.getAttribute("data-tier") === activeTier);
      });

      rows.forEach((r) => {
        r.classList.toggle("active", r.getAttribute("data-tier") === activeTier);
      });
    }

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const tier = tab.getAttribute("data-tier");

        tabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");

        rows.forEach((row) => {
          row.classList.toggle("active", row.getAttribute("data-tier") === tier);
        });
      });
    });
  })();