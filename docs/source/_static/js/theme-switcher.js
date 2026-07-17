/* Toggle directly between light and dark instead of using the system setting. */
document.addEventListener("DOMContentLoaded", () => {
  const applyTheme = (mode) => {
    document.documentElement.dataset.mode = mode;
    document.documentElement.dataset.theme = mode;
    localStorage.setItem("mode", mode);
    localStorage.setItem("theme", mode);

    document.querySelectorAll(".dropdown-menu").forEach((menu) => {
      menu.classList.toggle("dropdown-menu-dark", mode === "dark");
    });
  };

  if (document.documentElement.dataset.mode === "auto") {
    applyTheme(document.documentElement.dataset.theme);
  }

  document.querySelectorAll(".theme-switch-button").forEach((button) => {
    button.addEventListener(
      "click",
      (event) => {
        event.stopImmediatePropagation();
        applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
      },
      true,
    );
  });
});
