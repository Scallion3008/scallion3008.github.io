// Make MathJax highlight math inside single dollar signs
window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] } };

// Code block syntax highlighting
hljs.highlightAll();

// Open external links in new tab
for (const anchor of document.getElementsByTagName("a")) {
    if (new URL(anchor.href).host !== window.location.host) {
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
    }
}

// Update current year in copyright notice
document.getElementById("current-year").textContent = new Date().getFullYear();

// Panel toggles
for (const toggle of document.getElementsByClassName("panel-toggle")) {
    const [form, panelDiv] = toggle.children;

    const showHidePanels = (radioButton) => {
        const selectedPanelIdx = parseInt(radioButton.value);
        [...panelDiv.children].forEach((panel, i) => {
            panel.style.display = (i === selectedPanelIdx) ? "initial" : "none";
        });
    };

    const radioButtons = form.querySelectorAll("input[type='radio']");
    for (const radioButton of radioButtons) {
        radioButton.addEventListener("change", () => showHidePanels(radioButton));
    }
    showHidePanels(radioButtons[form.panel.value])
}
