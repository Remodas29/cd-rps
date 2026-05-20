document.addEventListener("DOMContentLoaded", function () {
    console.log("CD-RPS Frontend Loaded");

    const forms = document.querySelectorAll("form");

    forms.forEach(form => {
        form.addEventListener("submit", function () {
            console.log("Form Submitted");
        });
    });
});
