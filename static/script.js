document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const resultBox = document.getElementById("result");
  const errorBox = document.getElementById("error");

  const fields = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
  ];

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Reset UI
    resultBox.classList.add("hidden");
    errorBox.classList.add("hidden");
    resultBox.innerHTML = "";
    errorBox.innerHTML = "";

    // Build payload from form values
    const payload = {};
    for (const field of fields) {
      const el = document.getElementById(field);
      const value = el.value;
      if (value === "") {
        errorBox.textContent = `Please fill in the "${field}" field.`;
        errorBox.classList.remove("hidden");
        return;
      }
      payload[field] = Number(value);
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Predicting...";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Something went wrong while predicting.");
      }

      const isPositive = data.prediction.toLowerCase().includes("detected") &&
                          !data.prediction.toLowerCase().includes("no heart");

      resultBox.classList.remove("hidden");
      resultBox.classList.toggle("positive", isPositive);
      resultBox.classList.toggle("negative", !isPositive);

      const probabilityPct = (data.probability * 100).toFixed(1);
      resultBox.innerHTML = `
        <strong>${data.prediction}</strong>
        Model confidence: ${probabilityPct}% probability of heart disease
      `;
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.classList.remove("hidden");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Predict";
    }
  });
});
