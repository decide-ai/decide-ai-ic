import { model } from "declarations/model";

const tokensSlider = document.getElementById("tokens");
const tempSlider = document.getElementById("temp");
const tokensValue = document.getElementById("tokens-value");
const tempValue = document.getElementById("temp-value");

// Update the display when sliders change
tokensSlider.addEventListener("input", (e) => {
  tokensValue.textContent = e.target.value;
});

tempSlider.addEventListener("input", (e) => {
  tempValue.textContent = e.target.value;
});

document.querySelector("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = e.target.querySelector("button");
  const prompt = document.getElementById("prompt").value.toString();
  const tokens = parseInt(tokensSlider.value);
  const temperature = parseFloat(tempSlider.value);

  button.setAttribute("disabled", true);
  try {
    const result = await model.generate(prompt, tokens, temperature);
    document.getElementById("response").innerText = result.Ok || `Error: ${result.Err}`;
  } catch (error) {
    document.getElementById("response").innerText = `Error: ${error.message}`;
  } finally {
    button.removeAttribute("disabled");
  }
});