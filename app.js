// 👉 Configure aqui o endereço do seu Streamlit.
// Ex.: "http://localhost:8501" ou "http://192.168.1.172:8501".
// Se usar proxy com subcaminho (Nginx), pode ser "/app/".
const STREAMLIT_URL = "http://localhost:8501";

// Elements
const frame = document.getElementById("dashFrame");
const statusBadge = document.getElementById("statusBadge");
const liveText = document.getElementById("liveText");
const liveDot = document.getElementById("liveDot");
const openNew = document.getElementById("openNew");
const openNew2 = document.getElementById("openNew2");
const reloadBtn = document.getElementById("reloadBtn");
const themeToggle = document.getElementById("themeToggle");

// Set links + iframe src
function setTargets(url){
  frame.src = url;
  openNew.href = url;
  openNew2.href = url;
}
setTargets(STREAMLIT_URL);

// Status: marca online quando o iframe carrega
let loadedOnce = false;
frame.addEventListener("load", () => {
  loadedOnce = true;
  statusBadge.textContent = "Online";
  statusBadge.className = "badge ok";
  if(liveText) liveText.textContent = "Servidor ativo";
  if(liveDot) liveDot.style.background = "var(--ok)";
});

// Se em 4s não carregar, exibe aviso
setTimeout(() => {
  if(!loadedOnce){
    statusBadge.textContent = "Indisponível";
    statusBadge.className = "badge off";
    if(liveText) liveText.textContent = "Sem conexão — rode o Streamlit";
    if(liveDot) liveDot.style.background = "var(--error)";
  }
}, 4000);

// Recarregar o iframe (bypass cache com timestamp)
reloadBtn.addEventListener("click", () => {
  statusBadge.textContent = "Conectando…";
  statusBadge.className = "badge warn";
  frame.src = STREAMLIT_URL + (STREAMLIT_URL.includes("?") ? "&" : "?") + "ts=" + Date.now();
});

// Tema claro/escuro
const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
const savedTheme = localStorage.getItem("sst-theme");
if(savedTheme === "light" || (!savedTheme && prefersLight)) document.documentElement.classList.add("light");

themeToggle.addEventListener("click", () => {
  document.documentElement.classList.toggle("light");
  localStorage.setItem("sst-theme", document.documentElement.classList.contains("light") ? "light" : "dark");
});
