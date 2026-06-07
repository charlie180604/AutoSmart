// DONDE CAMBIAR LA CLAVE: edita este valor para cambiar el codigo secreto.
const CLAVE_SECRETA = "0906";

// DONDE AGREGAR CANCIONES: agrega objetos con nombre y archivo en assets/music/.
const CANCIONES = [
  { nombre: "Cancion 1", archivo: "assets/music/cancion1.mp3" },
  { nombre: "Cancion 2", archivo: "assets/music/cancion2.mp3" },
  { nombre: "Cancion 3", archivo: "assets/music/cancion3.mp3" },
  { nombre: "Cancion 4", archivo: "assets/music/cancion4.mp3" },
  { nombre: "Cancion 5", archivo: "assets/music/cancion5.mp3" },
  { nombre: "Cancion 6", archivo: "assets/music/cancion6.mp3" }
];

const sections = Array.from(document.querySelectorAll("[data-section]"));
const navLinks = Array.from(document.querySelectorAll("[data-section-link]"));
const menuToggle = document.getElementById("menuToggle");
const mainNav = document.getElementById("mainNav");
const siteHeader = document.getElementById("siteHeader");
const mainContent = document.getElementById("mainContent");
const lockGate = document.getElementById("lockGate");
const particles = document.getElementById("particles");

const accessCard = document.getElementById("accessCard");
const statusMessage = document.getElementById("statusMessage");
const dots = Array.from(document.querySelectorAll("#passcodeDots span"));
const keypad = document.querySelector(".keypad");
const letterText = document.getElementById("letterText");
const closeButton = document.getElementById("closeButton");

const musicAudio = document.getElementById("musicAudio");
const songSelect = document.getElementById("songSelect");
const currentSongName = document.getElementById("currentSongName");
const playPauseSong = document.getElementById("playPauseSong");
const prevSong = document.getElementById("prevSong");
const nextSong = document.getElementById("nextSong");
const playerStatus = document.getElementById("playerStatus");

let typedCode = "";
let isUnlocked = false;
let currentSongIndex = 0;

// DONDE EDITAR CADA SECCION DEL NAVBAR: cambia el HTML de cada <section data-section="..."> en index.html.
function showSection(sectionId) {
  sections.forEach((section) => {
    const isCurrent = section.dataset.section === sectionId;
    section.classList.toggle("is-active", isCurrent);
    section.hidden = !isCurrent;
  });

  navLinks.forEach((link) => {
    link.classList.toggle("is-active", link.dataset.sectionLink === sectionId);
  });

  mainNav.classList.remove("is-open");
  menuToggle.setAttribute("aria-expanded", "false");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function updateDots() {
  dots.forEach((dot, index) => {
    dot.classList.toggle("is-filled", index < typedCode.length);
  });
}

function addNumber(number) {
  if (isUnlocked || typedCode.length >= CLAVE_SECRETA.length) return;
  typedCode += number;
  statusMessage.textContent = "";
  statusMessage.classList.remove("is-success");
  updateDots();
}

function deleteLastNumber() {
  if (isUnlocked) return;
  typedCode = typedCode.slice(0, -1);
  statusMessage.textContent = "";
  updateDots();
}

function validateCode() {
  if (isUnlocked) return;

  if (typedCode === CLAVE_SECRETA) {
    isUnlocked = true;
    statusMessage.textContent = "Acceso concedido...";
    statusMessage.classList.add("is-success");
    lockGate.classList.add("is-unlocking");
    launchUnlockHearts();
    setTimeout(unlockPage, 760);
    return;
  }

  typedCode = "";
  updateDots();
  statusMessage.textContent = "Codigo incorrecto, intenta de nuevo.";
  accessCard.classList.remove("is-shaking");
  void accessCard.offsetWidth;
  accessCard.classList.add("is-shaking");
}

function unlockPage() {
  lockGate.hidden = true;
  lockGate.classList.remove("is-unlocking");
  siteHeader.hidden = false;
  mainContent.hidden = false;
  showSection("carta-secreta");
  revealLetterText();
}

function revealLetterText() {
  const paragraphs = Array.from(letterText.querySelectorAll("p"));
  paragraphs.forEach((paragraph) => {
    paragraph.classList.remove("is-visible");
  });

  paragraphs.forEach((paragraph, index) => {
    setTimeout(() => {
      paragraph.classList.add("is-visible");
    }, index * 360);
  });
}

function closeLetter() {
  isUnlocked = false;
  typedCode = "";
  updateDots();
  statusMessage.textContent = "";
  statusMessage.classList.remove("is-success");
  accessCard.classList.remove("is-shaking");
  musicAudio.pause();
  playPauseSong.textContent = "Play";

  siteHeader.hidden = true;
  mainContent.hidden = true;
  lockGate.hidden = false;
  showSection("carta-secreta");
}

function setupMusicPlayer() {
  CANCIONES.forEach((song, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = song.nombre;
    songSelect.appendChild(option);
  });

  loadSong(0);
}

function loadSong(index) {
  currentSongIndex = (index + CANCIONES.length) % CANCIONES.length;
  const song = CANCIONES[currentSongIndex];
  const wasPlaying = !musicAudio.paused;

  musicAudio.src = song.archivo;
  if (currentSongName) {
    currentSongName.textContent = song.nombre;
  }
  songSelect.value = String(currentSongIndex);
  if (playerStatus) {
    playerStatus.textContent = "";
  }
  playPauseSong.textContent = "Play";

  if (wasPlaying) {
    playCurrentSong();
  }
}

function playCurrentSong() {
  musicAudio.play()
    .then(() => {
      playPauseSong.textContent = "Pausar";
      if (playerStatus) {
        playerStatus.textContent = "";
      }
    })
    .catch(() => {
      if (playerStatus) {
        playerStatus.textContent = "Esta cancion no esta disponible todavia.";
      }
      playPauseSong.textContent = "Play";
    });
}

function toggleCurrentSong() {
  if (musicAudio.paused) {
    playCurrentSong();
  } else {
    musicAudio.pause();
    playPauseSong.textContent = "Play";
  }
}

function nextCurrentSong() {
  loadSong(currentSongIndex + 1);
}

function previousCurrentSong() {
  loadSong(currentSongIndex - 1);
}

function hideBrokenImage(image) {
  image.classList.add("image-hidden");

  const figure = image.closest("figure");
  if (figure && !figure.querySelector("figcaption")) {
    figure.classList.add("image-hidden");
  }
}

function tryAlternativeImage(image) {
  const currentSrc = image.getAttribute("src") || "";
  const alternativeSrc = /\.jpg$/i.test(currentSrc)
    ? currentSrc.replace(/\.jpg$/i, ".jpeg")
    : currentSrc.replace(/\.jpeg$/i, ".jpg");

  if (alternativeSrc !== currentSrc && image.dataset.triedAlternative !== "true") {
    image.dataset.triedAlternative = "true";
    image.src = alternativeSrc;
    return;
  }

  hideBrokenImage(image);
}

// DONDE CAMBIAR IMAGENES: edita los src="assets/img/..." en index.html.
function setupOptionalImages() {
  document.querySelectorAll(".optional-image").forEach((image) => {
    image.dataset.originalSrc = image.getAttribute("src") || "";

    if (image.complete && image.naturalWidth === 0) {
      tryAlternativeImage(image);
    }

    image.addEventListener("error", () => {
      tryAlternativeImage(image);
    });
  });
}

function createFloatingHearts() {
  const heartCount = 28;

  for (let index = 0; index < heartCount; index += 1) {
    const heart = document.createElement("span");
    heart.className = "heart";
    heart.style.left = `${Math.random() * 100}%`;
    heart.style.setProperty("--size", `${Math.random() * 12 + 10}px`);
    heart.style.setProperty("--duration", `${Math.random() * 7 + 9}s`);
    heart.style.setProperty("--drift", `${Math.random() * 120 - 60}px`);
    heart.style.animationDelay = `${Math.random() * -12}s`;
    particles.appendChild(heart);
  }
}

function launchUnlockHearts() {
  for (let index = 0; index < 24; index += 1) {
    const heart = document.createElement("span");
    heart.className = "unlock-heart";
    heart.textContent = "♥";
    heart.style.setProperty("--x", `${Math.random() * 360 - 180}px`);
    heart.style.setProperty("--y", `${Math.random() * 300 - 210}px`);
    heart.style.setProperty("--r", `${Math.random() * 140 - 70}deg`);
    document.body.appendChild(heart);
    setTimeout(() => heart.remove(), 950);
  }
}

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    showSection(link.dataset.sectionLink);
  });
});

menuToggle.addEventListener("click", () => {
  const isOpen = mainNav.classList.toggle("is-open");
  menuToggle.setAttribute("aria-expanded", String(isOpen));
});

keypad.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;

  if (button.dataset.number) addNumber(button.dataset.number);
  if (button.dataset.action === "delete") deleteLastNumber();
  if (button.dataset.action === "accept") validateCode();
});

document.addEventListener("keydown", (event) => {
  if (isUnlocked || lockGate.hidden) return;

  if (/^\d$/.test(event.key)) addNumber(event.key);
  if (event.key === "Backspace") deleteLastNumber();
  if (event.key === "Enter") validateCode();
});

closeButton.addEventListener("click", closeLetter);

songSelect.addEventListener("change", () => {
  loadSong(Number(songSelect.value));
});

playPauseSong.addEventListener("click", toggleCurrentSong);
nextSong.addEventListener("click", nextCurrentSong);
prevSong.addEventListener("click", previousCurrentSong);

musicAudio.addEventListener("ended", nextCurrentSong);
musicAudio.addEventListener("error", () => {
  if (playerStatus) {
    playerStatus.textContent = "Esta cancion no esta disponible todavia.";
  }
  playPauseSong.textContent = "Play";
});

setupMusicPlayer();
setupOptionalImages();
createFloatingHearts();
updateDots();
