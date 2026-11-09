let FilesetResolver = null;
let HandLandmarker = null;

const TASKS_VISION_MODULE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
const TASKS_VISION_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const CONFIDENCE_THRESHOLD = 0.65;
const HISTORY_LENGTH = 10;

const GESTURE_LABELS = {
  FIST: "Fist",
  OPEN_PALM: "Open palm",
  PEACE: "Peace",
  POINTING: "Pointing",
  THREE_FINGERS: "Three fingers",
  THUMBS_UP: "Thumbs up",
  UNKNOWN: "Unknown",
};

// MediaPipe's 21-point hand topology, index pairs.
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

const video = document.querySelector("#video");
const scene = document.querySelector("#scene");
const context = scene.getContext("2d");

const statusElement = document.querySelector("#status");
const statusText = document.querySelector("#statusText");
const stageEmpty = document.querySelector("#stageEmpty");
const startButton = document.querySelector("#startButton");
const startHint = document.querySelector("#startHint");
const liveBadge = document.querySelector("#liveBadge");
const gestureElement = document.querySelector("#gesture");
const confidenceValue = document.querySelector("#confidenceValue");
const confidenceFill = document.querySelector("#confidenceFill");
const historyElement = document.querySelector("#history");
const hintElement = document.querySelector("#hint");
const factTrees = document.querySelector("#factTrees");

let handLandmarker = null;
let gestureModel = null;
let stream = null;
let lastVideoTime = -1;
let lastTimestamp = 0;
let currentClass = null;
let currentConfidence = 0;

const history = [];

function setStatus(message, tone = "idle") {
  statusText.textContent = message;
  statusElement.dataset.tone = tone;
}

function loadGestureModel() {
  return fetch("./model/gesture_model.json").then((response) => {
    if (!response.ok) {
      throw new Error(
        `Could not load the gesture model (HTTP ${response.status}). ` +
          "Run `uv run export_model.py` first.",
      );
    }
    return response.json();
  });
}

async function loadMediaPipe() {
  const module = await import(TASKS_VISION_MODULE);
  FilesetResolver = module.FilesetResolver;
  HandLandmarker = module.HandLandmarker;
}

async function createHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(TASKS_VISION_URL);
  const options = {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.7,
    minTrackingConfidence: 0.7,
  };

  try {
    return await HandLandmarker.createFromOptions(vision, options);
  } catch (error) {
    // Some browsers/drivers cannot create a GPU delegate; CPU works everywhere.
    console.warn("GPU delegate unavailable, falling back to CPU.", error);
    options.baseOptions.delegate = "CPU";
    return HandLandmarker.createFromOptions(vision, options);
  }
}

async function initialise() {
  try {
    setStatus("Loading model…", "idle");
    const [model] = await Promise.all([loadGestureModel(), loadMediaPipe()]);

    gestureModel = model;
    setStatus("Starting tracker…", "idle");
    handLandmarker = await createHandLandmarker();
    factTrees.textContent = String(model.n_estimators);

    startButton.disabled = false;
    startButton.textContent = "Enable camera";
    startHint.textContent = "Your browser will ask for camera permission.";
    setStatus("Ready", "ready");
  } catch (error) {
    console.error(error);
    setStatus("Model failed to load", "error");
    startButton.disabled = false;
    startButton.textContent = "Retry";
    startHint.textContent = FilesetResolver
      ? error.message
      : "Could not load MediaPipe from the CDN. Check your internet connection.";
    hintElement.textContent =
      "The model could not be loaded. Serve this folder (for example with " +
      "`uv run python -m http.server 8000`) and open it via localhost.";
  }
}

async function startCamera() {
  if (!handLandmarker || !gestureModel) {
    return;
  }

  startButton.disabled = true;
  startButton.textContent = "Starting…";
  setStatus("Requesting camera…", "idle");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Camera access requires a secure context. Open this page through " +
          "http://localhost or an HTTPS URL.",
      );
    }

    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    video.srcObject = stream;
    await video.play();

    scene.width = video.videoWidth || 1280;
    scene.height = video.videoHeight || 720;
    scene.classList.remove("is-idle");

    stageEmpty.hidden = true;
    liveBadge.hidden = false;
    startButton.textContent = "Camera on";
    lastVideoTime = -1;

    setStatus("Starting tracker…", "ready");
    hintElement.textContent =
      "Hold a gesture steady. The result is smoothed over the last 10 frames.";

    requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    setStatus("Camera unavailable", "error");
    startButton.disabled = false;
    startButton.textContent = "Try again";
    startHint.textContent = error.message;
    hintElement.textContent =
      "Camera access needs permission and a localhost/HTTPS page.";
  }
}

function loop() {
  if (!stream) {
    return;
  }

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    drawMirroredFrame();

    const now = performance.now();
    const timestamp = now <= lastTimestamp ? lastTimestamp + 1 : now;
    lastTimestamp = timestamp;

    try {
      const result = handLandmarker.detectForVideo(scene, timestamp);
      handleDetection(result);
    } catch (error) {
      console.error("Hand tracking failed for this frame.", error);
    }
  }

  requestAnimationFrame(loop);
}

function drawMirroredFrame() {
  context.save();
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, scene.width, scene.height);
  context.translate(scene.width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, scene.width, scene.height);
  context.restore();
}

function handleDetection(result) {
  const landmarks = result?.landmarks?.[0];

  if (!landmarks) {
    history.length = 0;
    currentClass = null;
    currentConfidence = 0;
    renderReadout("NO_HAND");
    setStatus("No hand detected", "warning");
    return;
  }

  drawLandmarks(landmarks);

  const features = extractFeatures(landmarks);
  if (!features) {
    history.length = 0;
    currentClass = null;
    currentConfidence = 0;
    renderReadout("NO_HAND");
    return;
  }

  const prediction = classify(features);
  const confidence = Math.max(...prediction.probabilities);

  history.push(
    confidence >= CONFIDENCE_THRESHOLD ? prediction.className : "UNKNOWN",
  );
  if (history.length > HISTORY_LENGTH) {
    history.shift();
  }

  currentClass = mostCommon(history);
  currentConfidence = confidence;

  renderReadout(currentClass);
  setStatus("Tracking hand", "tracking");
}

// Same normalisation as collect_data.py / gesture_recognition.py:
// 1. subtract the wrist, 2. scale by the largest xy distance from the wrist.
function extractFeatures(landmarks) {
  const wrist = landmarks[0];
  const points = landmarks.map((point) => [
    point.x - wrist.x,
    point.y - wrist.y,
    point.z - wrist.z,
  ]);

  let scale = 0;
  for (const [x, y] of points) {
    scale = Math.max(scale, Math.hypot(x, y));
  }

  if (scale < 1e-6) {
    return null;
  }

  const features = new Float64Array(63);
  let index = 0;
  for (const [x, y, z] of points) {
    features[index] = x / scale;
    features[index + 1] = y / scale;
    features[index + 2] = z / scale;
    index += 3;
  }

  return features;
}

// Mirrors scikit-learn's RandomForestClassifier.predict_proba: every tree
// votes with its normalised leaf class distribution, then the votes are
// averaged and the best class is returned.
function classify(features) {
  const sums = new Float64Array(gestureModel.classes.length);

  for (const tree of gestureModel.trees) {
    let node = 0;
    while (tree.l[node] !== -1) {
      node =
        features[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    }
    const probabilities = tree.p[node];
    for (let index = 0; index < probabilities.length; index += 1) {
      sums[index] += probabilities[index];
    }
  }

  let best = 0;
  for (let index = 1; index < sums.length; index += 1) {
    if (sums[index] > sums[best]) {
      best = index;
    }
  }

  const probabilities = Array.from(
    sums,
    (value) => value / gestureModel.trees.length,
  );

  return { className: gestureModel.classes[best], probabilities };
}

function mostCommon(values) {
  const counts = new Map();
  let winner = values[0] ?? "UNKNOWN";
  let winnerCount = 0;

  for (const value of values) {
    const count = (counts.get(value) ?? 0) + 1;
    counts.set(value, count);
    if (count > winnerCount) {
      winner = value;
      winnerCount = count;
    }
  }

  return winner;
}

function renderReadout(state) {
  const isNoHand = state === "NO_HAND";
  const isUnknown = state === "UNKNOWN";

  gestureElement.textContent = isNoHand
    ? "No hand"
    : isUnknown
      ? "Unknown"
      : formatGesture(state);
  gestureElement.dataset.tone = isNoHand ? "muted" : isUnknown ? "unknown" : "";

  const percent = Math.max(0, Math.min(100, currentConfidence * 100));
  confidenceValue.textContent = `${percent.toFixed(1)}%`;
  confidenceFill.style.width = `${percent}%`;
  confidenceFill.style.background = isUnknown
    ? "linear-gradient(90deg, #f59e0b, #fbbf24)"
    : "linear-gradient(90deg, var(--accent-strong), var(--accent))";

  renderHistory();
}

function renderHistory() {
  historyElement.replaceChildren();

  for (const item of history) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent =
      item === "UNKNOWN" ? "Unknown" : formatGesture(item);
    chip.dataset.kind =
      item === "UNKNOWN"
        ? "unknown"
        : item === currentClass
          ? "match"
          : "neutral";
    historyElement.append(chip);
  }
}

function formatGesture(label) {
  return GESTURE_LABELS[label] ?? label.replaceAll("_", " ");
}

function drawLandmarks(landmarks) {
  const width = scene.width;
  const height = scene.height;
  const lineWidth = Math.max(2, Math.min(width, height) / 320);
  const pointRadius = Math.max(3.5, Math.min(width, height) / 180);

  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.lineWidth = lineWidth;
  context.strokeStyle = "rgba(94, 234, 212, 0.9)";
  context.beginPath();

  for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    context.moveTo(start.x * width, start.y * height);
    context.lineTo(end.x * width, end.y * height);
  }

  context.stroke();

  for (let index = 0; index < landmarks.length; index += 1) {
    const point = landmarks[index];
    const x = point.x * width;
    const y = point.y * height;
    const isTip = [4, 8, 12, 16, 20].includes(index);

    context.beginPath();
    context.arc(x, y, pointRadius, 0, Math.PI * 2);
    context.fillStyle = isTip ? "#5eead4" : "rgba(8, 15, 24, 0.92)";
    context.fill();
    context.lineWidth = lineWidth;
    context.strokeStyle = isTip ? "#99f6e4" : "rgba(94, 234, 212, 0.9)";
    context.stroke();
  }

  context.restore();
}

startButton.addEventListener("click", () => {
  if (!gestureModel || !handLandmarker) {
    initialise();
    return;
  }
  if (stream) {
    // The camera is already running; the button becomes a no-op after success.
    return;
  }
  startCamera();
});

window.addEventListener("pagehide", () => {
  stream?.getTracks().forEach((track) => track.stop());
});

initialise();
