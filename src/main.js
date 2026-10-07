import './style.css';
import { renderMesh3DMirror, symmetryData } from './mesh_mirror.js';

const $ = id => document.getElementById(id);

// MediaPipe Landmark Index definitions
// Central facial midline (forehead to chin)
const MIDLINE_INDICES = [
  10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152
];

// Outer boundary contour: Person's Right side (from chin 152 up to forehead 10)
const RIGHT_CONTOUR_UP = [
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
];

// Outer boundary contour: Person's Left side (from chin 152 up to forehead 10)
const LEFT_CONTOUR_UP = [
  152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10
];

// Eye landmark groups
const RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const LEFT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];

const state = {
  camera: false,
  tracking: false,
  mirrorEnabled: true,
  mirrorMode: 'mesh3d', // 'mesh3d' handles 3D head rotation; 'sagittal2d' is planar reflection
  sourceSide: 'left', // 'left' = person's left is healthy source; 'right' = person's right is healthy source
  strength: 1,
  feather: 20,
  preserveEye: false, // Default false so mirror movement & blink is immediately visible
  showMesh: false,
  landmarks: null,
};

let landmarker;
let stream;
let rafId = 0;
let lastVideoTime = -1;
let lastDraw = performance.now();
let frameCount = 0;

// Offscreen canvases for multi-pass composition
let frameCanvas;
let frameCtx;
let mirrorLayerCanvas;
let mirrorLayerCtx;
let maskCanvas;
let maskCtx;

const video = $('camera-video');
const canvas = $('mirror-canvas');
const ctx = canvas.getContext('2d', { alpha: false });
const debugCanvas = $('debug-canvas');
const debugCtx = debugCanvas.getContext('2d');

function setStatus(message, kind = '') {
  $('camera-status').textContent = message;
  $('camera-status').className = `camera-status ${kind}`;
  $('load-state').textContent = message;
}

function setPanels(enabled) {
  for (const id of ['mirror-panel', 'preserve-panel']) {
    const el = $(id);
    if (el) el.disabled = !enabled;
  }
}

function resizeCanvases() {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  const maxWidth = 1280;
  const scale = Math.min(1, maxWidth / width);
  const w = Math.round(width * scale);
  const h = Math.round(height * scale);

  for (const target of [canvas, debugCanvas]) {
    target.width = w;
    target.height = h;
  }

  if (!frameCanvas) frameCanvas = document.createElement('canvas');
  frameCanvas.width = w;
  frameCanvas.height = h;
  frameCtx = frameCanvas.getContext('2d', { alpha: false });

  if (!mirrorLayerCanvas) mirrorLayerCanvas = document.createElement('canvas');
  mirrorLayerCanvas.width = w;
  mirrorLayerCanvas.height = h;
  mirrorLayerCtx = mirrorLayerCanvas.getContext('2d');

  if (!maskCanvas) maskCanvas = document.createElement('canvas');
  maskCanvas.width = w;
  maskCanvas.height = h;
  maskCtx = maskCanvas.getContext('2d');
}

/**
 * Returns canvas coordinate for landmark index.
 * In selfie mirror mode, x is flipped so (1 - p.x) aligns with frameCanvas.
 */
function point(landmarks, index) {
  const p = landmarks?.[index];
  return p ? { x: (1 - p.x) * canvas.width, y: p.y * canvas.height, z: p.z || 0 } : null;
}

/**
 * Compute the reflection matrix across line (p1 -> p2).
 * For any point (x, y), reflection across line with unit normal (nx, ny):
 * p' = p - 2 * ((p - p1) . n) * n
 */
function getReflectionMatrix(p1, p2) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const len = Math.hypot(dx, dy);
  if (len < 1e-4) return null;

  const nx = -dy / len;
  const ny = dx / len;

  const a = 1 - 2 * nx * nx;
  const b = -2 * nx * ny;
  const c = -2 * nx * ny;
  const d = 1 - 2 * ny * ny;

  const k = nx * p1.x + ny * p1.y;
  const e = 2 * nx * k;
  const f = 2 * ny * k;

  return { a, b, c, d, e, f, nx, ny, p1, p2, len };
}

/**
 * Render the mirrored hemiface onto the target side.
 */
function applyHemifaceMirror(landmarks) {
  const pTop = point(landmarks, 10);
  const pBot = point(landmarks, 152);
  if (!pTop || !pBot) return false;

  const M = getReflectionMatrix(pTop, pBot);
  if (!M) return false;

  const w = canvas.width;
  const h = canvas.height;

  // In selfie view:
  // Person's Left is on viewer's right (x > midline)
  // Person's Right is on viewer's left (x < midline)
  const isHealthyLeft = state.sourceSide === 'left';
  // Target is the affected hemiface to be replaced
  const targetIndices = isHealthyLeft
    ? [...MIDLINE_INDICES, ...RIGHT_CONTOUR_UP] // Target is Person's Right
    : [...MIDLINE_INDICES, ...LEFT_CONTOUR_UP];  // Target is Person's Left

  const targetPoints = targetIndices.map(idx => point(landmarks, idx)).filter(Boolean);
  if (targetPoints.length < 10) return false;

  // 1. Draw reflected video frame onto mirrorLayerCanvas
  mirrorLayerCtx.clearRect(0, 0, w, h);
  mirrorLayerCtx.save();
  mirrorLayerCtx.setTransform(M.a, M.b, M.c, M.d, M.e, M.f);
  mirrorLayerCtx.drawImage(frameCanvas, 0, 0);
  mirrorLayerCtx.restore();

  // 2. Build soft alpha mask of the target hemiface
  maskCtx.clearRect(0, 0, w, h);
  maskCtx.save();
  if (state.feather > 0) {
    maskCtx.filter = `blur(${state.feather}px)`;
  }
  maskCtx.beginPath();
  targetPoints.forEach((p, i) => {
    if (i === 0) maskCtx.moveTo(p.x, p.y);
    else maskCtx.lineTo(p.x, p.y);
  });
  maskCtx.closePath();
  maskCtx.fillStyle = '#ffffff';
  maskCtx.fill();
  maskCtx.restore();

  // 3. Composite mask into mirrorLayerCanvas: keep only pixels inside target hemiface
  mirrorLayerCtx.save();
  mirrorLayerCtx.globalCompositeOperation = 'destination-in';
  mirrorLayerCtx.drawImage(maskCanvas, 0, 0);
  mirrorLayerCtx.restore();

  // 4. Draw mirrored layer onto the main canvas with adjustable strength
  ctx.save();
  ctx.globalAlpha = state.strength;
  ctx.drawImage(mirrorLayerCanvas, 0, 0);
  ctx.restore();

  // 5. If preserveEye is enabled, restore the original eye from frameCanvas
  if (state.preserveEye) {
    restoreEyeRegion(landmarks, isHealthyLeft ? RIGHT_EYE_INDICES : LEFT_EYE_INDICES);
  }

  return true;
}

function restoreEyeRegion(landmarks, eyeIndices) {
  const eyePoints = eyeIndices.map(idx => point(landmarks, idx)).filter(Boolean);
  if (eyePoints.length < 4) return;

  const minX = Math.min(...eyePoints.map(p => p.x));
  const maxX = Math.max(...eyePoints.map(p => p.x));
  const minY = Math.min(...eyePoints.map(p => p.y));
  const maxY = Math.max(...eyePoints.map(p => p.y));

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const rx = Math.max((maxX - minX) * 0.65, 12);
  const ry = Math.max((maxY - minY) * 0.75, 8);

  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.clip();
  ctx.drawImage(frameCanvas, 0, 0);
  ctx.restore();
}

function drawDebugOverlay(landmarks) {
  debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  if (!state.showMesh || !landmarks) return;

  const isHealthyLeft = state.sourceSide === 'left';

  // 1. In 3D mesh mode, draw the full anatomical wireframe
  if (state.mirrorMode === 'mesh3d') {
    debugCtx.save();
    debugCtx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
    debugCtx.lineWidth = 0.7;
    const tris = isHealthyLeft ? symmetryData.right_triangles : symmetryData.left_triangles;
    for (let i = 0; i < tris.length; i++) {
      const [v0, v1, v2] = tris[i];
      const p0 = point(landmarks, v0);
      const p1 = point(landmarks, v1);
      const p2 = point(landmarks, v2);
      if (p0 && p1 && p2) {
        debugCtx.beginPath();
        debugCtx.moveTo(p0.x, p0.y);
        debugCtx.lineTo(p1.x, p1.y);
        debugCtx.lineTo(p2.x, p2.y);
        debugCtx.closePath();
        debugCtx.stroke();
      }
    }
    debugCtx.restore();
  }

  const pTop = point(landmarks, 10);
  const pBot = point(landmarks, 152);
  if (!pTop || !pBot) return;

  // 2. Draw facial midline (dashed purple)
  debugCtx.save();
  debugCtx.strokeStyle = '#c084fc';
  debugCtx.lineWidth = 2.5;
  debugCtx.setLineDash([6, 6]);
  debugCtx.beginPath();
  const dx = pBot.x - pTop.x;
  const dy = pBot.y - pTop.y;
  debugCtx.moveTo(pTop.x - dx * 0.5, pTop.y - dy * 0.5);
  debugCtx.lineTo(pBot.x + dx * 0.5, pBot.y + dy * 0.5);
  debugCtx.stroke();
  debugCtx.restore();

  // 3. Draw target hemiface boundary
  const targetIndices = isHealthyLeft
    ? [...MIDLINE_INDICES, ...RIGHT_CONTOUR_UP]
    : [...MIDLINE_INDICES, ...LEFT_CONTOUR_UP];
  const targetPoints = targetIndices.map(idx => point(landmarks, idx)).filter(Boolean);

  debugCtx.save();
  debugCtx.strokeStyle = '#34d399';
  debugCtx.lineWidth = 1.5;
  debugCtx.beginPath();
  targetPoints.forEach((p, i) => {
    if (i === 0) debugCtx.moveTo(p.x, p.y);
    else debugCtx.lineTo(p.x, p.y);
  });
  debugCtx.closePath();
  debugCtx.stroke();

  // 4. Draw key landmarks
  debugCtx.fillStyle = '#38bdf8';
  for (const idx of [10, 152, 1, 168, 33, 263, 61, 291, 70, 300]) {
    const p = point(landmarks, idx);
    if (p) {
      debugCtx.beginPath();
      debugCtx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      debugCtx.fill();
    }
  }

  // 5. Direction label overlay
  debugCtx.font = 'bold 12px Inter, sans-serif';
  const sourceLabel = isHealthyLeft ? '← FONTE (Esquerda saudável)' : 'FONTE (Direita saudável) →';
  const targetLabel = isHealthyLeft ? 'ESPELHO 3D (Direita) →' : '← ESPELHO 3D (Esquerda)';

  debugCtx.fillStyle = '#10b981';
  debugCtx.fillText(isHealthyLeft ? targetLabel : sourceLabel, 20, 60);

  debugCtx.fillStyle = '#a855f7';
  debugCtx.fillText(isHealthyLeft ? sourceLabel : targetLabel, debugCanvas.width - 240, 60);

  debugCtx.restore();
}

function render(result) {
  if (!frameCtx) return;

  // Draw incoming video frame with horizontal flip (selfie mirror mode)
  frameCtx.save();
  frameCtx.setTransform(-1, 0, 0, 1, frameCanvas.width, 0);
  frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  frameCtx.restore();

  // Base canvas: start with the real camera frame
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.drawImage(frameCanvas, 0, 0);

  const landmarks = result?.faceLandmarks?.[0];
  state.landmarks = landmarks || null;

  if (landmarks && state.mirrorEnabled) {
    const isHealthyLeft = state.sourceSide === 'left';

    // 3D-Aware Facial Mesh Symmetry: follows head rotation (yaw, pitch, roll)
    const glCanvas = renderMesh3DMirror(landmarks, frameCanvas, isHealthyLeft, state.strength, state.feather, point);
    if (glCanvas) {
      ctx.save();
      ctx.drawImage(glCanvas, 0, 0);
      ctx.restore();
    }
    if (state.preserveEye) {
      restoreEyeRegion(landmarks, isHealthyLeft ? RIGHT_EYE_INDICES : LEFT_EYE_INDICES);
    }

    drawDebugOverlay(landmarks);
    const sideText = state.sourceSide === 'left' ? 'Esquerda → Direita' : 'Direita → Esquerda';
    $('mode-label').textContent = `Malha 3D · ${sideText} · ${state.preserveEye ? 'olho original preservado' : 'olho espelhado'}`;
  } else {
    debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
    $('mode-label').textContent = landmarks ? 'Espelhamento pausado' : 'Procurando rosto na câmera...';
  }
}

const startEpoch = Date.now() - performance.now();
function processFrame(now) {
  if (!state.camera) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const timestampMs = Math.floor(startEpoch + now);
      const result = landmarker.detectForVideo(video, timestampMs);
      state.tracking = Boolean(result.faceLandmarks?.length);
      setStatus(
        state.tracking ? 'Rastreando rosto' : 'Rosto não detectado',
        state.tracking ? 'tracking-ready' : 'tracking-error'
      );
      render(result);
    } catch (error) {
      console.warn('MediaPipe erro:', error);
      state.tracking = false;
      setStatus('Erro no processamento', 'tracking-error');
    }

    frameCount += 1;
    if (now - lastDraw >= 1000) {
      $('fps').textContent = `FPS: ${Math.round((frameCount * 1000) / (now - lastDraw))}`;
      frameCount = 0;
      lastDraw = now;
    }
  }

  rafId = requestAnimationFrame(processFrame);
}

async function createLandmarker() {
  if (landmarker) return landmarker;
  $('loading').hidden = false;

  const { FilesetResolver, FaceLandmarker } = await import(
    /* @vite-ignore */ 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs'
  );

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );

  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  $('loading').hidden = true;
  return landmarker;
}

async function startCamera() {
  $('start-camera').disabled = true;
  $('error').hidden = true;

  try {
    await createLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    resizeCanvases();
    state.camera = true;
    $('error').hidden = true;
    $('start-camera').textContent = 'Desativar webcam';
    $('start-camera').classList.add('active');
    setPanels(true);
    setStatus('Procurando rosto');

    lastVideoTime = -1;
    rafId = requestAnimationFrame(processFrame);
  } catch (error) {
    $('loading').hidden = true;
    $('error').hidden = false;
    $('error-message').textContent = error instanceof Error ? error.message : String(error);
    setStatus('Não foi possível iniciar', 'tracking-error');
  } finally {
    $('start-camera').disabled = false;
  }
}

function stopCamera() {
  cancelAnimationFrame(rafId);
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  video.srcObject = null;
  state.camera = false;
  state.tracking = false;
  setPanels(false);
  $('start-camera').textContent = 'Ativar webcam';
  $('start-camera').classList.remove('active');
  setStatus('Câmera desativada');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
}

// UI Event Listeners with safe checks
$('start-camera')?.addEventListener('click', () => (state.camera ? stopCamera() : startCamera()));

$('mirror-enabled')?.addEventListener('change', event => {
  state.mirrorEnabled = event.target.checked;
});


$('source-side')?.addEventListener('change', event => {
  state.sourceSide = event.target.value;
});

$('strength')?.addEventListener('input', event => {
  state.strength = Number(event.target.value);
  const out = $('strength-value');
  if (out) out.textContent = `${Math.round(state.strength * 100)}%`;
});

$('feather')?.addEventListener('input', event => {
  state.feather = Number(event.target.value);
  const out = $('feather-value');
  if (out) out.textContent = `${state.feather} px`;
});

$('preserve-eye')?.addEventListener('change', event => {
  state.preserveEye = event.target.checked;
});

$('show-mesh')?.addEventListener('change', event => {
  state.showMesh = event.target.checked;
});

$('reset')?.addEventListener('click', () => {
  state.mirrorEnabled = true;
  state.mirrorMode = 'mesh3d';
  state.sourceSide = 'left';
  state.strength = 1;
  state.feather = 16;
  state.preserveEye = false;
  state.showMesh = false;

  const mirrorCb = $('mirror-enabled');
  if (mirrorCb) mirrorCb.checked = true;
  const modeSel = $('mirror-mode');
  if (modeSel) modeSel.value = 'mesh3d';
  const sideSel = $('source-side');
  if (sideSel) sideSel.value = 'left';
  const str = $('strength');
  if (str) str.value = '1';
  const fea = $('feather');
  if (fea) fea.value = '16';
  const eye = $('preserve-eye');
  if (eye) eye.checked = false;
  const mesh = $('show-mesh');
  if (mesh) mesh.checked = false;
  const strVal = $('strength-value');
  if (strVal) strVal.textContent = '100%';
  const feaVal = $('feather-value');
  if (feaVal) feaVal.textContent = '16 px';
});

$('retry')?.addEventListener('click', () => {
  const err = $('error');
  if (err) err.hidden = true;
  startCamera();
});

window.addEventListener('pagehide', stopCamera);
