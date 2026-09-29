// 3D-Aware Facial Mesh Symmetry Renderer
// Uses WebGL to render 449 anatomical facial triangles in real-time.
// Features:
// 1. Premultiplied alpha rendering (zero dark borders / black fringes).
// 2. Soft midline feathering (prevents split/deformed nose when rotating head).
// 3. Boundary ring feathering (eliminates stretched hair/headphone black polygons).

import symmetryData from './face_symmetry_mesh.json';

const { partner, left_triangles, right_triangles, vertex_alpha } = symmetryData;

// Flattened index buffers
const rightIndices = new Uint16Array(right_triangles.flat());
const leftIndices = new Uint16Array(left_triangles.flat());

const defaultVertexAlphas = new Float32Array(vertex_alpha || new Array(468).fill(1.0));

let glCanvas = null;
let gl = null;
let program = null;
let positionBuffer = null;
let pixelPosBuffer = null;
let texCoordBuffer = null;
let alphaBuffer = null;
let indexBuffer = null;
let texture = null;

// Reusable vertex arrays
const positions = new Float32Array(468 * 2);
const pixelPositions = new Float32Array(468 * 2);
const texCoords = new Float32Array(468 * 2);
const alphas = new Float32Array(468);

const VS_SOURCE = `
attribute vec2 a_position;
attribute vec2 a_pixelPos;
attribute vec2 a_texCoord;
attribute float a_alpha;

varying vec2 v_texCoord;
varying vec2 v_pixelPos;
varying float v_alpha;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
  v_pixelPos = a_pixelPos;
  v_alpha = a_alpha;
}
`;

const FS_SOURCE = `
precision mediump float;
uniform sampler2D u_image;
uniform float u_strength;
uniform vec2 u_lineP1;
uniform vec2 u_lineP2;
uniform float u_midlineFeather;

varying vec2 v_texCoord;
varying vec2 v_pixelPos;
varying float v_alpha;

void main() {
  // Distance from facial midline line (P1 -> P2)
  vec2 d = u_lineP2 - u_lineP1;
  float len = length(d);
  float dist = 100.0;
  if (len > 0.001) {
    vec2 n = vec2(-d.y, d.x) / len;
    dist = abs(dot(v_pixelPos - u_lineP1, n));
  }

  // Smooth fade across the midline to prevent sharp nose/chin seams
  float midlineFade = smoothstep(0.0, max(u_midlineFeather, 1.0), dist);
  float totalAlpha = v_alpha * midlineFade * u_strength;

  vec4 color = texture2D(u_image, v_texCoord);

  // Output PREMULTIPLIED alpha so Canvas 2D composite produces ZERO black borders:
  gl_FragColor = vec4(color.rgb * totalAlpha, totalAlpha);
}
`;

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error('Shader compile error: ' + info);
  }
  return shader;
}

function initWebGL(width, height) {
  if (!glCanvas) {
    glCanvas = document.createElement('canvas');
  }
  glCanvas.width = width;
  glCanvas.height = height;

  // CRITICAL: premultipliedAlpha must be true so Canvas 2D drawImage does not darken the edges!
  gl = glCanvas.getContext('webgl', { alpha: true, premultipliedAlpha: true });
  if (!gl) return null;

  const vs = createShader(gl, gl.VERTEX_SHADER, VS_SOURCE);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, FS_SOURCE);
  program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error('Program link error: ' + gl.getProgramInfoLog(program));
  }

  gl.useProgram(program);

  positionBuffer = gl.createBuffer();
  pixelPosBuffer = gl.createBuffer();
  texCoordBuffer = gl.createBuffer();
  alphaBuffer = gl.createBuffer();
  indexBuffer = gl.createBuffer();

  texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  gl.enable(gl.BLEND);
  // Blending for premultiplied alpha:
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  return gl;
}

/**
 * Render 3D-aware mirrored hemiface.
 * @param {Array} landmarks - 468 MediaPipe normalized landmarks
 * @param {HTMLCanvasElement} frameCanvas - Unmodified video canvas in selfie view
 * @param {boolean} isHealthyLeft - true if person's left is healthy (viewer right)
 * @param {number} strength - 0.0 to 1.0 opacity
 * @param {number} feather - Midline feather radius in pixels
 * @param {Function} pointFn - function(landmarks, index) returning {x, y} in canvas space
 * @returns {HTMLCanvasElement|null} - rendered WebGL canvas
 */
export function renderMesh3DMirror(landmarks, frameCanvas, isHealthyLeft, strength, feather, pointFn) {
  const width = frameCanvas.width;
  const height = frameCanvas.height;

  if (!gl || glCanvas.width !== width || glCanvas.height !== height) {
    if (!initWebGL(width, height)) return null;
  }

  gl.viewport(0, 0, width, height);
  // Clear to transparent
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(program);

  // Compute position, pixel coords, UV, and boundary alpha for all 468 vertices
  for (let i = 0; i < 468; i++) {
    const dstPoint = pointFn(landmarks, i);
    const partnerId = partner[i];
    const srcPoint = pointFn(landmarks, partnerId);

    if (dstPoint && srcPoint) {
      // Screen clip space: [-1, 1]
      positions[i * 2] = (dstPoint.x / width) * 2 - 1;
      positions[i * 2 + 1] = 1 - (dstPoint.y / height) * 2;

      // Pixel positions [0..width, 0..height]
      pixelPositions[i * 2] = dstPoint.x;
      pixelPositions[i * 2 + 1] = dstPoint.y;

      // Texture UV space: [0, 1]
      texCoords[i * 2] = srcPoint.x / width;
      texCoords[i * 2 + 1] = srcPoint.y / height;

      // Boundary fade alpha (0.0 on face oval, 0.5 on inner border, 1.0 inside)
      alphas[i] = defaultVertexAlphas[i];
    } else {
      positions[i * 2] = 0;
      positions[i * 2 + 1] = 0;
      pixelPositions[i * 2] = 0;
      pixelPositions[i * 2 + 1] = 0;
      texCoords[i * 2] = 0;
      texCoords[i * 2 + 1] = 0;
      alphas[i] = 0;
    }
  }

  // Upload clip positions
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
  const aPos = gl.getAttribLocation(program, 'a_position');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  // Upload pixel positions
  gl.bindBuffer(gl.ARRAY_BUFFER, pixelPosBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, pixelPositions, gl.DYNAMIC_DRAW);
  const aPixelPos = gl.getAttribLocation(program, 'a_pixelPos');
  gl.enableVertexAttribArray(aPixelPos);
  gl.vertexAttribPointer(aPixelPos, 2, gl.FLOAT, false, 0, 0);

  // Upload UVs
  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.DYNAMIC_DRAW);
  const aTex = gl.getAttribLocation(program, 'a_texCoord');
  gl.enableVertexAttribArray(aTex);
  gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 0, 0);

  // Upload Alphas
  gl.bindBuffer(gl.ARRAY_BUFFER, alphaBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, alphas, gl.DYNAMIC_DRAW);
  const aAlpha = gl.getAttribLocation(program, 'a_alpha');
  gl.enableVertexAttribArray(aAlpha);
  gl.vertexAttribPointer(aAlpha, 1, gl.FLOAT, false, 0, 0);

  // Upload frame texture
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frameCanvas);

  // Midline line coordinates (forehead point 10 -> chin point 152)
  const pTop = pointFn(landmarks, 10) || { x: width * 0.5, y: 0 };
  const pBot = pointFn(landmarks, 152) || { x: width * 0.5, y: height };

  const uP1 = gl.getUniformLocation(program, 'u_lineP1');
  gl.uniform2f(uP1, pTop.x, pTop.y);

  const uP2 = gl.getUniformLocation(program, 'u_lineP2');
  gl.uniform2f(uP2, pBot.x, pBot.y);

  const uMidFeather = gl.getUniformLocation(program, 'u_midlineFeather');
  gl.uniform1f(uMidFeather, Math.max(feather, 12));

  const uStrength = gl.getUniformLocation(program, 'u_strength');
  gl.uniform1f(uStrength, strength);

  // Choose triangles for target hemiface
  const indices = isHealthyLeft ? rightIndices : leftIndices;

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);

  // Single GPU draw call for all 449 triangles
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);

  return glCanvas;
}

export { symmetryData };
