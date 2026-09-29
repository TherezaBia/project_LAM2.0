// 3D-Aware Facial Mesh Symmetry Renderer
// Uses WebGL to render 449 anatomical facial triangles in real-time.
// Fixes:
// 1. Nose solid attachment: Midline vertices are fully opaque and solidly anchored
//    to facial landmarks (no transparency, no double nostrils, no floating/loose mesh).
// 2. Dynamic Occlusion Handling: When turning toward the source side (e.g. looking left),
//    the occluded lateral cheek smoothly fades back to the user's REAL CAMERA SKIN.
// 3. Premultiplied alpha: Clean blending with Canvas 2D without any dark fringe.

import symmetryData from './face_symmetry_mesh.json';

const { partner, left_triangles, right_triangles, vertex_alpha, canonical_vertices } = symmetryData;

// Flattened index buffers
const rightIndices = new Uint16Array(right_triangles.flat());
const leftIndices = new Uint16Array(left_triangles.flat());

const defaultVertexAlphas = new Float32Array(vertex_alpha || new Array(468).fill(1.0));

// Precalculate normalized lateral distance from facial midline for each vertex:
// 0.0 = center (nose, lips), 1.0 = outer edge (ear, temple)
const lateralDistances = new Float32Array(468);
if (canonical_vertices) {
  for (let i = 0; i < 468; i++) {
    lateralDistances[i] = Math.abs(canonical_vertices[i][0]) / 7.7431;
  }
}

let glCanvas = null;
let gl = null;
let program = null;
let positionBuffer = null;
let texCoordBuffer = null;
let alphaBuffer = null;
let indexBuffer = null;
let texture = null;

// Reusable vertex arrays
const positions = new Float32Array(468 * 2);
const texCoords = new Float32Array(468 * 2);
const alphas = new Float32Array(468);

const VS_SOURCE = `
attribute vec2 a_position;
attribute vec2 a_texCoord;
attribute float a_alpha;

varying vec2 v_texCoord;
varying float v_alpha;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
  v_alpha = a_alpha;
}
`;

const FS_SOURCE = `
precision mediump float;
uniform sampler2D u_image;
uniform float u_strength;

varying vec2 v_texCoord;
varying float v_alpha;

void main() {
  vec4 color = texture2D(u_image, v_texCoord);
  float totalAlpha = v_alpha * u_strength;

  // Premultiplied alpha output ensures ZERO dark fringe or black borders
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

  // Premultiplied alpha for clean compositing with Canvas 2D
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
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  return gl;
}

/**
 * Render 3D-aware mirrored hemiface.
 * @param {Array} landmarks - 468 MediaPipe normalized landmarks
 * @param {HTMLCanvasElement} frameCanvas - Unmodified video canvas in selfie view
 * @param {boolean} isHealthyLeft - true if person's left is healthy (viewer right)
 * @param {number} strength - 0.0 to 1.0 opacity
 * @param {number} feather - Softness control (0..50)
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
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(program);

  // 1. Calculate head rotation (yaw) and source occlusion:
  // p1 = nose tip; p234 = Person's Right cheek; p454 = Person's Left cheek
  const p1 = pointFn(landmarks, 1);
  const p234 = pointFn(landmarks, 234);
  const p454 = pointFn(landmarks, 454);

  let sourceVisibility = 1.0;
  if (p1 && p234 && p454) {
    const dRight = Math.hypot(p1.x - p234.x, p1.y - p234.y);
    const dLeft = Math.hypot(p1.x - p454.x, p1.y - p454.y);

    // If healthy side is Person's Left, turning left compresses dLeft:
    const ratio = isHealthyLeft ? (dLeft / Math.max(dRight, 1)) : (dRight / Math.max(dLeft, 1));
    sourceVisibility = ratio;
  }

  // 2. Populate vertex buffers
  for (let i = 0; i < 468; i++) {
    const dstPoint = pointFn(landmarks, i);
    const partnerId = partner[i];
    const srcPoint = pointFn(landmarks, partnerId);

    if (dstPoint && srcPoint) {
      // Screen clip space: [-1, 1]
      positions[i * 2] = (dstPoint.x / width) * 2 - 1;
      positions[i * 2 + 1] = 1 - (dstPoint.y / height) * 2;

      // Texture UV space: [0, 1]
      texCoords[i * 2] = srcPoint.x / width;
      texCoords[i * 2 + 1] = srcPoint.y / height;

      // Alpha calculation:
      // Base alpha: 0.0 on face oval, 0.5 on inner ring, 1.0 inside
      let a = defaultVertexAlphas[i];

      // Dynamic Occlusion Handling:
      // When the head turns towards the source side (e.g. looking left with healthy left),
      // sourceVisibility drops. We dynamically fade out the lateral cheek vertices (lat > maxLat)
      // to 0.0, revealing the user's REAL CAMERA SKIN softly with zero dark hair/headphone stretching!
      const lat = lateralDistances[i];
      if (sourceVisibility < 1.05) {
        // When sourceVisibility is low (e.g. 0.4 - 0.7), maxLat contracts inward
        const maxLat = Math.min(0.85, Math.max(0.38, sourceVisibility * 0.95));
        if (lat > maxLat - 0.15) {
          const fade = Math.max(0.0, Math.min(1.0, (maxLat - lat + 0.15) / 0.20));
          a *= fade;
        }
      }

      alphas[i] = a;
    } else {
      positions[i * 2] = 0;
      positions[i * 2 + 1] = 0;
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

  const uStrength = gl.getUniformLocation(program, 'u_strength');
  gl.uniform1f(uStrength, strength);

  // Choose triangles for target hemiface:
  // If healthy side is Left -> target is Person's Right (right_triangles)
  // If healthy side is Right -> target is Person's Left (left_triangles)
  const indices = isHealthyLeft ? rightIndices : leftIndices;

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);

  // Single GPU draw call for all 449 triangles
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);

  return glCanvas;
}

export { symmetryData };
