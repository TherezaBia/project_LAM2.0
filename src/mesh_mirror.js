// 3D-Aware Facial Mesh Symmetry Renderer
// Uses WebGL to render 449 anatomical facial triangles in real-time,
// naturally accounting for 3D head rotation (yaw, pitch, roll) and perspective foreshortening.

import symmetryData from './face_symmetry_mesh.json';

const { partner, left_triangles, right_triangles, midline } = symmetryData;

// Flattened index buffers
const rightIndices = new Uint16Array(right_triangles.flat());
const leftIndices = new Uint16Array(left_triangles.flat());

let glCanvas = null;
let gl = null;
let program = null;
let positionBuffer = null;
let texCoordBuffer = null;
let indexBuffer = null;
let texture = null;

// Reusable vertex arrays (468 vertices * 2 floats)
const positions = new Float32Array(468 * 2);
const texCoords = new Float32Array(468 * 2);

const VS_SOURCE = `
attribute vec2 a_position;
attribute vec2 a_texCoord;
varying vec2 v_texCoord;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
}
`;

const FS_SOURCE = `
precision mediump float;
uniform sampler2D u_image;
uniform float u_strength;
varying vec2 v_texCoord;

void main() {
  vec4 color = texture2D(u_image, v_texCoord);
  gl_FragColor = vec4(color.rgb, color.a * u_strength);
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

  gl = glCanvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
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
  indexBuffer = gl.createBuffer();

  texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  return gl;
}

/**
 * Render 3D-aware mirrored hemiface.
 * @param {Array} landmarks - 468 MediaPipe normalized landmarks
 * @param {HTMLCanvasElement} frameCanvas - Unmodified video canvas in selfie view
 * @param {boolean} isHealthyLeft - true if person's left is healthy (viewer right)
 * @param {number} strength - 0.0 to 1.0 opacity
 * @param {Function} pointFn - function(landmarks, index) returning {x, y} in canvas space
 * @returns {HTMLCanvasElement|null} - rendered WebGL canvas
 */
export function renderMesh3DMirror(landmarks, frameCanvas, isHealthyLeft, strength, pointFn) {
  const width = frameCanvas.width;
  const height = frameCanvas.height;

  if (!gl || glCanvas.width !== width || glCanvas.height !== height) {
    if (!initWebGL(width, height)) return null;
  }

  gl.viewport(0, 0, width, height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(program);

  // Compute position and UV for all 468 vertices
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
    } else {
      positions[i * 2] = 0;
      positions[i * 2 + 1] = 0;
      texCoords[i * 2] = 0;
      texCoords[i * 2 + 1] = 0;
    }
  }

  // Upload positions
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

  // Upload frame texture
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frameCanvas);

  // Set strength uniform
  const uStrength = gl.getUniformLocation(program, 'u_strength');
  gl.uniform1f(uStrength, strength);

  // Choose triangles for target hemiface:
  // If healthy side is Left -> target is Person's Right (right_triangles)
  // If healthy side is Right -> target is Person's Left (left_triangles)
  const indices = isHealthyLeft ? rightIndices : leftIndices;

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);

  // Single GPU draw call for all 449 triangles!
  gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);

  return glCanvas;
}

export { symmetryData };
