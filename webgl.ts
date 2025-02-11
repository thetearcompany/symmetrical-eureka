// webgl.ts

const vertexShaderSource = `
  attribute vec4 a_position;
  void main() {
    gl_Position = a_position;
  }
`;

const fragmentShaderSource = `
  precision highp float;
  uniform sampler2D u_texture;
  void main() {
    gl_FragColor = texture2D(u_texture, vec2(0.5, 0.5));
  }
`;

function createShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return shader;
}

function createProgram(gl: WebGLRenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader): WebGLProgram {
  const program = gl.createProgram()!;
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  return program;
}

class WebGLTensor {
  gl: WebGLRenderingContext;
  texture: WebGLTexture;
  shape: number[];
  constructor(gl: WebGLRenderingContext, shape: number[], data: Float32Array) {
    this.gl = gl;
    this.shape = shape;
    this.texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, shape[0], shape[1], 0, gl.RED, gl.FLOAT, data);
  }
}

class WebGLNeuralNetwork {
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  constructor(canvas: HTMLCanvasElement) {
    this.gl = canvas.getContext("webgl")!;
    const vertexShader = createShader(this.gl, this.gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(this.gl, this.gl.FRAGMENT_SHADER, fragmentShaderSource);
    this.program = createProgram(this.gl, vertexShader, fragmentShader);
  }
  predict(input: WebGLTensor): WebGLTensor {
    return input;
  }
}