// DVD-JEPA inference in the browser. No framework: the model is three small
// MLPs, so the whole forward pass is a few matrix multiplies. Weights are
// loaded from weights.json (base64 float32, exported by dvd_jepa/train.py).

export async function loadModel(url = "weights.json") {
  const blob = await (await fetch(url)).json();
  const dec = (layers) =>
    layers.map((L) => ({
      out: L.shape[0],
      in: L.shape[1],
      w: b64f32(L.w),
      b: b64f32(L.b),
    }));
  return {
    meta: blob.meta,
    encoder: dec(blob.encoder),
    predictor: dec(blob.predictor),
    decoder: dec(blob.decoder),
  };
}

function b64f32(s) {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

// y = gelu/sigmoid?(x @ W^T + b);  W is row-major [out, in]
function linear(x, layer) {
  const { out, in: nin, w, b } = layer;
  const y = new Float32Array(out);
  for (let o = 0; o < out; o++) {
    let acc = b[o];
    const base = o * nin;
    for (let i = 0; i < nin; i++) acc += w[base + i] * x[i];
    y[o] = acc;
  }
  return y;
}

const C = Math.sqrt(2 / Math.PI);
function geluInPlace(x) {
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    x[i] = 0.5 * v * (1 + Math.tanh(C * (v + 0.044715 * v * v * v)));
  }
  return x;
}
function sigmoidInPlace(x) {
  for (let i = 0; i < x.length; i++) x[i] = 1 / (1 + Math.exp(-x[i]));
  return x;
}

// encoder:   lin gelu lin gelu lin
export function encode(m, obs) {
  let h = geluInPlace(linear(obs, m.encoder[0]));
  h = geluInPlace(linear(h, m.encoder[1]));
  return linear(h, m.encoder[2]);
}
// predictor: lin gelu lin   (the world model: one step forward in latent space)
export function predict(m, z) {
  const h = geluInPlace(linear(z, m.predictor[0]));
  return linear(h, m.predictor[1]);
}
// decoder:   lin gelu lin gelu lin sigmoid
export function decode(m, z) {
  let h = geluInPlace(linear(z, m.decoder[0]));
  h = geluInPlace(linear(h, m.decoder[1]));
  return sigmoidInPlace(linear(h, m.decoder[2]));
}

export function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s / a.length;
}

// The world: a DVD logo bouncing in a box. Physics identical to world.py.
export class World {
  constructor(meta) {
    this.H = meta.H;
    this.W = meta.W;
    this.sigma = meta.sigma;
    this.speed = meta.v * (meta.H - 1);
    this.reset();
  }
  _rand(a, b) {
    return a + Math.random() * (b - a);
  }
  reset() {
    const H = this.H;
    this.p = [this._rand(3, H - 4), this._rand(3, H - 4)];
    const s = () => (Math.random() < 0.5 ? -1 : 1) * this.speed;
    this.vel = [s(), s()];
    this.prev = this.render();
    this._advance();
    this.cur = this.render();
  }
  _advance() {
    const lims = [this.H - 2, this.W - 2];
    for (let d = 0; d < 2; d++) {
      this.p[d] += this.vel[d];
      if (this.p[d] > lims[d] || this.p[d] < 1) this.vel[d] *= -1;
      this.p[d] = Math.max(1, Math.min(lims[d], this.p[d]));
    }
  }
  teleport() {
    this.p = [this._rand(3, this.H - 4), this._rand(3, this.W - 4)];
  }
  step() {
    this.prev = this.cur;
    this._advance();
    this.cur = this.render();
  }
  // observation handed to the model: [older frame, newer frame] -> 2*H*W
  obs() {
    const o = new Float32Array(this.prev.length + this.cur.length);
    o.set(this.prev, 0);
    o.set(this.cur, this.prev.length);
    return o;
  }
  render() {
    const H = this.H,
      W = this.W,
      s2 = 2 * this.sigma * this.sigma;
    const f = new Float32Array(H * W);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const dy = y - this.p[0],
          dx = x - this.p[1];
        f[y * W + x] = Math.exp(-(dy * dy + dx * dx) / s2);
      }
    return f;
  }
}
