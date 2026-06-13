# Sharing / publishing DVD-JEPA

Already live (published automatically):

| Where | Link |
|---|---|
| GitHub (code, paper, releases) | https://github.com/mandarwagh9/dvd-jepa |
| Live demo (Vercel) | https://dvd-jepa.vercel.app |
| Live demo (🤗 Hugging Face Space) | https://huggingface.co/spaces/mandarwagh/dvd-jepa |
| Paper PDF + arXiv source (GitHub Release) | https://github.com/mandarwagh9/dvd-jepa/releases/latest |

The platforms below are tied to **your** account (posting as you, or agreeing to terms as you), so they
need your hands. Copy-paste drafts and exact steps follow.

---

## Hacker News — "Show HN"
Post at <https://news.ycombinator.com/submit> (URL = the demo; the title field is the headline).

**Title**
```
Show HN: DVD-JEPA – a JEPA world model that dreams a bouncing DVD logo (in-browser)
```
**URL**
```
https://dvd-jepa.vercel.app
```
**First comment (text)**
```
I wanted the smallest honest version of a JEPA world model — the LeCun-style architecture behind
I-JEPA / V-JEPA — so I trained one on the only physics that matters: a DVD logo bouncing in a box.

It predicts the *representation* of the next frame, not the pixels. A pure JEPA has no decoder, so it
understands the bounce perfectly and literally can't draw it — it hands you a 32-d vector. Bolt on an
optional decoder and it becomes a future-frame predictor (tracks the bounce, wall reflections and all,
for ~20 steps) and an anomaly detector (teleport the logo and the surprise signal spikes).

A frozen linear probe recovers the logo's exact position to <1px even though it never saw a coordinate.
Removing the stop-gradient collapses the representation to a constant — you can watch that in the paper's
ablation. It trains on a CPU in ~10 seconds, and the demo runs entirely client-side (the MLPs are
re-implemented in ~40 lines of JS).

Code + a short arXiv-style paper: https://github.com/mandarwagh9/dvd-jepa
It's a toy and a joke, but every part has a full-scale counterpart in V-JEPA 2. Happy to answer questions.
```

---

## Reddit — r/MachineLearning (use the `[P]` Project flair)
Post at <https://www.reddit.com/r/MachineLearning/submit>

**Title**
```
[P] DVD-JEPA: a minimal, reproducible JEPA world model (bouncing DVD logo) — in-browser demo + short paper
```
**Body**
```
JEPA (Joint-Embedding Predictive Architecture) predicts the *representation* of the future instead of
pixels. I built the smallest faithful instance I could: context encoder + EMA target encoder + latent
predictor + VICReg variance term, trained with no labels on a DVD logo bouncing in a 16x16 box.

Results (all from one CPU run, ~10s):
- A frozen linear probe recovers the logo's (y,x) to ~0.8px in a 16px box — the latent encodes exact
  world state with no coordinate supervision.
- Ablation: removing the stop-gradient collapses embedding std to 0; the EMA target prevents it.
- Add an optional decoder → future-frame prediction that tracks the bounce ~20 steps before latent drift.
- Run it as a 1-step monitor and prediction error becomes an anomaly signal (teleport → big surprise spike).

The fun part: a *pure* JEPA has no decoder, so it "understands" the bounce and can't render a pixel of it.

Interactive demo (client-side, no GPU): https://dvd-jepa.vercel.app
Code + paper + Colab: https://github.com/mandarwagh9/dvd-jepa

Built as a transparent on-ramp to I-JEPA/V-JEPA/V-JEPA 2 — feedback welcome.
```
> Also fits r/learnmachinelearning (educational angle). Don't cross-post the exact same text the same
> minute — space them out and tweak the intro so it doesn't read as spam.

---

## X / Twitter (thread)
```
1/ I trained a real JEPA world model on the most important physics in computing: a DVD logo bouncing in
a box. It predicts the FUTURE in representation space — not pixels. Runs in your browser, trains on a CPU
in ~10s. 🧵  https://dvd-jepa.vercel.app

2/ JEPA = encode, then predict the *embedding* of what comes next, and let the encoder throw away what it
can't predict. A pure JEPA has no decoder. So it understands the bounce perfectly and literally cannot
draw it — it just hands you a 32-d vector.

3/ Proof it actually learned the world: a frozen linear probe reads the logo's exact (x,y) out of that
vector to under 1 pixel — with zero coordinate supervision.

4/ Bolt on an optional decoder and it becomes a future-frame video predictor (tracks the bounce + wall
reflections ~20 steps) AND an anomaly detector — teleport the logo and the "surprise" signal spikes.

5/ It's a joke and it's also a correct, tiny instance of the architecture behind I-JEPA / V-JEPA /
V-JEPA 2. Code, a short paper, and a Colab: https://github.com/mandarwagh9/dvd-jepa
```

---

## LinkedIn
```
I built DVD-JEPA: the smallest honest version of a Joint-Embedding Predictive Architecture world model —
the LeCun-style approach behind I-JEPA and V-JEPA — trained on a DVD logo bouncing in a box.

Instead of predicting future pixels (and drowning in unpredictable detail), it predicts the *representation*
of the future. A frozen linear probe then recovers the logo's exact position to under a pixel, with no
coordinate labels — the model learned the world's state on its own. Add an optional decoder and it forecasts
future frames and flags anomalies when reality diverges from its prediction.

It trains on a CPU in ~10 seconds and runs entirely in the browser. A toy, deliberately — but a faithful,
inspectable on-ramp to the full-scale systems used for robot planning in V-JEPA 2.

Demo: https://dvd-jepa.vercel.app
Code + paper: https://github.com/mandarwagh9/dvd-jepa
```

---

## Self-serve publishing (no endorsement needed)

### Zenodo — free DOI (recommended; makes it citable today)
1. Sign in at <https://zenodo.org> with GitHub.
2. <https://zenodo.org/account/settings/github/> → flip the switch **ON** for `mandarwagh9/dvd-jepa`.
3. On GitHub, cut a new release (e.g. `v0.1.1`) — Zenodo archives it and mints a DOI automatically.
4. Add the DOI badge to `README.md`.

### PyPI — `pip install dvd-jepa`
The package is already configured (`pyproject.toml`). Once you have a PyPI account + API token:
```bash
pip install build twine
python -m build                 # creates dist/*.whl and dist/*.tar.gz
twine upload dist/*             # paste your PyPI token when prompted
```

### arXiv
See `paper/SUBMISSION.md` — package is built; you need an account + a cs.LG endorser (post-Jan-2026 policy).

### Hugging Face — model card (optional, in addition to the Space)
The Space (demo) is already live. If you also want the weights discoverable as a model:
`huggingface-cli upload mandarwagh/dvd-jepa-weights checkpoints/dvd_jepa.pt`.
