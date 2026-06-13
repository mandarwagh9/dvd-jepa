# Submitting DVD-JEPA to arXiv

Everything arXiv needs is prepared. arXiv has **no submission API** — uploading is a manual,
account-bound web process that agrees to author terms under your identity, so it has to be done by you.
This makes it a ~5-minute copy-paste.

## What you upload

`paper/dvd-jepa-arxiv.tar.gz` (38 KB) — the LaTeX source + figures. Rebuild it anytime with:

```bash
bash paper/build_arxiv.sh
```

It contains `main.tex`, `metrics.tex`, and the four figures. arXiv recompiles it on their servers
(default **pdfLaTeX**). The source is pdfLaTeX-compatible (lmodern + T1 fontenc, only standard packages,
no fontspec/XeLaTeX features), uses a manual `thebibliography` (so **no `.bbl` is needed**), and is well
under the 50 MB limit. It compiles cleanly from the tarball alone (verified).

## ⚠️ Step 0 — the real gate: an account + endorsement

As of **21 Jan 2026**, arXiv tightened its endorsement policy. An institutional email **no longer
qualifies on its own**. To submit to a category for the first time you need either:

1. an academic/research email **and** prior authorship of a paper already on arXiv in that category; **or**
2. a **personal endorsement** from an established arXiv author in that category (they enter a code for you).

As an independent author (gmail, no prior arXiv paper in `cs.LG`), you will most likely need **path 2**.
After you register at <https://arxiv.org/user/register> and start a `cs.LG` submission, arXiv shows you an
**endorsement code**; send it to someone who has published in `cs.LG` and ask them to endorse you at
<https://arxiv.org/auth/endorse>. Without this, the submission cannot be announced.

> If you don't have an endorser handy: the paper is already a public, linkable preprint via the
> [GitHub release](https://github.com/mandarwagh9/dvd-jepa/releases/latest) (PDF attached). You can also
> mint a DOI for it for free by pushing a release to **Zenodo** (no endorsement required), which many
> people cite instead of/while waiting on arXiv.

## Step-by-step (once you can submit)

1. Go to <https://arxiv.org/submit> and **Start a New Submission**.
2. **License:** choose **CC BY 4.0** (recommended — matches the repo's open-source spirit and allows reuse
   with attribution). Note: the license choice is permanent.
3. **Submission type:** article. Upload `dvd-jepa-arxiv.tar.gz`.
4. Let arXiv process/compile it; **review the generated PDF** (6 pages). If autoTeX errors, read the log —
   the source builds locally, so any issue is almost always a missing-file or engine note.
5. Paste the **metadata** below.
6. **Categories:** primary `cs.LG`; cross-list `cs.CV` (and optionally `cs.AI`).
7. Add the **comments** line. Submit. It then goes through moderation (often 1–2 business days) before
   it's announced.

## Ready-to-paste metadata

**Title**
```
DVD-JEPA: A Minimal, Reproducible Joint-Embedding Predictive Architecture World Model
```

**Authors**
```
Mandar Wagh
```

**Primary category:** `cs.LG`  **Cross-list:** `cs.CV`, `cs.AI`

**Comments**
```
6 pages, 5 figures. Code, trained weights, an interactive browser demo, and a Colab notebook:
https://github.com/mandarwagh9/dvd-jepa  |  Demo: https://dvd-jepa.vercel.app
```

**Abstract** (plain text)
```
A world model predicts how an environment evolves. The dominant failure mode when learning one from
video is generative: predicting future pixels forces the model to spend capacity on detail that is
fundamentally unpredictable. Joint-Embedding Predictive Architectures (JEPA) instead predict the
representation of the future and let the encoder discard what it cannot predict. We present DVD-JEPA,
the smallest honest instance of this idea we could construct: a context encoder, an exponential-moving-
average (EMA) target encoder, and a latent predictor are trained -- without labels and without a decoder
-- to predict the next observation of a bouncing DVD logo in a 32-dimensional representation space.
Despite never receiving a coordinate, a frozen linear probe recovers the logo's position to sub-pixel
accuracy in a 16-pixel box. An ablation shows that removing the stop-gradient collapses the
representation to a constant, while the EMA target prevents it. Adding an optional decoder turns the
model into a future-frame video predictor that tracks the bounce for about twenty steps before latent
rollout drift, and into a predictive anomaly detector whose surprise signal spikes by tens of times over
baseline on an injected teleport. The entire model trains on a CPU in about ten seconds and the trained
weights run client-side in a browser. DVD-JEPA is deliberately a toy, but every component has a
full-scale counterpart in I-JEPA, V-JEPA, and V-JEPA 2; it is meant as a transparent, runnable artifact
for understanding how JEPA world models work.
```

## After it's live
Add the arXiv ID to `README.md` (swap the paper badge for an arXiv badge) and to the BibTeX entry.
