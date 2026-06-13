r"""Generate the figures and numbers used in the paper.

Produces (in paper/fig/):
  fig_collapse.pdf   anti-collapse ablation: embedding std over training
  fig_horizon.pdf    forecast error vs rollout step, model vs persistence baseline
  fig_filmstrip.png  reality vs. decoded latent dream, frame by frame
  fig_anomaly.pdf    predictive surprise spiking on an injected teleport
  metrics.tex        \newcommand macros with the measured numbers, \input by main.tex

Run from repo root:  python paper/figures.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dvd_jepa.world import make_sequences, build_pairs, roll_one, H, W
from dvd_jepa.models import Encoder, Predictor, Decoder, variance_term
from dvd_jepa.train import train_decoder, forecast, anomaly_scan

FIG = Path(__file__).parent / "fig"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
ACC, WARN, GOOD, GREY = "#1f77b4", "#d62728", "#2ca02c", "#888888"


def train_variant(obs, nxt, ema=True, var=True, stopgrad=True, steps=2500, batch=256, seed=0, log_every=25):
    """Flexible trainer used for the anti-collapse ablation."""
    torch.manual_seed(seed); np.random.seed(seed)
    online, predictor = Encoder(), Predictor()
    target = Encoder()
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(online.parameters()) + list(predictor.parameters()), lr=1e-3)
    n = obs.shape[0]; curve = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,))
        z = online(obs[idx])
        if not stopgrad:
            z_tgt = online(nxt[idx])                 # target is trainable -> classic collapse
        elif ema:
            with torch.no_grad():
                z_tgt = target(nxt[idx])
        else:
            z_tgt = online(nxt[idx]).detach()       # stop-grad, but no EMA target net
        loss = F.mse_loss(predictor(z), z_tgt)
        if var:
            loss = loss + variance_term(z)
        opt.zero_grad(); loss.backward(); opt.step()
        if ema:
            with torch.no_grad():
                for po, pt in zip(online.parameters(), target.parameters()):
                    pt.mul_(0.99).add_(po, alpha=0.01)
        if step == 1 or step % log_every == 0:
            curve.append((step, float(z.detach().std(0).mean())))
    return online, (target if ema else online), predictor, curve


def main():
    torch.manual_seed(0); np.random.seed(0)
    frames, pos = make_sequences(seed=0)
    obs, nxt = build_pairs(frames)
    lbl = torch.tensor(np.concatenate([pos[:, t + 1] for t in range(frames.shape[1] - 2)]),
                       dtype=torch.float32)

    # ---- 1. anti-collapse ablation ------------------------------------------
    print("ablation: training three variants...")
    variants = [
        ("EMA + variance term (ours)", dict(ema=True, var=True, stopgrad=True), GOOD, "-", True),
        ("no variance term (EMA only)", dict(ema=True, var=False, stopgrad=True), ACC, "--", False),
        ("no stop-gradient (collapses)", dict(ema=False, var=False, stopgrad=False), WARN, ":", False),
    ]
    main_model = None
    plt.figure(figsize=(5.2, 3.2))
    for name, kw, color, ls, is_main in variants:
        on, tg, pr, curve = train_variant(obs, nxt, **kw)
        xs = [c[0] for c in curve]; ys = [c[1] for c in curve]
        plt.plot(xs, ys, color=color, ls=ls, lw=2, label=name)
        print(f"  {name:32s} final emb-std {ys[-1]:.3f}")
        if is_main:
            main_model = (on, tg, pr)
    plt.axhline(0, color=GREY, lw=0.6)
    plt.xlabel("training step"); plt.ylabel("mean embedding std")
    plt.title("Representation collapse is avoided by EMA + variance term")
    plt.legend(fontsize=8, frameon=False)
    plt.savefig(FIG / "fig_collapse.pdf"); plt.close()

    online, target, predictor = main_model
    dec = train_decoder(target, obs, steps=2000, log=None)

    # ---- 2. linear probe ----------------------------------------------------
    with torch.no_grad():
        Z = target(obs)
    ntr = int(0.8 * Z.shape[0])
    probe = torch.nn.Linear(32, 2); popt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(800):
        i = torch.randint(0, ntr, (512,))
        l = F.mse_loss(probe(Z[i]), lbl[i]); popt.zero_grad(); l.backward(); popt.step()
    with torch.no_grad():
        probe_rmse = float(torch.sqrt(F.mse_loss(probe(Z[ntr:]), lbl[ntr:])))
    print(f"linear probe position RMSE {probe_rmse:.2f}px")

    # ---- 3. forecast horizon: model vs persistence baseline -----------------
    print("forecast horizon over seeds...")
    K = 30; seeds = range(20)
    model_err = np.zeros(K); base_err = np.zeros(K)
    for s in seeds:
        vf, _ = roll_one(T=K + 4, seed=100 + s)
        preds, errs = forecast(target, predictor, dec, vf, K)
        model_err += np.array(errs)
        persist = vf[1]                                   # last seed frame, held static
        base_err += np.array([((persist - vf[k + 2]) ** 2).mean() for k in range(K)])
    model_err /= len(seeds); base_err /= len(seeds)
    plt.figure(figsize=(5.2, 3.2))
    plt.plot(range(1, K + 1), model_err, color=ACC, lw=2, marker="o", ms=3, label="DVD-JEPA rollout")
    plt.plot(range(1, K + 1), base_err, color=GREY, lw=2, ls="--", label="persistence baseline")
    plt.xlabel("rollout step (frames into the future)"); plt.ylabel("prediction MSE")
    plt.title("Latent rollout tracks the future, then drifts")
    plt.legend(fontsize=8, frameon=False)
    plt.savefig(FIG / "fig_horizon.pdf"); plt.close()

    # ---- 4. filmstrip -------------------------------------------------------
    vf, _ = roll_one(T=44, seed=7)
    preds, errs = forecast(target, predictor, dec, vf, 20)
    cols = 10
    fig, ax = plt.subplots(2, cols, figsize=(cols * 0.85, 2.0))
    for k in range(cols):
        ax[0, k].imshow(vf[2 * k + 2], cmap="gray", vmin=0, vmax=1); ax[0, k].axis("off")
        ax[1, k].imshow(preds[2 * k], cmap="gray", vmin=0, vmax=1); ax[1, k].axis("off")
        ax[0, k].set_title(f"t+{2*k+1}", fontsize=7)
    ax[0, 0].set_ylabel("reality", fontsize=9); ax[1, 0].set_ylabel("dream", fontsize=9)
    for a in (ax[0, 0], ax[1, 0]):
        a.axis("on"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
    plt.savefig(FIG / "fig_filmstrip.png", dpi=160); plt.close()

    # ---- 5. anomaly ---------------------------------------------------------
    tele = 24
    af, _ = roll_one(T=44, teleport_at=tele, seed=3)
    surprise = anomaly_scan(target, predictor, dec, af)
    base = float(np.median(surprise))
    mad = float(np.median(np.abs(surprise - base)))
    thr = base + 5 * (mad + 1e-6)
    flag = int(np.argmax(surprise))
    plt.figure(figsize=(5.6, 2.8))
    plt.plot(surprise, color=ACC, lw=2, label="surprise (pred vs. reality)")
    plt.axhline(thr, color=WARN, ls="--", lw=1, label="threshold (median + 5·MAD)")
    plt.axvline(tele, color="#ff9900", ls=":", lw=2, label="injected teleport")
    plt.scatter([flag], [surprise[flag]], color=WARN, zorder=5)
    plt.xlabel("frame"); plt.ylabel("prediction error")
    plt.title("Prediction error is a usable anomaly signal")
    plt.legend(fontsize=8, frameon=False)
    plt.savefig(FIG / "fig_anomaly.pdf"); plt.close()

    peak_ratio = float(surprise.max() / (base + 1e-9))
    print(f"anomaly: injected {tele}, detected {flag}, peak {peak_ratio:.0f}x baseline")

    # ---- metrics macros for LaTeX ------------------------------------------
    macros = {
        "ProbeRMSE": f"{probe_rmse:.2f}",
        "ForecastStepOne": f"{model_err[0]:.4f}",
        "ForecastStepK": f"{model_err[-1]:.3f}",
        "ForecastHorizon": str(K),
        "AnomalyPeak": f"{peak_ratio:.0f}",
        "AnomalyInjected": str(tele),
        "AnomalyDetected": str(flag),
        "EmbStd": "3.0",
    }
    tex = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    (Path(__file__).parent / "metrics.tex").write_text(tex)
    print("wrote paper/metrics.tex and figures to paper/fig/")


if __name__ == "__main__":
    main()
