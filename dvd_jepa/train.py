"""Train DVD-JEPA end to end and emit everything the project needs:

    checkpoints/dvd_jepa.pt   PyTorch weights (resume / fine-tune)
    web/weights.json          base64 float32 weights for the browser demo
    assets/dvd_jepa_dream.gif reality vs. the rendered latent dream
    assets/dvd_jepa_anomaly.png predictive surprise spiking on an anomaly
    assets/metrics.json       the numbers quoted in the paper

Run:  python -m dvd_jepa.train
"""
from __future__ import annotations
import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .world import H, W, EMB, SIGMA, make_sequences, roll_one, build_pairs
from .models import Encoder, Predictor, Decoder, variance_term

ROOT = Path(__file__).resolve().parent.parent
EMA = 0.99
V = 0.05


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_jepa(obs, nxt, steps=2500, batch=256, lr=1e-3, log=print):
    online, target, predictor = Encoder(), Encoder(), Predictor()
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(online.parameters()) + list(predictor.parameters()), lr=lr)
    n = obs.shape[0]
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,))
        z = online(obs[idx])
        with torch.no_grad():
            z_tgt = target(nxt[idx])
        loss = F.mse_loss(predictor(z), z_tgt) + variance_term(z)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(EMA).add_(po, alpha=1 - EMA)
        if log and (step % 500 == 0):
            log(f"  jepa step {step:4d} | loss {loss.item():.4f} | emb-std {z.std(0).mean():.2f}")
    return online, target, predictor


def train_decoder(target, obs, steps=2000, batch=256, lr=1e-3, log=print):
    dec = Decoder()
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    with torch.no_grad():
        Z = target(obs)
    tgt_frame = obs[:, H * W:]          # latest frame of the observation stack
    n = obs.shape[0]
    rloss = torch.tensor(0.0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,))
        rloss = F.mse_loss(dec(Z[idx]), tgt_frame[idx])
        opt.zero_grad(); rloss.backward(); opt.step()
    if log:
        log(f"  decoder recon MSE {rloss.item():.5f}")
    return dec


def linear_probe(target, obs, pos_label, steps=800, log=print):
    """Frozen-encoder linear readout of the true (y, x) position -> proves the
    latent encodes world state even though it was never given coordinates."""
    with torch.no_grad():
        Z = target(obs)
    n = Z.shape[0]; ntr = int(0.8 * n)
    probe = torch.nn.Linear(EMB, 2)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(steps):
        i = torch.randint(0, ntr, (512,))
        l = F.mse_loss(probe(Z[i]), pos_label[i])
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        rmse = torch.sqrt(F.mse_loss(probe(Z[ntr:]), pos_label[ntr:])).item()
    if log:
        log(f"  linear probe position RMSE {rmse:.2f} px")
    return probe, rmse


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def forecast(target, predictor, dec, vf, K=30):
    with torch.no_grad():
        z = target(torch.tensor(vf[0:2].reshape(1, -1), dtype=torch.float32))
        preds, errs = [], []
        for k in range(K):
            z = predictor(z)
            fr = dec(z).reshape(H, W).numpy()
            preds.append(fr)
            errs.append(float(((fr - vf[k + 2]) ** 2).mean()))
    return preds, errs


def anomaly_scan(target, predictor, dec, af):
    surprise = []
    with torch.no_grad():
        for t in range(len(af) - 2):
            z = target(torch.tensor(af[t:t + 2].reshape(1, -1), dtype=torch.float32))
            pred_next = dec(predictor(z)).reshape(H, W).numpy()
            surprise.append(float(((pred_next - af[t + 2]) ** 2).mean()))
    return np.array(surprise)


# --------------------------------------------------------------------------- #
# export weights for the browser (base64 float32, little-endian)
# --------------------------------------------------------------------------- #
def _linears(module):
    return [m for m in module.modules() if isinstance(m, torch.nn.Linear)]


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode()


def _dump(module):
    out = []
    for lin in _linears(module):
        w = lin.weight.detach().numpy()      # [out, in]
        b = lin.bias.detach().numpy()        # [out]
        out.append({"shape": list(w.shape), "w": _b64(w), "b": _b64(b)})
    return out


def export_weights(online, predictor, dec, path):
    blob = {
        "meta": {"H": H, "W": W, "EMB": EMB, "sigma": SIGMA, "v": V},
        # activation conventions are hard-coded in web/jepa.js:
        #   encoder:   lin gelu lin gelu lin
        #   predictor: lin gelu lin
        #   decoder:   lin gelu lin gelu lin sigmoid
        "encoder": _dump(online),
        "predictor": _dump(predictor),
        "decoder": _dump(dec),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(blob))
    return Path(path).stat().st_size


# --------------------------------------------------------------------------- #
# asset rendering
# --------------------------------------------------------------------------- #
def render_dream_gif(gt_frames, pred_frames, path, scale=9):
    from PIL import Image, ImageDraw
    gap, bar = 10, 18

    def to_rgb(frame, color):
        a = np.clip(frame, 0, 1)
        img = (np.stack([a * color[0], a * color[1], a * color[2]], -1) * 255).astype(np.uint8)
        return Image.fromarray(img, "RGB").resize((W * scale, H * scale), Image.NEAREST)

    gif = []
    for k in range(len(pred_frames)):
        canvas = Image.new("RGB", (W * scale * 2 + gap, H * scale + bar), (18, 18, 22))
        canvas.paste(to_rgb(gt_frames[k], (1, 1, 1)), (0, bar))
        canvas.paste(to_rgb(pred_frames[k], (0.3, 0.9, 1.0)), (W * scale + gap, bar))
        d = ImageDraw.Draw(canvas)
        d.text((4, 4), "reality", fill=(170, 170, 175))
        d.text((W * scale + gap + 4, 4), "JEPA dream", fill=(90, 210, 255))
        d.text((4, H * scale + bar - 12), f"t+{k + 1}", fill=(110, 110, 120))
        gif.append(canvas)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=130, loop=0)


def render_anomaly_png(surprise, tele, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = np.median(surprise)
    thresh = base + 5 * (np.median(np.abs(surprise - base)) + 1e-6)
    flag = int(np.argmax(surprise))
    plt.figure(figsize=(8, 3.2))
    plt.plot(surprise, color="#2bd4ff", lw=2, label="surprise (pred vs reality)")
    plt.axhline(thresh, color="#ff5d5d", ls="--", lw=1, label="anomaly threshold")
    plt.axvline(tele, color="#ffce54", ls=":", lw=2, label="injected teleport")
    plt.scatter([flag], [surprise[flag]], color="#ff5d5d", zorder=5)
    plt.title("DVD-JEPA as a predictive anomaly detector")
    plt.xlabel("frame"); plt.ylabel("prediction error"); plt.legend(fontsize=8)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=110)
    plt.close()
    return base, thresh, flag


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print("DVD-JEPA :: training\n")

    frames, _ = make_sequences(seed=0)
    obs, nxt = build_pairs(frames)
    # position labels for the probe (latest frame of each obs)
    _, pos = make_sequences(seed=0)
    lbl = []
    for t in range(frames.shape[1] - 2):
        lbl.append(pos[:, t + 1])
    lbl = torch.tensor(np.concatenate(lbl), dtype=torch.float32)

    online, target, predictor = train_jepa(obs, nxt)
    dec = train_decoder(target, obs)
    _, probe_rmse = linear_probe(target, obs, lbl)

    # evaluation
    vf, _ = roll_one(T=44, seed=7, v=V)
    K = 30
    preds, errs = forecast(target, predictor, dec, vf, K)
    gt = [vf[k + 2] for k in range(K)]

    tele = 24
    af, _ = roll_one(T=44, teleport_at=tele, seed=3, v=V)
    surprise = anomaly_scan(target, predictor, dec, af)

    # save checkpoint
    ckpt = ROOT / "checkpoints" / "dvd_jepa.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"online": online.state_dict(), "target": target.state_dict(),
                "predictor": predictor.state_dict(), "decoder": dec.state_dict()}, ckpt)

    # export for browser
    wsize = export_weights(online, predictor, dec, ROOT / "web" / "weights.json")

    # render assets
    render_dream_gif(gt, preds, ROOT / "assets" / "dvd_jepa_dream.gif")
    base, thresh, flag = render_anomaly_png(surprise, tele, ROOT / "assets" / "dvd_jepa_anomaly.png")

    metrics = {
        "probe_position_rmse_px": round(probe_rmse, 3),
        "forecast_mse_step1": round(errs[0], 5),
        "forecast_mse_stepK": round(errs[-1], 5),
        "forecast_mse_mean": round(float(np.mean(errs)), 5),
        "forecast_horizon": K,
        "anomaly_injected_at": tele,
        "anomaly_detected_at": int(flag),
        "anomaly_peak_over_baseline": round(float(surprise.max() / (base + 1e-9)), 1),
        "decoder_recon_mse": None,
        "weights_json_bytes": wsize,
    }
    (ROOT / "assets" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nresults")
    for k, vv in metrics.items():
        print(f"  {k:32s} {vv}")
    print(f"\nwrote: {ckpt.relative_to(ROOT)}, web/weights.json ({wsize/1024:.0f} KB),")
    print("       assets/dvd_jepa_dream.gif, assets/dvd_jepa_anomaly.png, assets/metrics.json")


if __name__ == "__main__":
    main()
