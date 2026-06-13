"""
DVD-JEPA  —  a Joint-Embedding Predictive Architecture world model for the
single most important phenomenon in computer history: a DVD logo bouncing
in a box. Built for absolutely no reason, as a joke.

It is a *real* JEPA. It has:
  - a context encoder            E_theta
  - an EMA target encoder        E_ema   (stop-grad)  <- the anti-collapse trick
  - a predictor P that predicts the FUTURE in REPRESENTATION space, not pixels
  - a VICReg-style variance term, just so we can watch collapse NOT happen

It is also a *real* world model. After training you can roll it forward purely
in latent space and it correctly dreams where the logo will go for many steps.

It has, by design and like every JEPA, NO DECODER. It predicts representations,
not pixels. So it understands the bounce in its bones and is physically
incapable of showing it to you. The only way we ever see the dream is by bolting
a tiny linear PROBE onto the frozen latents and asking "ok but where IS it" —
the JEPA itself just hands you a 32-d vector that, deep down, means
"yeah. still bouncing."

Trains on CPU in well under a minute.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

H = W = 16          # frame size (the logo deserves no more)
EMB = 32            # the entire understanding of the universe, in 32 floats
DEVICE = "cpu"


# ----------------------------------------------------------------------------
# The world: a DVD logo bouncing in a box. No actions. It just... bounces.
# ----------------------------------------------------------------------------
def make_sequences(n_seq=400, T=64, v=0.05):
    pos = np.zeros((n_seq, T, 2), np.float32)
    frames = np.zeros((n_seq, T, H, W), np.float32)
    p = np.random.uniform(2, H - 3, size=(n_seq, 2)).astype(np.float32)
    vel = (np.random.choice([-1, 1], size=(n_seq, 2)) * v * (H - 1)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for t in range(T):
        p = p + vel
        for d, lim_hi in ((0, H - 2), (1, W - 2)):
            flip = (p[:, d] > lim_hi) | (p[:, d] < 1.0)
            vel[flip, d] *= -1
            p[:, d] = np.clip(p[:, d], 1.0, lim_hi)
        pos[:, t] = p
        cy = p[:, 0][:, None, None]
        cx = p[:, 1][:, None, None]
        frames[:, t] = np.exp(-(((yy[None] - cy) ** 2 + (xx[None] - cx) ** 2) / (2 * 1.5 ** 2)))
    return frames, pos


def build_pairs(frames, pos):
    """An 'observation' is 2 stacked frames (so velocity is visible).
    obs_t = (frame_t, frame_{t+1});  next obs = (frame_{t+1}, frame_{t+2})."""
    n, T = frames.shape[0], frames.shape[1]
    obs, nxt, lbl = [], [], []
    for t in range(T - 2):
        obs.append(frames[:, t:t + 2].reshape(n, -1))
        nxt.append(frames[:, t + 1:t + 3].reshape(n, -1))
        lbl.append(pos[:, t + 1])                       # where the logo is "now"
    obs = torch.tensor(np.concatenate(obs), dtype=torch.float32)
    nxt = torch.tensor(np.concatenate(nxt), dtype=torch.float32)
    lbl = torch.tensor(np.concatenate(lbl), dtype=torch.float32)
    return obs, nxt, lbl


# ----------------------------------------------------------------------------
# The architecture
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * H * W, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, EMB),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB, 64), nn.GELU(),
            nn.Linear(64, EMB),
        )

    def forward(self, z):
        return self.net(z)


def variance_term(z, eps=1e-4):
    """VICReg hinge: push every embedding dim to have std >= 1. This is the
    'please do not collapse to a constant' regularizer, made visible."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()


# ----------------------------------------------------------------------------
# Train the JEPA
# ----------------------------------------------------------------------------
print("generating the bouncing universe...")
frames, pos = make_sequences()
obs, nxt, lbl = build_pairs(frames, pos)
N = obs.shape[0]
print(f"  {N} latent-transition pairs of a logo doing nothing of consequence\n")

online = Encoder().to(DEVICE)
target = Encoder().to(DEVICE)
target.load_state_dict(online.state_dict())
for p_ in target.parameters():
    p_.requires_grad_(False)
predictor = Predictor().to(DEVICE)

opt = torch.optim.Adam(list(online.parameters()) + list(predictor.parameters()), lr=1e-3)
EMA = 0.99
BATCH, STEPS = 256, 2500

print("training DVD-JEPA (predicting the future in representation space)...")
for step in range(1, STEPS + 1):
    idx = torch.randint(0, N, (BATCH,))
    z_online = online(obs[idx])
    with torch.no_grad():
        z_target = target(nxt[idx])          # EMA + stop-grad target
    pred = predictor(z_online)               # predict the FUTURE embedding
    inv = F.mse_loss(pred, z_target)         # ...and match it, in latent space
    var = variance_term(z_online)
    loss = inv + 1.0 * var
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():                    # EMA: target slowly follows online
        for po, pt in zip(online.parameters(), target.parameters()):
            pt.mul_(EMA).add_(po, alpha=1 - EMA)
    if step % 250 == 0 or step == 1:
        emb_std = z_online.detach().std(dim=0).mean().item()
        print(f"  step {step:4d} | loss {loss.item():.4f} | predict {inv.item():.4f} "
              f"| var-reg {var.item():.4f} | emb-std {emb_std:.3f}  "
              f"{'(NOT collapsed)' if emb_std > 0.3 else '(!! collapsing)'}")

# ----------------------------------------------------------------------------
# Does it secretly know the world? Freeze it, bolt on a linear PROBE: emb -> (y,x)
# ----------------------------------------------------------------------------
print("\nfreezing the JEPA and interrogating it with a linear probe...")
with torch.no_grad():
    Z = target(obs)                          # frozen latents for every observation
ntr = int(0.8 * N)
probe = nn.Linear(EMB, 2)
popt = torch.optim.Adam(probe.parameters(), lr=1e-2)
for _ in range(800):
    i = torch.randint(0, ntr, (512,))
    pl = F.mse_loss(probe(Z[i]), lbl[i])
    popt.zero_grad(); pl.backward(); popt.step()
with torch.no_grad():
    pred_pos = probe(Z[ntr:])
    rmse = torch.sqrt(F.mse_loss(pred_pos, lbl[ntr:])).item()
print(f"  linear probe recovers the logo's (y,x) to within {rmse:.2f} px "
      f"(box is {H}x{W}) — it knew exactly where it was the whole time.")

# ----------------------------------------------------------------------------
# Make it DREAM: roll the predictor forward in latent space, decode via probe.
# The JEPA never renders a pixel. The probe just reads its mind.
# ----------------------------------------------------------------------------
print("\nasking the world model to dream the future (latent rollout)...")
val_frames, val_pos = make_sequences(n_seq=1, T=40)
K = 28
with torch.no_grad():
    z = target(torch.tensor(val_frames[0, 0:2].reshape(1, -1), dtype=torch.float32))
    dreamed = []
    for _ in range(K):
        z = predictor(z)                     # step forward purely in latent space
        dreamed.append(probe(z)[0].numpy())  # probe peeks at the dream
dreamed = np.array(dreamed)
true = val_pos[0, 2:2 + K]
roll_rmse = float(np.sqrt(((dreamed - true) ** 2).mean()))
print(f"  {K}-step latent dream tracks the real bounce to {roll_rmse:.2f} px RMSE\n")

# ASCII: 'o' = reality, '*' = the JEPA's dream (decoded only via the probe)
grid = [[" "] * W for _ in range(H)]
for (ty, tx) in true:
    grid[int(np.clip(ty, 0, H - 1))][int(np.clip(tx, 0, W - 1))] = "o"
for (dy, dx) in dreamed:
    y, x = int(np.clip(dy, 0, H - 1)), int(np.clip(dx, 0, W - 1))
    grid[y][x] = "@" if grid[y][x] == "o" else "*"
print("   reality = o   dream = *   both = @")
print("  +" + "-" * W + "+")
for row in grid:
    print("  |" + "".join(row) + "|")
print("  +" + "-" * W + "+")

print("\nDVD-JEPA understands the bounce perfectly and will not, under any")
print("circumstances, draw it for you. It's a JEPA. It only does vectors.")
print("here, in lieu of a picture, is its complete mental state right now:")
print(" ", np.round(z[0].detach().numpy(), 2))
