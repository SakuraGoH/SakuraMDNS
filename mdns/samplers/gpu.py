"""GPU (XPU / CUDA / CPU-torch) parallel MCMC for 2D Ising and q-state Potts.

Two families, both written with plain torch ops so they run unchanged on
Intel XPU, CUDA and CPU:

``*_cb``     checkerboard (even/odd sublattice) Glauber.  Sites of one parity
             never share an edge, so a whole half-sweep is one vectorised
             update with no data-dependent control flow.  Extremely wide but
             suffers the usual critical slowing down of local updates.

``*_wolff``  one Wolff cluster per chain, grown as a level-synchronous BFS that
             is vectorised over the batch.  The *depth* of the cluster (O(L))
             is sequential, but every level is a batched tensor op, and the
             cluster flip removes critical slowing down.

Layout: Ising ``S`` is float32 in {-1,+1}, Potts ``S`` is int64 in 0..q-1,
both shaped ``(B, L, L)``.  Observables are returned as ``(B, sweeps)``.

Why not a custom kernel?  torch 2.10+xpu has no easy Triton path on this
machine and L is small, so launch overhead -- not arithmetic -- is the limit.
That is measured and reported rather than hidden.
"""

from __future__ import annotations

import math

import torch

KINDS = ("ising_cb", "potts_cb", "ising_wolff", "potts_wolff")


# ------------------------------------------------------------------ helpers
def _generator(device: torch.device, seed: int):
    try:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        return g
    except Exception:
        return None


def init_state(kind: str, L: int, B: int, q: int, device, seed: int = 0) -> torch.Tensor:
    dev = torch.device(device)
    gen = _generator(dev, seed)
    if kind.startswith("ising"):
        s = torch.randint(0, 2, (B, L, L), device=dev, generator=gen)
        return (s * 2 - 1).to(torch.float32)
    return torch.randint(0, q, (B, L, L), device=dev, generator=gen)


def energy(kind: str, S: torch.Tensor, J: float = 1.0, h: float = 0.0) -> torch.Tensor:
    """(B,) energy, same convention as mdns.physics."""
    if kind.startswith("ising"):
        nb = (torch.roll(S, 1, 1) + torch.roll(S, -1, 1)
              + torch.roll(S, 1, 2) + torch.roll(S, -1, 2))
        return (-J * (S * nb).sum((1, 2)) / 2.0 - h * S.sum((1, 2)))
    eq = (S == torch.roll(S, -1, 1)).sum((1, 2)) + (S == torch.roll(S, -1, 2)).sum((1, 2))
    return -J * eq.to(torch.float32)


def magnetization(kind: str, S: torch.Tensor, q: int = 3) -> torch.Tensor:
    """(B,) order parameter: |<s>| for Ising, dominant-state fraction for Potts."""
    B, L, _ = S.shape
    if kind.startswith("ising"):
        return S.mean((1, 2)).abs()
    flat = S.reshape(B, L * L)
    counts = torch.stack([(flat == c).sum(1) for c in range(q)], dim=1)
    return counts.max(1).values.to(torch.float32) / (L * L)


# ------------------------------------------------------------- checkerboard
def _ising_cb_sweep(S, beta, J, h, par, gen, dev):
    for p in (0, 1):
        nb = (torch.roll(S, 1, 1) + torch.roll(S, -1, 1)
              + torch.roll(S, 1, 2) + torch.roll(S, -1, 2))
        dE = 2.0 * J * S * nb + 2.0 * h * S
        # flip/unflip is symmetric, so the Metropolis rate is exactly a sigmoid
        acc = torch.rand(S.shape, device=dev, generator=gen) < torch.sigmoid(-beta * dE)
        S = torch.where((par == p) & acc, -S, S)
    return S


def _potts_cb_sweep(S, beta, J, q, par, gen, dev):
    states = torch.arange(q, device=dev, dtype=S.dtype).view(1, 1, 1, q)
    for p in (0, 1):
        nbs = torch.stack([torch.roll(S, 1, 1), torch.roll(S, -1, 1),
                           torch.roll(S, 1, 2), torch.roll(S, -1, 2)], dim=0)
        counts = (nbs.unsqueeze(-1) == states).sum(0).to(torch.float32)  # (B,L,L,q)
        logits = beta * J * counts
        u = torch.rand(logits.shape, device=dev, generator=gen).clamp_min(1e-20)
        gumbel = -torch.log(-torch.log(u))
        new = (logits + gumbel).argmax(-1)                              # (B,L,L)
        S = torch.where(par == p, new, S)
    return S


# ------------------------------------------------------------------- wolff
def _wolff_sweep(S, p_add, kind, q, gen, dev, hops: int = 4):
    B, L, _ = S.shape
    D = L * L
    # One independent Bernoulli per *edge*, drawn once per sweep.  Using one
    # draw per site is wrong: a site touching k frontier sites must get k
    # chances, otherwise the bond percolation is under-connected and the
    # stationary distribution is biased (measured: mean energy +8 on L=4).
    b_right = torch.rand((B, L, L), device=dev, generator=gen) < p_add  # (i,j)-(i,j+1)
    b_down = torch.rand((B, L, L), device=dev, generator=gen) < p_add   # (i,j)-(i+1,j)

    flat = S.reshape(B, D)
    seed_idx = torch.randint(D, (B,), device=dev, generator=gen)
    state = flat.gather(1, seed_idx[:, None]).squeeze(1)     # (B,)
    match = (flat == state[:, None]).view(B, L, L)
    in_c = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
    in_c.reshape(B, D).scatter_(1, seed_idx[:, None], True)
    front = in_c
    # The connected component containing the seed does not depend on traversal
    # order, so we may run several expansion hops per host sync.  A device sync
    # costs ~0.3 ms, far more than the ~0.02 ms of arithmetic per level, so
    # batching hops is what makes this kernel usable at all.
    for _ in range(D // max(hops, 1) + 1):
        for _h in range(hops):
            cand = ((torch.roll(front, -1, 2) & b_right)                      # right
                    | (torch.roll(front, 1, 2) & torch.roll(b_right, 1, 2))   # left
                    | (torch.roll(front, -1, 1) & b_down)                     # down
                    | (torch.roll(front, 1, 1) & torch.roll(b_down, 1, 1)))   # up
            add = cand & match & (~in_c)
            in_c = in_c | add
            front = add
        if not bool(front.any()):
            break

    if kind == "ising_wolff":
        return torch.where(in_c, -S, S)
    new = torch.randint(q - 1, (B,), device=dev, generator=gen)
    new = new + (new >= state).long()      # uniform over the q-1 other states
    return torch.where(in_c, new.view(B, 1, 1), S)


# --------------------------------------------------------------------- api
def run(kind: str, S: torch.Tensor, *, beta: float, J: float = 1.0, h: float = 0.0,
        q: int = 3, sweeps: int = 1, record: bool = True, seed: int = 0) -> dict:
    """Advance ``S`` by ``sweeps`` sweeps.  Returns a dict with the new state and,
    when ``record``, per-chain energy / magnetization series of shape (B, sweeps).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown gpu sampler {kind!r}; expected {KINDS}")
    dev = S.device
    gen = _generator(dev, seed)
    L = S.shape[1]
    if kind.endswith("_cb"):
        if L % 2:
            raise ValueError(
                f"checkerboard samplers need an even L (got {L}): with periodic "
                "boundaries an odd L is not bipartite, so the two sublattices are "
                "not independent and the update is biased.  Use *_wolff for odd L.")
        idx = torch.arange(L, device=dev)
        par = (idx[:, None] + idx[None, :]) % 2
        p_add = None
    else:
        par = None
        p_add = 1.0 - math.exp(-(2.0 * beta * J if kind.startswith("ising") else beta * J))

    es, ms = [], []
    for _ in range(int(sweeps)):
        if kind == "ising_cb":
            S = _ising_cb_sweep(S, beta, J, h, par, gen, dev)
        elif kind == "potts_cb":
            S = _potts_cb_sweep(S, beta, J, q, par, gen, dev)
        else:
            S = _wolff_sweep(S, p_add, kind, q, gen, dev)
        if record:
            es.append(energy(kind, S, J, h))
            ms.append(magnetization(kind, S, q))

    out = {"S": S}
    if record:
        out["energy"] = torch.stack(es, dim=1)
        out["mag"] = torch.stack(ms, dim=1)
    return out


def collect(kind: str, *, L: int, B: int, beta: float, J: float = 1.0, h: float = 0.0,
            q: int = 3, burn_in: int = 500, n_collect: int = 32, collect_every: int = 50,
            device=None, seed: int = 0):
    """Return (samples [n_collect*B, L*L], energies, mags) on CPU."""
    from .. import device as dev_mod

    dev = dev_mod.resolve(device)
    S = init_state(kind, L, B, q, dev, seed)
    if burn_in:
        S = run(kind, S, beta=beta, J=J, h=h, q=q, sweeps=burn_in, record=False, seed=seed)["S"]
    out, ens, mags = [], [], []
    for k in range(n_collect):
        r = run(kind, S, beta=beta, J=J, h=h, q=q, sweeps=collect_every, record=True,
                seed=seed + k + 1)
        S = r["S"]
        out.append(S.reshape(B, L * L).to("cpu"))
        ens.append(r["energy"][:, -1])
        mags.append(r["mag"][:, -1])
    samples = torch.cat(out, dim=0)
    if kind.startswith("ising"):
        # match the token convention used everywhere else: 0/1 int8
        samples = ((samples > 0).to(torch.int8))
    else:
        samples = samples.to(torch.int32)
    return (samples.numpy(), torch.cat(ens).cpu().numpy(), torch.cat(mags).cpu().numpy())
