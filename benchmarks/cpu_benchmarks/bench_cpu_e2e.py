"""End-to-end CPU benchmark, identical protocol for every method:
    Zt.grad = None; loss(Zt, yt, reduction='sum').backward()
Interleaved timings.  --consume additionally reads the whole gradient (Zt.grad.sum()) inside the timed
region for every method.  Writes bench_cpu_e2e[_consume].json (or --out)."""
import numpy as np, torch, time, json, os, argparse, platform
from fwcpp import familywise_loss, fw_loss_grad
p = argparse.ArgumentParser(); p.add_argument("--N", type=int, default=16)
p.add_argument("--kmax", type=float, default=1e7); p.add_argument("--stds", default="0.01,0.1,1,10")
p.add_argument("--consume", action="store_true"); p.add_argument("--quick", action="store_true"); p.add_argument("--out", default=""); p.add_argument("--kmin", type=float, default=100)
p.add_argument("--dist", default="gaussian", choices=["gaussian", "laplace", "mixture"]); p.add_argument("--mix-mean", type=float, default=2.0)
a = p.parse_args()
C = a.consume
def fw(Zt, yt):
    Zt.grad = None; familywise_loss(Zt, yt, "sum").backward()
    if C: Zt.grad.sum()
def ce(Zt, yt):
    Zt.grad = None; torch.nn.functional.cross_entropy(Zt, yt, reduction="sum").backward()
    if C: Zt.grad.sum()
def naive(Zt, yt):
    Zt.grad = None
    z_at_y = Zt[torch.arange(Zt.shape[0]), yt]
    h = torch.sort(Zt, dim=-1, descending=True).values.cumsum(-1).sub(1)
    h = h / torch.arange(1, Zt.shape[1] + 1, dtype=Zt.dtype)[None, :]
    (1 - z_at_y + h.max(-1).values).sum().backward()
    if C: Zt.grad.sum()
Ks = [int(round(k)) for k in np.logspace(np.log10(a.kmin), np.log10(a.kmax), int(round(2 * np.log10(a.kmax / a.kmin))) + 1)]
if a.quick: Ks = Ks[::2]
res = {"N": a.N, "threads": torch.get_num_threads(), "cpu": platform.platform() + ' / ' + platform.processor(),
       "torch": torch.__version__, "consume": C, "Ks": Ks, "dist": a.dist, "mix_mean": a.mix_mean, "runs": {}}
rng = np.random.default_rng(0)
def gen(K, std):
    """Logits [N, K].  gaussian: N(0, std^2).  laplace: Laplace with mean 0 and standard deviation std.
    mixture: each logit is drawn from N(0, 1) or from N(mix_mean, std^2) with probability 1/2 each."""
    if a.dist == "gaussian": return rng.standard_normal((a.N, K)) * std
    if a.dist == "laplace": return rng.laplace(0.0, std / np.sqrt(2.0), (a.N, K))
    pick = rng.random((a.N, K)) < 0.5
    return np.where(pick, rng.standard_normal((a.N, K)), a.mix_mean + std * rng.standard_normal((a.N, K)))
for std in [float(s) for s in a.stds.split(",")]:
    runs = {}
    for K in Ks:
        Z = gen(K, std).astype(np.float32); y = rng.integers(0, K, a.N)
        Zt = torch.from_numpy(Z).requires_grad_(True); yt = torch.from_numpy(y)
        fns = {"FW (ours)": fw, "CE (PyTorch)": ce}
        if K <= 1e6: fns["FW naive (PyTorch, App. D)"] = naive
        if K <= 1e6:                                   # same loss and gradient as the definition (float64 reference)
            Zt.grad = None; l1 = familywise_loss(Zt, yt, "sum"); l1.backward(); g1 = Zt.grad.clone()
            Zd = Zt.detach().double().requires_grad_(True); naive(Zd, yt)
            hd = torch.sort(Zd.detach(), dim=-1, descending=True).values.cumsum(-1).sub(1) / torch.arange(1, K + 1, dtype=torch.float64)
            l2 = (1 - Zd.detach()[torch.arange(a.N), yt] + hd.max(-1).values).sum().item()
            assert abs(l1.item() - l2) <= 1e-5 * max(1.0, abs(l2)), ("loss mismatch", l1.item(), l2)
            assert torch.allclose(g1.double(), Zd.grad, atol=1e-6), "gradient mismatch"
            del Zd, g1
        kbar = float(fw_loss_grad(Zt.detach(), yt)[2].double().mean())
        ts = {k: [] for k in fns}
        for f in fns.values(): f(Zt, yt)
        budget = (0.6 if a.quick else 2) if K < 1e6 else (2 if a.quick else 6)
        pair = [("FW (ours)", fw), ("CE (PyTorch)", ce)]
        t_end = time.perf_counter() + budget; rnd = 0
        while time.perf_counter() < t_end or rnd < 6:          # FW and CE interleaved, order alternates every round
            for k, f in (pair if rnd % 2 == 0 else pair[::-1]):
                t = time.perf_counter(); f(Zt, yt); ts[k].append(time.perf_counter() - t)
            rnd += 1
        if K <= 1e6:                                            # the naive version is timed separately
            t_end = time.perf_counter() + budget / 2
            while time.perf_counter() < t_end or len(ts["FW naive (PyTorch, App. D)"]) < 3:
                t = time.perf_counter(); naive(Zt, yt); ts["FW naive (PyTorch, App. D)"].append(time.perf_counter() - t)
        m = {k: float(np.median(v)) for k, v in ts.items()}
        line = f"std={std:g} K={K:>9d} reps={len(ts['FW (ours)']):>4d} CE/FW={m['CE (PyTorch)'] / m['FW (ours)']:6.2f}"
        for k, v in ts.items():
            v = np.array(v)
            runs.setdefault(k, []).append({"K": K, "med": float(np.median(v)), "q20": float(np.quantile(v, .2)),
                                           "q80": float(np.quantile(v, .8)), "reps": len(v), "kbar": kbar})
            line += f" | {k[:8]}={np.median(v) * 1e3:.3f}ms"
        print(line, flush=True)
    res["runs"][str(std)] = runs
    if a.out or not a.quick: json.dump(res, open(a.out or ("bench_cpu_e2e_consume.json" if C else "bench_cpu_e2e.json"), "w"), indent=1)
