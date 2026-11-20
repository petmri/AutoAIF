"""Noise-free simulation: how AIF errors propagate into Patlak / extended Tofts parameters.

Produces every number quoted in docs/population_aif_shrinkage.md (v2), sections 3-5.
Synthetic data only (Parker population AIF); reads no files. numpy only.

    python3 docs/sims/aif_error_ktrans.py
"""
import numpy as np

# ---------------------------------------------------------------- setup
DT = 0.1            # s, fine grid for the "true" continuous curves
T_TOTAL = 480.0     # s, scan length (32 frames x 15 s)
T_ARRIVAL = 60.0    # s, bolus arrival
FRAME = 15.0        # s, temporal resolution
HCT = 0.42

t = np.arange(0, T_TOTAL, DT)
n_frames = int(T_TOTAL / FRAME)
tf = (np.arange(n_frames) + 0.5) * FRAME          # frame centre times


def parker(tt, beta=0.1685, first_pass_gain=1.0):
    """Parker et al. 2006 population AIF (blood, mM), converted to plasma."""
    tm = (tt - T_ARRIVAL) / 60.0
    A1, A2, T1, T2, s1, s2 = 0.809, 0.330, 0.17046, 0.365, 0.0563, 0.132
    alpha, s, tau = 1.050, 38.078, 0.483
    gauss = (A1 / (s1 * np.sqrt(2 * np.pi)) * np.exp(-(tm - T1) ** 2 / (2 * s1 ** 2))
             + A2 / (s2 * np.sqrt(2 * np.pi)) * np.exp(-(tm - T2) ** 2 / (2 * s2 ** 2)))
    tail = alpha * np.exp(-beta * tm) / (1 + np.exp(-s * (tm - tau)))
    c = (first_pass_gain * gauss + tail) / (1 - HCT)
    c[tt < T_ARRIVAL] = 0.0
    return c


def shift(c, d):
    k = int(round(d / DT))
    out = np.zeros_like(c)
    out[k:] = c[:len(c) - k]
    return out


def disperse(c, mtt):
    h = np.exp(-np.arange(0, 10 * mtt, DT) / mtt)
    return np.convolve(c, h / h.sum())[:len(c)]


def to_frames(c):
    """Box-car average over each frame (same operator for AIF and tissue)."""
    return c[:n_frames * int(FRAME / DT)].reshape(n_frames, -1).mean(1)


def cumtrapz(y):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * FRAME)
    return out


def conv_exp(c, kep_s):
    out = np.zeros_like(c)
    e = np.exp(-kep_s * FRAME)
    for i in range(1, len(c)):
        out[i] = out[i - 1] * e + 0.5 * FRAME * (c[i] + c[i - 1] * e)
    return out


cp = parker(t)
I_fine = np.cumsum(cp) * DT


def tissue(K, vp, kep=0.0):
    """K, kep in 1/min. kep = 0 gives Patlak."""
    if kep == 0:
        return vp * cp + K / 60 * I_fine
    h = np.exp(-kep / 60 * np.arange(0, T_TOTAL, DT))
    return vp * cp + K / 60 * np.convolve(cp, h)[:len(t)] * DT


def lstsq2(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, np.sqrt(np.mean((y - X @ b) ** 2))


def fit_patlak(ct, ca):
    b, r = lstsq2(np.c_[ca, cumtrapz(ca)], ct)
    return b[1] * 60, b[0], r


def fit_etofts(ct, ca):
    best = None
    for kep in np.concatenate([[0], np.geomspace(1e-3, 3, 200)]):
        b, r = lstsq2(np.c_[ca, conv_exp(ca, kep / 60)], ct)
        if best is None or r < best[2]:
            best = (b[1] * 60, b[0], r)
    return best


def with_delay(fit):
    """Subject-level delay fit: shift the framed AIF (linear interpolation), keep best."""
    def f(ct, ca):
        best = None
        for d in np.arange(-20, 20.01, 1.0):
            res = fit(ct, np.interp(tf - d, tf, ca, left=0.0))
            if best is None or res[2] < best[2]:
                best = res
        return best
    return f


# ---------------------------------------------------------------- AIF errors
peak = cp.max()
c_half = peak * 0.7 / 0.3                        # saturation that cuts the peak by 30 %
first_pass = parker(t) - parker(t, first_pass_gain=0.0)
ERRORS = {
    "true AIF": cp,
    "scale x0.8 (partial volume, S0, T10, Hct)": 0.8 * cp,
    "first-pass area -30%": cp - 0.3 * first_pass,
    "peak saturation (peak -30%, nonlinear)": cp / (1 + cp / c_half),
    "delay +7.5 s": shift(cp, 7.5),
    "venous-like (4 s delay + 4 s dispersion)": disperse(shift(cp, 4), 4),
    "washout rate +30%": parker(t, beta=0.1685 * 1.3),
    "tail level +10% (>2 min after arrival)": cp * np.where(t > T_ARRIVAL + 120, 1.10, 1.0),
}
AIF = {k: to_frames(v) for k, v in ERRORS.items()}


def main():
    print(f"plasma AIF peak: {peak:.2f} mM fine grid, {AIF['true AIF'].max():.2f} mM at 15 s frames")
    print(f"first-pass area / AUC over scan: {first_pass.sum() * DT / I_fine[-1]:.2f}\n")

    # ---- Table 1: sensitivity (sections 3 and 5)
    cases = [("brain, Patlak", to_frames(tissue(0.002, 0.02)), with_delay(fit_patlak)),
             ("lesion, ext Tofts", to_frames(tissue(0.05, 0.03, kep=0.05 / 0.2)), with_delay(fit_etofts))]
    ref = {}
    rows = {k: [] for k in AIF}
    for name, ct, fit in cases:
        K0, v0, _ = fit(ct, AIF["true AIF"])
        ref[name] = (K0, v0)
        for k, ca in AIF.items():
            K, v, r = fit(ct, ca)
            rows[k] += [100 * (K / K0 - 1), 100 * (v / v0 - 1), 100 * r / ct.max()]
    print("Table 1. Change vs fit with the true AIF (delay fitted). resid = RMS residual, % of tissue peak")
    print(f"{'AIF error':44s} | {'brain dK%':>9s} {'dvp%':>7s} {'resid%':>7s} | {'lesion dK%':>10s} {'dvp%':>7s} {'resid%':>7s}")
    for k, r in rows.items():
        print(f"{k:44s} | {r[0]:9.1f} {r[1]:7.1f} {r[2]:7.2f} | {r[3]:10.1f} {r[4]:7.1f} {r[5]:7.2f}")

    # ---- Table 2: the 2x2 matrix M (section 4), fixed delay, exact
    c0 = AIF["true AIF"]
    Xtrue = np.c_[c0, cumtrapz(c0)]
    print("\nTable 2. Patlak, fixed delay: [vp_hat, K_hat] = M [vp, K]  (K in 1/min)")
    print(f"{'AIF error':44s} {'m11':>7s} {'m12(min)':>9s} {'m21(1/min)':>11s} {'m22':>7s} {'rho=m21/m11':>12s}")
    for k, ca in AIF.items():
        if "delay" in k:
            continue
        Xt = np.c_[ca, cumtrapz(ca)]
        M = np.linalg.solve(Xt.T @ Xt, Xt.T @ Xtrue)        # columns: [c, I_per_s]
        m11, m12, m21, m22 = M[0, 0], M[0, 1] / 60, M[1, 0] * 60, M[1, 1]
        print(f"{k:44s} {m11:7.3f} {m12:9.3f} {m21:+11.4f} {m22:7.3f} {m21 / m11:+12.4f}")

    # check exactness across many voxels
    rng = np.random.default_rng(0)
    vps, Ks = rng.uniform(0.005, 0.06, 300), rng.uniform(0, 0.004, 300)
    cts = np.array([to_frames(tissue(K, v)) for K, v in zip(Ks, vps)])
    ca = AIF["peak saturation (peak -30%, nonlinear)"]
    Kh = np.array([fit_patlak(ct, ca)[0] for ct in cts])
    A = np.c_[Ks, vps]
    coef, *_ = np.linalg.lstsq(A, Kh, rcond=None)
    r2 = 1 - np.sum((Kh - A @ coef) ** 2) / np.sum((Kh - Kh.mean()) ** 2)
    print(f"\nexactness check (300 voxels, saturated AIF): K_hat = {coef[0]:.3f} K {coef[1]:+.4f} vp, R^2 = {r2:.6f}")

    # ---- residual-blind direction (section 5)
    vp, K = 0.02, 0.002 / 60
    n = n_frames
    L = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, i + 1):
            L[i, j] += 0.5 * FRAME
            L[i, j - 1] += 0.5 * FRAME
    I0 = cumtrapz(c0)
    D = np.linalg.solve(vp * np.eye(n) + K * L, I0)
    tarr = tf - T_ARRIVAL
    i4 = np.argmin(abs(tarr - 240))
    D *= 0.05 * c0[i4] / D[i4]                                  # +5 % at 4 min
    Kh, vh, r = fit_patlak(to_frames(tissue(0.002, 0.02)), c0 + D)
    rel = [100 * D[np.argmin(abs(tarr - s))] / c0[np.argmin(abs(tarr - s))] for s in (30, 60, 120, 240, 420)]
    print("\nresidual-blind direction, relative AIF error at 0.5/1/2/4/7 min: "
          + ", ".join(f"{x:+.1f}%" for x in rel))
    print(f"  -> brain dK {100 * (Kh / 0.002 - 1):+.1f}%, dvp {100 * (vh / 0.02 - 1):+.1f}%, residual {r:.1e} mM")

    # ---- vessel-voxel anchor and its floor (section 5.5)
    print("\nVessel-voxel check: arterial AIF fitted to a venous voxel (true vp=1, K=0), delay fitted")
    vd = with_delay(fit_patlak)
    for d, mtt in [(4, 4), (6, 6), (4, 8)]:
        Kv, vv, _ = vd(to_frames(disperse(shift(cp, d), mtt)), c0)
        print(f"  correct AIF, vein delay {d} s dispersion {mtt} s: K_vessel {Kv:+.4f}/min, vp {vv:.3f}")
    Kv, vv, _ = vd(to_frames(disperse(shift(cp, 4), 4)), AIF["peak saturation (peak -30%, nonlinear)"])
    print(f"  saturated AIF (peak -30%), vein 4 s/4 s:        K_vessel {Kv:+.4f}/min, vp {vv:.3f}")


if __name__ == "__main__":
    main()
