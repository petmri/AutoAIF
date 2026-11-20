# Population-informed AIF estimation by empirical Bayes shrinkage

**Status:** design draft, 2026-11-20. Nothing here is implemented or validated yet.
Results and decisions from testing this go in eLabFTW (team VELA), not in this file.

**Rendering:** equations use GitHub/KaTeX math (`$...$` inline, `$$...$$` display).
They render on GitHub and in the VS Code Markdown preview (`Ctrl+Shift+V`).

---

## 1. The idea in one paragraph

When AutoAIF runs on a cohort, each subject's AIF is estimated on its own from a
handful of voxels. Those estimates are noisy, especially in amplitude (partial
volume, inflow, few voxels). But AIFs from the same population, same contrast
agent and same protocol should look alike. Borrowing strength across subjects, as
ComBat does across features, should cut that noise. Each subject's AIF is pulled
toward the population average **in proportion to how unreliable it is**. Then we can
use the population model again at the voxel level, so that among the voxels the
network rates highly, we prefer the ones that fit a plausible AIF. Finally we
iterate: re-estimate the population, re-shrink, re-select. This document puts that
on a formal footing. It shows that the whole procedure is an EM algorithm for a
three-level hierarchical model (voxel → subject → population), works out the
shrinkage formulas, and points out one failure mode the naive iteration hits
(collapse to the population mean) along with how to avoid it.

---

## 2. Notation and what AutoAIF computes today

| Symbol | Meaning |
|---|---|
| $s$ | site (scanner/protocol) |
| $j = 1,\dots,J$ | subject (or session) |
| $v$ | voxel |
| $t = 1,\dots,T$ | time frame |
| $y_{jv}(t)$ | signal of voxel $v$ in subject $j$ (later: relative enhancement or concentration) |
| $m_{jv} \in [0,1]$ | soft mask value from the network for voxel $v$ |
| $c_j(t)$ | true AIF of subject $j$ |
| $\hat c_j(t)$ | estimated AIF |

The network (`model_vif.py`, `computeCurve`) outputs a soft mask and forms the
AIF as the mask-weighted mean signal:

$$
\hat c_j(t) = \frac{\sum_v m_{jv}\, y_{jv}(t)}{\sum_v m_{jv}}
$$

At inference (`main_vif.py`, `inference_mode`) the mask is resampled back to native
resolution, and the top-20 (or top-5) voxels by $m_{jv}$ are also saved as a hard mask.
In both cases **each subject is estimated independently**. No information is shared
across the cohort.

Two details matter later:

- Training normalizes curves by their first frame (`y / y[:, 0]`), so the network
  cares about **shape** rather than absolute scale.
- Preprocessing resamples every series to $T = 32$ frames regardless of the real
  temporal resolution. A population model has to work in **physical time** (seconds
  from bolus arrival), not frame index (see §9.6).

---

## 3. How ComBat / neuroCombat works

References: Johnson, Li & Rabinovic (2007) for the method; Fortin et al. (2017,
2018) for neuroCombat. The formulas below follow the neuroCombat code
(`aprior`, `bprior`, `postmean`, `postvar`, `it_sol`).

### 3.1 The model

Index batch (site) $i$, sample (subject) $j$ within batch, feature $v$ (voxel,
ROI, gene). ComBat assumes

$$
y_{ijv} = \alpha_v + \mathbf{x}_{ij}^\top \boldsymbol\beta_v + \gamma_{iv} + \delta_{iv}\,\varepsilon_{ijv},
\qquad \varepsilon_{ijv} \sim \mathcal N(0, \sigma_v^2)
$$

- $\alpha_v$: feature grand mean.
- $\mathbf x_{ij}^\top \boldsymbol\beta_v$: biological covariates (age, sex,
  diagnosis) whose effects must be **preserved**.
- $\gamma_{iv}$: additive batch effect (shift in location).
- $\delta_{iv}$: multiplicative batch effect (change in scale).

The goal is to estimate $\gamma_{iv}$ and $\delta_{iv}$ and remove them.

### 3.2 Step 1: standardize

Fit $\alpha_v, \boldsymbol\beta_v$ and batch indicators by least squares,
feature by feature. Estimate a pooled residual variance $\hat\sigma_v^2$. Then

$$
z_{ijv} = \frac{y_{ijv} - \hat\alpha_v - \mathbf x_{ij}^\top \hat{\boldsymbol\beta}_v}{\hat\sigma_v}
$$

After this, every feature is on a common scale, so batch effects of different
features can be compared with each other. **That comparability is what makes
the next step possible.**

### 3.3 Step 2: naive per-batch, per-feature estimates

For batch $i$ with $n_i$ samples:

$$
\hat\gamma_{iv} = \frac{1}{n_i}\sum_j z_{ijv},
\qquad
\hat\delta^2_{iv} = \frac{1}{n_i - 1}\sum_j \left(z_{ijv} - \hat\gamma_{iv}\right)^2
$$

With small $n_i$ these are noisy, and there are thousands of them.

### 3.4 Step 3: the empirical Bayes prior, shared across features

The key modeling choice: **within one batch, the batch effects of different
features are exchangeable draws from a common distribution.**

$$
\gamma_{iv} \sim \mathcal N(\gamma_i,\ \tau_i^2),
\qquad
\delta^2_{iv} \sim \text{InvGamma}(\lambda_i,\ \theta_i)
$$

The hyperparameters $(\gamma_i, \tau_i^2, \lambda_i, \theta_i)$ are **per batch, shared over all features**.
They are estimated from the data by the method of moments across features (that
is what makes it *empirical* Bayes):

$$
\bar\gamma_i = \frac{1}{V}\sum_v \hat\gamma_{iv},
\qquad
\bar\tau_i^2 = \frac{1}{V-1}\sum_v \left(\hat\gamma_{iv} - \bar\gamma_i\right)^2
$$

For the inverse gamma, let $M$ and $S^2$ be the mean and variance of
$\{\hat\delta^2_{iv}\}_v$. If $X \sim \text{InvGamma}(\lambda, \theta)$ then

$$
\mathbb E[X] = \frac{\theta}{\lambda - 1},
\qquad
\operatorname{Var}[X] = \frac{\theta^2}{(\lambda-1)^2(\lambda-2)}
$$

so $\operatorname{Var}/\mathbb E^2 = 1/(\lambda - 2)$. Solving the two moment equations:

$$
\bar\lambda_i = \frac{M^2 + 2S^2}{S^2},
\qquad
\bar\theta_i = \frac{M^3 + M S^2}{S^2}
$$

These are exactly `aprior` and `bprior` in neuroCombat.

### 3.5 Step 4: posterior estimates (the shrinkage)

**Location.** Condition on $\delta^2_{iv}$. The samples give
$\bar z_{iv} = \hat\gamma_{iv}$ with likelihood $\mathcal N(\gamma_{iv}, \delta^2_{iv}/n_i)$.
Combined with the normal prior, a normal-normal conjugate update gives

$$
\gamma^{\ast}_{iv}
= \frac{\dfrac{\bar\gamma_i}{\bar\tau_i^2} + \dfrac{n_i\,\hat\gamma_{iv}}{\delta^2_{iv}}}{\dfrac{1}{\bar\tau_i^2} + \dfrac{n_i}{\delta^2_{iv}}}
= \frac{n_i \bar\tau_i^2\, \hat\gamma_{iv} + \delta^{2}_{iv}\, \bar\gamma_i}{n_i \bar\tau_i^2 + \delta^{2}_{iv}}
$$

This is a precision-weighted average: data precision $n_i/\delta^2$ against prior
precision $1/\tau^2$. Written as a shrinkage weight:

$$
\gamma^{\ast}_{iv} = w_{iv}\,\hat\gamma_{iv} + (1 - w_{iv})\,\bar\gamma_i,
\qquad
w_{iv} = \frac{\bar\tau_i^2}{\bar\tau_i^2 + \delta^2_{iv}/n_i}
$$

$w \to 1$ (trust the data) when the batch has many samples or the features truly
differ a lot ($\tau^2$ large). $w \to 0$ (trust the prior) when the data are noisy.

**Scale.** Condition on $\gamma_{iv}$. The inverse-gamma prior is conjugate for a
normal variance, so the conditional posterior is
$\text{InvGamma}\left(\lambda_i + n_i/2,\ \theta_i + \tfrac12 \sum_j (z_{ijv} - \gamma_{iv})^2\right)$,
whose mean is

$$
\delta^{2\ast}_{iv} = \frac{\bar\theta_i + \tfrac12 \sum_j \left(z_{ijv} - \gamma^{\ast}_{iv}\right)^2}{n_i/2 + \bar\lambda_i - 1}
$$

These are `postmean` and `postvar`.

**Iteration.** The two updates depend on each other ($\gamma^\ast$ needs $\delta^2$,
$\delta^{2\ast}$ needs $\gamma$). The priors are independent (not the joint
normal-inverse-gamma), so there is no closed form for the joint posterior.
neuroCombat (`it_sol`) starts from the naive estimates and alternates the two
conditional posterior means until the relative change is below $10^{-4}$. This is a
deterministic fixed-point scheme (like iterated conditional modes). It is not a
full posterior. It works well because it usually converges in a few iterations.

There is also a nonparametric variant (`parametric=False`) that replaces the
normal/inverse-gamma priors with the empirical distribution of the other
features' $(\hat\gamma, \hat\delta^2)$ and computes the posterior mean by weighting
each by its likelihood.

### 3.6 Step 5: adjust the data

$$
y^{\ast}_{ijv} = \frac{\hat\sigma_v}{\delta^{\ast}_{iv}}\left(z_{ijv} - \gamma^{\ast}_{iv}\right) + \hat\alpha_v + \mathbf x_{ij}^\top \hat{\boldsymbol\beta}_v
$$

### 3.7 Why it works: James–Stein and empirical Bayes

If you estimate $p \ge 3$ normal means at once with squared-error loss, the
separate sample means are **inadmissible**. James–Stein shrinkage toward a common
point has lower total risk for every value of the true means (James & Stein 1961).
Efron & Morris (1973, 1975) showed that the James–Stein estimator is what you get
from an empirical Bayes argument: assume the means come from a common normal,
estimate that normal from the data, and take posterior means. ComBat applies this
to thousands of batch-effect parameters.

Two points carry over to our problem and matter a lot:

1. **The guarantee is about total error across units, not each unit.** A unit
   that truly is an outlier gets biased toward the mean. On average you win.
2. **In ComBat the shrunk quantity is a nuisance** (the scanner effect), which you
   then remove. **For us, the shrunk quantity is the signal we want** (each
   subject's real AIF). That changes how much bias is acceptable (§9.1).

---

## 4. Mapping ComBat onto the AIF problem

The analogy works, but it is **transposed**:

| ComBat | Population AIF |
|---|---|
| batch $i$ | subject $j$ |
| samples $j$ within a batch | voxels $v$ within the subject's arterial ROI |
| feature $v$ | AIF component $k$ (a curve parameter or FPCA score) |
| $\hat\gamma_{iv}$ = batch mean of feature | $\hat\xi_{jk}$ = subject's component $k$, averaged over voxels |
| $\hat\delta^2_{iv}$ = within-batch variance | $\hat\delta^2_{jk}$ = scatter of component $k$ across the subject's voxels |
| prior shared **across features**, per batch | prior shared **across subjects**, per component |
| $n_i$ = samples in batch | $n_j$ = (effective) number of arterial voxels |
| covariates $\mathbf x$ preserved | covariates (dose/kg, age, hematocrit, cardiac output, site) set the prior mean |

So the textbook model for us is the **hierarchical normal model** (the "eight
schools" model, Gelman et al. BDA3 ch. 5), equivalently a linear mixed model with
subject random effects. ComBat is one well-known application of the same math.
The only real structural difference is which index the prior pools over.

---

## 5. Representing an AIF so that shrinkage makes sense

Shrinkage needs a representation $\boldsymbol\xi_j \in \mathbb R^K$ of each AIF in
which (a) a Gaussian prior is reasonable, (b) components are roughly independent
across subjects (so a diagonal prior is defensible, like ComBat's per-feature
treatment), and (c) every subject is on a common time axis and scale.

**Prerequisites, whichever representation we pick:**

- **Scale:** relative enhancement $(S(t) - S_0)/S_0$ at minimum. Concentration
  (needs $T_{10}$, flip angle, $r_1$) is better because then amplitude is
  physiological rather than sequence-dependent.
- **Time:** physical seconds, aligned to bolus arrival time (BAT). BAT itself
  (injection timing) is a nuisance. Keep it per subject; don't shrink it.

**Options:**

1. **Raw time samples** ($K = T$). Simple, but the components are strongly
   correlated, so a diagonal prior is wrong and a full $T\times T$ covariance
   needs many subjects. Not recommended alone.
2. **Parametric model** (e.g. the Parker 2006 population AIF form: two Gaussians plus
   a sigmoid-modulated exponential; or a gamma-variate plus washout). The parameters
   can be interpreted (peak amplitude, width, recirculation, washout rate) and can
   be log-transformed to look normal. The drawback: nonlinear fits, correlated
   parameters, and model misfit ends up in the residuals.
3. **Functional PCA (FPCA)** on aligned curves:
   $$
   c_j(t) \approx \mu(t) + \sum_{k=1}^{K} \xi_{jk}\, \phi_k(t)
   $$
   Scores are uncorrelated across subjects **by construction**, so a diagonal
   prior is justified. The model is linear, so everything below is closed form.
   This is also exactly the setting of PACE (Yao, Müller & Wang 2005), whose score
   estimates *are* shrinkage (BLUP) estimates.

**Recommendation:** FPCA for the method itself, because it is the cleanest match
to ComBat's math. Fit a parametric model on top only for reporting and checking
that results are physiologically plausible.

---

## 6. Stage 1: post-hoc empirical Bayes shrinkage of subject AIFs

This stage takes the voxels the network already chose (fixed weights $r_{jv}$, for
example $m_{jv}$ or the top-$N$ indicator) and only shrinks the resulting AIFs.

### 6.1 Voxel-level summary per subject

Project each selected voxel's curve onto the FPCA basis (orthonormal
$\phi_k$):

$$
\hat\xi_{jvk} = \left\langle y_{jv} - \mu,\ \phi_k \right\rangle
$$

Then, exactly as in ComBat step 2, with weights $r_{jv}$ and
$n_j = \sum_v r_{jv}$:

$$
\hat\xi_{jk} = \frac{1}{n_j}\sum_v r_{jv}\,\hat\xi_{jvk},
\qquad
\hat\delta^2_{jk} = \frac{1}{n_j - 1}\sum_v r_{jv}\left(\hat\xi_{jvk} - \hat\xi_{jk}\right)^2
$$

### 6.2 Hierarchical model

$$
\hat\xi_{jk} \mid \xi_{jk} \sim \mathcal N\!\left(\xi_{jk},\ \frac{\delta^2_{jk}}{n_j^{\text{eff}}}\right)
$$

$$
\xi_{jk} \sim \mathcal N\!\left(\mu_{s(j),k} + \mathbf x_j^\top \boldsymbol\beta_k,\ \tau_k^2\right),
\qquad
\delta^2_{jk} \sim \text{InvGamma}(\lambda_k,\ \theta_k)
$$

- The prior mean is per site $s(j)$ (and per covariate). Sites differ in dose
  protocol, injection rate and temporal resolution, so shrinking a site-A AIF
  toward a pooled mean that includes site B would introduce bias. This plays the
  role of ComBat's protected $\mathbf x^\top\boldsymbol\beta$.
- $n_j^{\text{eff}} \le n_j$ is the **effective** number of independent voxels
  (see §9.4). This matters: it sets how much we trust each subject.

### 6.3 Posterior (shrunk) scores

Same algebra as §3.5:

$$
\xi^{\ast}_{jk} = w_{jk}\,\hat\xi_{jk} + (1 - w_{jk})\left(\hat\mu_{s(j),k} + \mathbf x_j^\top\hat{\boldsymbol\beta}_k\right),
\qquad
w_{jk} = \frac{\hat\tau_k^2}{\hat\tau_k^2 + \delta^{2\ast}_{jk}/n_j^{\text{eff}}}
$$

$$
\delta^{2\ast}_{jk} = \frac{\hat\theta_k + \tfrac12\sum_v r_{jv}\left(\hat\xi_{jvk} - \xi^{\ast}_{jk}\right)^2}{n_j/2 + \hat\lambda_k - 1}
$$

The shrunk AIF is $c^{\ast}_j(t) = \hat\mu(t) + \sum_k \xi^{\ast}_{jk}\phi_k(t)$, and the posterior
variance of each score is

$$
V_{jk} = \left(\frac{1}{\hat\tau_k^2} + \frac{n_j^{\text{eff}}}{\delta^{2\ast}_{jk}}\right)^{-1}
$$

which gives every AIF a pointwise credible band for free. That is useful in itself
for propagating AIF uncertainty into $K^{\text{trans}}$.

**How the behavior reads:**

- **Subject with many clean arterial voxels:** $n^{\text{eff}}$ large, $w \approx 1$,
  so the AIF is essentially unchanged.
- **Subject with 3 partial-volumed voxels:** $w$ small, so the AIF is pulled toward
  the site mean, but only in the components where the voxels disagree with each
  other.
- **Component where subjects genuinely differ a lot** (large $\tau_k^2$, for
  example peak width): little shrinkage. **Component where subjects are all alike**
  (small $\tau_k^2$): strong shrinkage. The data decide this; we don't tune it by
  hand.
- **Variance shrinkage** (the $\delta^2$ part) stabilizes $w$ itself when a subject
  has so few voxels that its own scatter estimate is meaningless (top-5 voxels).

### 6.4 Two extremes as special cases

- $w_{jk} = 1$ for all $j,k$ gives individual AIFs (current AutoAIF).
- $w_{jk} = 0$ for all $j,k$ gives a site-specific population AIF.

The DCE literature has argued individual vs population AIF for years. Empirical
Bayes is the principled interpolation between the two, and the data choose where
to sit.

---

## 7. Stage 2: population-informed voxel selection (the mixture model)

This formalizes "among voxels the network rates highly, choose those that agree
better with the population".

### 7.1 Model

Each candidate voxel either is arterial ($z_{jv} = 1$) or is not (venous, tissue,
partial volume with tissue, noise):

$$
z_{jv} \sim \text{Bernoulli}(\pi_{jv}),
\qquad
\pi_{jv} = h(m_{jv})
$$

$$
y_{jv} \mid z_{jv} = 1 \sim \mathcal N\!\left(a_{jv}\, c_j,\ \sigma_j^2 I\right),
\qquad
y_{jv} \mid z_{jv} = 0 \sim g_j
$$

- **The network score is the prior probability of being arterial.** $h$ is a
  calibration map, estimated once on data with manual masks (for example Platt
  scaling), because a sigmoid output is not a calibrated probability.
- $a_{jv} \in (0, 1]$ is an optional per-voxel partial-volume scale. With $a$ in
  the model, voxel agreement is judged on **shape**, and amplitude is handled
  separately (§9.2).
- $g_j$ is a broad background density (a wide Gaussian around the subject's mean
  tissue curve, or a second FPCA-based component for "venous-like").
- $c_j$ has the prior from §6: $c_j = \mu + \Phi\boldsymbol\xi_j$ with
  $\boldsymbol\xi_j \sim \mathcal N(\mu_s, \operatorname{diag}(\tau^2))$.

### 7.2 EM within one subject (hyperparameters held fixed)

**E-step (voxel responsibilities):**

$$
r_{jv} = \frac{\pi_{jv}\,\mathcal N\!\left(y_{jv};\ a_{jv} c_j,\ \sigma_j^2 I\right)}{\pi_{jv}\,\mathcal N\!\left(y_{jv};\ a_{jv} c_j,\ \sigma_j^2 I\right) + (1 - \pi_{jv})\, g_j(y_{jv})}
$$

**M-step (subject AIF):** maximize
$\sum_v r_{jv}\log\mathcal N(y_{jv}; c_j, \sigma_j^2 I) + \log p(\boldsymbol\xi_j)$
(with $a = 1$ for clarity). The result is **exactly the shrinkage formula of §6.3**
with $n_j = \sum_v r_{jv}$:

$$
\xi^{\ast}_{jk} = \frac{\tau_k^2\, n_j\, \bar\xi_{jk} + \sigma_j^2\, \mu_{s,k}}{\tau_k^2\, n_j + \sigma_j^2},
\qquad
\bar\xi_{jk} = \frac{\sum_v r_{jv}\, \hat\xi_{jvk}}{\sum_v r_{jv}}
$$

This ties the two ideas together: **voxel selection and Bayesian shrinkage are the
E- and M-steps of one algorithm.** The current AutoAIF estimator is the special
case $r_{jv} = m_{jv}$ (no E-step) and $\tau^2 \to \infty$ (no prior).

### 7.3 The subtle point: agree with *what*?

In the E-step a voxel is judged against the **subject's own current AIF**
$c_j$, which is already shrunk. It is *not* judged against the population mean
directly. The population enters only through the prior on $c_j$. This matters:

- A subject whose real AIF is unusual but well supported (many consistent voxels)
  keeps its unusual AIF, because the voxels agree with each other and $w \approx 1$.
- A subject with conflicting candidate voxels (some arterial, some venous) gets
  pulled toward the population, and the prior **breaks the tie** in favor of the
  arterial-looking cluster. This is where we expect the method to help most.

Selecting voxels by distance to the population mean would discard real
between-subject variation. That is the naive reading of the idea, and we should
avoid it.

### 7.4 Hard selection is classification EM

"Pick the top-$N$ by $m$, then keep those closest to the curve" is the hard version:
$r_{jv} \in \{0,1\}$ by thresholding the posterior. This is *classification EM*
(Celeux & Govaert 1992). It is simpler and often fine for point estimates, but it is
known to **underestimate variances**: truncating to the conforming voxels makes the
within-subject scatter $\hat\delta^2$ too small, which inflates $n/\delta^2$ and makes
the subject look more certain than it is. Soft $r_{jv}$ avoids that. If we
want hard masks for downstream tools, compute them at the end from the converged
$r_{jv}$.

---

## 8. Stage 3: iterate, which is EM for the population hyperparameters

The outer loop ("update the population average, then update the shrinkage") is
the M-step for the hyperparameters $\Theta = \{\mu_s, \boldsymbol\beta, \tau^2, \lambda, \theta, \mu(t), \Phi\}$.

### 8.1 The correct update

For the variance components the EM update is

$$
\hat\tau_k^{2,\text{new}} = \frac{1}{J}\sum_j\left[\left(\xi^{\ast}_{jk} - \hat\mu_{s(j),k} - \mathbf x_j^\top\hat{\boldsymbol\beta}_k\right)^2 + V_{jk}\right]
$$

The $V_{jk}$ term (the posterior variance of each subject's score) is the
important part. It accounts for the fact that $\xi^{\ast}_{jk}$ are *shrunk*
estimates, which are less spread out than the true $\xi_{jk}$. Iterating this
increases the marginal likelihood $p(\text{all voxels} \mid \Theta)$ at every
step (Dempster, Laird & Rubin 1977), so the loop has a well-defined objective and
converges.

### 8.2 The naive update collapses

If you instead re-estimate the population variance from the shrunk AIFs as if they
were data (dropping $V_{jk}$), the loop drifts toward "every subject equals the
population mean". This is easy to show in the scalar case. Let every subject
have sampling variance $e = \delta^2/n$ and let $s^2$ be the observed spread of
$\hat\xi_j$. With $w = \tau^2/(\tau^2 + e)$:

**Naive:**
$$
\tau^2_{\text{new}} = w^2 s^2
\quad\Longrightarrow\quad
\text{fixed points: } \tau^2 = 0 \ \text{ and } \ (\tau^2 + e)^2 = \tau^2 s^2
$$

$\tau^2 = 0$ is always a **stable** fixed point: near zero,
$\tau^2_{\text{new}} \approx \tau^4 s^2/e^2 \ll \tau^2$. Nonzero fixed points exist only
if $s^2 \ge 4e$, and even then the upper one is biased low.

**EM:**
$$
\tau^2_{\text{new}} = w^2 s^2 + w e
\quad\Longrightarrow\quad
(\tau^2 + e)^2 = \tau^2 s^2 + e(\tau^2 + e)
\quad\Longrightarrow\quad
\tau^2 = s^2 - e
$$

which is the correct moment estimate: the observed spread minus the noise.

A quick numerical check ($J = 60$, 500 iterations) agrees:

| Case | Truth / MoM $\tau^2$ | Naive loop | EM loop |
|---|---|---|---|
| $e = 0.5$, $s^2 = 1.17$ ($s^2 < 4e$) | 0.67 | **0.00** from any start | 0.67 |
| $e = 0.1$, $s^2 = 0.78$ ($s^2 \ge 4e$) | 0.68 | 0.56 from a high start; **0.00** from a low start | 0.68 |

Voxel re-selection makes this worse: picking voxels that agree with the shrunk
curve reduces $s^2$ and $\hat\delta^2$ further every round. **The safeguards are not
optional:** soft responsibilities (§7.4), the $V_{jk}$ term, and judging voxels
against the subject's curve rather than the population mean (§7.3).

### 8.3 The full algorithm

```
initialize: r_jv = m_jv (current AutoAIF), fit mu(t), Phi by FPCA on current AIFs,
            hyperparameters by method of moments (as ComBat)
repeat until marginal log-likelihood changes < tol:
    for each subject j:                       # inner loop, hyperparameters fixed
        repeat:
            E-step:  r_jv  from  pi_jv, current c_j, sigma_j^2, g_j        (§7.2)
            M-step:  xi*_jk, delta2*_jk, V_jk  (ComBat-style it_sol)       (§6.3)
    hyper M-step: mu_s, beta, tau_k^2 (with V_jk), lambda_k, theta_k       (§8.1)
    (optionally) re-fit mu(t), Phi on posterior curves, with care
output: c*_j(t), pointwise bands from V_jk, final r_jv (soft voxel weights)
```

Re-fitting the FPCA basis inside the loop is the least principled step. Freezing
the basis after the first pass is the safe default.

---

## 9. Where the assumptions break

### 9.1 Real between-subject variation

AIFs genuinely differ: cardiac output, hematocrit, injection rate, body size,
vascular disease. Empirical Bayes keeps that variation only to the extent that
each subject's data are precise enough to show it. For downstream pharmacokinetics,
$K^{\text{trans}}$ and $v_p$ scale roughly as $1/\text{AIF amplitude}$, so any bias in amplitude
goes straight into the parameters.

- Put known drivers in the prior mean ($\mathbf x_j$): dose/kg, age, sex,
  hematocrit if available, site.
- **Group studies:** if disease status changes AIF shape (for example through cardiac
  output) and we shrink everyone toward a pooled mean, we partly erase a real
  group difference in AIF and **move it into $K^{\text{trans}}$**. Including group in
  $\mathbf x_j$ avoids this but makes the prior depend on the label being studied.
  This needs a decision (§11).

### 9.2 Shared bias is not fixed by shrinkage

Shrinkage reduces **variance** across subjects. It cannot remove a bias that every
subject shares. Partial volume underestimates peak amplitude in almost everyone,
so the population mean underestimates it too, and shrinking toward it doesn't
help. Amplitude needs an external anchor: a venous output function scaled to the
arterial curve, a physiological constraint (for example the first-pass area
implied by dose and cardiac output), or modeling $a_{jv}$ explicitly and taking the
amplitude from the voxels with the largest $a$.

### 9.3 Non-Gaussian populations and outliers

A subject with a failed injection or extravasation should not pull the population
mean, and should not be shrunk into looking normal. Options: a Student-$t$ prior
on $\xi_{jk}$ (heavier tails shrink outliers less), or the ComBat nonparametric
prior (§3.5). Also flag subjects with very low marginal likelihood for manual
review instead of silently shrinking them.

### 9.4 Correlated voxels: effective $n$

Neighboring voxels are correlated (point spread function, and AutoAIF's own
resample to $256\times256\times32$ and back), and their errors share systematic parts
(the same partial-volume pattern, the same inflow). Using the raw voxel count
makes subjects look too precise ($w$ too close to 1, too little shrinkage).
Estimate $n^{\text{eff}}$ by a spatial block bootstrap of $\hat\xi_{jk}$ over voxels, or
treat connected clusters (vessel segments) rather than voxels as the units.

### 9.5 Few subjects per site

The hyperparameters are themselves estimated. With small $J$ per site,
$\hat\tau_k^2$ is uncertain, and plug-in empirical Bayes understates uncertainty.
Options: share $\tau_k^2$ across sites and let only $\mu_s$ vary by site
(partial pooling of sites, a 3-level model); or full Bayes (PyMC/Stan) with
weak hyperpriors. Leave-one-out hyperparameters (each subject's prior estimated
without it) prevent a subject from pulling its own prior, which matters when $J$ is small.

### 9.6 Time axis

The network sees 32 resampled frames regardless of acquisition length or temporal
resolution. Curves from different sites are only comparable after mapping to
seconds and aligning on bolus arrival. The population model needs scan timing
metadata per site. Very coarse temporal resolution also smooths the peak, which is
a site-level effect for $\mu_s$ to absorb, not something to shrink away.

---

## 10. How we would know it works

No in-vivo ground truth exists, so we need several complementary checks:

1. **Simulation with known truth.** Synthetic DCE volumes with known AIFs drawn
   from a realistic population, with partial volume, noise and venous
   contamination added, at varying numbers of arterial voxels. Measure AIF error
   and $K^{\text{trans}}$ error for: individual (current), population, Stage 1, Stage 1+2.
   Expect the biggest gains for low-$n$ subjects and no loss for high-$n$ subjects.
2. **Test–retest / longitudinal reproducibility.** The data include repeat
   timepoints (e.g. `*_1st_timepoint`). Within-subject coefficient of variation of
   AIF peak, AUC and tissue $K^{\text{trans}}$ across sessions should drop.
3. **Agreement with manual AIFs** from the training masks (held-out subjects
   only).
4. **Held-out voxel prediction.** Split each subject's arterial voxels in two
   halves. Estimate the AIF from half A (with and without shrinkage) and score the
   predictive likelihood of half B. This is a direct, label-free test that
   shrinkage improves the estimate.
5. **Physiological plausibility.** $v_p$ in normal white matter and gray matter, and
   $K^{\text{trans}}$ in tissue with an intact barrier should have tighter
   distributions without shifting their means.
6. **Variance decomposition first (Stage 0).** Before building anything, measure
   how much of the observed AIF variance across subjects is within-subject voxel
   scatter versus real between-subject spread. If the typical $w_{jk}$ is above 0.95,
   shrinkage won't change anything and Stage 2 is where any gain would come from.
   This is cheap and tells us whether to proceed.

---

## 11. Open questions / decisions

- [ ] Representation: FPCA scores (recommended), Parker-model parameters, or both?
- [ ] Signal scale: relative enhancement now, concentration later? Do we have $T_{10}$
      maps for all sites?
- [ ] Prior structure: per-site means with a shared $\tau^2$, or a full 3-level model?
- [ ] Covariates in the prior mean: which are available cohort-wide (dose, weight,
      hematocrit, age)?
- [ ] Group studies: include diagnosis in $\mathbf x_j$ or not (§9.1)?
- [ ] Soft responsibilities or hard top-$N$ masks as the final output?
- [ ] How to calibrate $h(m)$: Platt scaling against manual masks?
- [ ] Amplitude anchor (§9.2): VOF scaling, $a_{jv}$ model, or accept the bias?
- [ ] Which dataset for Stage 0 (variance decomposition)?

---

## 12. References

- Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression
  data using empirical Bayes methods. *Biostatistics* 2007;8(1):118–127.
- Fortin J-P et al. Harmonization of multi-site diffusion tensor imaging data.
  *NeuroImage* 2017;161:149–170.
- Fortin J-P et al. Harmonization of cortical thickness measurements across
  scanners and sites. *NeuroImage* 2018;167:104–120.
- James W, Stein C. Estimation with quadratic loss. *Proc. 4th Berkeley Symp.* 1961.
- Efron B, Morris C. Stein's estimation rule and its competitors: an empirical
  Bayes approach. *JASA* 1973;68:117–130.
- Efron B, Morris C. Data analysis using Stein's estimator and its
  generalizations. *JASA* 1975;70:311–319.
- Morris CN. Parametric empirical Bayes inference: theory and applications.
  *JASA* 1983;78:47–55.
- Dempster AP, Laird NM, Rubin DB. Maximum likelihood from incomplete data via
  the EM algorithm. *JRSS B* 1977;39:1–38.
- Celeux G, Govaert G. A classification EM algorithm for clustering and two
  stochastic versions. *Comput. Stat. Data Anal.* 1992;14:315–332.
- Yao F, Müller H-G, Wang J-L. Functional data analysis for sparse longitudinal
  data. *JASA* 2005;100:577–590.
- Gelman A et al. *Bayesian Data Analysis*, 3rd ed., ch. 5 (hierarchical models).
- Parker GJM et al. Experimentally-derived functional form for a
  population-averaged high-temporal-resolution arterial input function for
  dynamic contrast-enhanced MRI. *MRM* 2006;56:993–1000.
- Saca L et al. Automatic detection of arterial input function for brain DCE-MRI
  in multi-site cohorts. *MRM* 2025;94(6):2732–2744.
