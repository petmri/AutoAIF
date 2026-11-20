# Review: population-informed AIF shrinkage design

**Reviews:** [population_aif_shrinkage.md](population_aif_shrinkage.md) (draft of 2026-11-20, commit `822e113`).
**Scope:** a critical read of the design, checked against the current code. Nothing was run on data.
The likelihood figures in §6 are my judgment, not measurements.

---

## 1. Summary

The math in the draft is correct, and the hierarchical-model framing is the right family of
methods. The problem is one level down, in what the model treats as "error".

**The method weights each subject by its within-subject voxel scatter. That scatter is not the
error that makes AIFs wrong.** The errors that dominate DCE AIFs (inflow, partial volume, signal
saturation, temporal undersampling, baseline and T1 errors) are shared by all of a subject's
selected voxels. They do not show up as disagreement between voxels, so the model counts them as
real between-subject variation and protects them from shrinkage. Meanwhile the one error it does
see, independent voxel noise, is already small after averaging.

The consequences:

1. **Stage 1 as written will most likely be a no-op** ($w \approx 1$ everywhere), or, once
   $n^{\text{eff}}$ is lowered to make it do something, a blend with the population AIF whose
   strength is set by a number we chose. Either way the claim "the data decide" does not hold.
2. **Shared bias is untouched**, as the draft says in §9.2. But this is most of the error budget,
   not a caveat.
3. **Stage 2 (voxel selection) has more potential and more risk.** Two specifics make it fragile:
   the free partial-volume scale $a_{jv}$ lets ordinary tissue voxels pass as arterial, and the
   prior cannot separate arterial from venous curves once bolus arrival is a free per-subject
   parameter.
4. **The population model is worth building anyway**, first as a quality-control tool, and second
   as a way to estimate how reliable individual AIFs are using information outside the vessel
   (tissue curves, repeat sessions). That reliability is the shrinkage weight the draft is
   trying to get from voxel scatter.

The rest of this document gives the reasoning, an error budget, the specific technical issues,
and a proposed restructure.

---

## 2. The central issue: two variance decompositions that do not coincide

The draft decomposes AIF variance as

- **between subjects** ($\tau^2$): treated as real, kept;
- **within subject, between voxels** ($\delta^2/n$): treated as error, shrunk away.

What we need is a different split:

- **physiology** (cardiac output, hematocrit, dose, injection): real, must be kept, because the
  tissue curves in that session saw it;
- **subject-level measurement error**: wrong, should be removed;
- **voxel-level noise**: wrong, averaged away.

Write the measured score as

$$
\hat\xi_j = \underbrace{\xi_j}_{\text{physiology}} + \underbrace{b_j}_{\text{subject-level error}} + \underbrace{\bar\varepsilon_j}_{\text{voxel noise}}
$$

The draft's $\hat\tau^2$ estimates $\operatorname{Var}(\xi) + \operatorname{Var}(b)$. The two are
not separable from vessel data alone. A subject whose 20 voxels all sit 30% low because of inflow
has small $\delta^2$, gets $w \approx 1$, and keeps its error. A subject with a genuinely unusual
AIF and a scattered mask gets pulled toward the mean. The weighting is driven by vessel geometry
and mask shape, not by how wrong the AIF is.

The ComBat analogy hides this. In ComBat the samples within a batch are independent draws around
the batch mean, so $\delta^2/n$ really is the sampling variance of an unbiased estimate. Here the
voxels are neither independent nor unbiased around the truth.

**Voxel scatter is also not noise around a common mean.** With partial volume, each voxel sees
$a_{jv}\,c_j$ with $a_{jv} \le 1$. The scatter is one-sided, the mean is biased low, and the best
estimate is nearer the top of the distribution than the middle. Stage 1 is built on the mean.

**Effective $n$ makes the weights nearly uniform.** The mask is linearly interpolated back to
native resolution ([utils_vif.py:61](../utils_vif.py:61)) and the top 20 voxels are taken, so they
are almost certainly one or two contiguous blobs. A realistic $n^{\text{eff}}$ is 1 to 3. At that
size $\hat\delta^2_{jk}$ carries almost no information, the inverse-gamma prior dominates, and
every subject ends up with about the same $w$. "Shrink in proportion to how unreliable it is"
is not achievable from this input. A block bootstrap over 20 voxels in one blob will not rescue it.

---

## 3. Error budget: what shrinkage can and cannot remove

"Visible" means the error appears as disagreement among a subject's selected voxels, which is the
only place the draft's model can see it.

| Source | Structure | Visible in voxel scatter? | Removed by this procedure? |
|---|---|---|---|
| Thermal noise | independent per voxel and frame | Yes | Yes, but already small after averaging |
| Partial volume | one-sided per voxel; subject-level through vessel size; shared across the population | Partly (heterogeneity only) | No. The mean and the population mean are both biased low |
| Inflow (unsaturated spins) | shared within subject, depends on slice position and slab; similar across a site | No | No. It goes into $\mu_s$ and is then shrunk toward |
| Signal saturation and $T_2^{\ast}$ loss at peak | nonlinear, dose-dependent, shared across the population | No | No |
| Temporal undersampling of the first pass | per subject, depends on where the frames fall on the bolus; site-level in the mean | No | Not as written. Yes in principle with a forward model (§5.4) |
| 32-frame resampling ([utils_vif.py:42](../utils_vif.py:42)) | linear `zoom` in time: no anti-alias filter when $T > 32$, interpolation when $T < 32$ | No | Avoidable: extract curves from native data |
| Baseline $S_0$ error | multiplies the whole relative-enhancement curve; inflow raises $S_0$ | Partly | No for the shared part |
| $T_{10}$ and $B_1^+$ error (concentration) | subject-level scale, spatially smooth | No | No |
| Hematocrit | real subject-level scale on plasma concentration | No | It is physiology. Shrinking it is a bias unless it is a covariate |
| Bolus arrival estimate | per subject, large at coarse temporal resolution | No | Leaks into shape scores (§4.3) |
| Motion | time-varying partial volume, per subject | Partly | No |
| Venous or tissue voxels in the mask | per voxel, clustered | Yes if mixed; No if the whole mask is venous | Stage 2 target, with the caveats in §4.5 |
| Selection effect | picking voxels by peak-like criteria favors positive noise; picking conforming voxels shrinks $\hat\delta^2$ | No | Introduced by Stage 2, soft or hard |

Reading down the last column: the procedure removes the smallest term, and for the venous
contamination case it may help. Everything else is either a shared bias, which no pooling across
subjects can fix, or a subject-level error with no within-subject signature.

**What would make the subject-level errors identifiable.** They need information from outside the
vessel ROI. Three sources exist in these data:

1. **Tissue curves from the same session.** In tissue with an intact barrier the curve is close
   to $v_p\,c_p(t)$. Across subjects, regress $\log(\text{tissue AUC in normal white matter})$ on
   $\log(\text{AIF AUC})$. If AIF amplitude variation is real, tissue follows it and the slope is
   near 1. If it is measurement error, tissue ignores it and the slope is near 0. Under the
   assumption that $v_p$ is independent of both, the slope is
   $\operatorname{Var}(\xi)/(\operatorname{Var}(\xi)+\operatorname{Var}(b))$, which is the correct
   shrinkage weight for amplitude. This needs no new method and no ground truth.
2. **Repeat sessions.** Between-session variance within a subject bounds
   $\operatorname{Var}(b)$ plus day-to-day physiology from above.
3. **Measurement-quality covariates** that predict $b_j$ rather than $\xi_j$: slice position of
   the mask relative to the slab edge (inflow), vessel diameter in voxels (partial volume),
   temporal resolution, baseline frame count. These can enter the *error* model, which is the
   opposite role from the physiological covariates the draft puts in the prior mean.

---

## 4. Specific technical issues

### 4.1 What the network output is

- **The loss is not scale-free in the way §2 says.** `loss_huber`
  ([model_vif.py:78](../model_vif.py:78)) divides by the first frame, so it compares
  $S(t)/S_0$. That removes absolute signal scale but keeps relative-enhancement amplitude. The
  network is trained to match the manual curve's enhancement ratio, not only its shape.
- **$m_{jv}$ is not a probability of being arterial.** The network is trained on the curve loss
  only (README: temporal loss alone). No voxel is ever supervised. The mask is whatever weighting
  reproduces the manual curve. It is then min-max normalized per subject
  ([model_vif.py:143](../model_vif.py:143)), so the maximum is always 1 regardless of confidence.
  Platt scaling against manual masks would give "probability a rater would pick this voxel",
  and the manual masks are a handful of chosen voxels, not an exhaustive arterial labeling.
  Usable as a prior, but §7.1 should say what it is.
- **Which estimator is the baseline?** The draft's formula is the soft-mask weighted mean over the
  whole resampled volume. Downstream use is presumably the top-20 hard mask on native data. These
  differ, and the soft sum over roughly two million voxels can be dominated by many small weights.
  Stage 1's "current AutoAIF" needs to name one.
- **Arterial or venous?** The code calls the curve a vascular function (`vif`) throughout. If the
  manual masks include the sagittal sinus at some sites, the "arterial versus venous" mixture in
  §7 is framed wrong and the population is bimodal by design. This needs settling before §7.

Side finding, not part of the design: at [main_vif.py:93](../main_vif.py:93),
`top20.all() > 0.9` tests the flat *indices* returned by `argsort`, not mask values. It is true
unless flat index 0 is among the top 20, so the top-5 branch is effectively never taken. The
draft's "top-20 (or top-5)" describes intent, not behavior.

### 4.2 FPCA claims

- "Scores are uncorrelated across subjects by construction" should read *across components*, and
  it holds only for the curves the basis was fit on, after site means and covariates are removed.
- FPCA diagonalizes the **total** covariance (physiology plus error). The draft needs the
  between-subject *true* covariance to be diagonal for the prior, and the within-subject error
  covariance to be diagonal for the likelihood. Neither follows.
- Under partial volume, voxel scatter is close to rank one: $(a_{jv} - \bar a_j)\,c_j$, so the
  scatter in every component is the same random variable times $\langle c_j, \phi_k\rangle$.
  Treating $\hat\delta^2_{jk}$ as separate per-component variances counts one degree of freedom
  $K$ times.
- The basis is fit on the same noisy, biased curves it is meant to improve, with small $J$.

### 4.3 Time alignment

After aligning on bolus arrival, the residual sub-frame shift is typically the second-largest mode
of variation, and it looks like the time derivative of the mean curve. Saying "do not shrink BAT"
does not keep timing error out of the scores. At coarse temporal resolution the arrival-time
error is a large fraction of the first-pass width. This is the standard amplitude-versus-phase
problem in functional data and needs explicit handling (curve registration, or a shift parameter
inside the likelihood).

### 4.4 The "one EM algorithm" claim

- §6 uses a per-component variance $\delta^2_{jk}$ with an inverse-gamma prior. §7 uses isotropic
  $\sigma_j^2 I$. These are different noise models, so the M-step in §7.2 is not "exactly" §6.3,
  and the loop in §8.3 (E-step with $\sigma_j^2$, M-step with $\delta^{2\ast}_{jk}$) is not EM for
  any single model.
- `it_sol`-style alternation of posterior means is not an EM step either, as §3.5 notes.
- So the statement in §8.1 that the marginal likelihood increases at every step does not apply to
  the algorithm as written. Either define one model and derive every update from it, or call it
  an EM-like coordinate scheme and monitor convergence empirically.
- $n_j$ versus $n_j^{\text{eff}}$ is used inconsistently between the two equations of §6.3.
  With fractional weights, the $n_j - 1$ denominator in §6.1 should use the Kish effective size
  $(\sum r)^2 / \sum r^2$.
- The EM fixed point $\tau^2 = s^2 - e$ is negative whenever noise exceeds observed spread. With
  small $J$ this will happen for higher components and gives full pooling at the boundary.

The collapse analysis in §8.2 is correct (I rechecked the fixed points and the table values). One
implication is not drawn: with voxels fixed, EM converges to the method-of-moments answer, so
iterating Stage 3 over Stage 1 alone buys nothing over a single pass. Iteration only matters
through voxel re-selection.

### 4.5 Stage 2 mixture

- **Free $a_{jv}$ admits tissue.** Normal brain tissue with an intact barrier has a curve close to
  $v_p\,c_p(t)$, which is the AIF scaled by a few percent. With $a_{jv}$ free in $(0,1]$ and
  agreement judged on shape, most of the brain is a valid "arterial voxel with $a = 0.03$". The
  model needs a lower bound or an informative prior on $a$, and a defined candidate set.
- **White-noise likelihood over $T$ frames is overconfident.** Residuals are strongly correlated
  in time (misfit, physiology). Treating 32 or more frames as independent drives responsibilities
  to 0 or 1, which turns soft EM into the hard version §7.4 warns against. Use a likelihood in
  score space, or temper it.
- **Background density $g_j$.** Results of a signal-plus-broad-background mixture depend heavily
  on $g$. A wide Gaussian in 32 dimensions has negligible density everywhere, so nearly every
  voxel is "arterial" unless $g$ is built from real non-arterial curves.
- **The prior does not break the arterial-venous tie.** A venous curve is a delayed, slightly
  dispersed AIF with *less* partial volume. With arrival time free per subject, the delay is
  absorbed, and the venous cluster looks like a clean, high-amplitude AIF. §7.3 expects the prior
  to favor the arterial cluster. It can do so only if timing is anchored to something else
  (for example arrival relative to whole-brain tissue enhancement).
- **Drift through selection.** The safeguards in §8.2 address hyperparameter collapse. They do
  not address this path: the curve is pulled toward the population, voxels near the pulled curve
  get higher responsibility, the re-estimated mean of those voxels sits nearer the population with
  *small* scatter and $w \approx 1$. The subject moves toward the mean while looking more certain.
  This is not an EM bug. A mixture with a flexible outlier class legitimately raises its
  likelihood by discarding inconvenient voxels. It needs a cap on how far selection can move a
  curve, and it must be tested in simulation with subjects that are true outliers.

### 4.6 Guarantees and uncertainty

- The James–Stein dominance result assumes equal, known variances and squared-error loss on the
  shrunk quantity. Here variances are unequal and estimated, and the loss we care about is error
  in tissue parameters per subject and bias in group comparisons. Treat it as motivation.
- If the AIF variation that gets shrunk is real, the tissue curves still carry it, so
  $K^{\text{trans}}$ and $v_p$ variance across subjects goes **up**, not down.
- The credible bands from $V_{jk}$ include only voxel scatter. They omit every shared and
  subject-level error in §3 and will be far too narrow. They should not be propagated into
  $K^{\text{trans}}$ uncertainty as they stand.

### 4.7 Validation plan (§10)

- **Test–retest of AIF peak and AUC is not evidence.** A pure population AIF has zero
  within-subject variation. Any shrinkage improves this metric. Keep only the tissue-parameter
  version, which penalizes over-shrinkage, and always include the two extremes ($w = 0$ and
  $w = 1$) as baselines. If the method does not beat both, stop.
- **Held-out voxel prediction** tests voxel noise only. The two halves share every subject-level
  bias and are spatially correlated.
- **Agreement with manual AIFs** tests agreement with the training target, which shares the same
  biases.
- **Simulation** will confirm the model's own assumptions unless it includes shared and
  subject-level errors (inflow by slice position, partial volume tied to vessel size, undersampled
  peaks, misestimated arrival, wholly venous masks).
- Add the tissue-regression reliability test from §3. It is the only check here that measures how
  much of the AIF variation is real.

---

## 5. How to structure it better

### 5.1 Separate what is being shrunk

Use a representation with three parts that get different treatment:

| Part | Treatment | Reason |
|---|---|---|
| Arrival time | Per subject, not pooled, anchored to tissue arrival | Nuisance, and the arterial-venous discriminator |
| Log amplitude | Not shrunk toward the population. Anchored externally (tissue regression, venous output, dose-based tail level) | Dominated by shared and subject-level bias; errors pass directly into $K^{\text{trans}}$ |
| Normalized shape | Shrunk, with weight from measured reliability | Mostly set by injection protocol and circulation; least affected by partial volume; where a population prior is most credible |

A log scale for amplitude also makes the partial-volume model additive and removes the rank-one
problem in §4.2.

### 5.2 Get the shrinkage weight from reliability, not voxel scatter

Estimate $w$ per component from repeat sessions and from the tissue regression, as a property of
the site and protocol, modulated by measurement-quality covariates. This is a single number per
component per site at first. It is cruder than per-subject weights but it measures the right
thing. Add a limited-translation rule (Efron and Morris) so no subject moves more than a set
distance, which protects true outliers.

### 5.3 Build the population model as quality control first

A site-level model of normal AIF shape, amplitude and timing gives a per-subject outlier score.
Flag low-likelihood subjects for review or fall back to the site population AIF. This uses the
same model, has no feedback loop, cannot erase a group difference silently, and catches the
failures that hurt most (wrong vessel, failed injection). It is the lowest-risk deliverable and
should precede any estimator.

### 5.4 Use a forward model where the error is known

Temporal undersampling is the one large error with a known mechanism. Fit the basis on
high-temporal-resolution sites, then fit low-resolution subjects through their actual sampling
operator (frame duration and timing) instead of treating the coarse curve as a noisy version of
the truth. This is the PACE setting the draft cites, and here pooling adds real information. It
requires matching injection protocol, which belongs in the covariates.

### 5.5 Document structure

- Lead with the hierarchical normal model. The draft itself says (§4) that this is the textbook
  model and ComBat is one application of it. The 150-line ComBat derivation and the transposed
  mapping table can move to an appendix; the analogy is where the misleading intuition
  (independent samples, nuisance parameter) comes from.
- Put the error budget and assumptions **before** the method, not in §9. The method should be
  derived from which errors are in scope.
- State the target and loss up front: the AIF that gives accurate tissue parameters in that
  session, judged on per-subject error and group-comparison bias.
- Move the collapse derivation (§8.2) to an appendix and keep the conclusion.
- Turn §10 item 6 into numbered gates with stop conditions (below).
- Resolve the baseline estimator and the arterial-or-venous question in §2.

### 5.6 Suggested gates

| Gate | Question | Stop or redirect if |
|---|---|---|
| 0a | Variance split: voxel, session, subject, site, by component | Voxel term is under about 5% of between-subject variance: drop voxel-scatter weighting |
| 0b | Tissue-regression slope for amplitude and for shape summaries | Slope near 1: individual AIFs are already reliable, shrinkage only adds bias |
| 0c | Population model as outlier detector against known failures | (deliverable in itself) |
| 1 | Shape shrinkage with measured reliability, against $w=0$ and $w=1$ on tissue-parameter test–retest | Does not beat both baselines |
| 2 | Voxel selection, in simulation with true outliers and venous masks, then on data | Outlier subjects drift toward the mean, or gains are matched by a trimmed mean of candidates |

---

## 6. How likely is it to work

Judgment only, to be replaced by Gate 0 results.

| Component | Chance of a useful gain | Most likely outcome |
|---|---|---|
| Stage 1 as written (voxel-scatter weights) | 10–20% | $w \approx 1$, no measurable change; or a hand-set blend once $n^{\text{eff}}$ is lowered |
| Removing shared bias (partial volume, inflow, saturation) by this procedure | About 0% | Unchanged, by construction |
| Shape-only shrinkage with reliability from repeat sessions or tissue | 40–50% | Modest gain in tissue-parameter repeatability, mainly at coarse-resolution sites |
| Stage 2 voxel selection beating a simple robust alternative | 30–40% | Helps mixed arterial and venous masks; needs constraints on $a_{jv}$ and timing to be safe |
| Population model as QC and fallback | 70–80% | Catches gross failures; useful regardless of the rest |
| Forward-model fitting for undersampled sites | 40–50%, depends on protocol match across sites | Better peak recovery where a high-resolution site shares the injection protocol |

The idea of borrowing strength across a cohort is sound. The draft applies it to the one error
that averaging already handles, using a reliability signal that does not track the errors that
matter. Redirecting the same machinery at shape, with reliability measured from tissue and repeat
scans, and using the population model first for QC, keeps most of the work and aims it where it
can pay off.

---

## 7. Questions to settle first

1. Are the manual masks strictly arterial, or do some sites use the sagittal sinus?
2. Which AIF is used downstream today: the network's soft-weighted curve or the top-20 native mask?
3. How many subjects have repeat sessions, per site?
4. Is a normal-appearing white matter mask available cohort-wide for the tissue regression?
5. Do any two sites share an injection protocol but differ in temporal resolution?
6. Is hematocrit recorded? If not, amplitude carries an unmodeled real 10–15% subject-level spread.
