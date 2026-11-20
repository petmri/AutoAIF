# AIF error and Ktrans: a population-informed plan

**Status:** design draft v2, 2026-11-20. Supersedes v1 (commit `822e113`). It takes in the
review in [population_aif_shrinkage_review.md](population_aif_shrinkage_review.md) and the PI's
comments of 2026-11-20. Nothing here has been run on cohort data. The numbers in §3–§5 come from a
noise-free simulation, [sims/aif_error_ktrans.py](sims/aif_error_ktrans.py), and depend on its
assumptions (§3.2). Results and decisions from testing this go in eLabFTW (team VELA), not in this file.

**Rendering:** equations use GitHub/KaTeX math (`$...$` inline, `$$...$$` display). They render on
GitHub and in the VS Code Markdown preview (`Ctrl+Shift+V`).

---

## 0. What changed since v1

| v1 | v2 | Why |
|---|---|---|
| Goal: an AIF closer to the truth | Goal: accurate $K^{\text{trans}}$ (and $v_p$) per subject, and no added bias in group comparisons | PI: $K^{\text{trans}}$ is the outcome |
| Shrink each subject's AIF toward the population | Estimate and correct each subject's **AIF error parameters**, the two numbers that actually move $K^{\text{trans}}$, and shrink *those* across subjects | Review §2: voxel scatter is not the error that matters. Shrinking a nuisance parameter is what ComBat actually does |
| Reliability from voxel scatter | Reliability from **physical consistency checks** (vessel voxels, tissue fits) | Review §3: shared and subject-level errors are invisible in voxel scatter |
| Voxel selection by agreement with the population | Candidate AIFs ranked by the $K^{\text{trans}}$-relevant checks | PI's tissue-fit idea (§5) |
| Arterial vs venous discrimination, peak recovery | Backlog for a general tool | PI: hard to resolve at ~15 s; minor for $K^{\text{trans}}$. The simulation agrees for arterial vs venous but **not for peak errors** (§3.4) |
| ComBat derivation and collapse analysis in the main text | Appendices A and B | Review §5.5 |

The v1 voxel-mixture EM formulation (v1 §6–§8) is parked in the backlog (§8). It is still in the
git history at `822e113`.

---

## 1. Target, regime and loss

- **Outcome:** $K^{\text{trans}}$ in brain, mostly subtle blood–brain barrier leakage
  ($K^{\text{trans}} \sim 0.5\text{–}3 \times 10^{-3}\ \text{min}^{-1}$, $v_p \sim 1\text{–}5\%$),
  with $v_p$ secondary. Lesions with larger leakage are a secondary case.
- **Data:** about 15 s temporal resolution, scans of roughly 8 min (to be confirmed per site),
  several sites.
- **Kinetic model:** Patlak for low-leakage brain, extended Tofts where leakage is larger.
- **Loss:** per-subject error in $K^{\text{trans}}$, plus bias in group comparisons. AIF fidelity
  matters only as far as it moves these.
- **Out of scope for now:** flow and first-pass perfusion parameters, and telling arterial from
  venous voxels.

---

## 2. Prerequisites in the current code

These come from the review (§4.1) and need settling before any measurement means anything:

1. **Name the baseline estimator.** The network's soft-weighted curve over the resampled volume, or
   the top-20 hard mask applied to native data? Downstream fits presumably use the latter.
2. **Bug:** [main_vif.py:93](../main_vif.py:93) tests `top20.all() > 0.9`, which checks the
   *indices* returned by `argsort`, not mask values. It is true unless flat index 0 is in the top 20,
   so the top-5 branch effectively never runs.
3. **Time axis:** curves for kinetic fitting must come from the native frames. Preprocessing resamples
   time to 32 frames with linear `zoom` ([utils_vif.py:42](../utils_vif.py:42)). That is fine for
   finding voxels, not for the curve.
4. **What $m_{jv}$ is:** the network is trained on the curve loss only and its output is min–max
   normalized per subject, so $m_{jv}$ is a voxel ranking, not a calibrated probability.
5. **Arterial or venous masks?** Whether the manual training masks include the sagittal sinus
   decides which vessel checks in §5.5 are independent of the AIF.

---

## 3. Which AIF errors move Ktrans

### 3.1 Model

Patlak, for tissue voxel $v$ with plasma AIF $c(t)$:

$$
C_v(t) = v_{p,v}\, c(t) + K_v \int_0^t c(u)\, du
$$

Fitted by linear least squares on the sampled frames, with a subject-level delay. Extended Tofts
replaces the integral with $\int_0^t c(u)\, e^{-k_{ep}(t-u)}\, du$.

### 3.2 Simulation setup

Parker population AIF (converted to plasma, Hct 0.42). 15 s frames (box-car average), 8 min,
arrival at 60 s. Brain voxel: $v_p = 0.02$, $K^{\text{trans}} = 0.002\ \text{min}^{-1}$ (Patlak).
Lesion voxel: $v_p = 0.03$, $K^{\text{trans}} = 0.05\ \text{min}^{-1}$, $v_e = 0.2$ (extended Tofts).
Errors are applied to the AIF only. Tissue is sampled with the same frames, so undersampling
alone is not an error. No noise. Changes are relative to the fit with the true AIF.

### 3.3 Results

| AIF error | Brain $\Delta K^{\text{trans}}$ | Brain $\Delta v_p$ | Lesion $\Delta K^{\text{trans}}$ | Lesion $\Delta v_p$ | Brain tissue-fit residual (% of tissue peak) |
|---|---|---|---|---|---|
| none | 0 | 0 | 0 | 0 | 0.05 |
| scale ×0.8 (partial volume, $S_0$, $T_{10}$, Hct) | **+25%** | +25% | **+25%** | +25% | 0.05 |
| first-pass area −30% | **−32%** | +28% | −4% | +65% | 3.20 |
| peak saturation (peak −30%, nonlinear) | **−17%** | +23% | +10% | +27% | 1.93 |
| delay +7.5 s (delay fitted) | +0.5% | +0.5% | +2.5% | −16% | 3.76 |
| venous-like: 4 s delay + 4 s dispersion | −3.8% | +3.6% | +1.3% | −9% | 5.35 |
| washout rate +30% | **+36%** | −1% | 0.1% | −0.4% | 0.13 |
| tail level +10% after 2 min | **−15%** | 0.3% | 1.1% | 0.1% | 0.57 |

### 3.4 What this says about the starting assumptions

- **Arterial vs venous: supported.** Once a delay is fitted, a venous-like AIF moves
  $K^{\text{trans}}$ by about 4% in brain and 1% in lesions.
- **Peak errors only matter for flow: supported for lesions, not for subtle leakage.** In
  low-leakage brain, a 30% loss of peak moves $K^{\text{trans}}$ by 17–32%. The mechanism is in §3.5.
  Pure temporal undersampling is harmless, because tissue is sampled the same way. What hurts is
  any **AIF-specific** distortion of the peak-to-tail ratio: $T_1$ saturation of the
  signal-to-concentration conversion, $T_2^{\ast}$ loss at peak, and concentration-dependent inflow.
- **Scale passes through one-to-one** in every case. It is the most common error (partial volume,
  baseline, $T_{10}$, hematocrit) and the largest.
- **Tail shape matters as much as scale in brain.** A 30% faster washout gives +36%, and a 10%
  higher late tail gives −15%.

So for subtle-leakage $K^{\text{trans}}$, the priority order is: **scale ≈ peak-to-tail and tail
shape ≫ timing ≈ arterial vs venous.**

### 3.5 Why low-leakage Ktrans is so sensitive: vp leverage

Write the measured AIF as $\tilde c = c + \Delta$ and its integral as
$\tilde I = I + L\Delta$, where $L$ is the integration operator. Let
$\tilde X = [\tilde c,\ \tilde I]$ and $G = (\tilde X^\top \tilde X)^{-1}\tilde X^\top$.
To first order:

$$
\frac{\Delta K}{K} \approx -\frac{v_p}{K}\,\big[G\,\Delta\big]_2 \;-\; \big[G\,L\,\Delta\big]_2
$$

The first term carries the factor $v_p/K^{\text{trans}}$. In brain that is about
$0.02 / 0.002\ \text{min}^{-1} = 10$ min; in the lesion it is 0.6 min. Tissue signal in
low-leakage brain is almost all $v_p\, c(t)$. $K^{\text{trans}}$ is a small slope on top of that,
so any part of the AIF error that the fit cannot absorb into $v_p$ gets amplified into
$K^{\text{trans}}$ by $v_p/K$. Longer scans reduce the leverage by giving the integral term more
weight.

---

## 4. Exact result: an AIF error acts through one 2×2 matrix per subject

For Patlak with a fixed delay, the fit is linear in the tissue curve, and the true tissue curve is
linear in $(v_p, K)$. For **any** measured AIF $\tilde c$, then, the fitted parameters of every
voxel are an exact linear map of the true ones:

$$
\begin{pmatrix}\hat v_p \\ \hat K\end{pmatrix}
= M(\tilde c)\begin{pmatrix} v_p \\ K\end{pmatrix},
\qquad
M(\tilde c) = (\tilde X^\top\tilde X)^{-1}\tilde X^\top\,[\,c,\ I\,]
$$

Without noise this is exact, not first order. (Simulation: $R^2 = 1.000000$ over 300 random
voxels.) With noise, a zero-mean term is added. Writing it out:

$$
\hat K_v = m_{22}\, K_v + m_{21}\, v_{p,v},
\qquad
\hat v_{p,v} = m_{11}\, v_{p,v} + m_{12}\, K_v
$$

| AIF error | $m_{11}$ | $m_{12}$ (min) | $m_{21}$ (min⁻¹) | $m_{22}$ | $\rho = m_{21}/m_{11}$ |
|---|---|---|---|---|---|
| none | 1 | 0 | 0 | 1 | 0 |
| scale ×0.8 | 1.250 | 0 | 0 | 1.250 | 0 |
| first-pass area −30% | 1.266 | 0.162 | −0.0394 | 1.074 | −0.031 |
| peak saturation −30% | 1.218 | 0.088 | −0.0262 | 1.090 | −0.022 |
| washout +30% | 0.999 | −0.068 | +0.0264 | 1.094 | +0.026 |
| tail level +10% | 1.000 | 0.036 | −0.0108 | 0.959 | −0.011 |

What follows from this:

1. **Within a subject, the AIF distorts the whole $K^{\text{trans}}$ map by a gain $m_{22}$ plus
   an offset proportional to $v_p$, with slope $m_{21}$.** Three numbers per subject
   ($m_{11}, m_{21}, m_{22}$; $m_{12}K$ is under 2% of $v_p$ here) describe the entire effect of
   the AIF on $K^{\text{trans}}$ and $v_p$. We do not need the whole AIF to be right. We need these
   numbers to be right, or known.
2. **The leverage error $\rho = m_{21}/m_{11}$ is the dangerous one.** With
   $m_{21} = -0.026\ \text{min}^{-1}$ (30% peak saturation) and $v_p = 0.02$, the offset is
   $-5.2\times10^{-4}\ \text{min}^{-1}$, about a quarter of a typical subtle-leakage
   $K^{\text{trans}}$ ($2\times10^{-3}$). It varies across voxels with $v_p$, so it also distorts regional contrasts
   (gray vs white matter).
3. **Scale errors give $M = s^{-1}\,\mathbb I$.** They leave $\rho = 0$ and multiply
   $K^{\text{trans}}$ and $v_p$ alike. The ratio $K^{\text{trans}}/v_p$ does not depend on scale.
4. **A correction formula.** Using $v_{p,v} \approx \hat v_{p,v}/m_{11}$:
   $$
   K_v \approx \frac{\hat K_v - \rho\, \hat v_{p,v}}{m_{22}}
   $$
   The $\rho$ part needs a shape anchor (§5.5). The $m_{22}$ part needs a scale anchor (§6.2).
   For pure scale errors $m_{22} = m_{11}$. For the shape errors above, $m_{22}$ stays within 10%
   of 1 while $m_{11}$ moves up to 27%. A scale anchor calibrated on $v_p$ therefore does not fully
   fix $m_{22}$.
5. **Extended Tofts** is nonlinear in $k_{ep}$, so $M$ is only a local linearization there. In
   low-leakage brain Patlak is the operative model anyway, and in lesions the leverage is small
   (Table §3.3).

---

## 5. The tissue-fit quality idea, analyzed

### 5.1 The proposal

Use each candidate AIF to fit tissue voxels, and score the candidate by fit quality. Tissue curves
are less biased than vessel curves: tissue is stationary, has no inflow, and its concentrations are
low enough to avoid saturation. Separately, look at the distribution of fitted parameters, since a
distribution that disagrees with expectation points to AIF error.

### 5.2 Formalization

For candidate $\tilde c$, with $\tilde P$ the projection onto $\operatorname{span}\{\tilde c, \tilde I\}$:

$$
R(\tilde c) = \sum_v \frac{\lVert y_v - \tilde X\hat\theta_v\rVert^2}{\sigma_v^2},
\qquad
\mathbb E\,R(\tilde c) = \sum_v \left[(T-2) + \frac{\lVert (\mathbb I - \tilde P)\,[c,\ I]\,\theta_v\rVert^2}{\sigma_v^2}\right]
$$

The excess residual measures the part of the true tissue curve that falls **outside** the span
generated by the candidate AIF. The $K^{\text{trans}}$ error is set by $M$, which is the projection
**onto** that span. **The residual and the $K^{\text{trans}}$ error are complementary pieces of
the same decomposition.** The residual cannot measure the $K^{\text{trans}}$ error in general.

### 5.3 What the residual is blind to

- **Scale:** exactly blind. $\tilde c = s\,c$ spans the same space.
- **A "tail ramp" direction:** the error $\Delta$ solving $(v_p + K L)\,\Delta \propto I$. This
  error grows with the running integral of the AIF. In the simulation it is +0.7% at 30 s, +1.5%
  at 1 min, +2.6% at 2 min, +5% at 4 min and +9% at 7 min. It moves brain $K^{\text{trans}}$ by
  **−9%**, and the fit is indistinguishable from the true AIF's (residual $5\times10^{-5}$ mM).
  The direction depends only on the voxel's $K/v_p$ ratio, so pooling voxels with similar ratios
  does not reveal it.
- **Near-blind in practice:** washout rate (+36% in $K^{\text{trans}}$, residual 0.13% of the tissue
  peak) and late tail level (−15%, residual 0.57%). These are close to the tail-ramp direction.

### 5.4 What the residual does see

Peak-shape distortions (first-pass −30%: 3.2%; saturation: 1.9%) and timing (3.8–5.4%, even
after the delay fit, because shifting 15 s frames by interpolation is crude). At ROI level, with thousands of white-matter voxels, a 2–3% systematic misfit is
measurable. Fit quality is therefore a good **detector of peak and timing problems and of gross
failures**, but a poor **measure of $K^{\text{trans}}$ error**. Washout gives a +36% $K^{\text{trans}}$
error with a smaller residual than a harmless half-frame delay.

Good practice if we use it:

- **Compare candidates paired on the same tissue.** Model inadequacy (Patlak vs a two-compartment
  exchange model), drift and CSF partial volume are shared by all candidates and largely cancel in
  $R(\tilde c_1) - R(\tilde c_2)$.
- **Use ROI means** (white matter, cortical gray matter) rather than single voxels. Residuals are
  correlated in time, and a white-noise likelihood over frames is overconfident (review §4.5).
- **Cross-fit:** choose the AIF using one set of regions and report $K^{\text{trans}}$ on another, so
  selection does not adapt to the outcome region's noise.

### 5.5 The parameter-distribution half: this is where the Ktrans information is

§4 says the AIF error shows up in the fitted parameters as $\hat K_v = m_{22}K_v + m_{21}v_{p,v}$.
Two ways to read $\rho$ off a subject:

**(a) Vessel voxels: a physical anchor.** A voxel fully or partly inside a large vessel has
$K = 0$ (no leakage from the lumen) and $v_p = f$ (its blood fraction, in blood units). Fitted with
the candidate AIF:

$$
\frac{\hat K_{\text{vessel}}}{\hat v_{p,\text{vessel}}} = \frac{m_{21} f}{m_{11} f} = \rho
$$

The ratio does not depend on the vessel voxel's partial volume $f$, or on hematocrit. Blood is well
mixed after the first pass, so a vein is a valid probe: the AIF error is *read through* it, not
added by it. In the simulation, the floor set by venous delay and dispersion is
$\lvert\hat K_{\text{vessel}}\rvert \le 0.008\ \text{min}^{-1}$ with the correct AIF, against
$-0.031\ \text{min}^{-1}$ for a 30% peak saturation. **The probe vessel must not be the voxels that
built the AIF**, because fitting an AIF to its own curve returns $\rho = 0$ trivially. Use a
different vessel (sagittal sinus vs an arterial AIF, or the reverse), or a held-out segment.

**(b) Tissue $K$–$v_p$ slope: a biological anchor.** Across tissue voxels,
$\hat K_v = m_{22}K_v + m_{21}v_{p,v}$, so regressing $\hat K$ on $\hat v_p$ gives
$\rho$ plus any true $K$–$v_p$ correlation. This needs the assumption that true leakage is unrelated
to blood volume, which may be false (more vessels, more surface). Use it as a cross-check on (a),
not as the anchor.

**Absolute values** (for example "white-matter $v_p$ should be about 2%") only inform scale (§6.2),
and they assume the very quantity a study might be measuring.

### 5.6 Relation to existing work

The idea that tissue curves determine the AIF is **multichannel blind identification**
(Riabkov & Di Bella 2002; Fluckiger et al. 2009; Schabel et al. 2010). Those papers show the AIF is
recoverable from multiple tissue curves *up to scale*, which matches §5.3. §5.3 adds that for
low-leakage Patlak there is a second near-blind direction, and that it is exactly the one that
moves $K^{\text{trans}}$. Low-level BBB leakage work commonly uses a sagittal sinus input
function, partly because it suffers less partial volume (for example Heye et al. 2016; Manning et
al. 2021). The $\rho$
check in §5.5(a) extends it from scale to shape.

---

## 6. Where the population (ComBat) idea fits now

### 6.1 Shrink the error parameters, not the AIF

Each subject $j$ has nuisance parameters $\phi_j = (\rho_j,\ \log s_j)$: leverage error and scale
error. They come from the measurement chain (vessel geometry, slab position, saturation at that
site's dose and sequence), so subjects at a site should resemble each other. They are also exactly
what we want to **remove**. This is the ComBat setting proper (Appendix A): a noisy per-unit
nuisance estimate, a shared prior across units, shrinkage, then removal. It answers the review's
main objection: the uncertainty being shrunk is now the uncertainty of an anchor-based error
estimate, not voxel scatter of the AIF.

For the leverage error:

$$
\hat\rho_j \mid \rho_j \sim \mathcal N(\rho_j,\ \text{se}_j^2),
\qquad
\rho_j \sim \mathcal N(\bar\rho_{s(j)} + \mathbf z_j^\top\boldsymbol\beta,\ \tau_\rho^2)
$$

$$
\rho^{\ast}_j = w_j\,\hat\rho_j + (1 - w_j)\,(\hat{\bar\rho}_{s(j)} + \mathbf z_j^\top\hat{\boldsymbol\beta}),
\qquad
w_j = \frac{\hat\tau_\rho^2}{\hat\tau_\rho^2 + \text{se}_j^2}
$$

- $\text{se}_j$ comes from bootstrapping over probe vessel voxels or segments, **plus** the
  dispersion floor of §5.5 as a known variance term. That floor is shared and would otherwise be
  ignored.
- $\mathbf z_j$ are **measurement-quality covariates**: vessel diameter in voxels, slice position
  of the vessel relative to the slab edge (inflow), dose, and sequence parameters. These are the
  error-model covariates the review proposed (§3, item 3). They are not physiological covariates.
- Subjects with no usable probe vessel get the site prediction: full shrinkage, $w_j = 0$.
- If $\hat\tau_\rho^2 \to 0$, the error is site-level. A single per-site correction then suffices,
  which is the simplest useful outcome.

### 6.2 Scale: triple collocation

Scale cannot be read from tissue fits (§5.3). It needs absolute anchors. Take three log-amplitude
measurements per subject with independent errors, for example:

- $X_1$: AutoAIF tail level;
- $X_2$: $\hat v_p$ of the least-partial-volumed probe vessel voxels (sinus maximum), converted so
  that a pure vessel reads 1;
- $X_3$: plasma level predicted from dose per kg and a population biexponential (Weinmann 1984),
  optionally adjusted for hematocrit and renal function.

Model each as $X_{kj} = \alpha_k + A_j + e_{kj}$, with true log amplitude $A_j$. Across subjects,
$\operatorname{Cov}(X_k, X_l) = \operatorname{Var}(A)$ for $k \ne l$, so every error variance is
identifiable without ground truth:

$$
\operatorname{Var}(e_1) = \operatorname{Var}(X_1) - \frac{\operatorname{Cov}(X_1,X_2)\,\operatorname{Cov}(X_1,X_3)}{\operatorname{Cov}(X_2,X_3)}
$$

and likewise for $e_2$ and $e_3$ (Stoffelen 1998, with general linear calibration). The best
combined amplitude weights each $X_k$ by $1/\operatorname{Var}(e_k)$. That is the empirical Bayes
shrinkage for scale, with weights estimated rather than assumed. This is the review's
tissue-regression idea (§3, item 1) made general.

**Circularity rule:** anything used as an anchor cannot be an outcome. Physical anchors (vessel
lumen, dose pharmacokinetics) come first. A white-matter $v_p$ anchor would partly normalize away
real differences in white-matter $v_p$ between groups, so use it only when that is not a study
question.

### 6.3 Candidate AIFs: the PI's voxel-selection idea, re-aimed

Generate candidate AIFs from the network ranking: top-$N$ for several $N$, connected components
separately, artery vs sinus, and the site population AIF. Rank them by:

1. $\lvert\hat\rho\rvert$ against an independent probe vessel (§5.5a). This is the
   $K^{\text{trans}}$-relevant criterion.
2. The paired tissue residual (§5.4), to catch peak and timing failures.
3. The population-model likelihood (§7), to catch gross failures.

Then apply the scale calibration (§6.2) to the chosen candidate. "Conform" now means agreeing with
physical consistency checks, not with the population mean, which avoids the drift-toward-the-mean
problem the review raised (§4.5).

If iterating site-level hyperparameters with re-selection, the collapse safeguards of Appendix B
apply: soft weights and posterior-variance terms in the variance update.

---

## 7. Population model as quality control

This is unchanged in spirit from the review (§5.3): a per-site model of AIF tail shape, amplitude
and timing, plus the distributions of $\hat\rho_j$ and the tissue residual. Subjects in the tails
are flagged for review or fall back to the site population AIF with site-level $\rho$ and scale
corrections. It has no feedback loop and is useful whatever happens to the rest.

---

## 8. Plan with gates

| Gate | Work | Needs | Stop or redirect if |
|---|---|---|---|
| **0** | Prerequisites in §2: baseline estimator, fix main_vif.py:93, native-frame curves, concentration conversion, Patlak fitting (ROCKETSHIP or Python) | code only | n/a |
| **1** | Re-run [the simulation](sims/aif_error_ktrans.py) with each site's real frame times, scan length, dose, and typical $v_p$, $K^{\text{trans}}$. Add realistic $T_1$ saturation for each site's sequence | protocol parameters, no subject data | Leverage $v_p/K$ is small at all sites: shape errors don't matter; go to Gate 3 only |
| **2** | Measure $\hat\rho_j$ on the cohort: fit probe vessel voxels (not in the AIF mask) with each subject's AutoAIF AIF. Report the implied offset $\hat\rho_j \cdot \bar v_p$ against the between-subject SD of $K^{\text{trans}}$. Cross-check with the tissue $K$–$v_p$ slope (§5.5b) | AutoAIF outputs, probe-vessel masks, Patlak maps | Implied offset under about 10% of between-subject $K^{\text{trans}}$ SD: leverage error is not a problem |
| **3** | Scale: triple collocation (§6.2) gives the error variance of AutoAIF amplitude per site | dose, weight, sinus voxels, optionally Hct | AutoAIF amplitude error variance is small next to biological spread |
| **4** | Tissue residual as a detector: corrupt real AIFs deliberately (saturation, tail ramp, delay, scale) and check that paired $\Delta R$ ranks them as §5.3–5.4 predicts | Gate 0 pipeline | It does not detect peak distortions on real data: drop it from candidate ranking |
| **5** | Correction: EB-shrunk $\rho^{\ast}_j$, scale calibration, candidate selection. Evaluate on $K^{\text{trans}}$ test–retest and against two baselines: uncorrected AutoAIF, and the site population AIF. Check that a known group difference is preserved | repeat sessions | It does not beat **both** baselines |

Gates 1–3 are measurement, not method development, and each can end the project cheaply.

**Backlog (for a general-purpose tool, not needed for subtle-leakage $K^{\text{trans}}$):**
the voxel-level mixture EM (v1 §7–§8, with the review's §4.5 fixes), arterial vs venous
discrimination, first-pass recovery for flow, the forward model for undersampled sites (review
§5.4), and FPCA shape shrinkage.

---

## 9. Open questions

1. Are the manual masks arterial, sinus, or mixed by site? (decides probe-vessel independence, §5.5)
2. Which AIF goes downstream today: soft-weighted curve or top-20 native mask?
3. Which kinetic fitting code and model are used on this cohort: ROCKETSHIP Patlak? Fitted delay?
4. Per site: frame duration, scan length, dose, flip angle and TR (Gate 1), and whether $T_{10}$
   maps exist.
5. Is the sagittal sinus in the field of view at every site? Is an artery?
6. How many subjects have repeat sessions, per site? (Gate 5)
7. Are hematocrit, weight and eGFR recorded? (scale anchor $X_3$)
8. Is white-matter $v_p$ ever an outcome? (decides whether it can be an anchor, §6.2)

---

## Appendix A. How ComBat / neuroCombat works

References: Johnson, Li & Rabinovic (2007) for the method; Fortin et al. (2017, 2018) for
neuroCombat. The formulas follow the neuroCombat code (`aprior`, `bprior`, `postmean`, `postvar`,
`it_sol`).

### A.1 Model

Batch (site) $i$, sample $j$ within batch, feature $v$:

$$
y_{ijv} = \alpha_v + \mathbf{x}_{ij}^\top \boldsymbol\beta_v + \gamma_{iv} + \delta_{iv}\,\varepsilon_{ijv},
\qquad \varepsilon_{ijv} \sim \mathcal N(0, \sigma_v^2)
$$

$\alpha_v$ is the feature mean, $\mathbf x^\top\boldsymbol\beta_v$ holds the biological covariates
to **preserve**, and $\gamma_{iv}$ and $\delta_{iv}$ are the additive and multiplicative batch
effects to **remove**.

### A.2 Standardize

Fit $\alpha_v$, $\boldsymbol\beta_v$ and batch indicators by least squares and pool the residual
variance $\hat\sigma_v^2$:

$$
z_{ijv} = \frac{y_{ijv} - \hat\alpha_v - \mathbf x_{ij}^\top \hat{\boldsymbol\beta}_v}{\hat\sigma_v}
$$

This puts all features on one scale, so their batch effects can be compared.

### A.3 Naive estimates

$$
\hat\gamma_{iv} = \frac{1}{n_i}\sum_j z_{ijv},
\qquad
\hat\delta^2_{iv} = \frac{1}{n_i - 1}\sum_j \left(z_{ijv} - \hat\gamma_{iv}\right)^2
$$

### A.4 Empirical Bayes prior, shared across features within a batch

$$
\gamma_{iv} \sim \mathcal N(\gamma_i,\ \tau_i^2),
\qquad
\delta^2_{iv} \sim \text{InvGamma}(\lambda_i,\ \theta_i)
$$

The hyperparameters are estimated by method of moments across features:
$\bar\gamma_i = \operatorname{mean}_v \hat\gamma_{iv}$ and
$\bar\tau_i^2 = \operatorname{var}_v \hat\gamma_{iv}$. With $M$ and $S^2$ the mean and variance of
$\{\hat\delta^2_{iv}\}_v$, the inverse-gamma moments $\mathbb E = \theta/(\lambda-1)$ and
$\operatorname{Var} = \theta^2/((\lambda-1)^2(\lambda-2))$ give

$$
\bar\lambda_i = \frac{M^2 + 2S^2}{S^2},
\qquad
\bar\theta_i = \frac{M^3 + M S^2}{S^2}
$$

### A.5 Posterior estimates

Normal–normal conjugacy, conditional on $\delta^2$:

$$
\gamma^{\ast}_{iv} = \frac{n_i \bar\tau_i^2\, \hat\gamma_{iv} + \delta^{2}_{iv}\, \bar\gamma_i}{n_i \bar\tau_i^2 + \delta^{2}_{iv}}
= w_{iv}\,\hat\gamma_{iv} + (1 - w_{iv})\,\bar\gamma_i,
\qquad
w_{iv} = \frac{\bar\tau_i^2}{\bar\tau_i^2 + \delta^2_{iv}/n_i}
$$

The inverse gamma is conjugate for a normal variance, conditional on $\gamma$:

$$
\delta^{2\ast}_{iv} = \frac{\bar\theta_i + \tfrac12 \sum_j \left(z_{ijv} - \gamma^{\ast}_{iv}\right)^2}{n_i/2 + \bar\lambda_i - 1}
$$

`it_sol` alternates these two conditional posterior means until the relative change is below
$10^{-4}$. This is a deterministic fixed-point scheme, not a full posterior. A nonparametric variant
replaces the priors with the empirical distribution of the other features' estimates.

### A.6 Adjust

$$
y^{\ast}_{ijv} = \frac{\hat\sigma_v}{\delta^{\ast}_{iv}}\left(z_{ijv} - \gamma^{\ast}_{iv}\right) + \hat\alpha_v + \mathbf x_{ij}^\top \hat{\boldsymbol\beta}_v
$$

### A.7 Justification and limits

This is James–Stein / empirical Bayes shrinkage (James & Stein 1961; Efron & Morris 1973, 1975):
for three or more normal means, pooling lowers *total* squared error. The guarantee assumes equal,
known variances and holds on average, not for each unit. A true outlier is biased toward the mean;
limited-translation rules (Efron & Morris 1972) cap that. In ComBat the shrunk quantity is a
nuisance that gets removed. §6 uses the same structure for the same reason.

---

## Appendix B. Why naive iteration of shrinkage collapses

Scalar case: each unit's estimate has sampling variance $e$, and the observed spread is $s^2$.
With $w = \tau^2/(\tau^2 + e)$:

**Naive** (re-estimate $\tau^2$ from shrunk values):
$\tau^2_{\text{new}} = w^2 s^2$. The fixed points are $\tau^2 = 0$, which is always **stable**
(near zero, $\tau^2_{\text{new}} \approx \tau^4 s^2/e^2$), and the roots of
$(\tau^2 + e)^2 = \tau^2 s^2$. Those roots exist only if $s^2 \ge 4e$, and the upper one is biased low.

**EM** (add the posterior variance):
$\tau^2_{\text{new}} = w^2 s^2 + w e$, whose only positive fixed point is $\tau^2 = s^2 - e$,
the moment estimate.

| Case ($J = 60$) | Truth / MoM | Naive | EM |
|---|---|---|---|
| $e = 0.5$, $s^2 = 1.17$ | 0.67 | 0.00 from any start | 0.67 |
| $e = 0.1$, $s^2 = 0.78$ | 0.68 | 0.56 from a high start, 0.00 from a low start | 0.68 |

With fixed inputs, EM converges to the one-pass moment estimate, so iterating buys nothing unless
the inputs (voxel or candidate selection) change. If they do, use soft weights and keep the
posterior-variance term.

---

## References

Citations are from memory. Check volume and page details before reusing them in a manuscript.

- Barnes SR et al. ROCKETSHIP: a flexible and modular software tool for the planning, processing
  and analysis of dynamic MRI studies. *BMC Med Imaging* 2015;15:19.
- Efron B, Morris C. Limiting the risk of Bayes and empirical Bayes estimators, part II: the
  empirical Bayes case. *JASA* 1972;67:130–139.
- Efron B, Morris C. Stein's estimation rule and its competitors: an empirical Bayes approach.
  *JASA* 1973;68:117–130.
- Efron B, Morris C. Data analysis using Stein's estimator and its generalizations. *JASA*
  1975;70:311–319.
- Fluckiger JU, Schabel MC, DiBella EVR. Model-based blind estimation of kinetic parameters in
  dynamic contrast enhanced (DCE)-MRI. *MRM* 2009;62:1477–1486.
- Fortin J-P et al. Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*
  2017;161:149–170.
- Fortin J-P et al. Harmonization of cortical thickness measurements across scanners and sites.
  *NeuroImage* 2018;167:104–120.
- Heye AK et al. Tracer kinetic modelling for DCE-MRI quantification of subtle blood–brain barrier
  permeability. *NeuroImage* 2016;125:446–455.
- James W, Stein C. Estimation with quadratic loss. *Proc. 4th Berkeley Symp.* 1961.
- Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using
  empirical Bayes methods. *Biostatistics* 2007;8:118–127.
- Manning C et al. Sources of systematic error in DCE-MRI estimation of low-level blood–brain
  barrier leakage. *MRM* 2021;86:1888–1903.
- Parker GJM et al. Experimentally-derived functional form for a population-averaged
  high-temporal-resolution arterial input function for dynamic contrast-enhanced MRI. *MRM*
  2006;56:993–1000.
- Patlak CS, Blasberg RG, Fenstermacher JD. Graphical evaluation of blood-to-brain transfer
  constants from multiple-time uptake data. *JCBFM* 1983;3:1–7.
- Riabkov DY, Di Bella EVR. Estimation of kinetic parameters without input functions: analysis of
  three methods for multichannel blind identification. *IEEE TBME* 2002;49:1318–1327.
- Saca L et al. Automatic detection of arterial input function for brain DCE-MRI in multi-site
  cohorts. *MRM* 2025;94:2732–2744.
- Schabel MC, Fluckiger JU, DiBella EVR. A model-constrained Monte Carlo method for blind arterial
  input function estimation in dynamic contrast-enhanced MRI: I. Simulations. *Phys Med Biol*
  2010;55:4783–4806.
- Stoffelen A. Toward the true near-surface wind speed: error modeling and calibration using
  triple collocation. *J Geophys Res* 1998;103:7755–7766.
- Tofts PS et al. Estimating kinetic parameters from dynamic contrast-enhanced T1-weighted MRI of
  a diffusable tracer: standardized quantities and symbols. *JMRI* 1999;10:223–232.
- Weinmann HJ, Laniado M, Mützel W. Pharmacokinetics of GdDTPA/dimeglumine after intravenous
  injection into healthy volunteers. *Physiol Chem Phys Med NMR* 1984;16:167–172.
