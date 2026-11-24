# AIF error and Ktrans: a population-informed plan

**Status:** design draft v2.1, 2026-11-24. v2 (2026-11-20) superseded v1 (commit `822e113`) and took
in the review in [population_aif_shrinkage_review.md](population_aif_shrinkage_review.md) and the PI's
comments of 2026-11-20. v2.1 adds the cohort's real protocols (§3.6), what the blind-estimation papers
add (§5.7), and answers from the data inventory ([data_inventory.md](data_inventory.md)) to §2 and §9.
No kinetic analysis has been run on cohort data yet; the inventory only reads metadata and fit logs. The
numbers in §3–§5 come from a noise-free simulation, [sims/aif_error_ktrans.py](sims/aif_error_ktrans.py),
and depend on its assumptions (§3.2). Results and decisions from testing this go in eLabFTW (team VELA),
not in this file.

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
- **Data:** USC-PPG, 933 DCE sessions from 642 subjects at five sites, in five protocol groups:
  frames of 7.9–15.6 s, 64 frames, scans of 8.4–16.6 min, Siemens and GE at 3 T
  ([data_inventory.md](data_inventory.md) §2.1). About 60% of sessions are 16-min scans.
- **Kinetic model:** Patlak for low-leakage brain, extended Tofts where leakage is larger.
- **Loss:** per-subject error in $K^{\text{trans}}$, plus bias in group comparisons. AIF fidelity
  matters only as far as it moves these.
- **Out of scope for now:** flow and first-pass perfusion parameters, and telling arterial from
  venous voxels.

---

## 2. Prerequisites in the current code

These come from the review (§4.1) and need settling before any measurement means anything:

1. **Baseline estimator (settled from the logs):** ROCKETSHIP averages the 20 top-voxel mask voxels
   on native frames, converts to plasma concentration with Hct 0.45, and **fits a 4-parameter model to
   that curve**. The fitted curve drives Patlak, which has no delay fit. The soft mask is not used. Any
   error analysis must use this fitted AIF, or switch it off with ROCKETSHIP `--aif-mode raw`.
2. **Bug (fixed in [petmri/AutoAIF#12](https://github.com/petmri/AutoAIF/pull/12), not merged):**
   [main_vif.py:93](../main_vif.py:93) tested `top20.all() > 0.9`, which checks argsort indices, not
   mask values. Every mask in the standard derivative is therefore top-20. Merging the fix changes the
   AIF for weak-prediction sessions, so the baseline for this project must name which version it uses.
3. **Time axis (settled):** kinetic fits already use native frames. The 32-frame resampling in
   AutoAIF ([utils_vif.py:42](../utils_vif.py:42)) only affects voxel selection.
4. **What $m_{jv}$ is:** the network is trained on the curve loss only and its output is min–max
   normalized per subject, so $m_{jv}$ is a voxel ranking, not a calibrated probability.
5. **Arterial or venous masks? (answered, PI 2026-11-24)** The manual masks are mostly arterial; a few
   are venous, usually the jugular. Venous probes are therefore independent of the AIF in most
   sessions (§5.5).
6. **Baseline window (settled):** the pre-contrast window ends around frame 2, usually 1–2 frames before
   the AIF peak. In about 340 sessions the ROCKETSHIP A-stage log prints a degenerate detector value
   ("image 1 to image 18"), but the logged concentrations show it is not the window used
   (data_inventory §3).
7. **Pre-contrast AIF $T_1$** has a median of 2.08 s in SIE16 and SIE10, against 1.5–1.7 s elsewhere and
   about 1.6–1.7 s expected for blood. It points to partial volume or a $T_1$ bias, which are scale
   errors.

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
shape ≫ timing ≈ arterial vs venous.** This is for an 8-min scan. With the cohort's 16-min scans, peak
errors drop out and the order becomes **scale ≈ tail shape ≫ the rest** (§3.6).

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

### 3.6 The cohort's protocols

The same brain case (Patlak, delay fitted), rerun with each protocol's frame duration and scan length
(`python3 docs/sims/aif_error_ktrans.py --protocols`; 64 frames, arrival 40–46 s). The cell is
$\Delta K^{\text{trans}}$, with the single-voxel fit residual in % of the tissue peak in brackets.

| Protocol | Scan | Scale ×0.8 | First-pass −30% | Peak saturation | Venous-like | Washout +30% | Tail +10% | $\rho$, saturation (min⁻¹) |
|---|---|---|---|---|---|---|---|---|
| SIE08 (CCIR) | 8.4 min | +25% (0.01) | **−29%** (2.2) | **−18%** (1.6) | −11% (3.1) | **+35%** (0.11) | −14% (0.37) | −0.022 |
| SIE10 | 10.5 min | +25% (0.04) | −17% (2.7) | −8% (1.6) | −7% (3.9) | **+35%** (0.16) | −12% (0.49) | −0.014 |
| GE10 (BANNER) | 11.3 min | +25% (0.03) | −14% (2.6) | −7% (1.6) | −3% (2.2) | **+34%** (0.16) | −11% (0.48) | −0.012 |
| SIE16 | 16.4 min | +25% (0.03) | −5% (2.7) | −0.6% (1.6) | −1.5% (4.3) | **+32%** (0.29) | −9% (0.54) | −0.007 |
| GE16 | 16.6 min | +25% (0.02) | −5% (2.6) | −0.5% (1.6) | −2.5% (5.3) | **+32%** (0.29) | −9% (0.52) | −0.007 |

- **The PI's intuition holds for the 16-min protocols**, about 60% of sessions: peak errors move brain
  $K^{\text{trans}}$ by under 5%, and arterial vs venous by under 3%.
- **It does not hold for the 8.4-min CCIR protocol**, where peak errors still move $K^{\text{trans}}$
  by 18–29%. Mid-length scans sit in between. Peak handling therefore differs by site, which is itself
  a site effect in $K^{\text{trans}}$.
- **Scale, washout and tail level matter at every site.** They are the priority everywhere.
- This is a protocol difference in the *leverage* $\rho$, so a pooled ComBat-style correction has to
  model it per protocol (§6.1). It also bounds how much a cross-site difference in
  $K^{\text{trans}}$ can come from AIF peak handling.

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
different vessel, or a held-out segment.

**Which probe vessel, in this cohort.** The coronal slabs reach the top of the brain, so the superior
sagittal sinus (SSS) and the jugular veins are both in view. Most AIF masks are arterial, so both veins
are independent probes. The two differ in inflow:

- **Jugular, preferred.** It runs in-plane, so it has little inflow effect. Use it for $\rho$ and as the
  late-blood scale anchor.
- **SSS, secondary.** It runs through-plane, so fresh unsaturated spins raise its signal. That breaks the
  $T_1$-based signal-to-concentration conversion in a flow-dependent way. The ratio
  $\hat K/\hat v_p$ cancels partial volume but not this nonlinearity. Agreement between the jugular
  and the SSS estimates of $\rho$ is a check on inflow.

In the few sessions whose AIF mask is jugular, probe with the SSS or with arterial voxels outside the
mask.

**(b) Tissue $K$–$v_p$ slope: a biological anchor.** Across tissue voxels,
$\hat K_v = m_{22}K_v + m_{21}v_{p,v}$, so regressing $\hat K$ on $\hat v_p$ gives
$\rho$ plus any true $K$–$v_p$ correlation. This needs the assumption that true leakage is unrelated
to blood volume, which may be false (more vessels, more surface). Use it as a cross-check on (a),
not as the anchor.

**Absolute values** (for example "white-matter $v_p$ should be about 2%") only inform scale (§6.2),
and they assume the very quantity a study might be measuring.

### 5.6 Relation to existing work

The idea that tissue curves determine the AIF is **multichannel blind identification**, or simultaneous
estimation (SIME) in PET (Riabkov & Di Bella 2002; Wang et al. 2004; Fluckiger et al. 2009;
Schabel et al. 2010; Zhu et al. 2024). Those papers show the AIF is recoverable from multiple tissue
curves *up to scale*, which matches §5.3. §5.3 adds that for low-leakage Patlak there is a second
near-blind direction, and that it is exactly the one that moves $K^{\text{trans}}$. §5.7 explains why,
using the literature's own identifiability condition. Low-level BBB leakage work commonly uses a sagittal sinus input
function, partly because it suffers less partial volume (for example Heye et al. 2016; Manning et
al. 2021). The $\rho$
check in §5.5(a) extends it from scale to shape.

### 5.7 What the blind-estimation papers add

Read for this plan: Riabkov & Di Bella 2002 (IEEE TBME), Wang et al. 2004 (ISBI), Fluckiger et al. 2009
(MRM), Zhu et al. 2024 (NeuroImage).

**1. Scale is never identifiable from tissue, and everyone fixes it with late blood.** Riabkov pins
the last two AIF points to measured blood. Fluckiger matches the mean of the last five points to a
nearby artery. Zhu scales the last point to a reference. The reason given each time is that late
concentrations are low, so saturation is small, and that a big vessel can be chosen for low partial
volume. This supports §6.2, with the **jugular tail** as the natural late-blood anchor in this cohort:
it is a large vessel with little inflow effect (§5.5), and the anchor is read where saturation is
least.

**2. Identifiability needs channels with different kinetics.** Riabkov's uniqueness proof requires
that the tissue transfer functions share no poles, meaning distinct $k_{ep}$. Zhu needs at least three
regions "of distinct shapes". Low-leakage brain under Patlak breaks this condition: every voxel has
$k_{ep} = 0$, so all tissue curves lie in $\operatorname{span}\{c, \int c\}$. Adding
$\Delta = \varepsilon \int c$ to the AIF changes the fit only in a residual term proportional to
$\varepsilon K$, which is tiny when $K$ is. **This is why the $\rho$ direction of §5.3 is near-blind
in brain, however many brain voxels are pooled.** Channels with $k_{ep} > 0$ restore it. The new check
in [the simulation](sims/aif_error_ktrans.py) adds a ramp $\propto \int c$ (pooled over 200 brain
voxels and 50 extracranial voxels; noise-free):

| Ramp at 4 min | Brain median $\Delta K^{\text{trans}}$ | Pooled brain residual (Patlak) | Pooled extracranial residual (ext. Tofts) |
|---|---|---|---|
| +5% | −0.0003 min⁻¹ (−15%) | 0.045 → 0.096% | 0.53 → 0.52% |
| +10% | −0.0006 min⁻¹ (−28%) | 0.045 → 0.145% | 0.53 → 0.87% |
| +20% | −0.0011 min⁻¹ (−55%) | 0.045 → 0.221% | 0.53 → 1.80% |

The extracranial residual rises 3–7 times more than the brain residual once the error is 10% or more.
Below that it is lost in the simulation's own discretization floor (0.53%), so this is only a lower
bound on what real noise allows. **Extracranial tissue with $k_{ep} > 0$** (temporalis muscle, scalp),
plus choroid plexus and any lesion, is the tissue that can see the error that moves brain
$K^{\text{trans}}$. In this cohort that tissue is in `desc-hmc_DCE` but masked out of the concentration
data (data_inventory §3).

**3. Tissue-fit quality does not rank AIFs, on independent evidence.** In Zhu et al., the carotid
image-derived input gave errors of 101% in $K_1$ and 186% in $v_b$, yet "all the methods were able to
fit the tissue TACs well". In Fluckiger et al., the dispersed, delayed input estimated from tissue alone
fits tissue *better* than the measured arterial one. A residual-based ranking therefore leans toward
local, dispersed input functions. This agrees with §5.3–5.4: use the residual as a detector of gross
and peak problems, not as the selection criterion.

**4. Constrain the AIF shape; a linear kernel form is best.** Without a shape model the blind AIF is
noisier than the tissue (Fluckiger), and nonlinear parametric forms fail to converge (Zhu, supplement).
Zhu's **kernel SIME** writes the AIF as $c = \mathbf K\alpha$. $\mathbf K$ is a Gaussian kernel matrix
built from features of an image-derived curve, and the AIF is linear in $\alpha$. It corrected a
partial-volumed carotid input to an MAE of 7–16% in kinetic parameters, against 37–101% for the raw
carotid curve. **AutoAIF's curve is exactly such an image-derived prior.** Kernel SIME on AutoAIF
features is a candidate correction method (§6.3) that uses the tissue to fix shape, including the peak
for short protocols. It still needs channels with distinct kinetics (point 2) and a late-blood scale
(point 1).

**5. Variable projection makes the whole-brain criterion cheap.** For fixed AIF and $k_{ep}$, the
per-voxel amplitudes are linear and can be profiled out. Riabkov's IQML and Wang's ML both reduce the
cost to

$$
L(\theta) = \operatorname{Tr}\{(\mathbb I - \mathbf P_{A(\theta)})\,\hat{\mathbf R}\},
\qquad \hat{\mathbf R} = \tfrac1N \textstyle\sum_v y_v y_v^\top
$$

Here $A(\theta)$ holds the AIF and its kinetic transforms. $\hat{\mathbf R}$ is a 64×64 matrix
computed once per region, so §5.2's $R(\tilde c)$ over all voxels costs a 64×64 projection per
candidate. The eigenvalues of $\hat{\mathbf R}$ also show the signal-subspace dimension, which is 2 for
pure Patlak. That makes it a model check for whether Patlak is adequate in a region.

**6. Practical details worth copying.**
- Fluckiger reduces voxels to about 8 k-means cluster means before fitting, for SNR. Results were stable
  for 4–10 clusters.
- Use a single global delay found by grid search (Zhu, Fluckiger).
- Initialize from a population AIF. Fluckiger's alternating method was robust (median 8% parameter error)
  from near-truth or zero starts, but poor at bound-valued starts. Our site prior (§6.1) is the right
  initializer.

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
- $X_2$: $\hat v_p$ of the least-partial-volumed probe vessel voxels (jugular maximum, which has less
  inflow than the SSS), converted so that a pure vessel reads 1;
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

**Circularity rule:** anything used as an anchor cannot be an outcome. White-matter $v_p$ has not
been an outcome in this cohort (PI, 2026-11-24), so it can be a fourth amplitude measure $X_4$ here. Physical anchors (vessel
lumen, dose pharmacokinetics) come first. A white-matter $v_p$ anchor would partly normalize away
real differences in white-matter $v_p$ between groups, so use it only when that is not a study
question.

### 6.3 Candidate AIFs: the PI's voxel-selection idea, re-aimed

Generate candidate AIFs from the network ranking: top-$N$ for several $N$, connected components
separately, artery vs sinus, the site population AIF, and a **kernel-SIME correction of the AutoAIF
curve** (§5.7 point 4) fitted to brain and extracranial channels. Rank them by:

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
| **0** | Prerequisites in §2. Done: baseline estimator identified, native frames confirmed, bug fix in PR #12. Left: decide pre- or post-fix masks for the baseline; choose fitted vs raw AIF; a Python Patlak refit that reproduces ROCKETSHIP's Ktrans on `smoke` | code; case list `smoke` | Refit does not reproduce ROCKETSHIP within noise: fix that first |
| **1** | Done for frame time and scan length (§3.6). Left: realistic $T_1$ saturation per protocol (FA 15°, each TR, 3.4 vs 5.7 relaxivity), each protocol's dose, and typical $v_p$, $K^{\text{trans}}$ from the standard derivative | protocol parameters; GM/WM medians from `dceprep-multihance_fix` | Leverage small at all sites: shape errors don't matter; go to Gate 3 only. Already true for peak errors at the 16-min sites |
| **2** | Measure $\hat\rho_j$ on the cohort: fit probe vessel voxels (not in the AIF mask) with each subject's AutoAIF AIF. Report the implied offset $\hat\rho_j \cdot \bar v_p$ against the between-subject SD of $K^{\text{trans}}$. Cross-check with the tissue $K$–$v_p$ slope (§5.5b) and with the extracranial multichannel residual (§5.7 point 2). Run on `small`, then `medium`, then `analyzable` | AutoAIF outputs, probe-vessel masks, Patlak maps, `desc-hmc_DCE` + unmasked VFAs for extracranial ROIs | Implied offset under about 10% of between-subject $K^{\text{trans}}$ SD: leverage error is not a problem |
| **3** | Scale: triple collocation (§6.2) gives the error variance of AutoAIF amplitude per site | dose, weight, jugular voxels (SSS as a check), optionally Hct | AutoAIF amplitude error variance is small next to biological spread |
| **4** | Tissue residual as a detector: corrupt real AIFs deliberately (saturation, tail ramp, delay, scale) and check that paired $\Delta R$ ranks them as §5.3–5.4 predicts | Gate 0 pipeline | It does not detect peak distortions on real data: drop it from candidate ranking |
| **5** | Correction: EB-shrunk $\rho^{\ast}_j$, scale calibration, candidate selection (including kernel SIME). Evaluate on $K^{\text{trans}}$ test–retest and against two baselines: uncorrected AutoAIF, and the site population AIF. Check that a known group difference is preserved. Use the 97 cross-site subjects to test that site differences shrink | case list `repeat` (187 subjects; within-site repeats at USCINI, USCSTEVENS, BANNER; none at CCIR) | It does not beat **both** baselines |

Gates 1–3 are measurement, not method development, and each can end the project cheaply.

**Backlog (for a general-purpose tool, not needed for subtle-leakage $K^{\text{trans}}$):**
the voxel-level mixture EM (v1 §7–§8, with the review's §4.5 fixes), arterial vs venous
discrimination, first-pass recovery for flow, the forward model for undersampled sites (review
§5.4), and FPCA shape shrinkage.

---

## 9. Open questions

Status after the data inventory ([data_inventory.md](data_inventory.md)):

1. **Answered (PI).** Manual masks are mostly arterial; a few are venous, usually the jugular. A
   per-session label still needs a look at the 118 masks, to find the venous ones.
2. **Answered.** Downstream uses the mean of the 20 top-voxel mask voxels on native frames, fitted to
   ROCKETSHIP's 4-parameter AIF model (§2 item 1). The soft mask is unused.
3. **Answered.** ROCKETSHIP (MATLAB, Gpufit), Patlak, voxel-wise, no delay fit, no smoothing,
   Hct 0.45, first frame dropped, automatic baseline and injection detection.
4. **Mostly answered.** Five protocol groups (data_inventory §2.1): FA 15°, frames 7.9–15.6 s,
   8.4–16.6 min, VFA $T_{10}$ maps with 5 flips (3 at GE16). Left: dose per site; the agent can only be
   inferred from relaxivity (5.7 MultiHance in 108 sessions, 3.4 otherwise).
5. **Answered (PI).** The coronal slabs cover to the top of the brain. The SSS is visible near the top
   (through-plane, strong inflow) and the jugular veins are visible (in-plane, less inflow). Arteries
   are in view; most AIF masks are arterial. Consequences: §5.5 probe choice.
6. **Answered.** 187 subjects have 2–5 sessions. Within-site repeats: USCINI 36 subjects, USCSTEVENS 10,
   BANNER 43, CCIR 0. 97 subjects have sessions at two or more sites.
7. **Answered.** Weight: yes, every session. Hematocrit and eGFR: no. Dose: not recorded.
8. **Answered (PI).** White-matter $v_p$ has not been an outcome, so it can serve as an extra scale
   anchor ($X_4$ in §6.2) for current studies. Revisit if a study makes it an outcome.
9. **Settled.** The pre-contrast window ends around frame 2. The "image 18" values in about 340 logs
   are a printing artifact of the detector (§2 item 6).
10. **Open.** Should the baseline for this project use the pre-fix (top-20 always) or post-fix
    (PR #12) AutoAIF masks?

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

Citations are from memory except Riabkov 2002, Fluckiger 2009, Wang 2004 and Zhu 2024, which were
checked against the PDFs (Drive folder `claude/AIF`). Check details before reusing them in a manuscript.

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
- Wang ZJ, Han Z, Liu KJR, Wang Y. Simultaneous estimation of kinetic parameters and the input
  function from DCE-MRI data: theory and simulation. *Proc. IEEE ISBI* 2004:996–999.
- Weinmann HJ, Laniado M, Mützel W. Pharmacokinetics of GdDTPA/dimeglumine after intravenous
  injection into healthy volunteers. *Physiol Chem Phys Med NMR* 1984;16:167–172.
- Zhu Y, Tran Q, Wang Y, Badawi RD, Cherry SR, Qi J, Abbaszadeh S, Wang G. Optimization-derived
  blood input function using a kernel method and its evaluation with total-body PET for brain
  parametric imaging. *NeuroImage* 2024;293:120611.
