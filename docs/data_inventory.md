# Data available for the population AIF work (USC-PPG)

**Status:** first inventory, 2026-11-24. It describes what exists on the network drive and what the
standard pipeline has already computed, so the work in
[population_aif_shrinkage.md](population_aif_shrinkage.md) reuses it instead of redoing it.
Numbers here are cohort-level counts from
[tools/data_inventory.py](../tools/data_inventory.py). Subject-level tables and case lists live on
the network drive (§6), not in this repo. Results go in eLabFTW (team VELA).

---

## 1. Where things are

| What | Path | Notes |
|---|---|---|
| BIDS root | `/media/network_mriphysics/USC-PPG/bids_ppg` | `CHANGES` is the curation log |
| Raw BIDS | `sourcedata/raw/sub-*/ses-*/{anat,dce}` | DCE, VFA flips, T1w. `participants.tsv`, `sub-*_sessions.tsv` |
| Session curation | `sourcedata/dicom_session_map.tsv` | Decides which DICOM folder becomes which session: include / exclude / hold |
| Excluded raw | `EXCLUDED/raw_2026-10-20/` | 16 duplicated, mislabeled or held sessions (CHANGES 1.2.0) |
| **Standard derivative** | `derivatives/dceprep-multihance_fix/` | DCEPrep + ROCKETSHIP (MATLAB, Gpufit). Fit logs dated June 2026 |
| Manual AIF masks | `derivatives/dceprep-manualAIF/`, `dceprep-manualAIF_refresh/` | 118 sessions with a manual or training mask |
| Other pipeline variants | `derivatives/dceprep-*` (34 folders) | AutoAIF loss variants (`autoAIF_huber*`), smoothing, no-lower-bound, gpufit versions, left/right |
| Cohort tables | `derivatives/spreadsheets/<variant>/dataset_ktrans-*.xlsx` | Regional Ktrans per session for each variant |
| Population QC reports | `derivatives/reports/EZQCreport*-<variant>_<date>.pdf` | Includes `_exclude` versions |
| FreeSurfer / FastSurfer | `derivatives/freesurfer*`, `derivatives/fastsurfer` | `wmparc` is already resampled into DCE space in the standard derivative |
| Earlier AIF population work | `derivatives/average_aif_curve.csv`; `code/all_aifs.nii.gz` (`code/compile_AIFs.sh`) | A 64-point mean AIF in signal-ratio units; a sum of top-voxel masks on one grid, not registered. Provenance not documented |
| **Our outputs** | `derivatives/autoaif-population/` | Created 2026-11-24. See §6 |

## 2. Cohort

- 642 subjects, 933 DCE sessions in raw BIDS. All 933 are `include` in the session map.
- Sessions per subject: 1: 455, 2: 116, 3: 44, 4: 21, 5: 6. **187 subjects have repeat sessions.**
- **97 subjects were scanned at more than one site.** Repeat subjects by site set: USCINI only 36,
  USCSTEVENS only 10, BANNER only 43, USCINI+USCSTEVENS 63, USCINI+USCRAD 27, other combinations 8.
  CCIR has no repeat sessions.
- 3 T throughout. Siemens 706 sessions, GE 227. Flip angle 15° throughout. Slices 5 mm.
- Body weight is recorded for every session (`sub-*_sessions.tsv`). Dose, hematocrit and eGFR are not
  in the BIDS data.

### 2.1 Protocols

The protocol group is set by vendor and frame duration. Frame duration follows the same sidecar logic
as ROCKETSHIP (`run_dce_cli.m`), and it matches the value ROCKETSHIP logged in all 908 sessions that have
a log.

| Group | Site (sessions) | Series | Frame | Frames | Scan | VFA flips | Agent (relaxivity) | Manual AIF |
|---|---|---|---|---|---|---|---|---|
| SIE16 | USCINI 313, USCSTEVENS 165 | `t1_vibe_cor_dynamic_waterext`, `t1_vibe_cor_dyn`, `t1_vibe_tra_dyn` | 15.36 s (some 15.6–15.8) | 64 | 16.4 min | 5 | mostly 3.4; 38 at 5.7 | 35 |
| SIE08 | CCIR 182 | `t1_vibe_cor_dyn` | 7.90 s | 64 | 8.4 min | 5 | 3.4 | 15 |
| SIE10 | USCINI 43, USCSTEVENS 3 | `t1_vibe_cor_dynamic_waterext` | 9.81 s | 64 | 10.5 min | 5 | 3.4 | 10 |
| GE10 | BANNER 140 | `DCE Hippo 64 phases+C` | 10.62 s | 64 | 11.3 min | 5 | 3.4 | 21 |
| GE16 | USCRAD 80, BANNER 4 | `DCE Hippo 64 phases` | 15.58 s (to 16.9) | 64 | 16.6 min | **3** | mostly 5.7 | 36 |
| other | BANNER 2, USCRAD 1 | GE | **683–691 s** (timing read wrong) | 64 | n/a | | | |

Relaxivity 5.7 (MultiHance) is used in 108 sessions: USCRAD 70 and USCINI 38. Everything else uses
3.4. Sites are DICOM `InstitutionName`, with `CENTER FOR CLINICAL IMAGING RESEARCH` and `CCIR` merged.

Matrix and voxel size: Siemens 320×320×14 (0.55×0.55×5 mm) or 240×320×16; GE 256×256×12
(0.70×0.70×5 mm). Coronal slabs of 60–80 mm (PI, 2026-11-24):

- The coronal acquisitions reach the top of the brain, so the **superior sagittal sinus** is visible
  near the top. It runs through the slice plane, so it has a strong inflow effect.
- The **jugular veins** are visible too. They run in-plane, so their inflow effect is smaller.
- The manual AIF masks are **mostly arterial**; a few are venous, usually the jugular.

## 3. What the standard derivative already computes

Per session, in `dceprep-multihance_fix/sub-*/ses-*/` (prefix `sub-X_ses-Y_` left out):

| Step | Files | Reuse for |
|---|---|---|
| Motion correction (mcflirt) | `dce/desc-hmc_DCE.nii.gz`, `desc-hmc_DCEref`, `.par`, `desc-hmc_maxdisp.txt` | **Only full field-of-view DCE**: extracranial tissue and vessels outside the brain mask |
| Bias correction, z-normalization | `dce/desc-bfc_DCE`, `desc-bfcz_DCE` (masked to brain + AIF), `desc-biases_DCE`, `desc-meanbias_DCE` | Brain curves |
| VFA registration, bias correction | `anat/flip-0N_space-DCEref_*VFA*`, `from-VFA0N_to-DCEref.mat` | $T_{10}$ outside the brain (unmasked VFAs are kept) |
| $T_{10}$ map | `anat/space-DCEref_T1map.nii` (+ `CIlow`, `CIhigh`, `rsquared`) | $T_{10}$ for concentration |
| Tissue masks | `anat/space-DCEref_label-{GM,WM,CSF}_mask`, `label-{GM,WM}QC`, `desc-wmparc`, `desc-brain_mask`, `desc-brainAIF_mask` | ROIs, probe-region exclusion |
| AutoAIF | `dce/desc-AIFfloat_mask.nii` (network output), `dce/desc-AIFtopvoxels_mask.nii` (20 voxels), `desc-AIF_T1map` | Baseline AIF, candidate generation |
| Concentration | `dce/dcedynamicCt.nii.gz` | Brain tissue curves |
| Patlak maps | `dce/Ktrans.nii`, `vp.nii`, `*_ci_low/high`, `sse.nii`, `seg-{GM,WM,CSF}_Ktrans` | Baseline outcome, $\hat K$–$\hat v_p$ slope |
| Fit record | `dce/A_dceR1info.log`, `B_dcefitted_R1info.log`, `dce_patlak_fit.log` (+ `.mat`), `AIF_values.txt` | Settings, AIF in mM, AIF fit $R^2$ |
| Reports | `reports/desc-casereport.html`, `figures/` | Visual QC |

**How Ktrans is fitted today** (from the ROCKETSHIP logs):

- AIF = mean of the **20 top-voxel** mask voxels, after an SNR filter (threshold 1; it removed no voxel
  in any session). Because of the `main_vif.py:93` bug
  ([petmri/AutoAIF#12](https://github.com/petmri/AutoAIF/pull/12)), every mask in this derivative is
  top-20. The 5-voxel fallback never ran.
- The soft (float) mask is **not** used downstream.
- Hematocrit fixed at 0.45. Relaxivity, TR and frame duration come from the DCE sidecar.
- First frame dropped (start repetition 2). Baseline end and injection time found automatically. The
  pre-contrast window ends around frame 2, usually 1–2 frames before the AIF peak (PI). In about 340
  sessions, mostly SIE16, the A-stage log prints a degenerate detector result instead
  (`Optimal t: 18`, local minimum and maximum both 18, "image 1 to image 18"). That printed value is not
  the window used: the logged AIF concentrations over those frames are well above zero (median 1 mM),
  which could not happen if they were the baseline. The inventory keeps the printed value as
  `log_baseline_end` and does not use it.
- **The AIF is fitted to a 4-parameter model before use** (`fit AIF to model: 1`, all 908 logged
  sessions). The fitted AIF, not the raw curve, drives Ktrans. ROCKETSHIP now has `--aif-mode raw`
  (commit `7127152`) to switch this.
- Patlak, voxel-wise, no delay fit, no temporal or spatial smoothing.

**Implications for the plan:**

- Kinetic fits already use the native frames. The 32-frame resampling in AutoAIF only affects voxel
  selection, which settles plan §2 item 3.
- Extracranial tissue (temporalis muscle, scalp) is in `desc-hmc_DCE` and the unmasked VFAs, but not
  in `bfcz_DCE`, `dcedynamicCt` or the $T_1$ map. Using it (plan §5.7) means converting those voxels
  to concentration ourselves.
- AIF errors reach Ktrans after passing through the 4-parameter AIF model. That model limits which
  error shapes can survive: a tail ramp has to look like a change in the fitted washout.

## 4. Session flags

`tools/data_inventory.py` sets these per session. **Hard** flags exclude a session by default;
**soft** flags mark it for troubleshooting.

| Flag | Kind | Sessions | Where | Meaning |
|---|---|---|---|---|
| `no_raw_dce` | hard | 26 | n/a | Derivative with no raw session: excluded or reconverted after the run (CHANGES 1.2.0). Stale |
| `no_ktrans` | hard | 25 | mostly SIE16 USCSTEVENS (13) | Raw session but no Ktrans map |
| `short_series` | hard | 14 | all groups | Fewer than 60 frames |
| `frame_implausible` | hard | 2 | GE (BANNER) | Frame duration outside 4–20 s (683–691 s read from the sidecar) |
| `deriv_older_than_raw` | hard | 0 | | Ktrans older than the raw NIfTI (none today; the reconverted `sub-1101590/ses-02` has no raw DCE now and is caught by `no_raw_dce`) |
| `aif_T1_out_of_range` | soft | 155 | all groups | Mean pre-contrast $T_1$ of the AIF voxels outside 1.2–2.5 s. Blood at 3 T is about 1.6–1.7 s |
| `no_t1map` | soft | 53 | | No $T_1$ map in the derivative |
| `aif_fit_r2_lt_0.8` | soft | 43 | | ROCKETSHIP's 4-parameter AIF fit has adjusted $R^2 < 0.8$ |

**Pre-contrast AIF $T_1$ by protocol:** SIE16 and SIE10 have a median of 2.08 s, the other groups
1.53–1.74 s. This is a hint of partial volume with longer-$T_1$ tissue or CSF, or of a $T_1$ mapping
bias at those sites.

## 5. What is not available

- **Dose.** Body weight is available for every session, but the injected dose and the agent per
  session are not; the agent can only be inferred from the relaxivity. The dose anchor in plan §6.2
  needs the protocol's standard dose confirmed per site.
- **Hematocrit and eGFR:** not in the data. Hct is fixed at 0.45 in every fit.
- **Per-session vessel labels.** Which manual masks are venous (jugular) is known only in general
  (§2.1); a per-session label needs a look at the 118 masks.
- Acquisition dates exist in `sub-*_sessions.tsv` (`acq_time`). Use them only as intervals computed
  in code; never print them.

## 6. Our outputs: `derivatives/autoaif-population/`

Rules:

1. **Never write into `dceprep-multihance_fix`** or any other existing derivative. Read their files in
   place by path.
2. Every output records the AutoAIF commit that made it. Commit and push before a run whose results
   go in eLabFTW. The scripts write `+uncommitted` when they were not committed.
3. Follow the BIDS and DCEPrep naming the lab already uses: `sub-X/ses-Y/{anat,dce}/`, `space-DCEref`,
   and a `desc-<method>` entity for each variant.

```
autoaif-population/
├── dataset_description.json          BIDS derivative description
├── inventory/session_inventory.tsv   one row per session (tools/data_inventory.py)
├── case_lists/
│   ├── case_lists.json               lists and problem lists (tools/build_case_lists.py)
│   └── manual_flags.tsv              hand-curated flags, merged on every rebuild
└── (planned) sub-X/ses-Y/dce/
        sub-X_ses-Y_desc-<method>_aif.tsv        AIF curve: time_s, conc_mM (+ sidecar .json)
        sub-X_ses-Y_desc-<method>_Ktrans.nii.gz  refits with an alternative AIF
        sub-X_ses-Y_desc-probe_mask.nii.gz       probe-vessel voxels (plan §5.5)
    group/<analysis>_<date>.tsv                  cohort tables (rho_j, scale anchors)
```

The planned names are a proposal. Rename the folder or the entities now, before anything depends on
them.

### 6.1 Case lists

`case_lists.json` holds `lists` and `problems`. Cases are `"sub-<label>/ses-<label>"` strings and
resolve against the raw root and every derivative root. Selection uses a fixed seed, so a rebuild
from the same inventory gives the same lists.

| List | Sessions | Use |
|---|---|---|
| `smoke` | 5 | One clean session per protocol group, with a manual mask where possible. Does the code run? |
| `small` | 20 | 4 per group, one session per subject; contains `smoke`. Quick comparisons |
| `medium` | 100 | 20 per group, one session per subject; contains `small`. Runs of about an hour |
| `manual_aif` | 117 | Analyzable sessions with a manual AIF mask. Reference comparisons |
| `repeat` | 444 | All analyzable sessions of subjects with ≥2 of them. Repeatability, cross-site pairs |
| `clean` | 704 | No flag at all |
| `analyzable` | 893 | No hard flag. The default full cohort |
| `exclude_default` | 66 | At least one hard flag |

`problems` has one entry per flag, with its severity and its cases.

To add a problem case by hand, append a row to `manual_flags.tsv`
(`participant_id  session_id  flag  severity  note`, where severity is `hard` or `soft`) and rebuild.
Hand flags survive rebuilds of the inventory.

```python
import json
L = json.load(open("/media/network_mriphysics/USC-PPG/bids_ppg/derivatives/autoaif-population/case_lists/case_lists.json"))
cases = L["lists"]["small"]["cases"]          # ["sub-.../ses-..", ...]
```

### 6.2 Regenerate

```bash
python3 tools/data_inventory.py
```

```bash
python3 tools/build_case_lists.py
```

Paths default to the network drive. Override them with `AUTOAIF_BIDS`, `AUTOAIF_STD` and
`AUTOAIF_DERIV`, or with `--bids`, `--std`, `--out` and `--deriv`.
