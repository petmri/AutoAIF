"""Per-session inventory of the USC-PPG DCE data and its standard DCEPrep derivative.

Writes one row per session (raw BIDS or standard derivative) to a TSV on the network drive and
prints only aggregate counts, so no subject-level values reach the terminal.

    python3 tools/data_inventory.py                       # defaults below
    python3 tools/data_inventory.py --out /path/to/session_inventory.tsv

Paths come from arguments or environment variables:
    AUTOAIF_BIDS   BIDS root            (default /media/network_mriphysics/USC-PPG/bids_ppg)
    AUTOAIF_STD    standard derivative  (default <BIDS>/derivatives/dceprep-multihance_fix)
    AUTOAIF_DERIV  our derivative       (default <BIDS>/derivatives/autoaif-population)

Column meanings are in docs/data_inventory.md. Standard library and numpy only.
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import os
import re
import statistics
import struct
import subprocess

import numpy as np

BIDS = os.environ.get("AUTOAIF_BIDS", "/media/network_mriphysics/USC-PPG/bids_ppg")

# DICOM InstitutionName -> short site code. The same site appears under more than one name.
SITES = {
    "USCINI": "USCINI",
    "USC Stevens Hall Institute": "USCSTEVENS",
    "CENTER FOR CLINICAL IMAGING RESEARCH": "CCIR",
    "CCIR": "CCIR",
    "Banner Alzheimers Institute": "BANNER",
    "USC Department of Radiology": "USCRAD",
}

# Hard flags exclude a session by default; soft flags mark it for troubleshooting.
HARD_FLAGS = ("no_raw_dce", "no_ktrans", "deriv_older_than_raw", "frame_implausible", "short_series")
SOFT_FLAGS = ("aif_fit_r2_lt_0.8", "aif_T1_out_of_range", "no_t1map")


def git_commit():
    """Short commit hash, with '+uncommitted' when this script differs from that commit."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", here, "status", "--porcelain", "--", os.path.abspath(__file__)],
                                        text=True).strip()
        return h + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"


def nifti_nt(path):
    try:
        with gzip.open(path, "rb") as f:
            h = f.read(348)
        dim = struct.unpack("<8h", h[40:56])
        return dim[4] if dim[0] >= 4 else 1
    except Exception:
        return None


def frame_duration(sidecar, nt):
    """Frame duration in s, following ROCKETSHIP run_dce_cli.m (the code that fits Ktrans)."""
    d = sidecar
    if "RepetitionTimeExcitation" in d:
        t, src = d.get("RepetitionTime"), "RepetitionTime"
    elif "TemporalResolution" in d:
        t, src = d["TemporalResolution"], "TemporalResolution"
    elif "AcquisitionDuration" in d:
        t, src = d["AcquisitionDuration"], "AcquisitionDuration"
    elif "TriggerDelayTime" in d and nt:
        t, src = d["TriggerDelayTime"] / nt / 1000, "TriggerDelayTime/nt"
    else:
        return None, "none (ROCKETSHIP default 15.36)"
    if "NumberOfAverages" in d and t is not None:
        t *= d["NumberOfAverages"]
    return t, src


def protocol_group(vendor, frame_s):
    if not frame_s:
        return "unknown"
    v = {"Siemens": "SIE", "GE": "GE"}.get(vendor, "OTHER")
    if 7.5 <= frame_s < 8.5:
        return f"{v}08"
    if 9.5 <= frame_s < 11.5:
        return f"{v}10"
    if 15.0 <= frame_s < 17.5:
        return f"{v}16"
    return "other"


def _num(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def parse_logs(dce_dir):
    """Settings and AIF diagnostics from ROCKETSHIP logs and AIF_values.txt."""
    out = {}
    try:
        a = open(f"{dce_dir}/A_dceR1info.log", errors="ignore").read()
        out["hct"] = _num(r"hematocit \(0 to 1\.0\):\s*\n\s*([\d.]+)", a)
        out["fit_relaxivity"] = _num(r"relaxivity \(/mM/sec\):\s*\n\s*([\d.]+)", a)
        out["fit_tr_ms"] = _num(r"TR \(ms\):\s*\n\s*([\d.]+)", a)
        # As printed by ROCKETSHIP. Unreliable: in ~340 sessions the detector prints a degenerate value
        # (image ~18) although the concentrations show the baseline ends before the bolus.
        out["log_baseline_end"] = _num(r"Steady state time selected from image \d+ to image (\d+)", a, int)
        m = re.search(r"has removed: (\d+) of (\d+) AIF voxels", a)
        if m:
            out["aif_vox_removed"], out["aif_vox"] = int(m.group(1)), int(m.group(2))
        out["aif_T1_s"] = _num(r"Average Filtered AIF T1:\s*\n\s*([\d.]+)", a)
    except OSError:
        pass
    try:
        b = open(f"{dce_dir}/B_dcefitted_R1info.log", errors="ignore").read()
        out["fit_frame_s"] = _num(r"time resolution \(sec\)\s*\n\s*([\d.]+)", b)
        out["fit_inj_min"] = _num(r"start injection \(min\):\s*\n\s*([\d.\-]+)", b)
        out["fit_aif_model"] = _num(r"fit AIF to model:\s*\n\s*(\d+)", b, int)
        r2 = re.findall(r"Adjusted R\^2 of AIF fit = ([\d.\-eE]+)", b)
        out["aif_fit_r2"] = float(r2[-1]) if r2 else None
        m = re.search(r"AIF mmol:\s*\n(.*?)\n\s*\n", b, re.S)
        if m:
            c = np.array([float(x) for x in m.group(1).split()])
            out["aif_mM_peak"] = round(float(c.max()), 3)
            out["aif_mM_tail"] = round(float(c[-5:].mean()), 3)
    except OSError:
        pass
    try:
        v = np.loadtxt(f"{dce_dir}/AIF_values.txt")
        out["aif_peak_frame"] = int(np.argmax(v))
    except (OSError, ValueError):
        pass
    return out


def session_sites(bids_raw):
    sites = {}
    for f in glob.glob(f"{bids_raw}/sub-*/sub-*_sessions.tsv"):
        sub = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            for s in csv.DictReader(fh, delimiter="\t"):
                inst = s.get("institution", "n/a")
                w = s.get("weight")
                sites[(sub, s["session_id"])] = (inst, SITES.get(inst, inst), w not in (None, "", "n/a"))
    return sites


def flags_for(r):
    f = []
    if not r["raw_dce"]:
        f.append("no_raw_dce")
    if r["raw_dce"] and not r.get("ktrans"):
        f.append("no_ktrans")
    if r.get("raw_mtime") and r.get("ktrans_mtime") and r["ktrans_mtime"] < r["raw_mtime"]:
        f.append("deriv_older_than_raw")
    fs = r.get("frame_s")
    if r["raw_dce"] and (fs is None or not 4 <= fs <= 20):
        f.append("frame_implausible")
    if r.get("nt") is not None and r["nt"] < 60:
        f.append("short_series")
    if r.get("aif_fit_r2") is not None and r["aif_fit_r2"] < 0.8:
        f.append("aif_fit_r2_lt_0.8")
    t1 = r.get("aif_T1_s")
    if t1 is not None and not 1.2 <= t1 <= 2.5:
        f.append("aif_T1_out_of_range")
    if r.get("deriv") and not r.get("t1map"):
        f.append("no_t1map")
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids", default=BIDS)
    ap.add_argument("--std", default=os.environ.get("AUTOAIF_STD"))
    ap.add_argument("--out", default=None, help="TSV path (default <deriv>/inventory/session_inventory.tsv)")
    args = ap.parse_args()
    raw = f"{args.bids}/sourcedata/raw"
    std = args.std or f"{args.bids}/derivatives/dceprep-multihance_fix"
    deriv = os.environ.get("AUTOAIF_DERIV", f"{args.bids}/derivatives/autoaif-population")
    out = args.out or f"{deriv}/inventory/session_inventory.tsv"

    keys = {tuple(p.split("/")[-4:-2]) for p in glob.glob(f"{raw}/sub-*/ses-*/dce/*_DCE.json")}
    keys |= {tuple(p.split("/")[-2:]) for p in glob.glob(f"{std}/sub-*/ses-*")}
    sites = session_sites(raw)
    manual_roots = [f"{args.bids}/derivatives/dceprep-manualAIF", f"{args.bids}/derivatives/dceprep-manualAIF_refresh"]

    rows = []
    for sub, ses in sorted(keys):
        r = {"participant_id": sub, "session_id": ses}
        js = sorted(glob.glob(f"{raw}/{sub}/{ses}/dce/*_DCE.json"))
        r["raw_dce"] = bool(js)
        if js:
            d = json.load(open(js[0]))
            nii = js[0][:-5] + ".nii.gz"
            r["nt"] = nifti_nt(nii)
            r["raw_mtime"] = int(os.path.getmtime(nii)) if os.path.exists(nii) else None
            r["vendor"] = d.get("Manufacturer")
            r["series"] = d.get("SeriesDescription")
            r["flip_deg"] = d.get("FlipAngle")
            r["relaxivity"] = d.get("relaxivity")
            fs, src = frame_duration(d, r["nt"])
            r["frame_s"] = round(fs, 3) if fs else None
            r["frame_src"] = src
            r["scan_min"] = round(fs * r["nt"] / 60, 2) if fs and r["nt"] else None
            r["protocol"] = protocol_group(r["vendor"], r["frame_s"])
            anat = f"{raw}/{sub}/{ses}/anat"
            r["n_vfa"] = len(glob.glob(f"{anat}/*VFA.nii.gz"))
            r["t1w"] = bool(glob.glob(f"{anat}/*T1w.nii.gz"))
        inst, site, has_w = sites.get((sub, ses), ("n/a", "n/a", False))
        r["institution"], r["site"], r["has_weight"] = inst, site, has_w
        dd = f"{std}/{sub}/{ses}"
        r["deriv"] = os.path.isdir(dd)
        if r["deriv"]:
            have = lambda pat: bool(glob.glob(f"{dd}/{pat}"))
            r.update(
                ktrans=have("dce/*_Ktrans.nii"), vp=have("dce/*_vp.nii"),
                aif_topvoxels=have("dce/*AIFtopvoxels_mask.nii*"), aif_float=have("dce/*AIFfloat_mask.nii*"),
                hmc_dce=have("dce/*desc-hmc_DCE.nii.gz"), bfcz_dce=have("dce/*desc-bfcz_DCE.nii.gz"),
                ct=have("dce/dcedynamicCt.nii.gz"), t1map=have("anat/*_space-DCEref_T1map.nii"),
                wmparc=have("anat/*desc-wmparc.nii.gz"), casereport=have("reports/*casereport.html"),
            )
            k = glob.glob(f"{dd}/dce/*_Ktrans.nii")
            r["ktrans_mtime"] = int(os.path.getmtime(k[0])) if k else None
            r.update(parse_logs(f"{dd}/dce"))
        r["manual_aif"] = any(
            glob.glob(f"{m}/{sub}/{ses}/dce/*{p}*")
            for m in manual_roots for p in ("desc-AIF_mask", "trainingAIF_mask", "label-AIF_mask")
        )
        r["flags"] = ";".join(flags_for(r))
        rows.append(r)

    fields = ["participant_id", "session_id", "protocol", "site", "institution", "vendor", "series", "frame_s",
              "frame_src", "nt", "scan_min", "flip_deg", "relaxivity", "n_vfa", "t1w", "has_weight", "raw_dce", "deriv",
              "ktrans", "vp", "aif_topvoxels", "aif_float", "hmc_dce", "bfcz_dce", "ct", "t1map", "wmparc",
              "casereport", "manual_aif", "hct", "fit_relaxivity", "fit_tr_ms", "fit_frame_s", "fit_inj_min",
              "fit_aif_model", "log_baseline_end", "aif_peak_frame", "aif_vox", "aif_vox_removed", "aif_T1_s",
              "aif_fit_r2", "aif_mM_peak", "aif_mM_tail", "raw_mtime", "ktrans_mtime", "flags"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        f.write(f"# generated by AutoAIF tools/data_inventory.py @ {git_commit()} from {args.bids} and {std}\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("n/a" if r.get(k) is None else r.get(k)) for k in fields})

    # ---- aggregates only
    C = collections.Counter
    rr = [r for r in rows if r["raw_dce"]]
    print(f"wrote {out}")
    print(f"sessions: {len(rows)} (raw DCE {len(rr)}, standard derivative {sum(r['deriv'] for r in rows)}); "
          f"subjects with raw DCE: {len({r['participant_id'] for r in rr})}")
    print("protocol x site:", dict(C((r["protocol"], r["site"]) for r in rr)))
    print("sessions per subject:", dict(sorted(C(C(r["participant_id"] for r in rr).values()).items())))
    for fl in HARD_FLAGS + SOFT_FLAGS:
        print(f"flag {fl}: {sum(fl in r['flags'].split(';') for r in rows)}")
    print("manual AIF masks:", sum(r["manual_aif"] for r in rows))
    for p in sorted({r["protocol"] for r in rr}):
        g = [r for r in rr if r["protocol"] == p]
        med = lambda k: statistics.median([r[k] for r in g if r.get(k) is not None] or [float("nan")])
        print(f"  {p}: n={len(g)} frame {med('frame_s'):.2f} s, scan {med('scan_min'):.1f} min, "
              f"AIF T1 {med('aif_T1_s'):.2f} s, AIF fit R2 {med('aif_fit_r2'):.3f}, "
              f"AIF peak {med('aif_mM_peak'):.2f} mM, tail {med('aif_mM_tail'):.2f} mM")


if __name__ == "__main__":
    main()
