"""Build nested case lists (smoke < small < medium < analyzable) and problem lists from the inventory.

    python3 tools/build_case_lists.py

Reads  <deriv>/inventory/session_inventory.tsv   (from tools/data_inventory.py)
       <deriv>/case_lists/manual_flags.tsv       (hand-curated; created empty if missing)
Writes <deriv>/case_lists/case_lists.json

<deriv> is $AUTOAIF_DERIV, default /media/network_mriphysics/USC-PPG/bids_ppg/derivatives/autoaif-population.
Selection is deterministic (fixed seed), so a rebuild from the same inventory gives the same lists.
The smoke, small and medium lists are nested and take one session per subject, stratified by protocol.
List meanings are in docs/data_inventory.md.
"""
import argparse
import csv
import datetime
import json
import os
import random
import subprocess
from collections import defaultdict

DERIV = os.environ.get("AUTOAIF_DERIV",
                       "/media/network_mriphysics/USC-PPG/bids_ppg/derivatives/autoaif-population")
HARD_FLAGS = ["no_raw_dce", "no_ktrans", "deriv_older_than_raw", "frame_implausible", "short_series"]
SOFT_FLAGS = ["aif_fit_r2_lt_0.8", "aif_T1_out_of_range", "no_t1map"]
PROTOCOLS = ["SIE16", "SIE08", "SIE10", "GE10", "GE16"]
PER_PROTOCOL = {"smoke": 1, "small": 4, "medium": 20}
SEED = 20261124
MANUAL_FLAGS_HEADER = ["participant_id", "session_id", "flag", "severity", "note"]


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


def read_tsv(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def key(r):
    return f"{r['participant_id']}/{r['session_id']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deriv", default=DERIV)
    args = ap.parse_args()
    inv_path = f"{args.deriv}/inventory/session_inventory.tsv"
    lists_dir = f"{args.deriv}/case_lists"
    manual_path = f"{lists_dir}/manual_flags.tsv"
    os.makedirs(lists_dir, exist_ok=True)
    if not os.path.exists(manual_path):
        with open(manual_path, "w") as f:
            f.write("\t".join(MANUAL_FLAGS_HEADER) + "\n")

    rows = read_tsv(inv_path)
    flags = {key(r): set(filter(None, r["flags"].split(";"))) for r in rows}
    hard = set(HARD_FLAGS)
    manual_notes = {}
    for m in read_tsv(manual_path):
        k = f"{m['participant_id']}/{m['session_id']}"
        flags.setdefault(k, set()).add(m["flag"])
        if m.get("severity", "").strip().lower() == "hard":
            hard.add(m["flag"])
        manual_notes.setdefault(m["flag"], {})[k] = m.get("note", "")

    by_key = {key(r): r for r in rows}
    analyzable = sorted(k for k in by_key if not flags[k] & hard)
    clean = sorted(k for k in analyzable if not flags[k])

    # nested stratified lists: per protocol, one session per subject, manual-mask sessions first
    rng = random.Random(SEED)
    order = {}
    for p in PROTOCOLS:
        cand = [k for k in clean if by_key[k]["protocol"] == p]
        rng.shuffle(cand)
        cand.sort(key=lambda k: by_key[k]["manual_aif"] != "True")      # stable: manual first, shuffled within
        seen, uniq = set(), []
        for k in cand:
            sub = by_key[k]["participant_id"]
            if sub not in seen:
                seen.add(sub)
                uniq.append(k)
        order[p] = uniq
    nested = {name: sorted(k for p in PROTOCOLS for k in order[p][:n]) for name, n in PER_PROTOCOL.items()}

    subs_multi = defaultdict(list)
    for k in analyzable:
        subs_multi[by_key[k]["participant_id"]].append(k)
    repeat = sorted(k for v in subs_multi.values() if len(v) >= 2 for k in v)

    lists = {
        "smoke": ("One clean session per protocol group; a manual AIF mask where one exists. For checking that "
                  "code runs end to end.", nested["smoke"]),
        "small": (f"{PER_PROTOCOL['small']} clean sessions per protocol group, one per subject, contains smoke. "
                  "For quick method comparisons.", nested["small"]),
        "medium": (f"{PER_PROTOCOL['medium']} clean sessions per protocol group (or all there are), one per "
                   "subject, contains small. For runs of an hour or so.", nested["medium"]),
        "manual_aif": ("Analyzable sessions that have a manual AIF mask (dceprep-manualAIF or _refresh). "
                       "Reference for AIF comparisons.",
                       sorted(k for k in analyzable if by_key[k]["manual_aif"] == "True")),
        "repeat": ("All analyzable sessions of subjects with two or more analyzable sessions. For "
                   "repeatability (Gate 5).", repeat),
        "clean": ("Sessions with no hard or soft flag.", clean),
        "analyzable": ("Sessions with no hard flag. The default full cohort.", analyzable),
        "exclude_default": ("Sessions with at least one hard flag. Leave out unless troubleshooting.",
                            sorted(k for k in by_key if flags[k] & hard)),
    }
    problems = {}
    for fl in sorted({f for s in flags.values() for f in s}):
        problems[fl] = {
            "severity": "hard" if fl in hard else "soft",
            "cases": sorted(k for k in by_key if fl in flags[k]),
        }
        if fl in manual_notes:
            problems[fl]["notes"] = manual_notes[fl]

    out = {
        "schema": "autoaif-population case lists v1",
        "generated": datetime.date.today().isoformat(),
        "generator": f"AutoAIF tools/build_case_lists.py @ {git_commit()}",
        "inventory": inv_path,
        "seed": SEED,
        "case_format": "sub-<label>/ses-<label>, relative to the BIDS raw root and to each derivative root",
        "protocol_groups": {
            "SIE16": "Siemens, ~15.4-15.8 s frames, 64 frames (~16.4 min)",
            "SIE08": "Siemens, ~7.9 s frames, 64 frames (~8.4 min)",
            "SIE10": "Siemens, ~9.8 s frames, 64 frames (~10.5 min)",
            "GE10": "GE, ~10.6 s frames, 64 frames (~11.3 min)",
            "GE16": "GE, ~15.6-16.9 s frames, 64 frames (~16.6 min)",
        },
        "flags": {"hard": sorted(hard), "soft": [f for f in SOFT_FLAGS if f not in hard]},
        "lists": {n: {"description": d, "n": len(c), "cases": c} for n, (d, c) in lists.items()},
        "problems": {fl: {**v, "n": len(v["cases"])} for fl, v in problems.items()},
    }
    path = f"{lists_dir}/case_lists.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")
    for n, v in out["lists"].items():
        print(f"  list {n}: {v['n']}")
    for n, v in out["problems"].items():
        print(f"  problem {n} ({v['severity']}): {v['n']}")


if __name__ == "__main__":
    main()
