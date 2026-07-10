#!/usr/bin/env python
"""
demo.py -- The Trust Layer, live in 60 seconds.
================================================

A single script for a live audience. It runs the whole trust layer on a
realistic small-n biology dataset and narrates what it finds:

    1. A scientist's leaky notebook is statically audited (no code executed).
    2. The dataset is audited against a leakage taxonomy.
    3. An honest, group-aware cross-validation ladder shows how each leak
       inflated the reported score.
    4. A calibrated model (linear vs nonlinear, auto-selected) reports the
       real, leakage-safe performance -- with a selection-honest check.
    5. The finale: on a dataset with pure nonlinear (XOR) structure, the tool
       automatically switches from the linear model to the nonlinear one.

Run it:

    python demo.py                 # full demo on the bundled sample data
    python demo.py --fast          # skip the nonlinear model (model="logreg")
    python demo.py --data DIR      # point at your own {expression_matrix,sample_metadata}.csv

Everything prints to the console; a full markdown report is written next to
the data (sample_dataset/trust_out/trust_report.md).
"""
import os
import sys
import time
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Console helpers -- make the live output easy to read from the back row.
# --------------------------------------------------------------------------
class C:
    B = "\033[1m"; DIM = "\033[2m"; R = "\033[0m"
    RED = "\033[31m"; GRN = "\033[32m"; YEL = "\033[33m"; CYN = "\033[36m"

    @staticmethod
    def off():
        for k in ("B", "DIM", "R", "RED", "GRN", "YEL", "CYN"):
            setattr(C, k, "")


if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    C.off()


def rule(title):
    print(f"\n{C.B}{C.CYN}{'='*72}{C.R}")
    print(f"{C.B}{C.CYN}  {title}{C.R}")
    print(f"{C.B}{C.CYN}{'='*72}{C.R}")


def step(msg):
    print(f"{C.B}> {msg}{C.R}")


def note(msg):
    print(f"{C.DIM}  {msg}{C.R}")


class timed:
    """Context manager that prints how long a stage took."""
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        print(f"{C.DIM}  ({self.label} took {time.time()-self.t:.1f}s){C.R}")


# --------------------------------------------------------------------------
# Module loading -- the trust layer's modules import each other by name, so we
# preload them into sys.modules in dependency order (works from any cwd).
# --------------------------------------------------------------------------
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_trust_layer():
    for name in ("trust_layer", "notebook_audit", "trust_tasks", "trust_audit"):
        path = os.path.join(HERE, f"{name}.py")
        if not os.path.exists(path):
            sys.exit(f"Missing {name}.py next to demo.py -- run from the project bundle.")
        mod = _load(name, path)
    return sys.modules["trust_audit"]


# --------------------------------------------------------------------------
def run_audit(ta, data_dir, model):
    import pandas as pd
    expr = os.path.join(data_dir, "expression_matrix.csv")
    meta = os.path.join(data_dir, "sample_metadata.csv")
    nb = os.path.join(data_dir, "analysis_leaky.py")
    X = pd.read_csv(expr).drop(columns=["sample_id"])
    md = pd.read_csv(meta)
    y = (md["diagnosis"] == "disease").astype(int).values

    rule("THE SCENARIO")
    step(f"A serum-proteomics study: {C.B}{X.shape[0]} samples x {X.shape[1]} proteins{C.R}.")
    note(f"{md['patient_id'].nunique()} patients, {md['replicate'].max()} replicates each; "
         f"run across {md['plate'].nunique()} plates.")
    note("A scientist wrote analysis_leaky.py and reports a near-perfect AUC.")
    note("Question for the audience: is that number real?")

    rule("WHAT THE TRUST LAYER DOES  (one function call)")
    step("trust_audit(X, y, groups=patient_id, batch=plate, notebook='analysis_leaky.py')")
    with timed("full audit"):
        res = ta.trust_audit(
            X, y,
            groups=md["patient_id"].values,
            batch=md["plate"].values,
            notebook=nb if os.path.exists(nb) else None,
            model=model,
            outdir=os.path.join(data_dir, "trust_out"),
        )

    # 1. code audit
    if "code_audit" in res:
        ca = res["code_audit"]
        rule("1. CODE AUDIT  (static -- no code was executed)")
        color = C.RED if ca["n_critical"] else C.GRN
        step(f"{color}{ca['n_critical']} leakage pattern(s){C.R} found by reading the notebook:")
        for f in ca["findings"]:
            if f["severity"] == "critical":
                loc = f"line {f['line']}" if f.get("line") else "code"
                print(f"    {C.RED}x{C.R} {loc}: {f['check']}")

    # 2. data audit
    da = res["data_audit"]
    rule("2. DATA AUDIT  (dataset vs leakage taxonomy)")
    step(f"{C.RED}{da['n_critical']} critical{C.R}, {C.YEL}{da['n_warning']} warning{C.R}:")
    for f in da["findings"]:
        if f["severity"] in ("critical", "warning"):
            mark = f"{C.RED}x{C.R}" if f["severity"] == "critical" else f"{C.YEL}!{C.R}"
            print(f"    {mark} {f['check']}: {f['message'][:78]}")

    # 3. honest CV ladder
    rule("3. HOW EACH LEAK INFLATED THE SCORE")
    names = {"naive": "Naive (preprocessing + selection on ALL data)",
             "groupkfold": "+ subject-safe (GroupKFold on patient)",
             "groupkfold_batchcorrected": "+ in-fold batch centering (fully honest)",
             "honest_randomsplit": "Honest (fit inside folds)"}
    for k, (m, s) in res["cv_ladder"].items():
        bar = "#" * int(round(m * 40))
        print(f"    {names.get(k, k):<44} {C.B}{m:.2f}{C.R} +/- {s:.2f}  {C.DIM}{bar}{C.R}")

    # 4. honest model
    m = res["model"]
    rule("4. THE HONEST NUMBER  (calibrated, leakage-safe)")
    naive = res["cv_ladder"]["naive"][0]
    step(f"Reported (naive):   {C.RED}AUC {naive:.2f}{C.R}")
    step(f"Honest (real):      {C.GRN}AUC {m['auc']:.2f}{C.R}   [{m['name']}]")
    print(f"\n    {C.B}=> Leakage inflated the AUC by "
          f"{C.RED}{res['inflation']:+.2f}{C.R}{C.B}. "
          f"That gap was an artifact, not biology.{C.R}")
    note(f"Brier score {m['brier']:.2f} (0.25 = uninformative) -- probabilities are calibrated.")
    cand = m.get("candidates") or {}
    if len(cand) > 1:
        print()
        step("Both a linear and a nonlinear model were fit through the same honest folds:")
        for name, cm in cand.items():
            win = f" {C.GRN}<- selected{C.R}" if name == m["name"] else ""
            print(f"    {name:<22} AUC {cm['auc']:.2f}{win}")
    ms = res.get("model_selection")
    if ms:
        note(f"Even choosing the better model is checked: nested selection AUC "
             f"{ms['nested_selection_auc']:.2f} (selection optimism "
             f"{ms['selection_optimism']:+.2f}).")

    print(f"\n{C.DIM}  Full markdown report written to "
          f"{os.path.join(data_dir, 'trust_out', 'trust_report.md')}{C.R}")
    return res


def run_finale(tl):
    """Show the tool adapting: on a pure nonlinear (XOR) signal it auto-switches
    from the linear model to the nonlinear one."""
    import numpy as np
    rule("FINALE: THE TOOL ADAPTS TO THE DATA")
    note("A dataset where the signal is a pure interaction (XOR) -- invisible to a")
    note("linear model, obvious to a tree ensemble. Same one call, model='auto'.")

    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = ((x1 > 0) ^ (x2 > 0)).astype(int)                    # XOR label
    X = np.column_stack([x1, x2, rng.standard_normal((n, 20))])  # + noise features

    with timed("auto model selection"):
        r = tl.fit_trust_model(X, y, model="auto")
    for name, cm in r["candidates"].items():
        win = f" {C.GRN}<- auto-selected{C.R}" if name == r["selected"] else ""
        col = C.GRN if name == r["selected"] else C.DIM
        print(f"    {col}{name:<22} AUC {cm['auc']:.2f}{C.R}{win}")
    step(f"The linear model is at chance; the tool correctly reports "
         f"{C.GRN}{r['selected']} (AUC {r['auc']:.2f}){C.R}.")


def main():
    ap = argparse.ArgumentParser(description="The Trust Layer -- live demo")
    ap.add_argument("--data", default=os.path.join(HERE, "sample_dataset"),
                    help="folder with expression_matrix.csv + sample_metadata.csv")
    ap.add_argument("--fast", action="store_true",
                    help="linear model only (skip the nonlinear fit)")
    ap.add_argument("--no-finale", action="store_true", help="skip the XOR finale")
    args = ap.parse_args()

    t0 = time.time()
    print(f"{C.B}THE TRUST LAYER{C.R} -- honest ML for small-data biology")
    ta = load_trust_layer()
    tl = sys.modules["trust_layer"]

    run_audit(ta, args.data, model="logreg" if args.fast else "auto")
    if not args.no_finale:
        run_finale(tl)

    rule("BOTTOM LINE")
    print(f"  A biologist points the tool at their data and notebook, and gets back:")
    print(f"  {C.B}the real performance, what inflated the reported one, and how "
          f"confident the model actually is{C.R} -- in plain language, in one call.")
    print(f"\n{C.DIM}  (whole demo ran in {time.time()-t0:.1f}s){C.R}\n")


if __name__ == "__main__":
    main()
