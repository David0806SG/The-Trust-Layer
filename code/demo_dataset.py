"""
demo_dataset.py
===============
Generate a canonical "small-n biology" dataset that carries the leakage traps
a bench scientist actually hits:

  * high-dimensional, small-n  (n=160 rows, p=800 features -- e.g. a proteomics
    or methylation panel)
  * repeated measures: 40 subjects, 4 replicate rows each (subject leakage)
  * a batch/plate effect that is partially CONFOUNDED with the outcome
    (batch leakage) and imprints a strong signature on many features
  * only a WEAK true biological signal in a handful of features

A naive analysis (global scaling + global feature selection + random CV) reads
the batch signature and the replicate structure and reports a near-perfect AUC.
An honest, group-aware analysis reveals the real (modest) signal.
"""
import numpy as np
import pandas as pd


def make_biology_dataset(seed=7):
    rng = np.random.default_rng(seed)

    n_subjects = 40
    reps = 4                       # replicate measurements per subject
    n = n_subjects * reps          # 160 rows
    p = 800                        # features >> samples

    # --- subject-level truth ------------------------------------------------
    subject_label = rng.integers(0, 2, size=n_subjects)          # disease vs control
    # batch assignment confounded with label: disease subjects mostly batch 1
    batch_prob = np.where(subject_label == 1, 0.62, 0.38)
    subject_batch = (rng.random(n_subjects) < batch_prob).astype(int)

    subject_ids = np.repeat(np.arange(n_subjects), reps)
    y = subject_label[subject_ids]
    batch = subject_batch[subject_ids]

    # --- features -----------------------------------------------------------
    X = rng.normal(0, 1, size=(n, p))

    # MODEST but real biological signal in first 25 features (effect ~1.3 sd)
    n_true = 25
    X[:, :n_true] += (y[:, None] * 1.3)

    # subject-level random effect: replicates of a subject are correlated
    # (this is what a random split exploits -- near-duplicate replicate rows)
    subj_effect = rng.normal(0, 0.5, size=(n_subjects, p))
    X += subj_effect[subject_ids]

    # STRONG batch signature on 200 features (this is what naive analysis reads)
    batch_feats = slice(50, 250)
    X[:, batch_feats] += (batch[:, None] * 2.5)

    cols = [f"feat_{i:03d}" for i in range(p)]
    Xdf = pd.DataFrame(X, columns=cols)
    meta = pd.DataFrame({"subject_id": subject_ids, "batch": batch, "y": y})
    truth = {"n_true_features": n_true, "true_effect_sd": 1.3,
             "batch_feature_range": (50, 250), "n_subjects": n_subjects,
             "reps_per_subject": reps}
    return Xdf, meta, truth


if __name__ == "__main__":
    X, meta, truth = make_biology_dataset()
    print(X.shape, dict(meta["y"].value_counts()), truth)
