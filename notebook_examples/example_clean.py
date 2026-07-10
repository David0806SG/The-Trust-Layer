import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("proteomics.csv")
groups = df["subject_id"].values
y = df["label"].values
X = df.drop(columns=["label", "subject_id"]).values

# preprocessing + selection live INSIDE a pipeline, fit per-fold only
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(f_classif, k=20)),
    ("clf", LogisticRegression()),
])

# group-aware CV: no subject on both sides
cv = GroupKFold(n_splits=5)
scores = cross_val_score(pipe, X, y, cv=cv, groups=groups)
print("honest AUC:", scores.mean())
