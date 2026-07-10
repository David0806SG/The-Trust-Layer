import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("proteomics.csv")
subject = df["subject_id"].values          # repeated measures present
y = df["label"].values
X = df.drop(columns=["label", "subject_id"]).values

# LEAK 1: scale the whole dataset before splitting
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# LEAK 2: select features on the whole dataset (uses y) before splitting
sel = SelectKBest(f_classif, k=20)
Xsel = sel.fit_transform(Xs, y)

# split AFTER preprocessing -- too late
X_tr, X_te, y_tr, y_te = train_test_split(Xsel, y, test_size=0.3, random_state=0)

clf = LogisticRegression().fit(X_tr, y_tr)
print("test AUC looks great!")

# LEAK 3: cross-validation ignores subject grouping
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, Xsel, y, cv=cv)
