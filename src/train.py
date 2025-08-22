# Training pipeline (heuristic extraction)

# =========================================================
# SETUP & IMPORTS
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report, confusion_matrix, f1_score
)
from sklearn.inspection import permutation_importance

# Applying XGBoost
XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Local config
DATA_PATH = "/Users/shyam/Desktop/Final Project/epl2022_player_stats.csv"  # <-- change as per your directory


# =========================================================
# ENSEMBLE WEIGHTS VIA 5-FOLD CV (on TRAIN)
# =========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ap_xgb, ap_mlp = [], []

for tr, va in cv.split(X_train, y_train):
    Xtr_raw, Xva_raw = X_train[tr], X_train[va]
    ytr, yva = y_train[tr], y_train[va]
    scaler_cv = StandardScaler().fit(Xtr_raw)
    Xtr_s, Xva_s = scaler_cv.transform(Xtr_raw), scaler_cv.transform(Xva_raw)

    if XGB_AVAILABLE:
        xgb_cv = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=2
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        xgb_cv = GradientBoostingClassifier(random_state=RANDOM_STATE)

    mlp_cv = MLPClassifier(
        hidden_layer_sizes=(32,16), activation="relu", solver="adam",
        alpha=1e-4, learning_rate_init=1e-3, max_iter=800,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, tol=1e-4, random_state=RANDOM_STATE
    )

    xgb_cv.fit(Xtr_raw, ytr)
    mlp_cv.fit(Xtr_s, ytr)

    p_x = xgb_cv.predict_proba(Xva_raw)[:,1] if hasattr(xgb_cv,"predict_proba") else xgb_cv.decision_function(Xva_raw)
    p_m = mlp_cv.predict_proba(Xva_s)[:,1]

    ap_xgb.append(average_precision_score(yva, p_x))
    ap_mlp.append(average_precision_score(yva, p_m))

w_xgb = float(np.mean(ap_xgb))
w_mlp = float(np.mean(ap_mlp))
w_sum = w_xgb + w_mlp
w_xgb /= w_sum
w_mlp /= w_sum
print({"w_xgb": round(w_xgb,4), "w_mlp": round(w_mlp,4)})


# =========================================================
# ENSEMBLE EVAL (TEST)
# =========================================================
proba_ens = w_xgb * proba_xgb + w_mlp * proba_mlp
res_ens = {
    "Model": f"Ensemble(w_xgb={w_xgb:.2f}, w_mlp={w_mlp:.2f})",
    "ROC_AUC": roc_auc_score(y_test, proba_ens),
    "AvgPrecision": average_precision_score(y_test, proba_ens),
    "F1@0.5": f1_score(y_test, (proba_ens>=0.5).astype(int), zero_division=0)
}
pd.DataFrame([
    {"Model": res_xgb["Model"], "ROC-AUC": res_xgb["ROC_AUC"], "AP": res_xgb["AvgPrecision"], "F1@0.5": res_xgb["F1"]},
    {"Model": res_mlp["Model"], "ROC-AUC": res_mlp["ROC_AUC"], "AP": res_mlp["AvgPrecision"], "F1@0.5": res_mlp["F1"]},
    {"Model": res_ens["Model"], "ROC-AUC": res_ens["ROC_AUC"], "AP": res_ens["AvgPrecision"], "F1@0.5": res_ens["F1@0.5"]},
]).round(4)
