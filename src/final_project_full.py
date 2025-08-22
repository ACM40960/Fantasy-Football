# Generated from Final Project.ipynb

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
# LOAD & AUDIT
# =========================================================
df = pd.read_csv(DATA_PATH)

audit = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "non_null": df.notnull().sum().values,
    "n_unique": [df[c].nunique() for c in df.columns]
})
display(audit.head(30))
display(df.head(8))


# =========================================================
# - Coerce numerics
# - Minutes > 0
# - Fill counting stats NaNs with 0
# - Target y = 1 if rating >= 7 OR (goals + assists) > 0
# - If xG missing, create it as zeros (warn)
# =========================================================
numeric_cols = ["minutes","rating","goals","assists","shots","passes","key_passes","yellow_cards","red_cards"]
if "xG" not in df.columns:
    print("WARNING: 'xG' column not found. Creating xG=0.0 (note data limitation).")
    df["xG"] = 0.0
numeric_cols.append("xG")

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df["minutes"].fillna(0) > 0].copy()

for c in ["goals","assists","shots","xG","passes","key_passes","yellow_cards","red_cards"]:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)

df["impact"] = ((df["rating"] >= 7.0) | ((df["goals"] + df["assists"]) > 0)).astype(int)
print("Class balance:", df["impact"].value_counts(normalize=True).round(3).to_dict())


# =========================================================
# Rules:
#   - drop zero-variance features
#   - drop "nearly all zeros" features (zero rate > ZERO_RATE_MAX)
#   - everything else stays
# =========================================================
ZERO_RATE_MAX = 0.995  # tweak if you want to be stricter/looser

def per90(series, minutes):
    m = minutes.values
    v = series.values
    return np.where(m > 0, v * 90.0 / m, 0.0)

# creating per-90s if inputs present
if {"shots","minutes"}.issubset(df.columns):   df["shots90"]  = per90(df["shots"], df["minutes"])
if {"passes","minutes"}.issubset(df.columns):  df["passes90"] = per90(df["passes"], df["minutes"])
if {"key_passes","minutes"}.issubset(df.columns): df["kp90"]  = per90(df["key_passes"], df["minutes"])
if {"yellow_cards","minutes"}.issubset(df.columns): df["yc90"]= per90(df["yellow_cards"], df["minutes"])
if {"red_cards","minutes"}.issubset(df.columns):    df["rc90"]= per90(df["red_cards"], df["minutes"])
if {"xG","minutes"}.issubset(df.columns):           df["xG90"]= per90(df["xG"], df["minutes"])

candidate_features = [
    c for c in [
        "minutes","shots","xG","passes","key_passes","yellow_cards","red_cards",
        "shots90","passes90","kp90","yc90","rc90","xG90"
    ] if c in df.columns
]

# computing stats to prune
stats = []
for c in candidate_features:
    s = df[c]
    zero_rate = float((s == 0).mean())
    var = float(s.var())
    keep = (var > 0) and (zero_rate <= ZERO_RATE_MAX)
    stats.append({"feature": c, "variance": var, "zero_rate": zero_rate, "keep": keep})
prune_df = pd.DataFrame(stats).sort_values("keep", ascending=True)
display(prune_df)

feature_cols = [r["feature"] for r in stats if r["keep"]]
print("Kept features:", feature_cols)

X = df[feature_cols].fillna(0.0).values
y = df["impact"].values


idx_all = np.arange(len(df))
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, idx_all, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# =========================================================
# MODELS (XGB/GradBoost + MLP with early stopping)
# =========================================================
if XGB_AVAILABLE:
    xgb = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=2
    )
else:
    from sklearn.ensemble import GradientBoostingClassifier
    xgb = GradientBoostingClassifier(random_state=RANDOM_STATE)

xgb.fit(X_train, y_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(32,16),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=800,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    tol=1e-4,
    random_state=RANDOM_STATE
)
mlp.fit(X_train_s, y_train)


# =========================================================
# EVAL HELPERS
# =========================================================
def evaluate(name, y_true, proba):
    pred = (proba >= 0.5).astype(int)
    rpt = classification_report(y_true, pred, output_dict=True, digits=3)
    return {
        "Model": name,
        "ROC_AUC": roc_auc_score(y_true, proba),
        "AvgPrecision": average_precision_score(y_true, proba),
        "F1": rpt["weighted avg"]["f1-score"],
        "cm": confusion_matrix(y_true, pred),
        "report": rpt
    }

def plot_roc(y_true, *probas_and_labels):
    plt.figure(figsize=(6,5))
    for proba, label in probas_and_labels:
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'--', label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)"); plt.legend(loc="lower right"); plt.tight_layout(); plt.show()

def plot_pr(y_true, *probas_and_labels):
    plt.figure(figsize=(6,5))
    for proba, label in probas_and_labels:
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        plt.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall (Test)"); plt.legend(loc="upper right"); plt.tight_layout(); plt.show()

def plot_cm(cm, title="Confusion Matrix (Norm)"):
    cm = cm.astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.5,4))
    im = ax.imshow(cm_norm, interpolation='nearest', aspect='auto')
    ax.set_title(title)
    ax.set_xticks([0,1]); ax.set_xticklabels(['No Impact','Impact'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['No Impact','Impact'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.colorbar(im)
    plt.tight_layout(); plt.show()


# =========================================================
# METRICS + ROC/PR + CONFUSION
# =========================================================
proba_xgb = xgb.predict_proba(X_test)[:,1] if hasattr(xgb,"predict_proba") else xgb.decision_function(X_test)
proba_mlp = mlp.predict_proba(X_test_s)[:,1]

res_xgb = evaluate("XGBoost" if XGB_AVAILABLE else "GradBoost", y_test, proba_xgb)
res_mlp = evaluate("NeuralNet(MLP)", y_test, proba_mlp)

metrics_df = pd.DataFrame([
    {"Model": res_xgb["Model"], "ROC-AUC": res_xgb["ROC_AUC"], "AvgPrecision": res_xgb["AvgPrecision"], "F1": res_xgb["F1"]},
    {"Model": res_mlp["Model"], "ROC-AUC": res_mlp["ROC_AUC"], "AvgPrecision": res_mlp["AvgPrecision"], "F1": res_mlp["F1"]},
]).round(4)
display(metrics_df)

plot_roc(y_test, (proba_xgb, res_xgb["Model"]), (proba_mlp, res_mlp["Model"]))
plot_pr(y_test, (proba_xgb, res_xgb["Model"]), (proba_mlp, res_mlp["Model"]))
plot_cm(res_xgb["cm"], f'{res_xgb["Model"]} — Confusion Matrix (Norm)')
plot_cm(res_mlp["cm"], f'{res_mlp["Model"]} — Confusion Matrix (Norm)')


# =========================================================
# FEATURE IMPORTANCES (dynamic columns)
# =========================================================
fi = None
if hasattr(xgb, "feature_importances_"):
    fi = pd.DataFrame({"feature": feature_cols, "importance": xgb.feature_importances_}) \
            .sort_values("importance", ascending=False)
    display(fi)
    plt.figure(figsize=(7,4))
    plt.bar(fi["feature"], fi["importance"])
    plt.xticks(rotation=45, ha="right")
    plt.title(f'{res_xgb["Model"]} Feature Importances'); plt.ylabel("Importance")
    plt.tight_layout(); plt.show()

perm = permutation_importance(mlp, X_test_s, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=2)
perm_df = pd.DataFrame({"feature": feature_cols, "perm_importance": perm.importances_mean}) \
            .sort_values("perm_importance", ascending=False)
display(perm_df)
plt.figure(figsize=(7,4))
plt.bar(perm_df["feature"], perm_df["perm_importance"])
plt.xticks(rotation=45, ha="right")
plt.title("Neural Net (Permutation Importance)")
plt.ylabel("Mean Decrease in Score"); plt.tight_layout(); plt.show()


# =========================================================
# "PREDICT THE GAME" HELPERS (using the stored test_idx)
# =========================================================
cols_keep = [c for c in ["match_id","team","player","position","minutes",
                         "shots","xG","passes","key_passes","yellow_cards","red_cards",
                         "shots90","passes90","kp90","yc90","rc90","xG90"]
             if c in df.columns]

test_mask = np.zeros(len(df), dtype=bool)
test_mask[test_idx] = True

pred_df = df.loc[test_mask, cols_keep].copy()
pred_df["proba_xgb"] = proba_xgb
pred_df["proba_mlp"] = proba_mlp
pred_df["proba_mean"] = pred_df[["proba_xgb","proba_mlp"]].mean(axis=1)
display(pred_df.head(12))

group_keys = [k for k in ["match_id","team"] if k in pred_df.columns]
if group_keys:
    team_match = pred_df.groupby(group_keys, as_index=False).agg(
        expected_impact_players=("proba_mean","sum"),
        mean_impact_prob=("proba_mean","mean")
    ).sort_values(group_keys)
    display(team_match.head(12))


# =========================================================
# THRESHOLD SWEEP (justifying the cutoff)
# =========================================================
def threshold_sweep(y_true, proba, metric="f1"):
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    from sklearn.metrics import precision_score, recall_score
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0)
        })
    return pd.DataFrame(rows)

sweep_x = threshold_sweep(y_test, proba_xgb)
sweep_m = threshold_sweep(y_test, proba_mlp)

plt.figure(figsize=(6,5))
plt.plot(sweep_x["threshold"], sweep_x["f1"], label=f'{res_xgb["Model"]} F1')
plt.plot(sweep_m["threshold"], sweep_m["f1"], label=f'{res_mlp["Model"]} F1')
plt.xlabel("Threshold"); plt.ylabel("F1-score"); plt.title("Threshold Sweep (F1)")
plt.legend(); plt.tight_layout(); plt.show()

print("Best F1 (XGB):", sweep_x.iloc[sweep_x["f1"].idxmax()].to_dict())
print("Best F1 (MLP):", sweep_m.iloc[sweep_m["f1"].idxmax()].to_dict())


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


# =========================================================
# WHAT REALLY AFFECTS IT (ENSEMBLE PERM + CONSENSUS)
# =========================================================
def metric_ap(y_true, proba): return average_precision_score(y_true, proba)

base_ap = metric_ap(y_test, proba_ens)
imp_drop = []
for j, col in enumerate(feature_cols):
    X_test_perm = X_test.copy()
    rng = np.random.RandomState(RANDOM_STATE + j)
    X_test_perm[:, j] = rng.permutation(X_test_perm[:, j])
    X_test_s_perm = scaler.transform(X_test_perm)

    p_x_perm = xgb.predict_proba(X_test_perm)[:,1] if hasattr(xgb,"predict_proba") else xgb.decision_function(X_test_perm)
    p_m_perm = mlp.predict_proba(X_test_s_perm)[:,1]
    p_ens_perm = w_xgb * p_x_perm + w_mlp * p_m_perm
    ap_perm = metric_ap(y_test, p_ens_perm)
    imp_drop.append(base_ap - ap_perm)

ens_perm = pd.DataFrame({"feature": feature_cols, "ensemble_perm_AP_drop": imp_drop}) \
            .sort_values("ensemble_perm_AP_drop", ascending=False)
display(ens_perm)

plt.figure(figsize=(7,4))
plt.bar(ens_perm["feature"], ens_perm["ensemble_perm_AP_drop"])
plt.xticks(rotation=45, ha="right")
plt.title("Ensemble Permutation Importance (AP drop)")
plt.ylabel("ΔAP vs baseline"); plt.tight_layout(); plt.show()

# Consensus rank (works even if fi is None due to pruning)
src = []
if 'fi' in globals() and fi is not None:
    t = fi[["feature","importance"]].copy()
    t.columns = ["feature","xgb_gain"]
    src.append(t)
n = perm_df[["feature","perm_importance"]].copy()
n.columns = ["feature","nn_perm"]
src.append(n)
src.append(ens_perm.rename(columns={"ensemble_perm_AP_drop":"ens_perm"}))

cons = src[0]
for s in src[1:]:
    cons = cons.merge(s, on="feature", how="outer")
for c in ["xgb_gain","nn_perm","ens_perm"]:
    if c in cons.columns:
        v = cons[c].values
        mn, mx = np.nanmin(v), np.nanmax(v)
        cons[c] = (v - mn) / (mx - mn + 1e-12)

score_cols = [c for c in ["xgb_gain","nn_perm","ens_perm"] if c in cons.columns]
cons["consensus_score"] = cons[score_cols].mean(axis=1)
cons_sorted = cons.sort_values("consensus_score", ascending=False).reset_index(drop=True)
display(cons_sorted)

plt.figure(figsize=(7.5,4.2))
plt.bar(cons_sorted["feature"], cons_sorted["consensus_score"])
plt.xticks(rotation=45, ha="right")
plt.title("Consensus Feature Importance (XGB + NN + Ensemble)")
plt.ylabel("Normalized score"); plt.tight_layout(); plt.show()
