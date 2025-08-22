# Fantasy Football Player Impact Prediction

## ACM40960 — Project in Mathematical Modelling

### *Predicting player impact with tree‑based models, neural networks and ensembling*

**Authors**: Shyam Pratap Singh Rathore (24222167), Charvie Kukreja (24202861) 
**Date**: August 2025

---

## Introduction

This project tackles the challenge of predicting which football players will deliver **impactful performances** in a match.  
Starting from raw player–match statistics for the 2022/23 English Premier League season, we build a full machine‑learning pipeline that:

- **Cleans and audits** the data, ensuring consistency and handling missing values.  
- **Engineers per‑90 metrics and context features** so that players with different playing times become comparable.  
- **Trains complementary models** — an interpretable tree‑based classifier (XGBoost/Gradient Boosting) and a flexible neural network (Multi‑Layer Perceptron) — before combining them in a weighted ensemble.  
- **Evaluates rigorously**, using ROC–AUC, average precision, F1‑score and threshold sweeps to select sensible decision cut‑offs.  
- **Interprets feature importance**, highlighting which statistics most consistently predict impact.

Although we demonstrate the workflow on EPL data, the approach is **generalisable**: any structured dataset of player statistics can be plugged into this pipeline to uncover recent performance patterns.  
With its combination of data engineering, model comparison and ensemble weighting, the project showcases a solid framework for **fantasy‑football optimisation and sports analytics research**.

---

## Objectives

- **Define “impact”**: label a player as impactful if they receive a match rating ≥7.0 or record a goal/assist.  
- **Clean and prepare the data**: convert numeric columns, drop players with zero minutes, and replace missing stats with zeros.  
- **Engineer per‑90 features** (e.g. shots90, passes90, xG90) and prune low‑variance columns.  
- **Split the data** into train and test sets with stratification and standardise features.  
- **Train and compare** an XGBoost/Gradient Boosting classifier and a neural net (MLP).  
- **Perform threshold sweeps** to understand precision–recall trade‑offs.  
- **Compute ensemble weights** via cross‑validation to combine models.  
- **Interpret feature importance** using gain‑based and permutation techniques.

---

## Dataset overview

The raw dataset contains **3 555 rows** (one per player‑match) with 626 unique players and 20 clubs.  
Key columns include minutes played, rating, goals, assists, shots, xG, passes, key passes and disciplinary cards.  
An audit of the data shows that the `match_id` and `team` columns are complete, while some performance statistics have missing values that are imputed with zeros.  
The distribution of unique values indicates diversity across teams and players, providing a rich basis for modelling.

---

##  Methodology

### 1 • Data preparation and feature engineering

- **Audit & conversion**: after loading the CSV, we inspect data types and counts.  Important identifiers (`match_id`, `team`, `player`) have non‑null counts of 3 555, while performance metrics like goals and assists are sparse.  
- **Minutes filter**: rows where a player didn’t play (`minutes = 0`) are dropped to avoid noise.  
- **Imputation**: missing numeric stats (goals, assists, xG, etc.) are set to 0.  
- **Target definition**: an **impact** flag is created when a player either has rating ≥7.0 or has any goals/assists.  
- **Per‑90 and contextual features**: statistics are normalised by minutes played and rolling metrics are added (e.g. form over previous matches).  Low‑variance columns are removed.

### 2 • Modelling

We split the data (75 % train, 25 % test) with stratification and standardise features.  Two complementary classifiers are trained:

1. **XGBoost** – a tree‑based ensemble effective on structured data.  
2. **Multi‑Layer Perceptron (MLP)** – a feed‑forward neural network with two hidden layers (32 and 16 neurons), ReLU activation, the Adam optimiser and early stopping.

Early stopping and cross‑validation guard against over‑fitting.  Evaluation functions compute ROC–AUC, average precision and F1, and produce ROC/PR curves and confusion matrices.

### 3 • Threshold analysis

Using threshold sweeps, we explore how precision and recall trade off across probability cut‑offs.  The best F1 for XGBoost occurs at **threshold ≈0.25** (precision ≈ 0.51, recall ≈ 0.85, F1 ≈ 0.64), while the MLP performs best at **threshold ≈0.40** (precision ≈ 0.60, recall ≈ 0.73, F1 ≈ 0.66).  This analysis informs decision rules beyond the naïve 0.5 cutoff.

### 4 • Ensembling

To combine the strengths of both models, we compute **ensemble weights** using 5‑fold cross‑validation.  The average precision on each fold determines weights **wₓgᵦ ≈ 0.49** and **wₘˡᵖ ≈ 0.51**, producing a blended probability for each player.  We then evaluate the ensemble on the test set.

### 5 • Feature importance

- **XGBoost gain importance**: key passes, minutes, shots and passes dominate, with key passes contributing over 34 % of the total importance.  
- **MLP permutation importance**: the neural network also highlights key passes, passes, minutes and shots, while penalising yellow cards and high per‑90 shot rates.  
Across all methods, **key passes** consistently emerge as the most decisive feature, while disciplinary statistics (yellow/red cards) contribute little to predictive power.

---

## Results

### Model performance

The table below summarises the performance of each model on the held‑out test set.  Values are rounded to four decimal places.  F1 scores are calculated at the default 0.5 threshold for comparability.

| Model               | ROC–AUC | Average Precision | F1 (threshold 0.5) |
|--------------------:|--------:|------------------:|--------------------:|
| **XGBoost**         | 0.7791  | 0.6775            | 0.7134|
| **Neural Net (MLP)**| **0.7865** | 0.6741         | **0.7225**|
| **Ensemble**        | **0.7865** | **0.6935**     | 0.6074|

**Key takeaways**:

- The MLP achieves the highest ROC–AUC and F1 at the default threshold, indicating better recall of impactful players.  
- XGBoost is more conservative but has slightly higher average precision when using its optimal threshold.  
- The **ensemble improves average precision** (0.6935 vs. 0.6775/0.6741) by balancing false positives and false negatives, although the F1 at 0.5 is lower because the ensemble probability distribution is more concentrated around mid‑range values.

![The ROC curve test](assets/ROC.png)

### Threshold sweep highlights

- **XGBoost** – best F1 of **0.6391** at threshold 0.25.  
- **Neural Net** – best F1 of **0.6557** at threshold 0.40.  
These custom cut‑offs provide better decision rules for fantasy‑football managers interested in maximising precision or recall.

![Description of the image](assets/Threshold.png)

### Ensemble weighting

Cross‑validation yields weights of **wₓgᵦ ≈ 0.49** and **wₘˡᵖ ≈ 0.51**, meaning the neural network contributes slightly more to the final prediction. This weighting is driven by marginally higher validation average precision of the MLP across folds.

![Ensemble](assets/ensemble.png)

### Feature importance insights

Across all importance measures, the following features consistently appear at the top:

- **Key passes** – the single most decisive indicator of impact.  
- **Minutes played** – players who stay on the pitch longer have more opportunity to contribute.  
- **Shots and passes** – both volume and per‑90 rates matter.  
- **Per‑90 key passes (kp90)** – emphasising creative contribution relative to minutes.  
- **Yellow/red cards** – negligible or even negative importance; disciplinary issues do not predict impact.

---

###  Project structure
```
fantasy_football/
├── data/                # All CSV files (raw and cleaned) used in the project
├── models/              # Saved model files, e.g. xgboost .pkl and neural net .h5
├── notebooks/           # Jupyter notebooks for analysis and reporting
│   ├── FinalProject.ipynb    # Main notebook you’ll submit
│   └── exploration.ipynb     # Optional: other analyses or EDA
├── src/                 # Python source code for data prep and training
│   ├── preprocessing.py  # Functions for cleaning and feature engineering
│   ├── train.py         # Script to train models
│   ├── evaluate.py      # Script to compute metrics and plots
│   └── utils.py         # Helper functions (e.g. loading models, plotting)
├── assets/              # Images for README (ROC curve, threshold plot, etc.)
├── README.md            # Project overview and documentation
├── requirements.txt     # List of Python dependencies
└── .gitignore           # Files/directories to exclude from version control
```
## Getting started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shyam7773/fantasy_football.git
   cd fantasy_football
   
2. **Set up a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   
3. **Prepare the data**
   Place the relevant CSV files (e.g. epl2022_player_stats.csv, epl2022_cleaned.csv, etc.) into the data/ directory. If you’re using different leagues or seasons, put those CSVs in
   data/ instead.
   
5. **Run the Notebook**
   ```bash
   jupyter notebook notebooks/FinalProject.ipynb

## Conclusion

This project demonstrates how combining a tree‑based model and a neural network can provide a more balanced, stable predictor of player impact. After cleaning and engineering per‑90 features from match‑level data, we trained both models and used cross‑validation to compute ensemble weights. Feature importance analysis consistently highlighted creative metrics (such as key passes) and playing time as the strongest predictors, while disciplinary statistics had little influence. The framework is fully adaptable to other leagues or seasons and serves as a solid foundation for fantasy‑football optimisation or broader sports analytics research.

## Potential extensions

- Real‑time updates: Integrate live match data to make weekly predictions for fantasy line‑ups.

- Richer contextual features: Incorporate team form, opponent strength, or match difficulty ratings.

- Hyperparameter optimisation: Use automated methods (e.g. Bayesian optimisation) to fine‑tune model parameters.

- Model explainability: Apply SHAP values or similar techniques to provide player‑level explanations for each prediction.

- User interface: Develop a simple web app where users can upload a squad and receive recommended transfers or line‑ups based on model outputs.
