import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# Load data and remove identical rows BEFORE splitting
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]
print("Rows after removing duplicates:", len(concrete))
# (a) Hold out 20% for testing, then CV on the training rows (linear baseline)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"
baseline_rmse = -cross_val_score(
    LinearRegression(), X_train, y_train, cv=folds, scoring=scoring
)
print("Baseline fold RMSEs:", baseline_rmse.round(2))
print(f"Baseline mean CV RMSE: {baseline_rmse.mean():.2f} MPa")
# (b) Tune polynomial + scaling + Ridge on the same folds
pipe = Pipeline(
    [
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge()),
    ]
)
grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100],
}
search = GridSearchCV(pipe, grid, cv=folds, scoring=scoring, refit=True)
search.fit(X_train, y_train)
results = pd.DataFrame(search.cv_results_)
top3 = results.sort_values("rank_test_score").head(3)
print("\nTop 3 settings:")
for _, row in top3.iterrows():
    print(
        f"Rank {int(row['rank_test_score'])}: {row['params']} "
        f"-> mean CV RMSE {-row['mean_test_score']:.2f} MPa"
    )
tuned_cv_rmse = -search.best_score_
print("\nBest parameters:", search.best_params_)
print(f"Tuned mean CV RMSE: {tuned_cv_rmse:.2f} MPa")
# (c) Test RMSE and MAE of the refitted winner
test_pred = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae = mean_absolute_error(y_test, test_pred)
print(f"Test RMSE: {test_rmse:.2f} MPa")
print(f"Test MAE: {test_mae:.2f} MPa")
# (c) Actual vs predicted plot with y = x reference line
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_pred, alpha=0.6, color="steelblue", label="Test predictions")
low = min(y_test.min(), test_pred.min())
high = max(y_test.max(), test_pred.max())
plt.plot([low, high], [low, high], color="red", linestyle="--", label="y = x")
plt.xlabel("Actual strength (MPa)")
plt.ylabel("Predicted strength (MPa)")
plt.title("Concrete Strength: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("task3_concrete.png", dpi=150)
print("Saved task3_concrete.png")
# (d) Baseline vs tuned
print(f"\nBaseline mean CV RMSE: {baseline_rmse.mean():.2f} MPa")
print(f"Tuned mean CV RMSE:    {tuned_cv_rmse:.2f} MPa")
print(f"Improvement: {baseline_rmse.mean() - tuned_cv_rmse:.2f} MPa")
