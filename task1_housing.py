import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data and make the feature (distance in units of 100 m)
homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]
print("Rows used:", len(homes))

# (a) Fit LinearRegression on all 414 rows
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope:.2f} price units per 100 m")
print(f"Intercept: {intercept:.2f} price units")

# (b) Predict at 500 m (= 5 hundred-metre units)
new_house = pd.DataFrame({"distance_100m": [5]})
pred_500 = model.predict(new_house)[0]
print(f"Predicted price per unit area at 500 m: {pred_500:.2f}")

# (c) Meaning of the slope
print(
    f"Each extra 100 m from the metro station changes the predicted price "
    f"by {slope:.2f} units (about {abs(slope) * 10000:,.0f} NT$ per ping)."
)

# (c) Plot observations and the fitted line
x_line = pd.DataFrame(
    {"distance_100m": [X["distance_100m"].min(), X["distance_100m"].max()]}
)
plt.figure(figsize=(7, 5))
plt.scatter(X["distance_100m"], y, alpha=0.5, color="steelblue", label="Observations")
plt.plot(
    x_line["distance_100m"],
    model.predict(x_line),
    color="red",
    linewidth=2,
    label="Fitted line",
)
plt.xlabel("Distance to nearest MRT station (100 m)")
plt.ylabel("House price per unit area (10,000 NT$ per ping)")
plt.title("Housing Price vs Distance to Metro")
plt.legend()
plt.tight_layout()
plt.savefig("task1_housing.png", dpi=150)
print("Saved task1_housing.png")
