import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# 1) Define column names for the NASA C-MAPSS dataset
cmapss_cols = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}"  for i in range(1, 22)]
)

# 2) Load & concatenate exactly the four training text files
frames = []
for path in glob.glob("train_FD00[1-4].txt"):
    # Use the Python engine so '\s+' splits on all whitespace
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df.columns = cmapss_cols
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
print(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")

# 3) Compute Remaining Useful Life (RUL)
data["max_cycle"] = data.groupby("engine_id")["cycle"].transform("max")
data["RUL"] = data["max_cycle"] - data["cycle"]

# 4) Prepare features (X) and target (y)
X = data.drop(columns=["RUL", "max_cycle"])
y = data["RUL"]

# 5) Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 6) Train a smaller Random Forest regressor
model = RandomForestRegressor(
    n_estimators=20,    # fewer trees
    max_depth=10,       # limit tree depth
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print(f"Test R² score: {model.score(X_test, y_test):.3f}")

# 7) Save the trained model with maximum compression
joblib.dump(model, "model.pkl", compress=9)
print("Training complete—saved model.pkl (compressed)")
