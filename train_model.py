import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# 1) Define column names for C-MAPSS
cmapss_cols = (
    ["engine_id", "cycle"] +
    [f"setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# 2) Load & concatenate all training text files
frames = []
for path in glob.glob("train_FD00*.txt"):
    df = pd.read_csv(path, sep=r"\\s+", header=None)
    df.columns = cmapss_cols
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

# 3) Compute Remaining Useful Life (RUL)
data["max_cycle"] = data.groupby("engine_id")["cycle"].transform("max")
data["RUL"] = data["max_cycle"] - data["cycle"]

# 4) Split features and target
X = data.drop(columns=["RUL", "max_cycle"])
y = data["RUL"]

# 5) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6) Train a Random Forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 7) Save the trained model
joblib.dump(model, "model.pkl")
print("Training complete—saved model.pkl")
