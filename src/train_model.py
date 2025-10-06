# src/train_model.py
import os
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save_model(outdir="models"):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {path}, accuracy={acc:.4f}")
    return path, acc

if __name__ == "__main__":
    train_and_save_model()
