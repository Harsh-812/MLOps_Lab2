import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a small but solid pipeline on Wine:
    StandardScaler -> RandomForestClassifier
    """
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, "../model/wine_model.pkl")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)

    # Check
    model = joblib.load("../model/wine_model.pkl")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Saved ../model/wine_model.pkl  |  Test accuracy: {acc:.3f}")
