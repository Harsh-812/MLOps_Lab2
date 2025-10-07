import joblib
import os


def _model_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "wine_model.pkl"))


def predict_data(X):
    model = joblib.load(_model_path())
    return model.predict(X)
