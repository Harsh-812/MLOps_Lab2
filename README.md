# FastAPI Lab – Wine Classifier
- Dataset: sklearn.load_wine()
- Model: StandardScaler + RandomForest (200 trees)
- Endpoints: / (health), /predict
- Instructions to run:
  1) source lab2_env/bin/activate
  2) python src/train.py
  3) uvicorn src.main:app --reload

