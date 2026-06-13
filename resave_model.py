import joblib
import sklearn
print(sklearn.__version__)  # versi lokal kamu

xgb = joblib.load('xgb_model_skripsi.pkl')
rf  = joblib.load('rf_model_skripsi.pkl')
sc  = joblib.load('scaler_skripsi.pkl')
print("Loaded OK")