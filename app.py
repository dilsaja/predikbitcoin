import pandas as pd
import numpy as np
import streamlit as st
import talib
import joblib  # <-- Ditambahkan untuk membaca file .pkl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import math

# ================================ #
# Streamlit Header
# ================================ #
st.set_page_config(page_title="Prediksi BTC", layout="wide")
st.title("📊 Prediksi Harga Bitcoin: XGBoost vs Random Forest")
st.caption("Komparasi model regresi menggunakan model terlatih  pada indikator teknikal (RSI & MA7).")

uploaded_file = st.file_uploader(
    "📥 Upload file CSV (wajib ada kolom: Date, Open, High, Low, Close, Volume)",
    type=["csv"]
)

if uploaded_file:
    # ================================ #
    # 1️⃣ Load Data
    # ================================ #
    df = pd.read_csv(uploaded_file)
    df_raw = df.copy()

    st.subheader("📄 Data Awal (Sebelum Preprocessing)")
    st.dataframe(df_raw.head(10))
    st.write(f"Jumlah data awal: {len(df_raw)}")

    # ================================ #
    # 2️⃣ Pemeriksaan Kualitas Data
    # ================================ #
    st.subheader("🧹 Pemeriksaan Kualitas Data")

    df.columns = df.columns.str.strip()
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan di file CSV: {missing_cols}")
        st.stop()

    duplicate_mask = df.duplicated()
    duplicate_count = duplicate_mask.sum()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Data Awal", len(df))
    with col2:
        st.metric("Jumlah Baris Duplikat", int(duplicate_count))

    if duplicate_count > 0:
        with st.expander("🔍 Lihat Baris Duplikat"):
            st.dataframe(df[df.duplicated(keep=False)])

    # Visualisasi duplikat
    st.subheader("📊 Visualisasi Duplikat")
    fig_dup, ax_dup = plt.subplots(figsize=(6, 4))
    dup_counts = [len(df) - duplicate_count, duplicate_count]
    dup_labels = ["Non-Duplikat", "Duplikat"]
    ax_dup.bar(dup_labels, dup_counts)
    ax_dup.set_title("Perbandingan Data Duplikat")
    ax_dup.set_ylabel("Jumlah Baris")
    for i, v in enumerate(dup_counts):
        ax_dup.text(i, v + max(dup_counts) * 0.01 if max(dup_counts) > 0 else 0.1, str(v), ha='center')
    st.pyplot(fig_dup)

    # Cek missing value
    missing_value = df.isnull().sum().reset_index()
    missing_value.columns = ["Kolom", "Jumlah Missing Value"]

    st.subheader("🧩 Missing Value")
    st.dataframe(missing_value)

    # Visualisasi missing value
    st.subheader("📉 Visualisasi Missing Value")
    fig_mv, ax_mv = plt.subplots(figsize=(10, 5))
    ax_mv.bar(missing_value["Kolom"], missing_value["Jumlah Missing Value"])
    ax_mv.set_title("Jumlah Missing Value per Kolom")
    ax_mv.set_ylabel("Jumlah Missing")
    ax_mv.set_xlabel("Kolom")
    plt.xticks(rotation=45)
    for i, v in enumerate(missing_value["Jumlah Missing Value"]):
        ax_mv.text(i, v + 0.1, str(v), ha='center')
    st.pyplot(fig_mv)

    # ================================ #
    # 3️⃣ Preprocessing Data
    # ================================ #
    st.subheader("⚙️ Preprocessing Data")

    df = df.drop_duplicates()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols)
    df = df.sort_values("Date").reset_index(drop=True)

    st.write(f"Jumlah data setelah preprocessing awal: {len(df)}")
    st.subheader("✅ Data Setelah Preprocessing Awal")
    st.dataframe(df.head(10))

    # ================================ #
    # 4️⃣ Feature Engineering
    # ================================ #

    df["MedianPrice"] = (df["High"] + df["Low"] + df["Open"]) / 3

    df["RSI14"] = talib.RSI(df["MedianPrice"], timeperiod=14)

    df["MA7"] = df["MedianPrice"].rolling(7).mean()

    df = df.dropna().reset_index(drop=True)

    st.subheader("📌 Data Final yang Digunakan Model")
    st.write(f"Jumlah data final untuk modeling: {len(df)}")
    st.dataframe(df[["Date", "Open", "High", "Low", "Close", "Volume", "MedianPrice", "RSI14", "MA7"]].tail(10))

    # ================================ #
    # 5️⃣ Split Data
    # ================================ #
    st.info("Sistem secara otomatis menggunakan kedua indikator (RSI14 & MA7) untuk memprediksi harga.")
    fitur = ["RSI14", "MA7"]
    X = df[fitur]
    y = df["Close"]

    # Train-test split (85% Training, 15% Testing sesuai format Colab lu)

    split_idx = int(len(df) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    df_test = df.iloc[split_idx:]

    # ================================ #
    # 6️⃣ Eksekusi Model (Memuat .pkl)
    # ================================ #
    st.subheader("🚀 Eksekusi Model")
    if st.button("🔮 Jalankan Prediksi dengan Model Terlatih"):

        with st.spinner("⏳ Memuat otak model...."):
            try:
                # Load model dan scaler
                xgb_model = joblib.load('xgb_model_skripsi.pkl')
                rf_model = joblib.load('rf_model_skripsi.pkl')
                scaler = joblib.load('scaler_skripsi.pkl')
                
                st.success("✅ Model berhasil dimuat")
            except FileNotFoundError:
                st.error("❌ File .pkl tidak ditemukan! Pastikan file xgb_model_skripsi.pkl, rf_model_skripsi.pkl, dan scaler_skripsi.pkl ada di folder yang sama dengan app.py.")
                st.stop()

            # Scaling data test menggunakan scaler bawaan dari Colab
            X_test_scaled_arr  = scaler.transform(X_test)
            last_row_scaled_arr = scaler.transform(X.iloc[[-1]])
 
            X_test_scaled_df  = pd.DataFrame(X_test_scaled_arr,  columns=fitur)
            last_scaled_df    = pd.DataFrame(last_row_scaled_arr, columns=fitur)

            colA, colB = st.columns(2)
            with colA:
                st.info("**Parameter Terbaik XGBoost**")
                # Menampilkan parameter bawaan dari model yang di-load
                st.json(xgb_model.get_params())
            with colB:
                st.info("**Parameter Terbaik Random Forest**")
                st.json(rf_model.get_params())

        # ================================ #
        # Evaluasi Metrik
        # ================================ #
        y_pred_xgb = xgb_model.predict(X_test_scaled_df)
        y_pred_rf  = rf_model.predict(X_test_scaled_df)

        # Metrik XGBoost
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = math.sqrt(mean_squared_error(y_test, y_pred_xgb))
        mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
        r2_xgb = r2_score(y_test, y_pred_xgb)

        # Prediksi Besok XGBoost
        last_scaled = scaler.transform(X.iloc[[-1]])
        pred_next_xgb = xgb_model.predict(last_scaled_df)[0]
        last_close = df["Close"].iloc[-1]
        pred_return_xgb = (pred_next_xgb - last_close) / last_close

        # Metrik Random Forest
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        rmse_rf = math.sqrt(mean_squared_error(y_test, y_pred_rf))
        mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100
        r2_rf = r2_score(y_test, y_pred_rf)

        # Prediksi Besok RF
        pred_next_rf = rf_model.predict(last_scaled_df)[0]
        pred_return_rf = (pred_next_rf - last_close) / last_close

        # ================================ #
        # 7️⃣ Tampilkan Hasil Perbandingan
        # ================================ #
        st.subheader("📊 Perbandingan Kinerja Model")
        hasil = pd.DataFrame({
            "Model": ["XGBoost", "Random Forest"],
            "MAE": [round(mae_xgb, 2), round(mae_rf, 2)],
            "RMSE": [round(rmse_xgb, 2), round(rmse_rf, 2)],
            "MAPE (%)": [round(mape_xgb, 2), round(mape_rf, 2)],
            "R2": [round(r2_xgb, 4), round(r2_rf, 4)],
            "Prediksi Close Besok": [round(pred_next_xgb, 2), round(pred_next_rf, 2)],
            "Prediksi Return": [f"{pred_return_xgb:.2%}", f"{pred_return_rf:.2%}"]
        })
        st.dataframe(hasil, use_container_width=True)

        # Tentukan model terbaik
        if mape_xgb < mape_rf:
            best_model = "XGBoost"
        elif mape_rf < mape_xgb:
            best_model = "Random Forest"
        else:
            best_model = "Seri (MAPE seimbang)"

        st.success(f"🎯 Model terbaik berdasarkan nilai error persentase (MAPE) terendah: **{best_model}**")

        # ================================ #
        # 8️⃣ Visualisasi Perbandingan
        # ================================ #
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(["XGBoost", "Random Forest"], [mape_xgb, mape_rf], color=["orange", "green"])
        ax.set_title("Perbandingan MAPE Antara XGBoost dan Random Forest")
        ax.set_ylabel("MAPE (%)")
        for i, val in enumerate([mape_xgb, mape_rf]):
            ax.text(i, val + 0.1, f"{val:.2f}%", ha='center', fontweight='bold')
        st.pyplot(fig)

        # Visualisasi Prediksi vs Aktual
        st.subheader("📈 Perbandingan Harga Aktual vs Prediksi")
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        ax2.plot(df_test["Date"], y_test.values, label="Actual", color="black", linewidth=2)
        ax2.plot(df_test["Date"], y_pred_xgb, label="XGBoost", color="orange", linestyle="--")
        ax2.plot(df_test["Date"], y_pred_rf, label="Random Forest", color="green", linestyle="--")
        ax2.set_title("Prediksi vs Aktual pada Data Testing")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Harga Close")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # Visualisasi Error Harian
        st.subheader("📉 Error Model pada Data Testing")
        error_df = pd.DataFrame({
            "Date": df_test["Date"],
            "Error_XGB": np.abs(y_test.values - y_pred_xgb),
            "Error_RF": np.abs(y_test.values - y_pred_rf)
        })
        fig3, ax3 = plt.subplots(figsize=(16, 6))
        ax3.plot(error_df["Date"], error_df["Error_XGB"], label="XGBoost Error", color="orange")
        ax3.plot(error_df["Date"], error_df["Error_RF"], label="RF Error", color="green")
        ax3.set_title("Perbandingan Error Harian")
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("Absolute Error")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)

        # Feature Importance XGBoost
        colC, colD = st.columns(2)
        with colC:
            st.subheader("📊 Feature Importance - XGBoost")
            importance_xgb = pd.DataFrame({
                "Fitur": fitur,
                "Importance": xgb_model.feature_importances_
            }).sort_values(by="Importance", ascending=True)
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.barh(importance_xgb["Fitur"], importance_xgb["Importance"], color="orange")
            ax4.set_title("Pengaruh Indikator - XGBoost")
            st.pyplot(fig4)

        # Feature Importance Random Forest
        with colD:
            st.subheader("📊 Feature Importance - Random Forest")
            importance_rf = pd.DataFrame({
                "Fitur": fitur,
                "Importance": rf_model.feature_importances_
            }).sort_values(by="Importance", ascending=True)
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            ax5.barh(importance_rf["Fitur"], importance_rf["Importance"], color="green")
            ax5.set_title("Pengaruh Indikator - Random Forest")
            st.pyplot(fig5)