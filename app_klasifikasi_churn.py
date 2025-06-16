import streamlit as st
import pandas as pd
import joblib # Untuk memuat model dan preprocessor
import numpy as np # Diperlukan untuk operasi array

# --- Judul Aplikasi ---
st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="centered")
st.title('Prediksi Churn Pelanggan Telco')
st.markdown("""
Aplikasi ini memprediksi apakah seorang pelanggan Telco kemungkinan akan `churn` (berhenti berlangganan) 
berdasarkan karakteristik layanannya.
""")

# --- Memuat Model dan Objek Pra-pemrosesan ---
# Pastikan file .pkl berada di direktori yang sama dengan script ini
try:
    preprocessing_assets = joblib.load('preprocessing_assets_klasifikasi.pkl')
    model_klasifikasi = joblib.load('model_klasifikasi.pkl')
    
    # Ekstrak objek-objek dari dictionary preprocessing_assets
    total_charges_mean = preprocessing_assets['total_charges_mean']
    label_encoders = preprocessing_assets['label_encoders']
    minmax_scaler = preprocessing_assets['minmax_scaler']
    standard_scaler = preprocessing_assets['standard_scaler']
    pca_transformer = preprocessing_assets['pca']

    st.success("Model dan objek pra-pemrosesan berhasil dimuat!")
except FileNotFoundError:
    st.error("""
    Error: File model atau pra-pemrosesan (.pkl) tidak ditemukan.
    Pastikan 'preprocessing_assets_klasifikasi.pkl' dan 'model_klasifikasi.pkl' 
    berada di direktori yang sama dengan script ini.
    """)
    st.stop() # Hentikan eksekusi jika file tidak ditemukan
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat file: {e}")
    st.stop()


# --- Fungsi untuk Pra-pemrosesan Input Pengguna ---
# Fungsi ini harus MENCERMINKAN SECARA PERSIS alur pra-pemrosesan yang Anda lakukan saat melatih model
def preprocess_input(input_df):
    df_processed = input_df.copy()

    # 1. Penanganan 'TotalCharges': Konversi ke numerik dan isi NaN
    # Karena input dari Streamlit adalah angka, NaN hanya akan muncul jika pengguna mengosongkan (default ke 0)
    # Tapi kita tetap pertahankan alur ini untuk konsistensi dengan pelatihan.
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed['TotalCharges'].fillna(total_charges_mean, inplace=True) # Menggunakan mean dari data pelatihan

    # 2. Label Encoding untuk kolom kategorikal
    # Pastikan urutan kolom sama dengan saat pelatihan
    # Loop melalui label_encoders yang sudah kita simpan
    for col, le in label_encoders.items():
        if col in df_processed.columns:
            # Penting: gunakan .transform() bukan .fit_transform()
            df_processed[col] = le.transform(df_processed[col])
        else:
            st.warning(f"Kolom '{col}' dari LabelEncoder tidak ditemukan di input. Mungkin ada kesalahan kolom.")

    # 3. Normalisasi (MinMaxScaler)
    columns_to_scale_minmax = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    
    # Pastikan semua kolom ada di df_processed sebelum scaling
    # Buat DataFrame sementara untuk kolom yang akan diskalakan
    df_for_minmax_scaling = df_processed[columns_to_scale_minmax]
    
    # Lakukan transformasi menggunakan scaler yang sudah fit
    scaled_data_minmax = minmax_scaler.transform(df_for_minmax_scaling)
    
    # Masukkan kembali hasil scaling ke df_processed
    df_processed[columns_to_scale_minmax] = scaled_data_minmax

    # 4. Standarisasi (StandardScaler)
    # StandardScaler diterapkan pada semua fitur setelah MinMaxScaler (kecuali ID dan target)
    # df_processed harus memiliki urutan dan jumlah kolom yang sama dengan X_resampled saat pelatihan
    # Anda perlu memastikan urutan kolom di X_resampled_scaled sama dengan df_processed di sini.
    # Cara terbaik adalah memastikan input_df memiliki urutan kolom yang sama dengan data asli Anda
    # setelah Label Encoding dan MinMaxScaler
    
    # Untuk memastikan urutan kolom, kita bisa membuat ulang dataframe dengan kolom-kolom yang sama
    # dengan X (sebelum SMOTE) yang digunakan saat pelatihan.
    # Asumsi: X (sebelum SMOTE) punya urutan kolom yang benar.
    # Karena kita tidak menyimpan X, kita bisa membuat list urutan kolom secara manual.
    # Urutan kolom yang diharapkan oleh StandardScaler dan PCA adalah
    # urutan kolom dari X sebelum SMOTE (yaitu df_cleaned setelah semua preprocessing awal dan drop Churn)
    
    # Dapatkan urutan kolom yang benar dari LabelEncoder (jika ada kategorikal yang di encode)
    # Ini adalah urutan kolom yang diharapkan oleh StandardScaler dan PCA
    expected_features_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    # Reindex df_processed untuk memastikan urutan kolom
    df_processed_reindexed = df_processed[expected_features_order]

    # Lakukan standarisasi
    scaled_data_standard = standard_scaler.transform(df_processed_reindexed)

    # 5. Reduksi Dimensi (PCA)
    pca_data = pca_transformer.transform(scaled_data_standard)

    return pca_data


# --- Input Pengguna untuk Fitur ---
st.header('Masukkan Data Pelanggan:')

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    senior_citizen = st.selectbox('Warga Senior', ['No', 'Yes'])
    partner = st.selectbox('Punya Pasangan', ['No', 'Yes'])
    dependents = st.selectbox('Punya Tanggungan', ['No', 'Yes'])
    tenure = st.slider('Lama Berlangganan (bulan)', 0, 72, 12)

with col2:
    phone_service = st.selectbox('Layanan Telepon', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multi Saluran', ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox('Layanan Internet', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Keamanan Online', ['No internet service', 'No', 'Yes'])
    online_backup = st.selectbox('Pencadangan Online', ['No internet service', 'No', 'Yes'])

with col3:
    device_protection = st.selectbox('Proteksi Perangkat', ['No internet service', 'No', 'Yes'])
    tech_support = st.selectbox('Dukungan Teknis', ['No internet service', 'No', 'Yes'])
    streaming_tv = st.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'])
    streaming_movies = st.selectbox('Streaming Film', ['No internet service', 'No', 'Yes'])
    contract = st.selectbox('Jenis Kontrak', ['Month-to-month', 'One year', 'Two year'])
    
    paperless_billing = st.selectbox('Tagihan Tanpa Kertas', ['No', 'Yes'])
    payment_method = st.selectbox('Metode Pembayaran', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Biaya Bulanan', min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input('Total Biaya', min_value=0.0, max_value=10000.0, value=1000.0)


# --- Tombol Prediksi ---
st.markdown("---")
if st.button('Prediksi Churn'):
    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0], # Konversi 'Yes'/'No' ke 1/0
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Lakukan pra-pemrosesan pada input data
    try:
        processed_input = preprocess_input(input_data.copy())
        
        # Prediksi
        prediction = model_klasifikasi.predict(processed_input)
        prediction_proba = model_klasifikasi.predict_proba(processed_input)[:, 1]

        st.subheader('Hasil Prediksi:')
        if prediction[0] == 1:
            st.error(f'Pelanggan ini kemungkinan akan **Churn** dengan probabilitas: **{prediction_proba[0]:.2f}**')
            st.write("Disarankan untuk mengambil tindakan retensi pelanggan.")
        else:
            st.success(f'Pelanggan ini kemungkinan **TIDAK Churn** dengan probabilitas: **{1 - prediction_proba[0]:.2f}**')
            st.write("Pelanggan ini kemungkinan besar akan tetap berlangganan.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input atau melakukan prediksi: {e}")
        st.write("Harap periksa kembali input yang dimasukkan.")