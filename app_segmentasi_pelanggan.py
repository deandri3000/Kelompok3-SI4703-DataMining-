import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Judul Aplikasi ---
st.set_page_config(page_title="Segmentasi Pelanggan", layout="centered")
st.title('Aplikasi Segmentasi Pelanggan Telco')
st.markdown("""
Aplikasi ini mengelompokkan pelanggan ke dalam segmen (cluster) berdasarkan karakteristik layanan mereka.
""")

# --- Memuat Model dan Objek Pra-pemrosesan ---
try:
    preprocessing_assets_clustering = joblib.load('preprocessing_assets_clustering.pkl')
    model_kmeans = joblib.load('model_kmeans.pkl')
    
    total_charges_mean_clustering = preprocessing_assets_clustering['total_charges_mean']
    label_encoders_clustering = preprocessing_assets_clustering['label_encoders']
    standard_scaler_clustering = preprocessing_assets_clustering['standard_scaler']
    pca_clustering = preprocessing_assets_clustering['pca']
    
    # --- Tambahan: Muat Profil Cluster ---
    cluster_profiles_df = pd.DataFrame.from_dict(preprocessing_assets_clustering['cluster_profiles_df'])


    st.success("Model dan objek pra-pemrosesan untuk clustering berhasil dimuat!")
except FileNotFoundError:
    st.error("""
    **Error:** File model atau pra-pemrosesan (.pkl) untuk clustering tidak ditemukan.
    Pastikan 'preprocessing_assets_clustering.pkl' dan 'model_kmeans.pkl' 
    berada di direktori yang sama dengan script ini.
    """)
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat file: {e}")
    st.stop()


# --- Fungsi untuk Pra-pemrosesan Input Pengguna (sesuai alur clustering) ---
def preprocess_input_for_clustering(input_df):
    df_processed = input_df.copy()

    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed['TotalCharges'].fillna(total_charges_mean_clustering, inplace=True)

    for col, le in label_encoders_clustering.items():
        if col in df_processed.columns:
            df_processed[col] = le.transform(df_processed[col])
        else:
            st.warning(f"Kolom '{col}' dari LabelEncoder tidak ditemukan di input. Mungkin ada kesalahan kolom.")

    expected_features_order_clustering = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    df_processed_reindexed = df_processed[expected_features_order_clustering]

    scaled_data_standard = standard_scaler_clustering.transform(df_processed_reindexed)

    pca_data = pca_clustering.transform(scaled_data_standard)

    return pca_data


# --- Input Pengguna untuk Fitur ---
st.header('Masukkan Data Pelanggan untuk Segmentasi:')

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


# --- Tombol Tentukan Segmen ---
st.markdown("---")
if st.button('Tentukan Segmen Pelanggan'):
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
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

    try:
        processed_input = preprocess_input_for_clustering(input_data.copy())
        segment_prediction = model_kmeans.predict(processed_input)
        
        predicted_cluster_id = segment_prediction[0]

        st.subheader('Hasil Segmentasi:')
        st.success(f'Pelanggan ini termasuk dalam **Segmen Cluster {predicted_cluster_id}**')
        st.info("Setiap nomor cluster mewakili karakteristik pelanggan yang berbeda.")
        
        # --- Tambahan: Tampilkan Profil Cluster Terkait ---
        if predicted_cluster_id in cluster_profiles_df.index:
            st.markdown(f"### Karakteristik Segmen Cluster {predicted_cluster_id}:")
            
            # Mendapatkan profil untuk cluster yang diprediksi
            profile = cluster_profiles_df.loc[predicted_cluster_id].to_frame().T
            
            # Mengatur ulang kolom untuk tampilan yang lebih baik jika perlu
            # profile = profile[['Jumlah Pelanggan', 'Rata-rata Lama Berlangganan (bulan)',
            #                    'Rata-rata Biaya Bulanan ($)', 'Rata-rata Total Biaya ($)',
            #                    '% Warga Senior', 'Mayoritas Gender', 'Kontrak Dominan']]

            st.table(profile) # Menampilkan DataFrame sebagai tabel

        else:
            st.warning("Profil untuk cluster ini tidak ditemukan. Pastikan proses profiling berjalan dengan benar.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input atau melakukan segmentasi: {e}")
        st.write("Harap periksa kembali input yang dimasukkan.")