import streamlit as st
import pandas as pd
import joblib
import os

# 1. Pengaturan Halaman Utama
st.set_page_config(page_title="Prediksi Churn E-Commerce", page_icon="🎯", layout="wide")

# 2. Fungsi Memuat Model
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'churn_prediction_model.pkl')
    return joblib.load(model_path)

model = load_model()

# 3. Antarmuka Pengguna (UI) - Header
st.title("🎯 Sistem Prediksi Risiko Churn Pelanggan")
st.markdown("Masukkan seluruh parameter operasional pelanggan di bawah ini untuk mengevaluasi probabilitas retensi secara *real-time*.")
st.divider()

# 4. Formulir Input Pengguna (Menggunakan Tabs agar rapi)
tab1, tab2, tab3 = st.tabs(["📋 Profil & Demografi", "📱 Interaksi & Perangkat", "🛍️ Transaksi & Kepuasan"])

with tab1:
    st.subheader("Data Demografi Pelanggan")
    col1_a, col1_b = st.columns(2)
    with col1_a:
        gender = st.selectbox("Jenis Kelamin (Gender)", ["Male", "Female"])
        marital = st.selectbox("Status Pernikahan (MaritalStatus)", ["Married", "Single", "Divorced"])
        city_tier = st.selectbox("Tingkat Kota (CityTier)", [1, 2, 3])
    with col1_b:
        tenure = st.number_input("Masa Berlangganan / Tenure (Bulan)", min_value=0.0, value=1.0, step=1.0)
        address_count = st.number_input("Jumlah Alamat Terdaftar (NumberOfAddress)", min_value=1.0, value=1.0, step=1.0)
        warehouse_dist = st.number_input("Jarak Gudang ke Rumah / WarehouseToHome (km)", min_value=1.0, value=10.0, step=1.0)

with tab2:
    st.subheader("Perilaku Penggunaan Aplikasi")
    col2_a, col2_b = st.columns(2)
    with col2_a:
        login_device = st.selectbox("Perangkat Login (PreferredLoginDevice)", ["Mobile Phone", "Computer", "Phone"])
        payment_mode = st.selectbox("Metode Pembayaran (PreferredPaymentMode)", ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery", "CC"])
    with col2_b:
        device_reg = st.number_input("Jumlah Perangkat Terdaftar (NumberOfDeviceRegistered)", min_value=1.0, value=1.0, step=1.0)
        hour_spend = st.number_input("Jam Dihabiskan di Aplikasi (HourSpendOnApp)", min_value=0.0, value=2.0, step=1.0)

with tab3:
    st.subheader("Riwayat Transaksi dan Evaluasi Layanan")
    col3_a, col3_b = st.columns(2)
    with col3_a:
        order_cat = st.selectbox("Kategori Pesanan Utama (PreferedOrderCat)", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
        order_count = st.number_input("Total Pesanan (OrderCount)", min_value=1.0, value=1.0, step=1.0)
        order_hike = st.number_input("Kenaikan Jumlah Pesanan dari Tahun Lalu (%)", min_value=0.0, value=15.0, step=1.0)
        coupon_used = st.number_input("Kupon Digunakan (CouponUsed)", min_value=0.0, value=1.0, step=1.0)
    with col3_b:
        day_since_order = st.number_input("Hari Sejak Pesanan Terakhir (DaySinceLastOrder)", min_value=0.0, value=5.0, step=1.0)
        cashback = st.number_input("Rata-rata Cashback (CashbackAmount)", min_value=0.0, value=150.0, step=10.0)
        satisfaction = st.slider("Skor Kepuasan (SatisfactionScore)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
        complain = st.selectbox("Riwayat Komplain (Complain)", ["Tidak Ada", "Ada"])
        complain_val = 1.0 if complain == "Ada" else 0.0

st.divider()

# 5. Logika Tombol Prediksi
if st.button("Jalankan Prediksi Risiko", type="primary", use_container_width=True):
    
    # Menyusun data input mentah sesuai urutan kolom pada saat pelatihan
    input_dict = {
        'Tenure': [tenure],
        'PreferredLoginDevice': [login_device],
        'CityTier': [city_tier],
        'WarehouseToHome': [warehouse_dist],
        'PreferredPaymentMode': [payment_mode],
        'Gender': [gender],
        'HourSpendOnApp': [hour_spend],
        'NumberOfDeviceRegistered': [device_reg],
        'PreferedOrderCat': [order_cat],
        'SatisfactionScore': [satisfaction],
        'MaritalStatus': [marital],
        'NumberOfAddress': [address_count],
        'Complain': [complain_val],
        'OrderAmountHikeFromlastYear': [order_hike],
        'CouponUsed': [coupon_used],
        'OrderCount': [order_count],
        'DaySinceLastOrder': [day_since_order],
        'CashbackAmount': [cashback]
    }
    
    # Membuat DataFrame dari input pengguna
    input_df = pd.DataFrame(input_dict)
    
    try:
        # Karena 'model' adalah Pipeline (termasuk preprocessor), kita dapat langsung memasukkan DataFrame mentah
        # Pipeline akan melakukan get_dummies/scaling secara otomatis di latar belakang
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        # Menampilkan Hasil
        st.subheader("Hasil Analisis Sistem:")
        if prediction == 1:
            st.error(f"⚠️ PELANGGAN BERISIKO TINGGI (CHURN) | Probabilitas: {probability:.1f}%")
            st.info("Saran Tindakan: Segera alokasikan program retensi, evaluasi riwayat komplain, atau berikan insentif khusus.")
        else:
            st.success(f"✅ PELANGGAN AMAN (LOYAL) | Probabilitas Churn: {probability:.1f}%")
            st.info("Saran Tindakan: Pertahankan kualitas layanan berdasarkan skor kepuasan saat ini.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat pemrosesan data: {e}")