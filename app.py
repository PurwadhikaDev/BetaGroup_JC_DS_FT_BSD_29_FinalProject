import streamlit as st
import pandas as pd
import pickle
import os

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. LOAD MODEL
# ==============================================================================
@st.cache_resource
def load_model():
    # Jalur file model (pastikan file ada di folder 'model')
    model_path = 'model/final_churn_model_knn.pkl'
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"❌ File model tidak ditemukan di path: {model_path}")
        st.stop()

model = load_model()

# ==============================================================================
# 3. USER INTERFACE (SIDEBAR - INPUT DATA)
# ==============================================================================
st.sidebar.header("📝 Input Data Pelanggan")
st.sidebar.write("Masukkan informasi pelanggan di bawah ini:")

def user_input_features():
    # --- Kelompok 1: Demografi & Akun ---
    st.sidebar.subheader("Profil Pelanggan")
    tenure = st.sidebar.slider("Tenure (Bulan Berlangganan)", 0, 61, 12)
    city_tier = st.sidebar.selectbox("City Tier (Tingkat Kota)", [1, 2, 3])
    
    # --- Kelompok 2: Perilaku Transaksi ---
    st.sidebar.subheader("Aktivitas Belanja")
    order_count = st.sidebar.number_input("Total Order Count (Jumlah Pesanan)", min_value=1, value=5)
    cashback = st.sidebar.number_input("Cashback Amount (Rata-rata Cashback)", min_value=0.0, value=150.0)
    hike_score = st.sidebar.number_input("Kenaikan Belanja dr Thn Lalu (%)", min_value=0.0, value=15.0)
    coupon_used = st.sidebar.number_input("Jumlah Kupon Digunakan", min_value=0, value=1)
    
    # --- Kelompok 3: Interaksi Aplikasi ---
    st.sidebar.subheader("Interaksi Aplikasi")
    hour_spend = st.sidebar.slider("Jam di App (HourSpendOnApp)", 0, 5, 3)
    num_device = st.sidebar.slider("Jumlah Device Terdaftar", 1, 6, 2)
    satisfaction = st.sidebar.slider("Skor Kepuasan (1-5)", 1, 5, 3)
    complain = st.sidebar.selectbox("Pernah Komplain?", ["Tidak", "Ya"])
    
    # --- Kelompok 4: Kategori ---
    st.sidebar.subheader("Preferensi")
    login_device = st.sidebar.selectbox("Login Device", ['Mobile Phone', 'Computer'])
    payment_mode = st.sidebar.selectbox("Metode Pembayaran", ['Debit Card', 'Credit Card', 'E wallet', 'UPI', 'COD'])
    order_cat = st.sidebar.selectbox("Kategori Belanja Utama", ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'])
    
    # Konversi Input Komplain ke 0/1
    complain_val = 1 if complain == "Ya" else 0

    # Bungkus data ke dalam DataFrame
    data = {
        'Tenure': tenure,
        'CityTier': city_tier,
        'WarehouseToHome': 15, # Default value (karena tidak terlalu signifikan di input user)
        'HourSpendOnApp': hour_spend,
        'NumberOfDeviceRegistered': num_device,
        'SatisfactionScore': satisfaction,
        'NumberOfAddress': 3, # Default value
        'OrderAmountHikeFromlastYear': hike_score,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': 5, # PENTING: Fitur ini sudah didrop di pipeline, tapi kadang pipeline butuh dummy input
        'CashbackAmount': cashback,
        'Complain': complain_val,
        'PreferredLoginDevice': login_device,
        'PreferredPaymentMode': payment_mode,
        'PreferedOrderCat': order_cat,
        'Gender': 'Male', # Default (tidak dipakai model)
        'MaritalStatus': 'Single' # Default (tidak dipakai model)
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ==============================================================================
# 4. HALAMAN UTAMA (OUTPUT)
# ==============================================================================
st.title("🛒 E-Commerce Churn Prediction")
st.markdown("""
Aplikasi ini memprediksi apakah seorang pelanggan berpotensi **Churn (Berhenti)** atau **Stay (Setia)** berdasarkan profil dan riwayat transaksinya.
""")

st.write("---")

# Tampilkan data yang diinput user (Preview)
st.subheader("📋 Review Data Pelanggan")
st.dataframe(input_df)

# Tombol Prediksi
if st.button("🔍 Prediksi Sekarang", type="primary"):
    
    # Lakukan Prediksi
    # Catatan: Pipeline kita di dalam pickle akan otomatis handle scaling & encoding
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # Ambil probabilitas Churn (kelas 1)
    churn_prob = probability[0][1]
    
    st.write("---")
    st.subheader("📊 Hasil Analisis")
    
    # Tampilan Hasil
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("⚠️ **PREDIKSI: CHURN (Berisiko Pergi)**")
            st.write(f"Pelanggan ini memiliki probabilitas **{churn_prob*100:.1f}%** untuk berhenti berlangganan.")
        else:
            st.success("✅ **PREDIKSI: STAY (Setia)**")
            st.write(f"Pelanggan ini diprediksi aman. Probabilitas churn hanya **{churn_prob*100:.1f}%**.")

    with col2:
        st.metric(label="Probabilitas Churn", value=f"{churn_prob:.2%}")
        st.progress(churn_prob)

    # Rekomendasi Bisnis (Logika Sederhana)
    st.write("---")
    st.subheader("💡 Rekomendasi Tindakan")
    
    if prediction[0] == 1:
        st.warning("""
        **Saran Strategi:**
        1. Segera kirimkan voucher diskon khusus atau penawaran cashback.
        2. Jika pelanggan pernah komplain, hubungi secara personal untuk memastikan kepuasan.
        3. Tawarkan program loyalitas prioritas.
        """)
    else:
        st.info("""
        **Saran Strategi:**
        1. Pertahankan engagement dengan newsletter berkala.
        2. Tawarkan produk cross-selling (rekomendasi produk lain).
        """)