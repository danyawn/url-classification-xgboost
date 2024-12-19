import streamlit as st
import pandas as pd
import numpy as np
import re
import ipaddress
from urllib.parse import urlparse, parse_qs
import tldextract
from joblib import load
from category_encoders import TargetEncoder
from xgboost import XGBClassifier

# ======================================
# 1. Konfigurasi Aplikasi Streamlit
# ======================================

st.set_page_config(page_title="🔍 URL Classification App",
                   layout="centered", initial_sidebar_state="auto")

# ======================================
# 2. Daftar TLD Terpercaya
# ======================================

# Daftar TLD terpercaya
from constant_datas import TRUSTED_TLDS

# ======================================
# 3. Fungsi Feature Engineering
# ======================================

def secure_http(url):
    """Mendeteksi apakah URL menggunakan HTTPS."""
    try:
        return 1 if url.lower().startswith('https://') else 0
    except:
        return 0

def url_length(url):
    """Menghitung panjang URL tanpa skema (http/https)."""
    try:
        return len(url.replace('http://', '', 1).replace('https://', '', 1))
    except:
        return 0

def subdomain_length(url):
    """Menghitung panjang subdomain dalam URL."""
    try:
        ext = tldextract.extract(url)
        return len(ext.subdomain) if ext.subdomain else 0
    except:
        return 0

def domain_entropy(url):
    """Menghitung entropi domain."""
    try:
        ext = tldextract.extract(url)
        domain = ext.domain
        prob = [float(domain.count(c)) / len(domain) for c in set(domain)]
        return -sum([p * np.log2(p) for p in prob])
    except:
        return 0

def digit_letter_ratio(url):
    """Menghitung rasio digit terhadap huruf dalam URL."""
    try:
        digits = sum(1 for c in url if c.isdigit())
        letters = sum(1 for c in url if c.isalpha())
        return digits / letters if letters > 0 else 0
    except:
        return 0

def special_char_count(url):
    """Menghitung jumlah karakter khusus dalam URL."""
    try:
        return sum(1 for c in url if not c.isalnum())
    except:
        return 0

def contains_cyrillic(url):
    """Mendeteksi apakah URL mengandung karakter Cyrillic."""
    try:
        return 1 if re.search('[\u0400-\u04FF]', url) else 0
    except:
        return 0

def contains_at_symbol(url):
    """Mendeteksi apakah URL mengandung '@'."""
    try:
        return 1 if '@' in url else 0
    except:
        return 0

def subdomain_dot_count(url):
    """Menghitung jumlah titik dalam subdomain."""
    try:
        ext = tldextract.extract(url)
        return ext.subdomain.count('.') if ext.subdomain else 0
    except:
        return 0

def query_param_count(url):
    """Menghitung jumlah parameter query dalam URL."""
    try:
        query = urlparse(url).query
        return len(parse_qs(query))
    except:
        return 0

def path_zero_count(url):
    """Menghitung jumlah angka nol dalam path URL."""
    try:
        path = urlparse(url).path
        return path.count('0')
    except:
        return 0

def path_special_characters(url):
    """Menghitung jumlah karakter khusus dalam path."""
    try:
        path = urlparse(url).path
        return sum(1 for c in path if not c.isalnum())
    except:
        return 0

def domain_hyphen_count(url):
    """Menghitung jumlah tanda hubung dalam domain."""
    try:
        domain = tldextract.extract(url).domain
        return domain.count('-')
    except:
        return 0

def uppercase_lowercase_ratio(url):
    """Menghitung rasio huruf besar terhadap huruf kecil dalam URL."""
    try:
        uppercase = sum(1 for c in url if c.isupper())
        lowercase = sum(1 for c in url if c.islower())
        return uppercase / lowercase if lowercase > 0 else 0
    except:
        return 0

def domain_length(url):
    """Menghitung panjang domain utama."""
    try:
        domain = tldextract.extract(url).domain
        return len(domain)
    except:
        return 0

def query_string_length(url):
    """Menghitung panjang query string."""
    try:
        query = urlparse(url).query
        return len(query)
    except:
        return 0

def suspicious_words(url):
    """Mendeteksi kata-kata mencurigakan dalam URL."""
    try:
        pattern = re.compile(
            r'\b(login|signin|bank|account|update|secure|verify|admin|password|confirm|validate|authenticate|customer|click|win|prize|claim|urgent|important|alert|notice|support|help|access|limited|offer|special|deal|promo|discount|gift|reward)\b',
            re.IGNORECASE)
        return 1 if pattern.search(url) else 0
    except:
        return 0

# ======================================
# 4. Load Model and Encoders
# ======================================

@st.cache_resource
def load_model_and_encoders(model_path, encoder_path, label_encoder_path):
    """Memuat model XGBoost dan encoder yang telah dilatih."""
    try:
        model = load(model_path)
        encoder = load(encoder_path)
        label_encoder = load(label_encoder_path)
        return model, encoder, label_encoder
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau encoder: {e}")
        st.stop()

def validate_and_align_features(features, expected_columns):
    """Validasi dan sesuaikan fitur dengan kolom yang diharapkan model."""
    for col in expected_columns:
        if col not in features.columns:
            features[col] = 0  # Tambahkan kolom yang hilang dengan nilai default
    features = features[expected_columns]  # Hanya gunakan kolom yang diharapkan
    return features

# ======================================
# 5. Feature Extraction
# ======================================

def extract_features(url):
    """Ekstraksi fitur dari URL."""
    features = {
        'secure_http': secure_http(url),
        'url_length': url_length(url),
        'subdomain_length': subdomain_length(url),
        'domain_entropy': domain_entropy(url),
        'digit_letter_ratio': digit_letter_ratio(url),
        'special_char_count': special_char_count(url),
        'contains_cyrillic': contains_cyrillic(url),
        'contains_at_symbol': contains_at_symbol(url),
        'subdomain_dot_count': subdomain_dot_count(url),
        'query_param_count': query_param_count(url),
        'path_zero_count': path_zero_count(url),
        'path_special_characters': path_special_characters(url),
        'domain_hyphen_count': domain_hyphen_count(url),
        'uppercase_lowercase_ratio': uppercase_lowercase_ratio(url),
        'domain_length': domain_length(url),
        'query_string_length': query_string_length(url),
        'suspicious_words': suspicious_words(url)
    }
    return pd.DataFrame([features])

# ======================================
# 6. Streamlit App Layout
# ======================================

MODEL_PATH = "./models-balanced/final_xgb_model.pkl"
ENCODER_PATH = "./models-balanced/target_encoder.pkl"
LABEL_ENCODER_PATH = "./models-balanced/label_encoder.pkl"
DATASET_PATH = "./balanced_urls.csv"

st.title("🔍 URL Classification App")
st.write("Masukkan URL di bawah ini untuk memprediksi apakah URL tersebut **MALICIOUS** atau **BENIGN**.")

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=['URL', 'Prediction'])

tab1, tab2, tab3 = st.tabs(["Prediksi URL", "Riwayat Prediksi", "📊 Dataset"])

with tab1:
    user_input = st.text_input("Masukkan URL:", "https://www.example.com")

    if st.button("Prediksi"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan URL yang valid.")
        else:
            with st.spinner('Memproses URL...'):
                model, encoder, label_encoder = load_model_and_encoders(
                    MODEL_PATH, ENCODER_PATH, LABEL_ENCODER_PATH)

                features = extract_features(user_input)

                try:
                    # Validasi dan sesuaikan fitur
                    expected_columns = encoder.get_feature_names_out()
                    features_aligned = validate_and_align_features(features, expected_columns)

                    # Transformasi fitur
                    X_encoded = encoder.transform(features_aligned)

                    # Prediksi
                    prediction_encoded = model.predict(X_encoded)
                    prediction = label_encoder.inverse_transform(prediction_encoded)

                    st.session_state.prediction_history = pd.concat(
                        [st.session_state.prediction_history,
                         pd.DataFrame({'URL': [user_input], 'Prediction': [prediction[0].upper()]})],
                        ignore_index=True
                    )

                    if prediction[0].lower() == 'malicious':
                        st.error(f"**{prediction[0].upper()}**")
                    else:
                        st.success(f"**{prediction[0].upper()}**")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

with tab2:
    st.subheader("📜 Riwayat Prediksi")
    st.dataframe(st.session_state.prediction_history, width=1000)

with tab3:
    st.subheader("📊 Dataset: 1000 Baris Pertama")
    try:
        data = pd.read_csv(DATASET_PATH)
        st.write(f"Menampilkan 1000 baris dari dataset (Total: {data.shape[0]} baris)")
        st.dataframe(data.head(1000))
    except FileNotFoundError:
        st.error(f"File `{DATASET_PATH}` tidak ditemukan di direktori proyek.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
