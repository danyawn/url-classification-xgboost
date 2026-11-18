from constant_datas import TRUSTED_TLDS
import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse, parse_qs
import tldextract
from joblib import load

# ======================================
# 1. Konfigurasi Aplikasi Streamlit
# ======================================

st.set_page_config(page_title="🔍 URL Classification App",
                   layout="centered", initial_sidebar_state="auto")

# ======================================
# 2. Daftar TLD Terpercaya
# ======================================

# Daftar TLD terpercaya

# ======================================
# 3. Fungsi Feature Engineering
# ======================================

# 1. HTTPS detection


def secure_http(url):
    """Check if the URL uses HTTPS."""
    try:
        return 1 if url.lower().startswith('https://') else 0
    except Exception as e:
        print(f"Error in secure_http: {e}")
        return 0

# 2. URL Length


def url_length(url):
    """Calculate the length of the URL excluding the scheme."""
    try:
        stripped_url = re.sub(r'^https?://', '', url)
        return len(stripped_url)
    except Exception as e:
        print(f"Error in url_length: {e}")
        return 0

# 3. Long Subdomain Detection


def is_long_subdomain(url, threshold=20):
    """Check if the subdomain length exceeds a certain threshold."""
    try:
        subdomain = tldextract.extract(url).subdomain
        return 1 if len(subdomain) > threshold else 0
    except Exception as e:
        print(f"Error in is_long_subdomain: {e}")
        return 0

# 4. Domain Entropy


def domain_entropy(url):
    """Calculate entropy of the domain name."""
    try:
        ext = tldextract.extract(url)
        domain = ext.domain
        prob = [float(domain.count(c)) / len(domain) for c in set(domain)]
        return -sum([p * np.log2(p) for p in prob])
    except Exception as e:
        print(f"Error in domain_entropy: {e}")
        return 0

# 5. Protocol Count


def protocol_count(url):
    """Count the occurrences of protocols like 'http' or 'https'."""
    try:
        return url.lower().count('http')
    except Exception as e:
        print(f"Error in protocol_count: {e}")
        return 0

# 6. Suspicious Path Words


def suspicious_path_words(url):
    """Check for sensitive words in the URL path."""
    try:
        path = urlparse(url).path.lower()
        sensitive_terms = [
            "sales",
            "admin",
            "user",
            "verify",       # Mengelabui korban agar memverifikasi akun
            "secure",       # Berpura-pura menjamin keamanan
            "update",       # Menarik korban untuk memperbarui data
            "checkout",     # Menargetkan pembayaran palsu
            "password",     # Mencoba mendapatkan kata sandi korban
            "payment",      # Menargetkan pembayaran korban
            "signin",       # Halaman login palsu
            "reset",        # Mereset kata sandi palsu
            "support",      # Menarik korban dengan klaim layanan pelanggan
            "transfer",     # Menargetkan transaksi bank
            "validate",     # Meminta validasi data
            "submit",       # Halaman pengumpulan informasi
            "details",      # Meminta informasi pribadi
            "auth",         # Autentikasi palsu
            "confirm",      # Menipu pengguna untuk konfirmasi
            "free",         # Menarik dengan tawaran gratis
            "deal",         # Menarik dengan penawaran diskon
            "offer",        # Memancing pengguna dengan promosi
            "service",      # Mengelabui dengan klaim layanan resmi
            "tracking",     # Sering muncul dalam URL phishing pengiriman barang
            "id",           # Sering digunakan untuk meniru ID pengguna
            "dashboard",    # Menargetkan akun dashboard
            "security",     # Klaim keamanan palsu
            "invoice"       # Menargetkan korban dengan klaim tagihan palsu
        ]
        return 1 if any(term in path for term in sensitive_terms) else 0
    except Exception as e:
        print(f"Error in suspicious_path_words: {e}")
        return 0

# 7. TLD Reputation


def tld_reputation(url):
    """Assign a reputation score based on the TLD using TRUSTED_TLDS."""
    try:
        ext = tldextract.extract(url)
        tld = ext.suffix.lower()
        return 1 if tld in TRUSTED_TLDS else 0
    except Exception as e:
        print(f"Error in tld_reputation: {e}")
        return 0

# 8. Digit-to-Letter Ratio


def digit_letter_ratio(url):
    """Calculate the ratio of digits to letters in the URL."""
    try:
        digits = sum(1 for c in url if c.isdigit())
        letters = sum(1 for c in url if c.isalpha())
        return digits / letters if letters > 0 else 0
    except Exception as e:
        print(f"Error in digit_letter_ratio: {e}")
        return 0

# 9. Number of Special Characters


def special_char_count(url):
    """Count the number of special characters in the URL."""
    try:
        special_chars = sum(1 for c in url if not c.isalnum())
        return special_chars
    except Exception as e:
        print(f"Error in special_char_count: {e}")
        return 0

# 10. Path Length


def path_length(url):
    """Calculate the length of the path in the URL."""
    try:
        return len(urlparse(url).path)
    except Exception as e:
        print(f"Error in path_length: {e}")
        return 0

# 11. Presence of Cyrillic Characters


def contains_cyrillic(url):
    """Check if the URL contains Cyrillic characters."""
    try:
        return 1 if re.search('[\u0400-\u04FF]', url) else 0
    except Exception as e:
        print(f"Error in contains_cyrillic: {e}")
        return 0

# 12. Presence of '@' in URL


def contains_at_symbol(url):
    """Check if the URL contains '@'."""
    try:
        return 1 if '@' in url else 0
    except Exception as e:
        print(f"Error in contains_at_symbol: {e}")
        return 0

# 13. Number of Dots in Subdomain


def subdomain_dot_count(url):
    """Count the number of dots in the subdomain."""
    try:
        subdomain = tldextract.extract(url).subdomain
        return subdomain.count('.') if subdomain else 0
    except Exception as e:
        print(f"Error in subdomain_dot_count: {e}")
        return 0

# 14. Number of Query Parameters


def query_param_count(url):
    """Count the number of query parameters in the URL."""
    try:
        query = urlparse(url).query
        return len(parse_qs(query))
    except Exception as e:
        print(f"Error in query_param_count: {e}")
        return 0

# 15. Presence of Zeroes in Path


def path_zero_count(url):
    """Count the number of zeroes in the URL path."""
    try:
        path = urlparse(url).path
        return path.count('0')
    except Exception as e:
        print(f"Error in path_zero_count: {e}")
        return 0

# 16. Path Special Characters


def path_special_characters(url):
    """Count the number of special characters in the path."""
    try:
        path = urlparse(url).path
        special_chars = sum(1 for c in path if not c.isalnum())
        return special_chars
    except Exception as e:
        print(f"Error in path_special_characters: {e}")
        return 0

# 17. Number of Hyphens in Domain


def domain_hyphen_count(url):
    """Count the number of hyphens in the domain."""
    try:
        domain = tldextract.extract(url).domain
        return domain.count('-')
    except Exception as e:
        print(f"Error in domain_hyphen_count: {e}")
        return 0

# 18. Ratio of Uppercase to Lowercase Characters


def uppercase_lowercase_ratio(url):
    """Calculate the ratio of uppercase to lowercase characters in the URL."""
    try:
        uppercase = sum(1 for c in url if c.isupper())
        lowercase = sum(1 for c in url if c.islower())
        return uppercase / lowercase if lowercase > 0 else 0
    except Exception as e:
        print(f"Error in uppercase_lowercase_ratio: {e}")
        return 0

# 19. Length of the Primary Domain


def domain_length(url):
    """Calculate the length of the primary domain."""
    try:
        domain = tldextract.extract(url).domain
        return len(domain)
    except Exception as e:
        print(f"Error in domain_length: {e}")
        return 0

# 20. Query String Length


def query_string_length(url):
    """Calculate the length of the query string."""
    try:
        query = urlparse(url).query
        return len(query)
    except Exception as e:
        print(f"Error in query_string_length: {e}")
        return 0


# Feature list for integration into XGBoost or other ML models

# advanced_features = [
#     "secure_http",
#     "url_length",
#     "subdomain_length",
#     "domain_entropy",
#     "protocol_count",
#     "suspicious_path_words",
#     "tld_reputation",
#     "digit_letter_ratio",
#     "special_char_count",
#     "path_length",
#     "contains_cyrillic",
#     "contains_at_symbol",
#     "subdomain_dot_count",
#     "query_param_count",
#     "path_zero_count",
#     "path_special_characters",
#     "domain_hyphen_count",
#     "uppercase_lowercase_ratio",
#     "domain_length",
#     "query_string_length",
#     # "trusted_and_https",
#     # "entropy_special_char",
# ]

advanced_features = [
    'secure_http',  # 1
    'url_length',  # 2
    'domain_entropy',  # 3
    'contains_cyrillic',  # 4
    'special_char_count',  # 5
    'tld_reputation',  # 6
    'path_length',  # 7
    'digit_letter_ratio',  # 8
    'contains_at_symbol',  # 9
    'subdomain_dot_count',  # 10
    'query_param_count',  # 11
    'path_zero_count',  # 12
    'path_special_characters',  # 13
    'domain_hyphen_count',  # 14
    'uppercase_lowercase_ratio',  # 15
    'domain_length',  # 16
    'query_string_length',  # 17
    "protocol_count",  # 18
    "suspicious_path_words",  # 19
    "is_long_subdomain",  # 20


    # combination features
    # 'trusted_and_https',
    # 'entropy_special_char',
    # 'long_url_long_subdomain',
    # 'path_sensitive_score',
    # 'upper_lower_special_ratio',
    # 'query_complexity',
    # 'entropy_domain_length',
    # 'secure_path_score',
    # 'tld_upper_ratio'

    # 'suspicious_words', # 18
    # 'domain_token_count', # 19
    # 'subdomain_length' # 20
]
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
            # Tambahkan kolom yang hilang dengan nilai default
            features[col] = 0
    # Hanya gunakan kolom yang diharapkan
    features = features[expected_columns]
    return features

# ======================================
# 5. Feature Extraction
# ======================================


def extract_features(url):
    """Ekstraksi fitur dari URL."""
    features = {
        'secure_http': secure_http(url),
        'url_length': url_length(url),
        'is_long_subdomain': is_long_subdomain(url),
        'protocol_count': protocol_count(url),
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
        'path_length': path_length(url),
        'query_string_length': query_string_length(url),
        'tld_reputation': tld_reputation(url),
        'suspicious_path_words': suspicious_path_words(url),
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
    st.session_state.prediction_history = pd.DataFrame(
        columns=pd.Index(['URL', 'Prediction']))

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
                    features_aligned = validate_and_align_features(
                        features, expected_columns)

                    # Transformasi fitur
                    X_encoded = encoder.transform(features_aligned)

                    # Prediksi
                    prediction_encoded = model.predict(X_encoded)
                    prediction = label_encoder.inverse_transform(
                        prediction_encoded)

                    st.session_state.prediction_history = pd.concat(
                        [st.session_state.prediction_history,
                         pd.DataFrame({'URL': [user_input], 'Prediction': [prediction[0].upper()]})],
                        ignore_index=True
                    )

                    # Tampilkan hasil prediksi
                    if prediction[0].lower() == 'malicious':
                        st.error(f"**{prediction[0].upper()}**")
                    else:
                        st.success(f"**{prediction[0].upper()}**")

                    # **Tambahkan Feature Importance Berdasarkan Weight**
                    st.subheader("🔑 Feature Importance Berdasarkan Weight")
                    feature_importances_weight = model.get_booster().get_score(importance_type='weight')
                    feature_importances_weight = pd.Series(
                        feature_importances_weight).sort_values(ascending=False)

                    # Tampilkan dalam tabel
                    st.write(
                        "Berikut adalah feature importance berdasarkan weight:")
                    st.dataframe(feature_importances_weight.reset_index().rename(
                        columns={'index': 'Feature', 0: 'Importance'}), width=1000)

                    # Visualisasi feature importance berdasarkan weight
                    st.write("Visualisasi Feature Importance Berdasarkan Weight:")
                    st.bar_chart(feature_importances_weight)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")


with tab2:
    st.subheader("📜 Riwayat Prediksi")
    st.dataframe(st.session_state.prediction_history, width=1000)

with tab3:
    st.subheader("📊 Dataset: 1000 Baris Pertama")
    
    try:
        data = pd.read_csv(DATASET_PATH)
        random_sample = data.sample(n=1000, random_state=42)
        st.write(
            f"Menampilkan 1000 baris acak dari dataset (Total: {data.shape[0]} baris)")
        st.dataframe(random_sample, width=1000)
    except FileNotFoundError:
        st.error(
            f"File `{DATASET_PATH}` tidak ditemukan di direktori proyek.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
