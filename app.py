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
# 1. Daftar TLD Terpercaya
# ======================================

TRUSTED_TLDS = {
    # ... (daftar TLD tetap sama seperti sebelumnya)
}

# ======================================
# 2. Fungsi Feature Engineering
# ======================================

def secure_http(url):
    """Menghitung jumlah kemunculan 'https' dalam URL."""
    try:
        return url.lower().count('https')
    except:
        return 0

def url_length(url):
    """Menghitung panjang URL tanpa skema (http/https)."""
    try:
        return len(url.replace('http://', '', 1).replace('https://', '', 1))
    except:
        return 0

def find_subdomain(url):
    """Ekstrak subdomain dari URL."""
    try:
        ext = tldextract.extract(url)
        return ext.subdomain if ext.subdomain else 'tidak_ada'
    except:
        return 'tidak_ada'

def get_tld_func(url):
    """Ekstrak top-level domain (TLD) dari URL."""
    try:
        ext = tldextract.extract(url)
        return ext.suffix if ext.suffix else 'tidak_ada'
    except:
        return 'tidak_ada'

def domain_token_count(url):
    """Menghitung jumlah token dalam domain."""
    try:
        parsed_url = urlparse(url if urlparse(url).scheme else 'http://' + url)
        domain = parsed_url.netloc
        if domain.startswith('[') and domain.endswith(']'):
            domain = domain[1:-1]
        return len(domain.split('.')) if domain else 0
    except:
        return 0

def hostname_length(url):
    """Menghitung panjang hostname."""
    try:
        parsed_url = urlparse(url if urlparse(url).scheme else 'http://' + url)
        hostname = parsed_url.hostname
        return len(hostname) if hostname else 0
    except:
        return 0

def extract_root_domain(url):
    """Ekstrak root domain dari URL."""
    try:
        extracted = tldextract.extract(url)
        return extracted.domain if extracted.domain else 'tidak_ada'
    except:
        return 'tidak_ada'

def count_dot(url):
    """Menghitung jumlah titik dalam URL."""
    try:
        return url.count('.')
    except:
        return 0

def no_of_dir(url):
    """Menghitung jumlah direktori dalam path URL."""
    try:
        path = urlparse(url).path
        return path.count('/')
    except:
        return 0

def digit_count(url):
    """Menghitung jumlah digit dalam URL."""
    try:
        return sum(c.isdigit() for c in url)
    except:
        return 0

def letter_count(url):
    """Menghitung jumlah huruf dalam URL."""
    try:
        return sum(c.isalpha() for c in url)
    except:
        return 0

def fd_length(url):
    """Menghitung panjang direktori pertama dalam path URL."""
    try:
        path = urlparse(url).path
        parts = path.split('/')
        return len(parts[1]) if len(parts) > 1 else 0
    except:
        return 0

def path_length(url):
    """Menghitung panjang path URL."""
    try:
        path = urlparse(url).path
        return len(path)
    except:
        return 0

def num_query_components(url):
    """Menghitung jumlah komponen query dalam URL."""
    try:
        query = urlparse(url).query
        return len(parse_qs(query))
    except:
        return 0

def shortening_service(url):
    """Mendeteksi apakah URL menggunakan layanan pemendek URL."""
    try:
        pattern = re.compile(
            r'(bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|'
            r'cli\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|'
            r'cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|'
            r'vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net)', re.IGNORECASE)
        return 1 if pattern.search(url) else 0
    except:
        return 0

def abnormal_url(url):
    """Mendeteksi apakah hostname muncul lebih dari sekali dalam URL."""
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if not hostname:
            return 0
        occurrences = len(re.findall(re.escape(hostname), url))
        return 1 if occurrences > 1 else 0
    except:
        return 0

def suspicious_words(url):
    """Mendeteksi kata-kata mencurigakan dalam URL."""
    try:
        pattern = re.compile(
            r'(login|signin|bank|account|update|free|bonus|secure|verify|'
            r'password|confirm|validate|authenticate|prize|gift|win|urgent|'
            r'important|alert|notice|support|limited|offer|special|promo|'
            r'discount|reward|survey|contest|trial|membership|billing|invoice|'
            r'purchase|shipment|delivery|tracking|refund|exchange|feedback|'
            r'security|protection|privacy|breach|compromise|alert|warning|'
            r'danger|risk|threat|malware|virus|spyware|phishing|scam|fraud|'
            r'fake|deceptive|misleading|trick|trap|spoof|hack|breach|attack)', re.IGNORECASE)
        return 1 if pattern.search(url) else 0
    except:
        return 0

def having_ip_address(url):
    """Mendeteksi apakah URL mengandung alamat IP."""
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if hostname:
            hostname = hostname.strip('[]')
            ipaddress.ip_address(hostname)
            return 1
        return 0
    except ValueError:
        return 0
    except:
        return 0

def contains_file(url):
    """Memeriksa apakah URL mengandung ekstensi file tertentu."""
    try:
        file_extensions = ['.html', '.txt', '.pdf', '.jpg', '.png', '.doc', '.xls',
                           '.csv', '.mp3', '.mp4', '.exe', '.docx', '.xlsx', '.zip',
                           '.rar', '.js']
        path = urlparse(url).path.lower()
        return 1 if any(ext in path for ext in file_extensions) else 0
    except:
        return 0

def is_trusted_tld(url, weight=10, https_bonus=5):
    """Menentukan skor berdasarkan TLD terpercaya dan penggunaan HTTPS."""
    try:
        ext = tldextract.extract(url)
        tld = ext.suffix.lower()
        is_trusted = tld in TRUSTED_TLDS
        uses_https = url.lower().startswith('https://')
        tld_score = weight if is_trusted else 0
        https_score = https_bonus if uses_https else 0
        total_score = tld_score + https_score
        return total_score
    except:
        return 0

# ======================================
# 3. Load Model and Encoders
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

# ======================================
# 4. Feature Extraction
# ======================================

def extract_features(url, feature_columns):
    """Ekstraksi fitur dari URL."""
    try:
        features = {}
        features['secure_http'] = secure_http(url)
        features['url_length'] = url_length(url)
        features['find_subdomain'] = find_subdomain(url)
        features['tld'] = get_tld_func(url)
        features['domain_token_count'] = domain_token_count(url)
        features['hostname_length'] = hostname_length(url)
        features['root_domain'] = extract_root_domain(url)
        features['count.'] = count_dot(url)
        features['count_dir'] = no_of_dir(url)
        features['count%'] = url.count('%') if isinstance(url, str) else 0
        features['count?'] = url.count('?') if isinstance(url, str) else 0
        features['count-'] = url.count('-') if isinstance(url, str) else 0
        features['count='] = url.count('=') if isinstance(url, str) else 0
        features['count-digits'] = digit_count(url)
        features['count-letters'] = letter_count(url)
        features['fd_length'] = fd_length(url)
        features['path_length'] = path_length(url)
        features['NumQueryComponents'] = num_query_components(url)
        features['short_url'] = shortening_service(url)
        features['abnormal_url'] = abnormal_url(url)
        features['sus_url'] = suspicious_words(url)
        features['use_of_ip'] = having_ip_address(url)
        features['contains_file'] = contains_file(url)
        features['is_trusted_tld'] = is_trusted_tld(url, weight=10, https_bonus=5)
        features['trusted_and_https'] = features['is_trusted_tld'] * features['secure_http']

        # Membuat DataFrame
        df = pd.DataFrame([features])

        # Mengisi nilai kosong pada fitur kategorikal
        df[['find_subdomain', 'tld', 'root_domain']] = df[['find_subdomain', 'tld', 'root_domain']].fillna('tidak_ada')

        # Memastikan semua kolom fitur tersedia
        df = df[feature_columns]

        return df
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return pd.DataFrame()

# ======================================
# 5. Streamlit App Layout
# ======================================

# File paths
MODEL_PATH = "./models/final_xgb_model.pkl"
ENCODER_PATH = "./models/target_encoder.pkl"
LABEL_ENCODER_PATH = "./models/label_encoder.pkl"

def main():
    # Mengatur konfigurasi halaman
    st.set_page_config(page_title="URL Classification", layout="centered", initial_sidebar_state="auto")

    # Judul aplikasi
    st.title("🔍 URL Classification App")
    st.write("Masukkan URL di bawah ini untuk memprediksi apakah URL tersebut **MALICIOUS** atau **BENIGN**.")

    # Inisialisasi history prediksi dalam session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(columns=['URL', 'Prediction'])

    # Membuat tab
    tab1, tab2 = st.tabs(["Prediksi URL", "Riwayat Prediksi"])

    with tab1:
        # Input URL
        user_input = st.text_input("Masukkan URL:", "https://www.example.com")

        if st.button("Prediksi"):
            if user_input.strip() == "":
                st.warning("Silakan masukkan URL yang valid.")
            else:
                with st.spinner('Memproses URL...'):
                    # Load model dan encoders
                    model, encoder, label_encoder = load_model_and_encoders(MODEL_PATH, ENCODER_PATH, LABEL_ENCODER_PATH)

                    # Ekstrak fitur
                    feature_columns = [
                        'secure_http', 'url_length', 'find_subdomain', 'tld', 'domain_token_count',
                        'hostname_length', 'root_domain', 'count.', 'count_dir', 'count%', 'count?',
                        'count-', 'count=', 'count-digits', 'count-letters', 'fd_length',
                        'path_length', 'NumQueryComponents', 'short_url', 'abnormal_url',
                        'sus_url', 'use_of_ip', 'contains_file', 'is_trusted_tld', 
                        'trusted_and_https'
                    ]
                    features = extract_features(user_input, feature_columns)

                    if features.empty:
                        st.error("Gagal mengekstrak fitur dari URL. Pastikan URL valid.")
                    else:
                        # Tampilkan tabel fitur
                        st.subheader("📋 Fitur yang Diekstrak")
                        st.dataframe(features)

                        # Tampilkan detail fitur dengan penjelasan
                        st.subheader("📝 Detail Fitur")
                        feature_descriptions = {
                            'secure_http': "Jumlah kemunculan 'https' dalam URL.",
                            'url_length': "Panjang URL tanpa skema (http/https).",
                            'find_subdomain': "Subdomain yang digunakan dalam URL.",
                            'tld': "Top-level domain (TLD) dari URL.",
                            'domain_token_count': "Jumlah token dalam domain.",
                            'hostname_length': "Panjang hostname dari URL.",
                            'root_domain': "Root domain dari URL.",
                            'count.': "Jumlah titik dalam URL.",
                            'count_dir': "Jumlah direktori dalam path URL.",
                            'count%': "Jumlah karakter '%' dalam URL.",
                            'count?': "Jumlah karakter '?' dalam URL.",
                            'count-': "Jumlah karakter '-' dalam URL.",
                            'count=': "Jumlah karakter '=' dalam URL.",
                            'count-digits': "Jumlah digit dalam URL.",
                            'count-letters': "Jumlah huruf dalam URL.",
                            'fd_length': "Panjang direktori pertama dalam path URL.",
                            'path_length': "Panjang path dalam URL.",
                            'NumQueryComponents': "Jumlah komponen query dalam URL.",
                            'short_url': "Apakah URL menggunakan layanan pemendek URL.",
                            'abnormal_url': "Apakah hostname muncul lebih dari sekali dalam URL.",
                            'sus_url': "Apakah URL mengandung kata-kata mencurigakan.",
                            'use_of_ip': "Apakah URL menggunakan alamat IP.",
                            'contains_file': "Apakah URL mengandung ekstensi file.",
                            'is_trusted_tld': "Skor berdasarkan TLD terpercaya dan penggunaan HTTPS.",
                            'trusted_and_https': "Kombinasi skor TLD terpercaya dan HTTPS."
                        }
                        for col in feature_columns:
                            st.markdown(f"**{col}:** {features.at[0, col]} - {feature_descriptions.get(col, '')}")

                        # Encode fitur kategorikal
                        try:
                            X_encoded = encoder.transform(features)
                        except Exception as e:
                            st.error(f"Error encoding features: {e}")
                            return

                        # Prediksi
                        try:
                            prediction_encoded = model.predict(X_encoded)
                            prediction = label_encoder.inverse_transform(prediction_encoded)

                            # Menambahkan hasil prediksi ke history
                            new_entry = {'URL': user_input, 'Prediction': prediction[0].upper()}
                            st.session_state.prediction_history = pd.concat(
                                [st.session_state.prediction_history, pd.DataFrame([new_entry])],
                                ignore_index=True
                            )

                            # Menampilkan hasil prediksi
                            if prediction[0].lower() == 'malicious':
                                st.error(f"**{prediction[0].upper()}**")
                            else:
                                st.success(f"**{prediction[0].upper()}**")

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

    with tab2:
        # Menampilkan history prediksi
        st.subheader("📜 Riwayat Prediksi")
        st.dataframe(st.session_state.prediction_history, width=1000)

if __name__ == "__main__":
    main()
