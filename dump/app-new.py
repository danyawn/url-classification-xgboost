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
from sklearn.model_selection import GridSearchCV

# Keep all the TRUSTED_TLDS and feature engineering functions the same
# [Previous TRUSTED_TLDS definition and all feature engineering functions remain unchanged]

# ======================================
# 1. Daftar TLD Terpercaya
# ======================================

# Daftar TLD terpercaya
TRUSTED_TLDS = {
    # TLD Generik Umum
    'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'info', 'co', 'biz', 'me', 'pro',
    'ai', 'dev', 'shop', 'tech', 'store', 'io', 'online', 'site', 'website', 'app', 'club',
    'news', 'health', 'travel', 'blog', 'design', 'media', 'agency',

    # TLD Indonesia yang terpercaya
    'ac.id', 'sch.id', 'go.id', 'or.id', 'web.id', 'co.id', 'my.id', 'ponpes.id',
    'biz.id', 'net.id', 'com.id', 'desa.id',

    # TLD Internasional lainnya yang terpercaya
    # United Kingdom
    'co.uk', 'gov.uk', 'ac.uk', 'org.uk', 'me.uk', 'ltd.uk', 'plc.uk', 'net.uk',

    # Australia
    'edu.au', 'gov.au', 'com.au', 'org.au', 'net.au', 'id.au', 'asn.au',

    # Kanada
    'ca', 'gc.ca', 'co.ca', 'org.ca', 'net.ca',

    # Jepang
    'co.jp', 'ac.jp', 'ad.jp', 'ed.jp', 'go.jp', 'or.jp', 'ne.jp',

    # Jerman
    'de', 'ac.de', 'gov.de', 'edu.de', 'org.de', 'net.de',

    # Prancis
    'fr', 'ac.fr', 'gov.fr', 'asso.fr', 'org.fr', 'com.fr',

    # Amerika Serikat
    'us', 'edu.us', 'gov.us', 'mil.us', 'net.us', 'org.us',

    # Belanda
    'nl', 'bv.nl', 'voornaam.nl', 'achternaam.nl',

    # Singapura
    'sg', 'com.sg', 'net.sg', 'org.sg', 'gov.sg', 'edu.sg',

    # Korea Selatan
    'kr', 'co.kr', 'ne.kr', 'or.kr', 're.kr', 'pe.kr',

    # Cina
    'cn', 'com.cn', 'net.cn', 'org.cn', 'gov.cn', 'edu.cn',

    # Brasil
    'br', 'com.br', 'net.br', 'org.br', 'gov.br', 'edu.br',

    # Rusia
    'ru', 'com.ru', 'net.ru', 'org.ru', 'gov.ru', 'edu.ru',

    # India
    'in', 'co.in', 'net.in', 'org.in', 'firm.in', 'gen.in', 'ind.in',

    # Afrika Selatan
    'za', 'co.za', 'org.za', 'net.za', 'gov.za', 'mil.za', 'ac.za',

    # Uni Emirat Arab
    'ae', 'co.ae', 'net.ae', 'org.ae', 'gov.ae', 'sch.ae',

    # Seluruh Dunia
    'eu', 'asia', 'mobi', 'museum', 'travel', 'jobs', 'name', 'post', 'tel', 'tv', 'cc',
    'me', 'us', 'ca', 'uk', 'de', 'jp', 'fr', 'au', 'cn', 'br', 'ru', 'in', 'za', 'sg',
    'kr', 'it', 'es', 'nl', 'se', 'no', 'fi', 'dk', 'be', 'ch', 'at', 'pl', 'cz', 'hu',
    'sk', 'si', 'ro', 'bg', 'gr', 'tr', 'il', 'my', 'th', 'vn', 'id', 'ph', 'tw', 'hk',
}

# ======================================
# 2. Fungsi Feature Engineering
# ======================================

# Add GridSearch functionality
@st.cache_resource
def perform_grid_search(X_train, y_train):
    """Perform GridSearchCV for XGBoost model."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1]
    }
    
    base_xgb = XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search

@st.cache_resource
def load_model_and_encoders(model_path, encoder_path, label_encoder_path):
    """Load trained XGBoost model and encoders."""
    try:
        model = load(model_path)
        encoder = load(encoder_path)
        label_encoder = load(label_encoder_path)
        return model, encoder, label_encoder
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        st.stop()

def extract_features_batch(urls, feature_columns):
    """Extract features for multiple URLs."""
    features_list = []
    for url in urls:
        features = extract_features(url, feature_columns)
        features_list.append(features)
    return pd.concat(features_list, ignore_index=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="🔍 URL Classification App",
        layout="centered",
        initial_sidebar_state="auto"
    )

    # App title
    st.title("🔍 URL Classification App")
    st.write("Enter a URL below to predict if it's **MALICIOUS** or **BENIGN**.")

    # Initialize prediction history in session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(
            columns=['URL', 'Prediction']
        )

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Predict URL", "Prediction History", "📊 Dataset", "🔧 Model Tuning"]
    )

    with tab1:
        # URL input
        user_input = st.text_input("Enter URL:", "https://www.example.com")

        if st.button("Predict"):
            if user_input.strip() == "":
                st.warning("Please enter a valid URL.")
            else:
                with st.spinner('Processing URL...'):
                    # Load model and encoders
                    model, encoder, label_encoder = load_model_and_encoders(
                        MODEL_PATH, ENCODER_PATH, LABEL_ENCODER_PATH
                    )

                    # Extract features
                    feature_columns = [
                        'secure_http', 'url_length', 'find_subdomain', 'tld', 
                        'domain_token_count', 'hostname_length', 'root_domain', 
                        'count.', 'count_dir', 'count%', 'count?', 'count-', 
                        'count=', 'count-digits', 'count-letters', 'fd_length',
                        'path_length', 'NumQueryComponents', 'short_url', 
                        'abnormal_url', 'sus_url', 'use_of_ip', 'contains_file', 
                        'is_trusted_tld', 'trusted_and_https'
                    ]
                    features = extract_features(user_input, feature_columns)

                    if features.empty:
                        st.error("Failed to extract features from URL.")
                    else:
                        try:
                            X_encoded = encoder.transform(features)
                            prediction_encoded = model.predict(X_encoded)
                            prediction = label_encoder.inverse_transform(prediction_encoded)

                            # Add to prediction history
                            new_entry = {'URL': user_input, 'Prediction': prediction[0].upper()}
                            st.session_state.prediction_history = pd.concat(
                                [st.session_state.prediction_history, pd.DataFrame([new_entry])],
                                ignore_index=True
                            )

                            # Display prediction
                            if prediction[0].lower() == 'malicious':
                                st.error(f"**{prediction[0].upper()}**")
                            else:
                                st.success(f"**{prediction[0].upper()}**")

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

                        # Display feature details
                        st.subheader("📝 Feature Details")
                        feature_descriptions = {
                            'secure_http': "Number of 'https' occurrences",
                            'url_length': "URL length without scheme",
                            'find_subdomain': "Subdomain used",
                            'tld': "Top-level domain",
                            # Add descriptions for other features...
                        }
                        
                        feature_details = pd.DataFrame({
                            'Feature': feature_columns,
                            'Value': [features.at[0, col] for col in feature_columns],
                            'Description': [feature_descriptions.get(col, '') for col in feature_columns]
                        })

                        st.table(feature_details)

    with tab2:
        # Display prediction history
        st.subheader("📜 Prediction History")
        st.dataframe(st.session_state.prediction_history, width=1000)

    with tab3:
        # Display dataset
        st.subheader("📊 Dataset: First 1000 Rows")
        try:
            data = pd.read_csv(DATASET_PATH)
            st.write(f"Showing 1000 rows from dataset (Total: {data.shape[0]} rows)")
            st.dataframe(data.head(1000))
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    with tab4:
        st.subheader("🔧 Model Hyperparameter Tuning")
        
        if st.button("Start Grid Search"):
            with st.spinner('Performing Grid Search...'):
                try:
                    # Load and prepare data
                    data = pd.read_csv(DATASET_PATH)
                    
                    feature_columns = [
                        'secure_http', 'url_length', 'find_subdomain', 'tld', 
                        'domain_token_count', 'hostname_length', 'root_domain', 
                        'count.', 'count_dir', 'count%', 'count?', 'count-', 
                        'count=', 'count-digits', 'count-letters', 'fd_length',
                        'path_length', 'NumQueryComponents', 'short_url', 
                        'abnormal_url', 'sus_url', 'use_of_ip', 'contains_file', 
                        'is_trusted_tld', 'trusted_and_https'
                    ]
                    
                    X = extract_features_batch(data['url'].tolist(), feature_columns)
                    
                    # Load encoders and prepare target variable
                    _, _, label_encoder = load_model_and_encoders(
                        MODEL_PATH, ENCODER_PATH, LABEL_ENCODER_PATH
                    )
                    y = label_encoder.fit_transform(data['type'])
                    
                    # Perform grid search
                    grid_search = perform_grid_search(X, y)
                    
                    # Display results
                    st.success("Grid Search completed!")
                    st.write("Best parameters:", grid_search.best_params_)
                    st.write("Best score:", grid_search.best_score_)
                    
                    # Display CV results
                    cv_results = pd.DataFrame(grid_search.cv_results_)
                    st.write("Cross-validation Results:")
                    st.dataframe(cv_results)
                    
                    # Save best model
                    best_model = grid_search.best_estimator_
                    dump(best_model, MODEL_PATH)
                    st.success(f"Best model saved to {MODEL_PATH}")
                    
                except Exception as e:
                    st.error(f"Error during grid search: {e}")

if __name__ == "__main__":
    main()