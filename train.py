# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOptionalMemberAccess=false, reportCallIssue=false, reportOptionalSubscript=false
import joblib
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBClassifier, plot_importance
from joblib import dump
from category_encoders import TargetEncoder
from urllib.parse import urlparse, parse_qs
from constant_datas import TRUSTED_TLDS
import numpy as np
import re

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import tldextract

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn import metrics

# Set plot style for better aesthetics
sns.set(style="whitegrid")
# 2. Configuration

# Set up the configuration, including the path to your dataset. Ensure you replace `'./malicious_phish.csv'` dengan path sebenarnya ke file CSV Anda yang berisi data URL.
# Path to your dataset CSV file
DATA_PATH = './balanced_urls.csv'

# Path to save models and encoders
MODEL_PATH = './models-balanced/final_xgb_model.pkl'
ENCODER_PATH = './models-balanced/target_encoder.pkl'
LABEL_ENCODER_PATH = './models-balanced/label_encoder.pkl'
# 3. Feature Engineering Functions

# Define a series of functions to extract meaningful features from URLs. These features will help the model distinguish between malicious and benign URLs.
# Daftar TLD terpercaya

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

# 3. Subdomain Length


def subdomain_length(url):
    """Calculate the length of the subdomain."""
    try:
        ext = tldextract.extract(url)
        return len(ext.subdomain) if ext.subdomain else 0
    except Exception as e:
        print(f"Error in subdomain_length: {e}")
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
        sensitive_terms = ['login', 'reset', 'password', 'admin', 'signin']
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
advanced_features = [
    'secure_http', 'url_length', 'domain_entropy', 'contains_cyrillic',
    'suspicious_words', 'domain_token_count', 'special_char_count',
    'tld_reputation', 'path_length', 'digit_letter_ratio',
    'contains_at_symbol', 'subdomain_dot_count', 'query_param_count',
    'path_zero_count', 'path_special_characters', 'domain_hyphen_count',
    'uppercase_lowercase_ratio', 'domain_length', 'query_string_length'
]

# ======================================
# 4. Loading the Dataset
# ======================================
# Load Data
try:
    data = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully with shape: {data.shape}")

    # Remove duplicates
    data = data.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {data.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}. Please check the path.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit(1)
# ======================================
# 5. Exploring the Dataset
# ======================================
print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nClass Distribution:")
print(data['label'].value_counts())
# ======================================
# 6. Visualizing Word Clouds
# ======================================

# Generate word clouds for each class to visualize common patterns in URLs.


def plot_wordcloud(data_subset, title):
    """Generate a word cloud for a given subset of data."""
    combined_urls = " ".join(i for i in data_subset.url if isinstance(i, str))
    return WordCloud(width=1600, height=800, colormap='Paired',
                     background_color='black').generate(combined_urls)
    # plt.figure(figsize=(12, 14), facecolor='k')
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.title(title, color='white')
    # plt.tight_layout(pad=0)
    # plt.show()


# Plot wordclouds for each class
classes = data['label'].unique()
for cls in classes:
    subset = data[data.label == cls]
    # plot_wordcloud(subset, f"{cls.capitalize()} URLs Wordcloud")

# *This will display word clouds for each class (`benign`, `malicious`, etc.), highlighting the most frequent words in the URLs.*
# ======================================
# 7. Feature Engineering
# ======================================

# Apply the previously defined feature engineering functions to extract features from the URLs.

print("\nStarting feature engineering...")

# Apply feature engineering functions
data['secure_http'] = data['url'].apply(secure_http)
data['url_length'] = data['url'].apply(url_length)
data['subdomain_length'] = data['url'].apply(subdomain_length)
data['domain_entropy'] = data['url'].apply(domain_entropy)
data['protocol_count'] = data['url'].apply(protocol_count)
data['suspicious_path_words'] = data['url'].apply(suspicious_path_words)
data['tld_reputation'] = data['url'].apply(tld_reputation)
data['digit_letter_ratio'] = data['url'].apply(digit_letter_ratio)
data['special_char_count'] = data['url'].apply(special_char_count)
data['path_length'] = data['url'].apply(path_length)
data['contains_cyrillic'] = data['url'].apply(contains_cyrillic)
data['contains_at_symbol'] = data['url'].apply(contains_at_symbol)
data['subdomain_dot_count'] = data['url'].apply(subdomain_dot_count)
data['query_param_count'] = data['url'].apply(query_param_count)
data['path_zero_count'] = data['url'].apply(path_zero_count)
data['path_special_characters'] = data['url'].apply(path_special_characters)
data['domain_hyphen_count'] = data['url'].apply(domain_hyphen_count)
data['uppercase_lowercase_ratio'] = data['url'].apply(
    uppercase_lowercase_ratio)
data['domain_length'] = data['url'].apply(domain_length)
data['query_string_length'] = data['url'].apply(query_string_length)

# Add combination features
data['trusted_and_https'] = data['secure_http'] * data['tld_reputation']
data['entropy_special_char'] = data['domain_entropy'] * \
    data['special_char_count']

# Select Feature Columns
# Remove columns that are not present in the DataFrame
feature_columns = [col for col in advanced_features if col in data.columns]

# ======================================
# 8. Feature Selection
# ======================================

# Select the relevant features for model training
X = data[feature_columns]
y = data['label']

print("\nFeature matrix X shape:", X.shape)
print("Target vector y distribution:")
print(y.value_counts())
# ======================================
# 9. Label Encoding
# ======================================

# Encode the target labels into numerical values suitable for model training.
print("\nEncoding target labels...")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Classes found and their corresponding encoded labels:")
for idx, class_label in enumerate(label_encoder.classes_):
    print(f"{class_label}: {idx}")

# Save the label encoder for future use
dump(label_encoder, LABEL_ENCODER_PATH)
print(f"Label encoder saved to {LABEL_ENCODER_PATH}")
# ======================================
# 10. Handling Missing Values for Categorical Features
# ======================================

# Mengisi nilai null di fitur kategorikal sebelum melakukan encoding.
# Handling missing values
print("\nHandling missing values...")

# Fill missing values for numerical columns with the median
numerical_columns = X.select_dtypes(include=[np.number]).columns
X[numerical_columns] = X[numerical_columns].fillna(
    X[numerical_columns].median())

# Fill missing values for categorical columns with the mode
categorical_columns = X.select_dtypes(include=[object]).columns
if not categorical_columns.empty:
    X[categorical_columns] = X[categorical_columns].fillna(
        X[categorical_columns].mode().iloc[0])

print("Missing values handled successfully.")
# ======================================
# 11. Encoding Categorical Features with TargetEncoder
# ======================================

# Gunakan `TargetEncoder` untuk meng-encode fitur kategorikal.

# Ensure the columns to be encoded are present in X
cols_to_encode = ['trusted_and_https', 'entropy_special_char',
                  'path_special_characters', 'subdomain_dot_count',
                  'contains_cyrillic', 'domain_hyphen_count']

# Add missing columns with default values
for col in cols_to_encode:
    if col not in X.columns:
        print(f"Adding missing column: {col}")
        X.loc[:, col] = 0  # Ensure addition is explicit using .loc

# Ensure there are no null values in columns to be encoded
for col in cols_to_encode:
    if X[col].isnull().any():
        print(f"Filling null values in column: {col}")
        X.loc[:, col] = X[col].fillna('missing')  # Explicitly fill nulls

# Ensure y is a pandas Series
if not isinstance(y, pd.Series):
    print("Converting y to pandas Series...")
    y = pd.Series(y, name='label')

# Convert y to numerical using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Classes and their corresponding encoded labels:")
for class_label, encoded_label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{class_label}: {encoded_label}")

# Initialize TargetEncoder
encoder = TargetEncoder(cols=cols_to_encode)

# Fit encoder on training data
try:
    encoder.fit(X, y_encoded)
    print("TargetEncoder fitted successfully.")
except Exception as e:
    print(f"Error during encoder fitting: {e}")
    raise

# Transform training data
try:
    X_encoded = encoder.transform(X)
    print("TargetEncoder transformation successful.")
except Exception as e:
    print(f"Error during encoder transformation: {e}")
    raise

# Save the encoder and label encoder for future use
try:
    dump(encoder, ENCODER_PATH)
    dump(label_encoder, LABEL_ENCODER_PATH)
    print("Encoders saved successfully.")
except Exception as e:
    print(f"Error saving encoders: {e}")
    raise

# Debugging step: Preview transformed features
print("\nEncoding categorical features completed. Here's a preview of the encoded features:")
print(X_encoded.head())
# ======================================
# 12. Cross-Validation with XGBoost
# ======================================

# Perform k-fold cross-validation to evaluate the model's performance across different subsets of the data.

# Initialize k-fold cross-validator
num_folds = 8
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists to store metrics
accuracy_scores = []
confusion_matrices = []
f1_scores = []
roc_auc_scores = []
feature_importances_list = []

fold_number = 1

for train_index, test_index in kf.split(X_encoded, y_encoded):
    print(f"\n--- Fold {fold_number} ---")

    X_train_fold, X_test_fold = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
    y_train_fold, y_test_fold = y_encoded[train_index], y_encoded[test_index]

    # Ensure y_train_fold and y_test_fold are 1D arrays
    y_train_fold = np.ravel(y_train_fold)
    y_test_fold = np.ravel(y_test_fold)

    # Initialize XGBoost Classifier without use_label_encoder
    xgb_model = XGBClassifier(
        n_estimators=100,  # Tuning n_estimators
        learning_rate=0.1,  # Tuning learning_rate
        max_depth=6,  # Tuning max_depth
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )

    # Train the model
    xgb_model.fit(X_train_fold, y_train_fold)

    # Predict on test fold
    y_pred_xgb = xgb_model.predict(X_test_fold)

    # Confusion Matrix
    cm_fold = metrics.confusion_matrix(
        y_test_fold, y_pred_xgb, labels=np.arange(len(label_encoder.classes_)))
    confusion_matrices.append(cm_fold)

    # Classification Report
    print("\nClassification Report:")
    print(metrics.classification_report(y_test_fold,
          y_pred_xgb, target_names=label_encoder.classes_))

    # Accuracy Score
    score = metrics.accuracy_score(y_test_fold, y_pred_xgb)
    print(f"Accuracy: {score:.3f}")
    accuracy_scores.append(score)

    # F1 Score
    f1_score_fold = metrics.f1_score(
        y_test_fold, y_pred_xgb, average='weighted')
    print(f"F1 Score: {f1_score_fold:.3f}")
    f1_scores.append(f1_score_fold)

    # ROC AUC Score (for multi-class classification)
    probs = xgb_model.predict_proba(X_test_fold)

    # Compute ROC AUC Score for multi-class classification
    if len(label_encoder.classes_) > 2:  # Multi-class scenario
        roc_auc_score_fold = metrics.roc_auc_score(
            y_test_fold, probs, multi_class='ovr')
    else:  # Binary classification
        roc_auc_score_fold = metrics.roc_auc_score(y_test_fold, probs[:, 1])

    print(f"ROC AUC Score: {roc_auc_score_fold:.3f}")
    roc_auc_scores.append(roc_auc_score_fold)

    # Confusion Matrix Heatmap
    cm_df_fold = pd.DataFrame(cm_fold, index=label_encoder.classes_,
                              columns=label_encoder.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df_fold, annot=True, fmt="d", cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold_number}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Feature Importances using XGBoost's plot_importance
    plot_importance(xgb_model, importance_type='weight', max_num_features=10)
    plt.title(f'Feature Importances - Fold {fold_number}')
    plt.show()

    # Store feature importances
    feat_importances = pd.Series(
        xgb_model.feature_importances_, index=X_train_fold.columns)
    feature_importances_list.append(feat_importances)

    fold_number += 1

# Print average scores across all folds
print("\n--- Final Results ---")
print(f"Average Accuracy: {np.mean(accuracy_scores):.3f}")
print(f"Average F1 Score: {np.mean(f1_scores):.3f}")
print(f"Average ROC AUC Score: {np.mean(roc_auc_scores):.3f}")
# ======================================
# 13. Summary of Cross-Validation Metrics
# ======================================

# Calculate and visualize the average accuracy and feature importances across all folds.
# Average Accuracy
average_accuracy = np.mean(accuracy_scores)
print(f"\nAverage Accuracy Across {num_folds} Folds: {average_accuracy:.3f}")

# Aggregate Feature Importances
average_feature_importances = pd.concat(
    feature_importances_list, axis=1).mean(axis=1).sort_values()
average_feature_importances.plot(kind="barh", figsize=(10, 6))
plt.title('Average Feature Importances Across Folds')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
# ======================================
# 14. Training the Final Model
# ======================================

# Train the final XGBoost model on the entire dataset after balancing it with oversampling.

# Define parameter grid for GridSearchCV with reduced combinations
param_grid = {
    'n_estimators': [100, 150],  # Reduce number of estimators to limit fits
    'max_depth': [4, 6],         # Lower max_depth options for simplicity
    'learning_rate': [0.01, 0.1],
    'gamma': [0.1, 0.2],         # Fewer gamma values to explore
    'subsample': [0.8],          # Fix subsample to one value
    'colsample_bytree': [0.8],   # Fix colsample_bytree to one value
    'min_child_weight': [1, 2]   # Reduce min_child_weight options
}

# Initialize base XGBoost model
base_xgb = XGBClassifier(
    random_state=42,
    n_jobs=-1
)

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("\nPerforming GridSearchCV...")
grid_search.fit(X_encoded, y_encoded)

# Output best parameters and score
print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"\nBest cross-validation score: {grid_search.best_score_:.3f}")

# Train the final model using the best parameters
final_xgb = grid_search.best_estimator_
final_xgb.fit(X_encoded, y_encoded)

# Save the final model to disk using dump
dump(final_xgb, MODEL_PATH)
print("Final model saved as 'final_xgb_model.pkl'")

# Feature Importances for Final Model
print('=================================')
plt.figure(figsize=(10, 6))
plot_importance(final_xgb, importance_type='gain', max_num_features=20)
plt.title('Feature Importances for Final Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Evaluate the final model
print("\nEvaluating the final model...")
y_pred = final_xgb.predict(X_encoded)
print(metrics.classification_report(
    y_encoded, y_pred, target_names=label_encoder.classes_))
cm = metrics.confusion_matrix(y_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Final Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ======================================
# 15. Predicting on Test URLs
# ======================================

# Prepare a set of test URLs, apply feature engineering, and use the trained model to predict their classifications.

print("\nStarting feature engineering on test data...")

# Prepare test URLs
test_urls = [
    'https://www.google.com/info',
    'https://www.facebook.com',
    'https://www.amazon.com',
    'https://www.netflix.com',
    'https://www.reddit.com',
    'https://www.youtube.com/',
    'https://www.twitter.com',
    'https://www.linkedin.com',
    'https://www.instagram.com',
    'https://www.microsoft.com',
    'https://www.apple.com',
    'https://www.github.com',
    'https://www.wikipedia.org',
    'https://www.quora.com',
    'https://www.pinterest.com',
    'https://www.ebay.com',
    'https://www.dropbox.com',
    'https://www.paypal.com',
    'https://www.salesforce.com',
    'https://www.tesla.com',
    "allmusic.com/album/crazy-from-the-heat-r16990",
    "http://peluqueriadeautor.com/index.php?option=com_virtuemart&page=shop.browse&category_id=31&Itemid=70",
    "http://9779.info/%E5%84%BF%E7%AB%A5%E7%AB%8B%E4%BD",
    "http://9779.info/%E5%84%BF%E7%AB%A5%E7%AB%8B%E4%BD",
    "http://alexpay2.beget.tech",
    'http://www.pn-wuppertal.de/links/2-linkseite/5-httpwwwkrebshilfede',
    "https://spada.upnyk.ac.id/login/index.php",
    "https://bima.upnyk.ac.id/",
    "https://bankindonesia.co.id/",  # Indonesian bank with cyrillic characters
    "https://xn--80aa0cboq.xn--p1ai",  # Cyrillic domain with Russian TLD
    "https://bаnkindonesia.co.id/",
    "https://www.bi.go.id/id/",
    "https://www.bі.go.іԁ/іԁ/"
]

# Convert URLs to DataFrame
test_data = pd.DataFrame({'url': test_urls})

# Apply feature engineering functions
test_data['secure_http'] = test_data['url'].apply(secure_http)
test_data['url_length'] = test_data['url'].apply(url_length)
test_data['subdomain_length'] = test_data['url'].apply(subdomain_length)
test_data['domain_entropy'] = test_data['url'].apply(domain_entropy)
test_data['protocol_count'] = test_data['url'].apply(protocol_count)
test_data['suspicious_path_words'] = test_data['url'].apply(
    suspicious_path_words)
test_data['tld_reputation'] = test_data['url'].apply(tld_reputation)
test_data['digit_letter_ratio'] = test_data['url'].apply(digit_letter_ratio)
test_data['special_char_count'] = test_data['url'].apply(special_char_count)
test_data['path_length'] = test_data['url'].apply(path_length)
test_data['contains_cyrillic'] = test_data['url'].apply(contains_cyrillic)
test_data['contains_at_symbol'] = test_data['url'].apply(contains_at_symbol)
test_data['subdomain_dot_count'] = test_data['url'].apply(subdomain_dot_count)
test_data['query_param_count'] = test_data['url'].apply(query_param_count)
test_data['path_zero_count'] = test_data['url'].apply(path_zero_count)
test_data['path_special_characters'] = test_data['url'].apply(
    path_special_characters)
test_data['domain_hyphen_count'] = test_data['url'].apply(domain_hyphen_count)
test_data['uppercase_lowercase_ratio'] = test_data['url'].apply(
    uppercase_lowercase_ratio)
test_data['domain_length'] = test_data['url'].apply(domain_length)
test_data['query_string_length'] = test_data['url'].apply(query_string_length)

# Add combination features
test_data['trusted_and_https'] = test_data['secure_http'] * \
    test_data['tld_reputation']
test_data['entropy_special_char'] = test_data['domain_entropy'] * \
    test_data['special_char_count']

# Select Feature Columns
feature_columns = [
    'secure_http', 'url_length', 'subdomain_length', 'domain_entropy', 'protocol_count',
    'suspicious_path_words', 'tld_reputation', 'digit_letter_ratio', 'special_char_count',
    'path_length', 'contains_cyrillic', 'contains_at_symbol', 'subdomain_dot_count',
    'query_param_count', 'path_zero_count', 'path_special_characters', 'domain_hyphen_count',
    'uppercase_lowercase_ratio', 'domain_length', 'query_string_length',
    'trusted_and_https', 'entropy_special_char'
]

# Select features
test_features = test_data[feature_columns]

# Handling missing values
print("\nHandling missing values...")

# Load TargetEncoder and XGBoost model
try:
    encoder_loaded = joblib.load(ENCODER_PATH)
    label_encoder_loaded = joblib.load(LABEL_ENCODER_PATH)
    final_xgb_loaded = joblib.load(MODEL_PATH)
    print("Encoders and model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the encoder and model files exist.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading encoders and model: {e}")
    exit(1)

# Align test features with training features
expected_feature_columns = encoder_loaded.get_feature_names_out()
missing_cols = set(expected_feature_columns) - set(test_features.columns)
for col in missing_cols:
    print(f"Adding missing column: {col}")
    test_features[col] = 0  # Default value for missing features

extra_cols = set(test_features.columns) - set(expected_feature_columns)
if extra_cols:
    print(f"Removing unexpected columns: {extra_cols}")
    test_features = test_features.drop(columns=extra_cols)

# Reorder columns to match the model's training data
test_features = test_features[expected_feature_columns]

# Transform test data using loaded encoder
X_test_encoded = encoder_loaded.transform(test_features)

print("\nFeature engineering on test data completed. Here's a preview:")
print(X_test_encoded.head())

# Predict using the loaded model
test_predictions_encoded = final_xgb_loaded.predict(X_test_encoded)

# Decode predictions to original labels
test_predictions = label_encoder_loaded.inverse_transform(
    test_predictions_encoded)

# Display predictions
test_data['Predicted_Type'] = test_predictions
print("\nTest URL Predictions:")
print(test_data[['url', 'Predicted_Type']])
