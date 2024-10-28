# 🔍 URL Classification Project

Welcome to the **URL Classification Project**! This repository houses two powerful applications designed to **classify URLs as malicious or benign**. Whether you're looking to understand the data processing and model training process or want an interactive web interface for real-time URL classification, you've come to the right place! 🚀

---

## 📜 Table of Contents

- [🎯 Overview](#-overview)
- [🛠️ Setup Instructions](#️-setup-instructions)
  - [1. Create a Virtual Environment](#1-create-a-virtual-environment)
  - [2. Activate the Virtual Environment](#2-activate-the-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
- [🏃‍♂️ Running the Applications](#-running-the-applications)
  - [A. Data Processing and Model Training](#a-data-processing-and-model-training)
  - [B. URL Classification Web App](#b-url-classification-web-app)
- [📁 Project Structure](#-project-structure)
- [📝 Notes](#-notes)
- [🙏 Credits](#-credits)

---

## 🎯 Overview

This project consists of two main components:

1. **Data Processing and Model Training (`url_classification.py`)**  
   This script handles data loading, feature engineering, model training with cross-validation, and saving the trained model along with necessary encoders.

2. **URL Classification Web App (`app.py`)**  
   An interactive Streamlit application that allows users to input URLs and receive instant classifications as **MALICIOUS** or **BENIGN**.

---

## 🛠️ Setup Instructions

Follow these steps to set up and run the applications seamlessly.

### 1. Create a Virtual Environment

Using a virtual environment ensures that dependencies are managed effectively without interfering with your system's Python installation.

Open your terminal or command prompt and navigate to your project directory:

```bash
python -m venv venv
```

This command creates a new virtual environment named `venv` in your project directory.

### 2. Activate the Virtual Environment

Activate the virtual environment using the appropriate command based on your operating system:

- **For Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **For macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

After activation, your terminal prompt will show `(venv)` indicating that the virtual environment is active.

### 3. Install Dependencies

Ensure you're in the project directory where `requirements.txt` is located. Install all necessary packages using:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This command installs all the required packages listed in the `requirements.txt` file.

---

## 🏃‍♂️ Running the Applications

### A. Data Processing and Model Training

**Script:** `url_classification.py`

This script performs the following tasks:

- **Data Loading:** Reads the dataset containing URLs labeled as malicious or benign.
- **Feature Engineering:** Extracts meaningful features from URLs to aid in classification.
- **Model Training:** Trains an XGBoost classifier using cross-validation and handles class imbalance with oversampling.
- **Saving Artifacts:** Saves the trained model and encoders for future predictions.

**Steps to Run:**

1. **Ensure the Dataset is Available:**

   - Place your dataset CSV file (e.g., `malicious_phish.csv`) in a directory named `data` within your project folder.
   - Verify that the `DATA_PATH` variable in `url_classification.py` points to the correct location of your dataset.

2. **Run the Script:**

   ```bash
   python url_classification.py
   ```

   Sit back and relax! ☕ The script will process the data, train the model, and save the necessary artifacts. This might take some time depending on your system's performance.

3. **Outputs:**

   - **Trained Model:** Saved as `final_xgb_model.pkl` in the `models` directory.
   - **Encoders:**
     - `target_encoder.pkl` for categorical feature encoding.
     - `label_encoder.pkl` for target label encoding.

   Ensure that the `models` directory exists. If not, create it manually or modify the script to handle directory creation.

### B. URL Classification Web App

**Script:** `app.py`

An interactive Streamlit app that lets you input URLs and get instant classifications! 🌐

**Steps to Run:**

1. **Ensure the Trained Model and Encoders are Available:**

   - Make sure `final_xgb_model.pkl`, `target_encoder.pkl`, and `label_encoder.pkl` are present in the `models` directory.

2. **Run the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

   The app will automatically open in your default web browser. If it doesn't, navigate to `http://localhost:8501`.

3. **Using the App:**

   - **Input URL:** Enter the URL you want to classify.
   - **Predict:** Click on the "Prediksi" button to see the magic happen! ✨
   - **Extracted Features:** View the features extracted from your URL.
   - **Prediction History:** Check out the "Riwayat Prediksi" tab to see all your past predictions.

---

## 📁 Project Structure

Your project directory should look like this:

```
project/
├── app.py                     # Streamlit application script
├── url_classification.py      # Data processing and model training script
├── requirements.txt           # List of required Python packages
├── malicious_phish.csv        # Your dataset
├── models/
│   ├── final_xgb_model.pkl    # Trained XGBoost model
│   ├── target_encoder.pkl     # Target encoder for categorical features
│   └── label_encoder.pkl      # Label encoder for target variable
└── venv/                      # Virtual environment directory
```

---

## 📝 Notes

- **Dataset Availability:** Ensure that the dataset is placed correctly in the `data` directory.
- **Adjusting Paths:** Update paths in `url_classification.py` and `app.py` if your directories differ.
- **Python Version:** Developed and tested on **Python 3.10.15**.
- **Deactivating Virtual Environment:** When done, deactivate with:

  ```bash
  deactivate
  ```

- **Dependencies:** All required packages are in `requirements.txt`. Update if necessary.
- **Model Retraining:** To retrain with new data or parameters, rerun `url_classification.py`.
- **Need Help?** Open an issue or reach out if you face any problems!

---

## 🙏 Credits

Developed with ❤️ by **Wayan Danu Tirta** (NIM: 123200025)  
**Universitas Pembangunan Nasional Veteran Yogyakarta**

---

**Enjoy exploring the world of URL classification!** 🌐🔒