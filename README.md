# M.I.N.D. – Mental Health Screening App

## Overview

**M.I.N.D.** is an interactive web-based questionnaire built with Streamlit that estimates the probability of several common mental health conditions based on user-reported symptoms.

The application uses a trained machine learning model to provide probabilistic insights for:

* Depression
* Anxiety
* ADHD
* Autism Spectrum Disorder (ASD)
* Obsessive-Compulsive Disorder (OCD)
* Schizophrenia

**Disclaimer:** This tool is for informational purposes only and does not provide medical diagnosis.

---

## Features

* Guided questionnaire with structured inputs
* Multi-page flow (intro → questions → results)
* Real-time progress tracking
* Probabilistic predictions using a trained ML model
* Interactive UI powered by Streamlit
* External educational links for each condition

---

## Tech Stack

* Python
* Streamlit
* Pandas
* XGBoost
* Scikit-learn
* Joblib

---

## Project Structure

```
project/
│
├── front_end.py              				# Main Streamlit application (UI + prediction logic)
├── regressor_generator.py   				# Script used to train and generate the ML model
├── mental_health_multilabel_dataset.csv              	# AI-generated dataset used for training
├── model.joblib             				# Trained ML model
├── feature.joblib           				# Feature columns
├── label.joblib             				# Label columns
└── README.md		               			# Project documentation 
```

---

## Dataset

The dataset used in this project is **AI-generated**.

It was created to simulate realistic symptom patterns across multiple mental health conditions and is intended for:

* experimentation
* prototyping
* model training demonstrations

Because the dataset is synthetic, predictions **should not be considered clinically accurate**.

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/mind-app.git
cd mind-app
```

### 2. Create a virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install streamlit pandas xgboost scikit-learn joblib
```

---

## Running the App

### Option 1: Using Streamlit

```
streamlit run front_end.py
```

### Option 2: Using a batch file (Windows)

Create a file named `run.bat`:

```
@echo off
cd /d "%~dp0"
python -m streamlit run front_end.py
pause
```

---

## Model Training

To regenerate or retrain the model, run:

```
python regressor_generator.py
```

This script:

* processes the dataset
* trains the multi-output classifier
* exports:

  * `model.joblib`
  * `feature.joblib`
  * `label.joblib`

---

## How It Works

1. The user answers a series of structured questions.
2. Responses are encoded into numerical features.
3. The trained model predicts probabilities for each condition.
4. Results are displayed with visual indicators.

---

## Model Details

* Base model: XGBoost Classifier
* Wrapper: MultiOutputClassifier (scikit-learn)
* Output: Probability scores for multiple conditions simultaneously

---

## Important Disclaimer

This application:

* Is **not a diagnostic tool**
* Should **not replace professional medical advice**
* Uses a **synthetic (AI-generated) dataset**
* Is intended for **educational and experimental purposes only**

If you experience severe symptoms or distress, consult a licensed professional.

---

## Author

Developed as a machine learning and health-tech project.

---

## Support

For issues or suggestions, feel free to open an issue or contribute.
