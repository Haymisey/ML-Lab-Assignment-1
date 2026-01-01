## Name: Haymanot Abera
## ID: UGR/9265/15


# Diabetes Prediction ML Application

A FastAPI-based web application for diabetes prediction using Decision Tree and Logistic Regression models.

## Features

- **Dual Model Predictions**: Compare results from Decision Tree and Logistic Regression
- **UI**: Responsive interface with dark theme
- **Real-time Predictions**: Instant diabetes risk assessment
- **RESTful API**: Well-documented endpoints for integration

## Setup Instructions

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at: `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```

### Predictions
```
POST /predict/decision-tree
POST /predict/logistic-regression
POST /predict/both
```

## Input Features

The model expects 8 features:

1. **Pregnancies**: Number of times pregnant (0-20)
2. **Glucose**: Plasma glucose concentration (0-300 mg/dL)
3. **Blood Pressure**: Diastolic blood pressure (0-200 mm Hg)
4. **Skin Thickness**: Triceps skin fold thickness (0-100 mm)
5. **Insulin**: 2-Hour serum insulin (0-900 mu U/ml)
6. **BMI**: Body mass index (0-70)
7. **Diabetes Pedigree Function**: Diabetes pedigree function (0-3)
8. **Age**: Age in years (1-120)

