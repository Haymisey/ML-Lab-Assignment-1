from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os
from typing import Literal

app = FastAPI(
    title="Diabetes Prediction API",
    description="ML models for diabetes prediction using Decision Tree and Logistic Regression",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models
MODEL_DIR = "models"
decision_tree_model = None
logistic_regression_model = None
scaler = None

@app.on_event("startup")
async def load_models():
    global decision_tree_model, logistic_regression_model, scaler
    try:
        # Load Decision Tree model
        dt_path = os.path.join(MODEL_DIR, "decision_tree_model.joblib")
        if os.path.exists(dt_path):
            decision_tree_model = joblib.load(dt_path)
            print(f"✓ Decision Tree model loaded from decision_tree_model.joblib")
        else:
            print("⚠ Warning: Decision Tree model not found")
        
        # Load Logistic Regression model
        lr_path = os.path.join(MODEL_DIR, "logistic_regression_model.joblib")
        if os.path.exists(lr_path):
            logistic_regression_model = joblib.load(lr_path)
            print(f"✓ Logistic Regression model loaded from logistic_regression_model.joblib")
        else:
            print("⚠ Warning: Logistic Regression model not found")
        
        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded from scaler.joblib")
        else:
            print("⚠ Warning: Scaler not found (predictions may be inaccurate if models were trained with scaled data)")
            
    except Exception as e:
        print(f"Error loading models: {e}")

# Pydantic models for request validation
class DiabetesInput(BaseModel):
    """Input features for diabetes prediction"""
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=300, description="Glucose level")
    blood_pressure: float = Field(..., ge=0, le=200, description="Blood pressure (mm Hg)")
    skin_thickness: float = Field(..., ge=0, le=100, description="Skin thickness (mm)")
    insulin: float = Field(..., ge=0, le=900, description="Insulin level (mu U/ml)")
    bmi: float = Field(..., ge=0, le=70, description="Body Mass Index")
    diabetes_pedigree: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    age: int = Field(..., ge=1, le=120, description="Age in years")

    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 6,
                "glucose": 148.0,
                "blood_pressure": 72.0,
                "skin_thickness": 35.0,
                "insulin": 0.0,
                "bmi": 33.6,
                "diabetes_pedigree": 0.627,
                "age": 50
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_type: str
    prediction: int
    prediction_label: str
    probability: float = None

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    info = {
        "status": "healthy",
        "models_loaded": {
            "decision_tree": decision_tree_model is not None,
            "logistic_regression": logistic_regression_model is not None,
            "scaler": scaler is not None
        }
    }
    
    # Add scaler feature names if available
    if scaler is not None:
        try:
            feature_names = getattr(scaler, 'feature_names_in_', None)
            if feature_names is not None:
                info["scaler_features"] = feature_names.tolist()
        except:
            pass
    
    # Add model info if available
    if decision_tree_model is not None:
        try:
            info["decision_tree_info"] = {
                "max_depth": getattr(decision_tree_model, 'tree_', {}).max_depth if hasattr(decision_tree_model, 'tree_') else None,
                "n_features": getattr(decision_tree_model, 'n_features_in_', None)
            }
        except:
            pass
    
    # Add model info if available
    if decision_tree_model is not None:
        try:
            if hasattr(decision_tree_model, 'tree_'):
                info["decision_tree_info"] = {
                    "max_depth": decision_tree_model.tree_.max_depth,
                    "n_features": getattr(decision_tree_model, 'n_features_in_', None)
                }
        except:
            pass
    
    return info

@app.get("/test-models")
async def test_models():
    """Test endpoint to verify models work with different inputs"""
    if decision_tree_model is None or logistic_regression_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Test with different inputs
    test_cases = [
        {"pregnancies": 6, "glucose": 148.0, "blood_pressure": 72.0, "skin_thickness": 35.0, 
         "insulin": 0.0, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50},
        {"pregnancies": 1, "glucose": 85.0, "blood_pressure": 66.0, "skin_thickness": 29.0, 
         "insulin": 0.0, "bmi": 26.6, "diabetes_pedigree": 0.351, "age": 31},
        {"pregnancies": 8, "glucose": 183.0, "blood_pressure": 64.0, "skin_thickness": 0.0, 
         "insulin": 0.0, "bmi": 23.3, "diabetes_pedigree": 0.672, "age": 32}
    ]
    
    results = []
    for i, test_case in enumerate(test_cases):
        input_data = DiabetesInput(**test_case)
        
        # Decision Tree
        features_dt = np.array([[
            input_data.pregnancies, input_data.glucose, input_data.blood_pressure,
            input_data.skin_thickness, input_data.insulin, input_data.bmi,
            input_data.diabetes_pedigree, input_data.age
        ]])
        features_df_dt = pd.DataFrame(features_dt, columns=['Pregnancies', 'Glucose', 'BloodPressure', 
                      'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        features_scaled_dt = scaler.transform(features_df_dt)
        dt_pred = decision_tree_model.predict(features_scaled_dt)[0]
        dt_proba = decision_tree_model.predict_proba(features_scaled_dt)[0] if hasattr(decision_tree_model, 'predict_proba') else None
        
        # Logistic Regression
        features_lr = np.array([[
            input_data.pregnancies, input_data.glucose, input_data.blood_pressure,
            input_data.skin_thickness, input_data.insulin, input_data.bmi,
            input_data.diabetes_pedigree, input_data.age
        ]])
        features_df_lr = pd.DataFrame(features_lr, columns=['Pregnancies', 'Glucose', 'BloodPressure', 
                      'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        features_scaled_lr = scaler.transform(features_df_lr)
        lr_pred = logistic_regression_model.predict(features_scaled_lr)[0]
        lr_proba = logistic_regression_model.predict_proba(features_scaled_lr)[0] if hasattr(logistic_regression_model, 'predict_proba') else None
        
        results.append({
            "test_case": i + 1,
            "input": test_case,
            "decision_tree": {
                "prediction": int(dt_pred),
                "probability": float(dt_proba[1]) if dt_proba is not None else None
            },
            "logistic_regression": {
                "prediction": int(lr_pred),
                "probability": float(lr_proba[1]) if lr_proba is not None else None
            }
        })
    
    return {"test_results": results}

@app.post("/predict/decision-tree", response_model=PredictionResponse)
async def predict_decision_tree(input_data: DiabetesInput):
    """Predict using Decision Tree model"""
    if decision_tree_model is None:
        raise HTTPException(status_code=503, detail="Decision Tree model not loaded")
    
    try:
        # Prepare input features as numpy array
        # Note: Decision Trees are scale-invariant and typically don't need scaling
        # If your Decision Tree was trained with scaled data, uncomment the scaling section below
        features = np.array([[
            input_data.pregnancies,
            input_data.glucose,
            input_data.blood_pressure,
            input_data.skin_thickness,
            input_data.insulin,
            input_data.bmi,
            input_data.diabetes_pedigree,
            input_data.age
        ]])
        
        # Decision Trees are scale-invariant, so we use raw features
        # If your Decision Tree was trained with scaled data, uncomment these lines:
        if scaler is not None:
            features_df = pd.DataFrame(features, columns=['Pregnancies', 'Glucose', 'BloodPressure', 
                         'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            features = scaler.transform(features_df)
        
        # Make prediction
        prediction = int(decision_tree_model.predict(features)[0])
        
        # Get probability if available
        probability = None
        if hasattr(decision_tree_model, 'predict_proba'):
            proba = decision_tree_model.predict_proba(features)[0]
            probability = float(proba[1])
        
        return PredictionResponse(
            model_type="Decision Tree",
            prediction=prediction,
            prediction_label="Diabetic" if prediction == 1 else "Non-Diabetic",
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/logistic-regression", response_model=PredictionResponse)
async def predict_logistic_regression(input_data: DiabetesInput):
    """Predict using Logistic Regression model"""
    if logistic_regression_model is None:
        raise HTTPException(status_code=503, detail="Logistic Regression model not loaded")
    
    try:
        # Prepare input features as DataFrame (to match scaler's expected format)
        features_df = pd.DataFrame([[
            input_data.pregnancies,
            input_data.glucose,
            input_data.blood_pressure,
            input_data.skin_thickness,
            input_data.insulin,
            input_data.bmi,
            input_data.diabetes_pedigree,
            input_data.age
        ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Apply scaling (Logistic Regression requires scaled features)
        if scaler is not None:
            features_scaled = scaler.transform(features_df)
        else:
            raise HTTPException(status_code=503, detail="Scaler not loaded. Logistic Regression model requires scaled features.")
        
        # Make prediction
        prediction = int(logistic_regression_model.predict(features_scaled)[0])
        
        # Get probability
        probability = None
        if hasattr(logistic_regression_model, 'predict_proba'):
            proba = logistic_regression_model.predict_proba(features_scaled)[0]
            probability = float(proba[1])
        
        return PredictionResponse(
            model_type="Logistic Regression",
            prediction=prediction,
            prediction_label="Diabetic" if prediction == 1 else "Non-Diabetic",
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/both")
async def predict_both_models(input_data: DiabetesInput):
    """Get predictions from both models"""
    results = {}
    
    if decision_tree_model is not None:
        try:
            dt_result = await predict_decision_tree(input_data)
            results["decision_tree"] = dt_result
        except Exception as e:
            results["decision_tree"] = {"error": str(e)}
    
    if logistic_regression_model is not None:
        try:
            lr_result = await predict_logistic_regression(input_data)
            results["logistic_regression"] = lr_result
        except Exception as e:
            results["logistic_regression"] = {"error": str(e)}
    
    if not results:
        raise HTTPException(status_code=503, detail="No models available")
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
