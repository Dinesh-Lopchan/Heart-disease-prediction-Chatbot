from django.shortcuts import render
from django.http import JsonResponse
import joblib
from pathlib import Path
from transformers import pipeline
import numpy as np


# Load the Hugging Face conversational AI model
conversational_pipeline = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'chatbot' / 'models' / 'svm_pipeline.joblib'
SCALER_PATH = BASE_DIR / 'chatbot' / 'models' / 'scaler.joblib'
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES_MAPPING = {
    'age': 'age',
    'sex': 'sex',
    'resting blood pressure': 'trestbps',
    'cholesterol': 'chol',
    'fasting blood sugar': 'fbs',
    'maximum heart rate achieved': 'thalach',
    'exercise induced angina': 'exang',
    'depression induced by exercise relative to rest': 'oldpeak',
    'slope of the peak exercise ST segment': 'slope',
    'number of major vessels colored by fluoroscopy': 'ca',
    'chest pain type 1': 'cp_1',
    'chest pain type 2': 'cp_2',
    'chest pain type 3': 'cp_3',
    'resting electrocardiographic results 1': 'restecg_1',
    'resting electrocardiographic results 2': 'restecg_2',
    'thalassemia type 1': 'thal_1',
    'thalassemia type 2': 'thal_2',
    'thalassemia type 3': 'thal_3'
}

def predict_heart_disease(features):
    try:
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return prediction[0]
    except Exception as e:
        return f"Error during prediction: {str(e)}"
    
def parse_features(user_input):
    try:
        data = {}
        for part in user_input.split(","):
            key, value = part.strip().split(":")
            data[key.strip().lower()] = value.strip()

        features = [
            float(data.get("age", 0)),
            1 if data.get("sex", "male").lower() in ["male", "1"] else 0,
            float(data.get("trestbps", 0)),
            float(data.get("chol", 0)),
            int(data.get("fbs", 0)),
            float(data.get("thalach", 0)),
            int(data.get("exang", 0)),
            float(data.get("oldpeak", 0)),
            int(data.get("slope", 0)),
            int(data.get("ca", 0)),
            int(data.get("cp_1", 0)),
            int(data.get("cp_2", 0)),
            int(data.get("cp_3", 0)),
            int(data.get("restecg_1", 0)),
            int(data.get("restecg_2", 0)),
            int(data.get("thal_1", 0)),
            int(data.get("thal_2", 0)),
            int(data.get("thal_3", 0)),
        ]
        return features
    except Exception as e:
        return f"Error parsing features: {str(e)}"


def ask_huggingface(message):
    try:
        if not message.strip():
            return "The input message is empty. Please provide a valid question or input."

        response = conversational_pipeline(message)
        return response[0]["generated_text"]
    except Exception as e:
        return f"An error occurred while fetching the response from Hugging Face: {str(e)}"


def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')

        if any(keyword in message.lower() for keyword in FEATURES_MAPPING.keys()):
            try:
                features = parse_features(message)

                if isinstance(features, list):
                    prediction = predict_heart_disease(features)

                    if prediction == 1:
                        response = "The model predicts a high risk of heart disease."
                    elif prediction == 0:
                        response = "The model predicts a low risk of heart disease."
                    else:
                        response = f"Unexpected prediction result: {prediction}"
                else:
                    response = features
            except Exception as e:
                response = f"An error occurred during prediction: {str(e)}"
        else:
            response = ask_huggingface(message)
            if "error" in response.lower():
                print(f"Error in Hugging Face model: {response}")

        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html')