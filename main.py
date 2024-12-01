# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O puedes especificar una lista de orígenes específicos, por ejemplo: ["http://localhost:4200"]
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Load and preprocess the dataset
file_path = "./Automobile_data.csv"
data = pd.read_csv(file_path)

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Convert columns to numeric where applicable
numeric_columns = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values
data['normalized-losses'] = data['normalized-losses'].fillna(data['normalized-losses'].median())
data['num-of-doors'] = data['num-of-doors'].fillna(data['num-of-doors'].mode()[0])

for col in ['bore', 'stroke', 'horsepower', 'peak-rpm']:
    data[col] = data[col].fillna(data[col].mean())


# Drop rows with missing target variable
data.dropna(subset=['price'], inplace=True)

# One-hot encode categorical columns
categorical_columns = [
    'make', 'fuel-type', 'body-style', 'drive-wheels', 'engine-type',
    'num-of-cylinders', 'fuel-system', 'aspiration', 'num-of-doors', 'engine-location'
]
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm',
                      'engine-size', 'curb-weight', 'width', 'height', 'length']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Separate features and target
X = data.drop(columns=['price'])
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Define input data model
class CarFeatures(BaseModel):
    normalized_losses: float
    bore: float
    stroke: float
    horsepower: float
    peak_rpm: float
    engine_size: float
    curb_weight: float
    width: float
    height: float
    length: float
    make: str
    fuel_type: str
    body_style: str
    drive_wheels: str
    engine_type: str
    num_of_cylinders: str
    fuel_system: str
    aspiration: str
    num_of_doors: str
    engine_location: str

# Load the model and scaler
with open("model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

@app.post("/predict")
def predict_price(features: CarFeatures):
    try:
        # Convert features to DataFrame
        input_data = pd.DataFrame([features.dict()])

        # One-hot encode categorical columns
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Scale numerical features
        input_data[numerical_features] = loaded_scaler.transform(input_data[numerical_features])

        # Predict price
        prediction = loaded_model.predict(input_data)
        return {"predicted_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
