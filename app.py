from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Cargar el modelo y el escalador
model = joblib.load("modelo_random_forest.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos JSON de la solicitud
    data = request.json

    # Convertir los datos a un DataFrame
    input_data = pd.DataFrame([data])

    # Escalar los datos de entrada
    input_data_scaled = scaler.transform(input_data)

    # Hacer la predicción
    predicted_rent = model.predict(input_data_scaled)

    # Retornar la predicción en formato JSON
    return jsonify({"predicted_rent": predicted_rent[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
