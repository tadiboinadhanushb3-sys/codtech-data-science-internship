from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return "House Price Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.json
    
    features = np.array(data["features"]).reshape(1, -1)
    
    prediction = model.predict(features)
    
    return jsonify({
        "predicted_price": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)