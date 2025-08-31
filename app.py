from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("iris_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    features = [float(data["sepal_length"]), float(data["sepal_width"]),
                float(data["petal_length"]), float(data["petal_width"])]
    
    prediction = model.predict([features])[0]
    
    return jsonify({"prediction": prediction})

@app.route("/results")
def results():
    # Read accuracy from metrics file
    try:
        with open("static/metrics.txt", "r") as f:
            accuracy = f.read()
    except:
        accuracy = "Accuracy not available. Run training again."
    return render_template("results.html", accuracy=accuracy)
    
if __name__ == "__main__":
    app.run(debug=True)
