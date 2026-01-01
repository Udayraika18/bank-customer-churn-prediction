from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("Customer_Churn_Prediction.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    CreditScore = int(request.form["CreditScore"])
    Age = int(request.form["Age"])
    Tenure = int(request.form["Tenure"])
    Balance = float(request.form["Balance"])
    NumOfProducts = int(request.form["NumOfProducts"])
    HasCrCard = int(request.form["HasCrCard"])
    IsActiveMember = int(request.form["IsActiveMember"])
    EstimatedSalary = float(request.form["EstimatedSalary"])

    Geography = request.form["Geography_Germany"]
    Geography_Germany = 1 if Geography == "Germany" else 0
    Geography_Spain = 1 if Geography == "Spain" else 0
    # France is baseline (dropped)

    Gender = request.form["Gender_Male"]
    Gender_Male = 1 if Gender == "Male" else 0

    features = [[
        CreditScore, Age, Tenure, Balance, NumOfProducts,
        HasCrCard, IsActiveMember, EstimatedSalary,
        Geography_Germany, Geography_Spain, Gender_Male
    ]]

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "The Customer will leave the bank"
    else:
        result = "The Customer will not leave the bank"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
