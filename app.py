from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# 🟣 Load the trained ML model
model = joblib.load("stock_suggestion_model.pkl")

# 🟣 Encoding maps (same logic used in training)
mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Beginner": 0,
    "Intermediate": 1,
    "Expert": 2,
    "Short-term": 0,
    "Medium-term": 1,
    "Long-term": 2,
    "18-25": 0,
    "26-40": 1,
    "41-60": 2,
    "60+": 3,
    "Below 5L": 0,
    "5L-10L": 1,
    "10L-25L": 2,
    "25L+": 3,
}

# 🟣 Stock map for predictions 0–9
stock_map = {
    0: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],  # Tech
    1: [
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "RELIANCE.NS",
        "ICICIBANK.NS",
    ],  # India Blue Chips
    2: ["TSLA", "META", "NFLX", "ADBE", "AMD"],  # Growth
    3: ["BA", "NKE", "KO", "PEP", "MCD"],  # Consumer
    4: ["JPM", "GS", "BAC", "MS", "V"],  # Finance
    5: ["UNH", "PFE", "JNJ", "MRK", "LLY"],  # Healthcare
    6: ["XOM", "CVX", "BP", "SHEL", "TOT"],  # Energy
    7: ["PG", "COST", "WMT", "TGT", "HD"],  # Retail
    8: ["INTC", "IBM", "ORCL", "SAP", "CSCO"],  # Enterprise Tech
    9: ["AMAT", "ASML", "LRCX", "TXN", "QCOM"],  # Semiconductors
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form inputs
        age_group = request.form["age_group"]
        investment_horizon = request.form["investment_horizon"]
        financial_experience = request.form["financial_experience"]
        annual_income = request.form["annual_income"]
        risk_tolerance = request.form["risk_tolerance"]

        # Encode features
        features = np.array(
            [
                mapping.get(age_group, 0),
                mapping.get(investment_horizon, 0),
                mapping.get(financial_experience, 0),
                mapping.get(annual_income, 0),
                mapping.get(risk_tolerance, 0),
            ]
        ).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        print("Model prediction:", prediction)

        try:
            pred_class = int(prediction)
        except:
            pred_class = 0

        # Get suggested stocks
        suggested = stock_map.get(pred_class, stock_map[0])

        return render_template("result.html", stocks=suggested)

    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
