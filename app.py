"""
Bank Customer Churn Prediction and Retention Intelligence System
Flask web app — Phase 7 of the project.

Loads the trained, calibrated model saved from the notebook (churn_model.joblib)
and serves a form for entering a customer's details, returning a churn
probability, risk level, and retention recommendation.
"""

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "churn_model.joblib")
model = joblib.load(MODEL_PATH)

# This threshold was derived by running a precision/recall sweep against the
# exact final_model in the notebook (Phase 6), chosen to catch ~81% of actual
# churners in the test set. If you regenerate churn_model.joblib with a
# different calibration approach, re-run that sweep and update this value.
CHURN_THRESHOLD = 0.16

# Geography is deliberately excluded as a model input: it stands in for national
# origin, a protected characteristic under fair-lending law (ECOA). Using it to
# differentially flag customers for retention outreach risks the same kind of
# profiling that law exists to prevent, regardless of its predictive value.


def classify(proba_pct, flagged):
    """Mirrors the predict_churn() risk-bucket logic from the notebook."""
    if proba_pct >= 70:
        return "High", "Contact this customer with a retention offer and review their account activity."
    elif proba_pct >= 40:
        return "Medium", "Monitor this customer and consider a proactive engagement touchpoint."
    elif flagged:
        return "Elevated", "Add to retention outreach queue — model recall threshold met, review before outreach."
    else:
        return "Low", "No action needed at this time."


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    customer = {
        "CreditScore": int(form["credit_score"]),
        "Gender": form["gender"],
        "Age": int(form["age"]),
        "Tenure": int(form["tenure"]),
        "Balance": float(form["balance"]),
        "NumOfProducts": int(form["num_products"]),
        "HasCrCard": 1 if form.get("has_cr_card") == "on" else 0,
        "IsActiveMember": 1 if form.get("is_active_member") == "on" else 0,
        "EstimatedSalary": float(form["estimated_salary"]),
    }

    input_df = pd.DataFrame([customer])
    proba = model.predict_proba(input_df)[0, 1]
    proba_pct = round(proba * 100, 1)
    flagged = bool(proba >= CHURN_THRESHOLD)
    risk_level, recommendation = classify(proba_pct, flagged)

    # Needle rotation: -90deg (0%) to 90deg (100%) across the semicircle gauge
    needle_rotation = -90 + (proba_pct / 100) * 180

    return render_template(
        "result.html",
        customer=customer,
        probability=proba_pct,
        flagged=flagged,
        risk_level=risk_level,
        recommendation=recommendation,
        needle_rotation=needle_rotation,
    )


if __name__ == "__main__":
    app.run(debug=True)
