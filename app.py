from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# -----------------------------
# Load Model & Artifacts
# -----------------------------
model = joblib.load("models/best_model.joblib")
# scaler = joblib.load("scaler.joblib")  # Pre-fitted scaler from training
# Load the dataset for preprocessing reference
top_20_features = joblib.load("models/top_20_features.joblib")  # List of 20 input feature names
file_path = 'PCOS_data_without_infertility.csv'
sheet_name = 'Full_new'
df = pd.read_excel(file_path, sheet_name=sheet_name)
# Ensure numerical conversion and handle missing values
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except (ValueError, TypeError):
        pass

df.drop_duplicates(inplace=True)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
numerical_features = df.select_dtypes(include=['number']).columns
df[numerical_features] = imputer.fit_transform(df[numerical_features])
scaler = StandardScaler()
df[top_20_features] = scaler.fit_transform(df[top_20_features])
# -----------------------------
# CSV Files for Deployment
# -----------------------------
DEPLOY_CSV = "PCOS_test_deploy_set.csv"
USER_CSV = "pcos_user_test.csv"

# Initialize CSV files if they do not exist
for f in [DEPLOY_CSV, USER_CSV]:
    if not os.path.exists(f):
        pd.DataFrame(columns=["Index", "PCOS"] + list(top_20_features)).to_csv(f, index=False)

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/patient", methods=["GET", "POST"])
def patient():
    """
    Patient route:
    - Collects the 20 feature values (raw input).
    - Uses the saved scaler to transform the raw inputs.
    - Predicts PCOS using the pre-loaded model.
    - Checks if this record already exists in the CSV; if not, saves the raw inputs and the prediction.
    """
    if request.method == "POST":
        # 1) Collect user input (raw values)
        user_data = []
        for feature in top_20_features:
            val = request.form.get(feature, "0")
            try:
                user_data.append(float(val))
            except ValueError:
                user_data.append(0.0)
        
        # 2) Scale the user input using the saved scaler
        user_df = pd.DataFrame([user_data], columns=top_20_features)
        user_scaled = scaler.transform(user_df)

        # 3) Predict using the model
        prediction = model.predict(user_scaled)[0]          # Output 0 or 1
        prediction_proba = model.predict_proba(user_scaled)[0]  # e.g., [0.77, 0.23]

        # 4) Check if this record already exists in DEPLOY_CSV (compare raw values)
        deploy_df = pd.read_csv(DEPLOY_CSV)
        existing = False
        existing_index = -1
        for _, row in deploy_df.iterrows():
            try:
                row_features = row[list(top_20_features)].values.astype(float)
                if np.allclose(row_features, user_data, atol=1e-6):
                    existing = True
                    existing_index = int(row["Index"])
                    break
            except Exception:
                continue

        # 5) Save new record if not existing
        if not existing:
            new_index = 0 if deploy_df.empty else int(deploy_df["Index"].max()) + 1
            new_row_df = pd.DataFrame([[new_index, prediction] + user_data],
                                      columns=["Index", "PCOS"] + list(top_20_features))
            deploy_df = pd.concat([deploy_df, new_row_df], ignore_index=True)
            deploy_df.to_csv(DEPLOY_CSV, index=False)

            user_csv_df = pd.read_csv(USER_CSV)
            user_csv_df = pd.concat([user_csv_df, new_row_df], ignore_index=True)
            user_csv_df.to_csv(USER_CSV, index=False)
            final_index = new_index
        else:
            final_index = existing_index

        return render_template("patient_result.html",
                               prediction=prediction,
                               prob=prediction_proba,
                               index=final_index)
    return render_template("patient.html", features=top_20_features)

@app.route("/gyno", methods=["GET", "POST"])
def gyno():
    """
    Gynaecologist route:
    - On GET: show a form that allows entering a patient index.
    - On POST: load that record from DEPLOY_CSV, transform the raw inputs using the saved scaler, predict, and display results.
    """
    df_test = pd.read_csv(DEPLOY_CSV)
    max_index = int(df_test['Index'].max()) if not df_test.empty else 0
    
    if request.method == "POST":
        idx_str = request.form.get("index", "0")
        try:
            idx = int(idx_str)
            if idx < 0 or idx > max_index:
                return render_template("gyno.html", max_index=max_index, error=True)
        except ValueError:
            return render_template("gyno.html", max_index=max_index, error=True)
        
        row = df_test[df_test["Index"] == idx]
        if row.empty:
            return render_template("gyno.html", max_index=max_index, error=True)
        
        try:
            # Extract raw feature values from CSV; ensure you select only the input features.
            values = [float(row.iloc[0][f]) for f in top_20_features]
            
            input_df = pd.DataFrame([values], columns=top_20_features)
            user_scaled = scaler.transform(input_df)
            prediction = model.predict(user_scaled)[0]
            prediction_proba = model.predict_proba(user_scaled)[0]
            
            return render_template("gyno_result.html",
                                row=row.iloc[0],
                                prediction=prediction,
                                prob=prediction_proba,
                                features=top_20_features,
                                values=values,
                                index=idx)
        except Exception as e:
            return render_template("gyno.html", max_index=max_index, error=True)
    
    return render_template("gyno.html", max_index=max_index)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
