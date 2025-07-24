from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from numpy.linalg import norm

app = Flask(__name__)

model = None
preprocessor = None

# Schema configuration
required_cols = ["Item", "Category", "Purchase Amount", "Gender"]
optional_cols = ["Size", "Color", "Discount Planned", "Age"]
expected_cols = required_cols + optional_cols

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/score", methods=["POST"])
def score():
    global model, preprocessor

    # TRAINING PHASE
    file = request.files.get("file")
    if file and file.filename.endswith(".csv"):
        df = pd.read_csv(file)

        # Normalize column names
        df.columns = [col.strip().title() for col in df.columns]

        # Validate required columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return f"Missing required columns: {missing}"

        # Fill optional columns
        for col in optional_cols:
            if col not in df.columns:
                df[col] = ""

        # Clean numeric data
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0)
        df["Purchase Amount"] = pd.to_numeric(df["Purchase Amount"], errors="coerce").fillna(0)

        # Define pipelines
        numeric = ["Purchase Amount", "Age"]
        categoric = ["Item", "Category", "Size", "Color", "Gender", "Discount Planned"]

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoric)
        ])

        X = preprocessor.fit_transform(df)

        # Train model with k=3
        model = KMeans(n_clusters=3, init="k-means++", n_init=20, max_iter=300, random_state=42)
        model.fit(X)

    # SCORING PHASE
    if not model or not preprocessor:
        return "Model not trained. Upload a valid CSV first."

    # Extract form input safely
    try:
        purchase_amount = float(request.form.get("amount", "0").strip())
    except ValueError:
        return "Error: Purchase Amount must be numeric."

    age_input = request.form.get("age", "").strip()
    try:
        age = int(age_input) if age_input else 0
    except ValueError:
        age = 0

    new_product = {
        "Item": request.form.get("item", ""),
        "Category": request.form.get("category", ""),
        "Purchase Amount": purchase_amount,
        "Size": request.form.get("size", ""),
        "Color": request.form.get("color", ""),
        "Age": age,
        "Gender": request.form.get("gender", ""),
        "Discount Planned": request.form.get("discount", "")
    }

    input_df = pd.DataFrame([new_product])

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = ""

    try:
        X_new = preprocessor.transform(input_df)
    except Exception as e:
        return f"Preprocessing error: {str(e)}"

    # Cluster prediction and dynamic scoring
    cluster = model.predict(X_new)[0]
    centroid = model.cluster_centers_[cluster]

    # Handle sparse matrix if needed
    if hasattr(X_new, "toarray"):
        X_array = X_new.toarray()
    else:
        X_array = X_new

    distance = norm(X_array - centroid)

    # Compare to other centroids to get relative confidence
    other_centroids = [model.cluster_centers_[i] for i in range(model.n_clusters) if i != cluster]
    max_distance = max(norm(X_array - center) for center in model.cluster_centers_)
    confidence = 1 - (distance / max_distance)
    confidence = max(0, min(confidence, 1))

    # Score range per cluster
    score_ranges = {
        0: (85, 95),
        1: (65, 75),
        2: (45, 60)
    }

    low, high = score_ranges.get(cluster, (50, 70))
    score = round(low + confidence * (high - low), 1)

    return render_template("result.html", cluster=cluster, score=score)

if __name__ == "__main__":
    app.run(debug=True)
