import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request

# -----------------------------
# 1. Load datasets
# -----------------------------

train_data = pd.read_csv("data/Training.csv")
test_data = pd.read_csv("data/Testing.csv")

print("Training dataset loaded successfully!")
print("Training shape:", train_data.shape)

print("Testing dataset loaded successfully!")
print("Testing shape:", test_data.shape)


# -----------------------------
# 2. Remove unnecessary columns
# -----------------------------

train_data = train_data.loc[
    :, ~train_data.columns.str.contains("^Unnamed")
]

test_data = test_data.loc[
    :, ~test_data.columns.str.contains("^Unnamed")
]


# -----------------------------
# 3. Separate input and target
# -----------------------------

X = train_data.drop("prognosis", axis=1)
y = train_data["prognosis"]

print("Input features:", X.shape)
print("Target:", y.shape)

print("Number of diseases:", y.nunique())


# -----------------------------
# 4. Train Decision Tree
# -----------------------------

model = DecisionTreeClassifier()
model.fit(X, y)

print("Decision Tree model trained successfully!")


# -----------------------------
# 5. Train Random Forest
# -----------------------------

rf_model = RandomForestClassifier()
rf_model.fit(X, y)

print("Random Forest model trained successfully!")


# -----------------------------
# 6. Test both models
# -----------------------------

X_test = test_data.drop("prognosis", axis=1)
y_test = test_data["prognosis"]

# Keep same feature order
X_test = X_test[X.columns]

dt_predictions = model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nModel Accuracy:")
print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)


# -----------------------------
# 7. Flask application
# -----------------------------

app = Flask(__name__)


# Home page
@app.route("/")
def home():
    return render_template(
        "index.html",
        symptoms=X.columns.tolist()
    )


# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    # Get symptoms entered/selected by user
    user_symptoms = request.form.getlist("symptoms")

    # Create input row with all symptoms = 0
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=X.columns
    )

    # Set selected symptoms to 1
    for symptom in user_symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1

    # Predictions
    dt_result = model.predict(input_data)[0]
    rf_result = rf_model.predict(input_data)[0]

    # Use Random Forest result as the main prediction
    disease = rf_result

    # Find symptoms associated with predicted disease
    disease_rows = train_data[
        train_data["prognosis"] == disease
    ]

    disease_symptoms = []

    if not disease_rows.empty:
        for column in X.columns:
            if disease_rows[column].sum() > 0:
                disease_symptoms.append(column)

    return render_template(
        "index.html",
        symptoms=X.columns.tolist(),
        disease=disease,
        dt_result=dt_result,
        rf_result=rf_result,
        disease_symptoms=disease_symptoms,
        dt_accuracy=round(dt_accuracy * 100, 2),
        rf_accuracy=round(rf_accuracy * 100, 2)
    )


# -----------------------------
# 8. Run Flask
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)