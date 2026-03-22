from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and target
X = data[['attendance', 'previous_marks', 'assignments', 'internal_marks']]
y = data['final_marks']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        attendance = float(request.form['attendance'])
        previous_marks = float(request.form['previous_marks'])
        assignments = float(request.form['assignments'])
        internal_marks = float(request.form['internal_marks'])

        prediction = model.predict([[attendance, previous_marks, assignments, internal_marks]])

        return render_template(
            "index.html",
            prediction_text=f"Predicted Marks: {prediction[0]:.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error: Please enter valid inputs"
        )

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)