from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        unit_price = float(request.form['unit_price'])
        quantity = float(request.form['quantity'])
        competitor_price = float(request.form['competitor_price'])
        cogs = float(request.form['cogs'])

         # Create price difference
        price_gap = unit_price - competitor_price

        # Create profit margin (avoid divide by zero)
        profit_margin = (unit_price - cogs) / cogs if cogs != 0 else 0

        # Combine features into an array
        features = np.array([[unit_price, quantity, competitor_price, cogs, price_gap, profit_margin]])

        # Make prediction
        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return f"Something went wrong: {e}"


    # Run the app
if __name__ == '__main__':
    app.run(debug=True)