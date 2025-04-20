from flask import Flask, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def home():
    # For health checks on Render or similar
    if request.method == 'HEAD':
        return '', 200

    result_html = ""
    if request.method == 'POST':
        try:
            # Collect form inputs
            unit_price = float(request.form['unit_price'])
            quantity = float(request.form['quantity'])
            competitor_price = float(request.form['competitor_price'])
            cogs = float(request.form['cogs'])

            # Derived features
            price_gap = unit_price - competitor_price
            profit_margin = (unit_price - cogs) / cogs if cogs != 0 else 0

            # Combine features into an array
            features = np.array([[unit_price, quantity, competitor_price, cogs, price_gap, profit_margin]])

            # Make prediction
            prediction = model.predict(features)[0]
            prediction = round(prediction, 2)

            result_html = f"<h3>Predicted Value: {prediction}</h3>"
        except Exception as e:
            result_html = f"<h3 style='color:red;'>Error: {str(e)}</h3>"

    # HTML page with form + result (if any)
    return f"""
    <html>
        <head><title>Prediction Form</title></head>
        <body>
            <h2>Enter Input Values</h2>
            <form method="post">
                <label>Unit Price:</label><br>
                <input type="text" name="unit_price" required><br><br>

                <label>Quantity:</label><br>
                <input type="text" name="quantity" required><br><br>

                <label>Competitor Price:</label><br>
                <input type="text" name="competitor_price" required><br><br>

                <label>COGS:</label><br>
                <input type="text" name="cogs" required><br><br>

                <input type="submit" value="Predict">
            </form>
            <br>
            {result_html}
        </body>
    </html>
    """

# Optional for local testing
# if __name__ == '__main__':
#     app.run(debug=True)
