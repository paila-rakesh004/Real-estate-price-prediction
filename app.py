from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('real_estate_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    stories = int(request.form['stories'])
    prediction = model.predict([[area, bedrooms,stories]])
    return render_template('index.html', prediction_text=f'Estimated Price: ₹{prediction[0]:.2f}Lakhs')

if __name__ == '__main__':
    app.run(debug=True)