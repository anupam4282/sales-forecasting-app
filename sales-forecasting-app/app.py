from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('model/forecast.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    periods = int(request.form.get('periods', 30))
    
    future = model.make_future_dataframe(periods=periods)
    future['promo'] = 0 
    
    forecast = model.predict(future)
    
    results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    results['ds'] = results['ds'].dt.strftime('%Y-%m-%d')
    
    return jsonify(results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)