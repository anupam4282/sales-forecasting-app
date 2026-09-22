import pandas as pd
from prophet import Prophet
import pickle

# Load dataset
df = pd.read_csv('data/historical_sales.csv')

# Initialize and train Prophet model
model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
model.add_regressor('promo')
model.fit(df)

# Save the trained model
with open('model/forecast.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Model trained and saved successfully!")