import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Sample data (Area in sqft, Bedrooms, Age in years, Price in ₹L)
data = pd.read_csv('real_estate.csv')
X = data[['area', 'bedrooms', 'stories']]  # Fixed bracket typo here
y = data['price']
model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'real_estate_model.pkl')  # Fixed file extension from 'pk1' to 'pkl'
print("Model saved as real_estate_model.pkl")