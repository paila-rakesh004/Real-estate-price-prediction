import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Sample data (Area in sqft, Bedrooms, number of floors(stories), Price in ₹L)
data = pd.read_csv('real_estate.csv')
X = data[['area', 'bedrooms', 'stories']]  
y = data['price']
model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'real_estate_model.pkl')  
print("Model saved as real_estate_model.pkl")
