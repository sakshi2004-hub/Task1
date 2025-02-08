# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Example dataset: replace this with actual house price data
# DataFrame format: columns could include 'Size', 'Bedrooms', 'Bathrooms', 'Location', 'Price'
data = {
    'Size': [1500, 1800, 2400, 3000, 3500, 4000],
    'Bedrooms': [3, 4, 3, 5, 4, 5],
    'Bathrooms': [2, 3, 2, 3, 3, 4],
    'Location': [1, 2, 2, 1, 1, 3],  # Numerical representation of location (e.g., zip codes or categorical data)
    'Price': [400000, 500000, 600000, 650000, 700000, 750000]  # The target variable: house price
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Feature columns: 'Size', 'Bedrooms', 'Bathrooms', 'Location'
X = df[['Size', 'Bedrooms', 'Bathrooms', 'Location']]

# Target variable: 'Price'
y = df['Price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Displaying the predictions vs. actual prices
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# Optional: Plotting the actual vs predicted prices (just for visual understanding)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
