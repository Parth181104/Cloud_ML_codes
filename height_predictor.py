import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example data: [Age, Weight] as input features
X = np.array([
    [5, 20],
    [10, 35],
    [15, 50],
    [20, 65],
    [25, 70],
    [30, 75],
    [35, 80],
    [40, 85]
])

# Corresponding heights (in cm) as target output
y = np.array([105, 125, 145, 160, 165, 170, 172, 174])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predict height for a new person (e.g., Age=28, Weight=72)
new_person = np.array([[28, 72]])
predicted_height = model.predict(new_person)
print("Predicted Height (cm):", predicted_height[0])
