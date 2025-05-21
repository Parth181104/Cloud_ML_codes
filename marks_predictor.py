import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Input feature: Hours studied
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Target output: Marks obtained
y = np.array([35, 40, 45, 50, 55, 60, 65, 70, 75, 80])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predict marks for a student who studied 7.5 hours
new_hours = np.array([[7.5]])
predicted_marks = model.predict(new_hours)
print(f"Predicted Marks for 7.5 hours of study: {predicted_marks[0]:.2f}")
