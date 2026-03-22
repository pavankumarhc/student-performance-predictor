import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Input features
X = data[['attendance','previous_marks','assignments','internal_marks']]

# Output
y = data['final_marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100)

model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = r2_score(y_test,y_pred)

print("Model Accuracy:", accuracy)