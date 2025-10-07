# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, split into features (CGPA, IQ) and target (placement), then divide it into training and testing sets.

2.Standardize feature values using StandardScaler to normalize input data for better model performance.

3.Initialize and train the SGDClassifier on the scaled training set, applying stochastic gradient descent optimization for efficient linear classification.

4.Predict placement outcomes on the test set, compute accuracy, and generate a confusion matrix to evaluate model performance.

5.Extract and display feature importance from model coefficients, then predict placement and decision function value for a new scaled student input.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: VIJAYARAGHAVAN M
RegisterNumber:  25017872
*/

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the placement dataset
df = pd.read_csv('placement.csv')
print(df.head())

# Split features and target
X = df[['cgpa', 'iq']]
y = df['placement']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = sgd_clf.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
           xticklabels=['Not Placed', 'Placed'],
           yticklabels=['Not Placed', 'Placed'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Feature importance
print("\nFeature Importance:")
feature_names = ['CGPA', 'IQ']
for i, feature in enumerate(feature_names):
    print(f"{feature}: {sgd_clf.coef_[0][i]:.4f}")

# Make a prediction for a new student
new_student = [[8.5, 120]]  # Example values for cgpa and iq
new_student_scaled = scaler.transform(new_student)
prediction = sgd_clf.predict(new_student_scaled)
probability = sgd_clf.decision_function(new_student_scaled)

print("\nPrediction for new student:")
print(f"Decision function value: {probability[0]:.4f}")
print(f"Prediction: {'Placed' if prediction[0] == 1 else 'Not Placed'}")

```

## Output:
![prediction of iris species using SGD Classifier](sam.png)
<img width="1056" height="747" alt="Screenshot 2025-10-07 113231" src="https://github.com/user-attachments/assets/c63b85f0-4a37-4071-a7e9-5ae87522e7ab" />
<img width="526" height="178" alt="Screenshot 2025-10-07 113246" src="https://github.com/user-attachments/assets/88a5e8cb-197e-496d-8f18-b899145612a9" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
