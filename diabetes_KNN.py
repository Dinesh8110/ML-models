import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the data
data = pd.read_csv('./diabetes1.csv')  # Update file path if needed

# Step 2: Preprocess the data
# Replace zeros in columns where zero is invalid, except for Pregnancies and Outcome
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_to_replace] = data[columns_to_replace].replace(0, pd.NA)

# Fill missing values (NaN) with column means
data.fillna(data.mean(), inplace=True)

# Step 3: Split the data into features and target labels
X = data.drop('Outcome', axis=1)  # Features (all columns except Outcome)
y = data['Outcome']  # Target label (Outcome column)

# Optionally, scale the features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train the k-Nearest Neighbors Classifier
knn_clf = KNeighborsClassifier(n_neighbors=20)  # You can change the value of k (number of neighbors)
knn_clf.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = knn_clf.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
