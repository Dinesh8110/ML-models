import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Data
data = pd.read_csv("C:/Users/dinesh/learning materials/Machine_learning_SEM-5/ML_assignment1/diabetes1.csv")  # Replace with your actual CSV file path

# Step 2: Preprocess the Data
# Optionally handle missing values, e.g., replace zeros with NaNs then impute with mean
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, pd.NA)

# Fill NaN with the mean of each column
data.fillna(data.mean(), inplace=True)

# Step 3: Split the Data into Features and Labels
X = data.drop('Outcome', axis=1)  # Features (all columns except Outcome)
y = data['Outcome']  # Labels (Outcome column)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

