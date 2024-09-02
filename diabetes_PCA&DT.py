import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


# Step 1: Load the dataset
data = pd.read_csv('./diabetes1.csv')  # Update this with the correct path

# Step 2: Preprocess the data
# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X_scaled)

# Step 5: Show the explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Display the first few rows of the PCA-transformed dataset
print("PCA-transformed data (first 5 rows):\n", X_pca[:20])

# # Step 6: Split the PCA data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# # Step 7: Train a Logistic Regression model on the reduced data
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Step 8: Predict and evaluate
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy after PCA:", accuracy)

# # Print classification report
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

