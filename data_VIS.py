import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# # Step 1: Load the Data
# data = pd.read_csv("C:/Users/dinesh/learning materials/Machine_learning_SEM-5/ML_assignment1/diabetes1.csv")  # Replace with your actual CSV file path

# # Step 2: Preprocess the Data
# # Optionally handle missing values, e.g., replace zeros with NaNs then impute with mean
# data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, pd.NA)

# # Fill NaN with the mean of each column
# data.fillna(data.mean(), inplace=True)

# # Step 3: Split the Data into Features and Labels
# X = data.drop('Outcome', axis=1)  # Features (all columns except Outcome)
# y = data['Outcome']  # Labels (Outcome column)

# # Split the data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Train the Decision Tree Classifier
# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X_train, y_train)

# # Step 5: Make Predictions
# y_pred = clf.predict(X_test)

# # Step 6: Evaluate the Model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Step 7: Feature Importance Visualization
# feature_importances = clf.feature_importances_
# features = X.columns

# # Create a bar plot of feature importances
# plt.figure(figsize=(10, 6))
# plt.barh(features, feature_importances, color='skyblue')
# plt.xlabel('Feature Importance')
# plt.ylabel('Features')
# plt.title('Feature Importance in Decision Tree')
# plt.show()

# # Step 8: Decision Tree Visualization
# plt.figure(figsize=(16, 12))
# plot_tree(clf, feature_names=features, class_names=['0', '1'], filled=True, rounded=True, fontsize=10)
# plt.title("Decision Tree Visualization")
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

# For demonstration, generate a synthetic dataset with 2 features
# If you have more than 2 features, use PCA or similar to reduce dimensions
data = pd.read_csv("C:/Users/dinesh/learning materials/Machine_learning_SEM-5/ML_assignment1/diabetes1.csv")
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Use PCA to reduce to 2 features for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Split the PCA-transformed data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train the classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the mesh grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap='coolwarm', marker='o', s=50)
plt.title('Decision Boundary and Training Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
