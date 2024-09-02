import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv('./diabetes1.csv')  # Update this with the correct path

# Step 2: Preprocess the data
# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply LDA
lda = LDA(n_components=1)  # Reduce to 1 linear discriminant component
X_lda = lda.fit_transform(X_scaled, y)

# Step 5: Show the explained variance ratio
print("Explained variance ratio (Not applicable for LDA): N/A")  # LDA does not provide variance ratio like PCA

# Display the first few rows of the LDA-transformed dataset
print("LDA-transformed data (first 5 rows):\n", X_lda[:40])

# Step 6: Split the LDA data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

# Step 7: Train a Logistic Regression model on the reduced data
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy after LDA:", accuracy)

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
