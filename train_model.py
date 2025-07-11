from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model to pickle file
joblib.dump(model, 'iris_model.pkl')
print("Model saved as iris_model.pkl")
