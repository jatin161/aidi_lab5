import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv(r"Fish.csv")

# Preprocess data
# Assuming 'Species' is a categorical column and needs to be encoded
df = pd.get_dummies(df, drop_first=True)

# Handle missing values (if any)
df = df.dropna()

# Split features and target
X = df.drop(['Weight'], axis=1)  # Adjust this based on your target column
y = df['Weight']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to numpy arrays
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the feature names
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as file:
    pickle.dump(feature_names, file)
