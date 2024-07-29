from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the feature names
with open('feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# Load dataset for unique values
df = pd.read_csv(r"Fish.csv")

# Extract unique values for each parameter
unique_species = df['Species'].unique()
unique_length1 = df['Length1'].unique()
unique_length2 = df['Length2'].unique()
unique_length3 = df['Length3'].unique()
unique_height = df['Height'].unique()
unique_width = df['Width'].unique()

@app.route('/')
def home():
    return render_template('index.html', 
                           unique_species=unique_species, 
                           unique_length1=unique_length1,
                           unique_length2=unique_length2,
                           unique_length3=unique_length3,
                           unique_height=unique_height,
                           unique_width=unique_width)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])
    species = request.form['species']

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Length1': [length1],
        'Length2': [length2],
        'Length3': [length3],
        'Height': [height],
        'Width': [width],
        'Species': [species]
    })

    # One-hot encode species
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Ensure all expected features are present, even if some are zero
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Reorder columns to match training data
    input_data = input_data[feature_names]

    # Convert to numpy array
    features = input_data.values

    # Make prediction
    prediction = model.predict(features)[0]

    # Render the result page with the prediction
    return render_template('result.html', result=f'The predicted weight is {prediction:.2f} grams.')

if __name__ == '__main__':
    app.run(debug=True)
