import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(data):
    # Convert categorical columns (e.g., Sex) to numeric
    data['Sex'] = data['Sex'].map({'Male': 0, 'Female': 1})
    
    # Extract numerical values from 'Blood Pressure'
    data['Systolic_BP'] = data['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
    data['Diastolic_BP'] = data['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
    
    # Select features and target
    features = data[['Age', 'Sex', 'Cholesterol', 'Systolic_BP', 'Diastolic_BP',
                     'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
                     'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'BMI', 
                     'Triglycerides']]
    target = data['Heart Attack Risk']
    
    return features, target

def train_model(data):
    # Preprocess the data
    features, target = preprocess_data(data)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict_risk(input_data, model, scaler):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input data to match model training
    input_df['Sex'] = input_df['Sex'].map({'Male': 0, 'Female': 1})
    input_df['Systolic_BP'] = input_df['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
    input_df['Diastolic_BP'] = input_df['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
    
    # Select only the necessary columns for prediction
    input_df = input_df[['Age', 'Sex', 'Cholesterol', 'Systolic_BP', 'Diastolic_BP',
                         'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
                         'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'BMI',
                         'Triglycerides']]
    
    # Scale the data
    input_scaled = scaler.transform(input_df)
    
    # Predict the risk
    prediction = model.predict(input_scaled)
    return prediction[0]
