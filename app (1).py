# from flask import Flask, render_template, request
# import pandas as pd
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the trained model and scaler (if needed)
# with open('model (2).pkl', 'rb') as f:
#      model = pickle.load(f)  # Assuming both model and scaler are saved together

# # Define the predict_risk function
# def predict_risk(input_data):
#     # Convert the input_data dictionary into a DataFrame
#     input_df = pd.DataFrame([input_data])
    
#     # Scale the input data
#     # scaled_data = scaler.transform(input_df)
    
#     # Make prediction
#     # prediction = model.predict(scaled_data)
#     # return prediction

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get input values from the form
#         form_data = request.form
#         input_data = {
#             'Age': int(form_data['age']),
#             'Sex': form_data['sex'],  # Assuming 'Sex' is a categorical variable, keep as string
#             'Cholesterol': float(form_data['cholesterol']),
#             'Blood Pressure':int(form_data['blood_pressure']),
#             'Heart Rate': int(form_data['heart_rate']),
#             'Diabetes': int(form_data['diabetes']),
#             'Family History': int(form_data['family_history']),
#             'Smoking': int(form_data['smoking']),
#             'Obesity': int(form_data['obesity']),
#             'Physical Activity Days Per Week': int(form_data['physical_activity']),
#             'Sleep Hours Per Day': int(form_data['sleep_hours']),
#             'BMI': float(form_data['bmi']),
#             'Triglycerides': float(form_data['triglycerides']),
#         }

#         # If 'Sex' is categorical, you may need to encode it. Example:
#         # For demonstration, we'll just assume a mapping for 'Sex' if required
#         # Handle categorical variable 'Sex'
#         if input_data['Sex'] == 'Male':
#             input_data['Sex'] = 1
#         elif input_data['Sex'] == 'Female':
#             input_data['Sex'] = 0
#         else:
#             raise ValueError("Invalid value for 'Sex'. Must be 'Male' or 'Female'.")


#         # Make prediction using the loaded model
#         prediction = predict_risk(input_data)

#         # Return the prediction result
#         if prediction[0] == 0:
#             return render_template('result.html', prediction_text='NO')
#         else:
#             return render_template('result.html', prediction_text='YES')

# if __name__ = '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model (2).pkl', 'rb') as f:
    model = pickle.load(f)

# Define the predict_risk function
def predict_risk(input_data):
    # Convert the input_data dictionary into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # You can preprocess or scale the data here if necessary (commented out for now)
    # scaled_data = scaler.transform(input_df)  # Uncomment if a scaler is required
    
    # Make prediction
    prediction = model.predict(input_df)  # Assuming input_df has the correct feature names
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        form_data = request.form
        Age = int(form_data['age'])
        Sex = form_data['sex'] # Assuming 'Sex' is a categorical variable, keep as string
        Cholesterol = float(form_data['cholesterol'])
        Blood_Pressure_Category = int(form_data['blood_pressure_Category'])  # Assuming Blood Pressure is one value (e.g., systolic)
        Heart_Rate = int(form_data['heart_rate'])
        Diabetes = int(form_data['diabetes'])
        Family_History = int(form_data['family_history'])
        Smoking = int(form_data['smoking'])
        Obesity = int(form_data['obesity'])
        Alcohol_Consumption = int(form_data['Alcohol Consumption'])
        Exercise_Hours_Per_Week = float(form_data['Exercise Hours Per Week'])
        Diet = int(form_data['Diet'])
        Previous_Heart_Problem = int(form_data[' Previous Heart Problem'])
        Medication_Use = int(form_data['Medication Use'])
        Stress_Level = int(form_data['Stress Level'])
        Sedentary_Hours_Per_Day = float(form_data['Sedentary_Hours_Per Day'])
        Income = int(form_data['Income'])
        BMI = float(form_data['bmi'])
        Triglycerides = float(form_data['triglycerides'])
        Physical_Activity_Days_Per_Week = int(form_data['physical_activity'])
        Sleep_Hours_Per_Day = int(form_data['sleep_hours'])
           
    #    Blood_Pressure_Category_High = 0
    #    Blood_Pressure_Category_Low = 0
    #    Blood_Pressure_Category_Normal = 0 
        
        # if Blood_Pressure_Category =='Blood_Pressure_Category_High':
        #    Blood_Pressure_Category_High = 1
        # elif Blood_Pressure_Category =='Blood_Pressure_Category_Low':
        #     Blood_Pressure_Category_Low = 1
        # elif Blood_Pressure_Category == 'Blood_Pressure_Category_Normal':
        #     Blood_Pressure_Category_Normal = 1
             
        Diet_Average = 0
        Diet_Healthy = 0
        Diet_Unhealthy = 0

        if Diet == 'Diet_Average':
            Diet_Average = 1
        elif Diet == 'Diet_Healthy':
            Diet_Healthy = 1
        elif Diet == 'Diet_UnHealthy':
            Diet_UnHealthy = 1
        
                 
        # Handle categorical variable 'Sex'
        if Sex == 'Male':
            Sex = 1
        elif Sex == 'Female':
            Sex = 0
        else:
            return render_template('result.html', prediction_text='Error: Invalid value for Sex. Must be Male or Female.')

        input_data =[[Age, Sex, Cholesterol, Blood_Pressure_Category_High,
       Blood_Pressure_Category_Low,Blood_Pressure_Category_Normal , Heart_Rate, Diabetes,
       Family_History , Smoking , Obesity , Alcohol_Consumption,
       Exercise_Hours_Per_Week , Diet_Average,Diet_Healthy,
       Diet_Unhealthy ,Previous_Heart_Problems,
       Medication_Use, Stress_Level , Sedentary_Hours_Per_Day, Income,
       BMI , Triglycerides , Physical_Activity_Days_Per_Week,
       Sleep_Hours_Per_Day]]
        prediction = model.predict(input_df)  # Assuming input_df has the correct feature names


        # Based on the prediction, return the appropriate result
        # result = 'At Risk' if prediction[0] == 1 else 'Not At Risk'
        
        # return render_template('result.html', prediction_text=f'Heart Attack Risk Prediction: {result}')

        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run()
