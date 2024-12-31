import streamlit as st
import joblib
import pickle
import pandas as pd
from category_encoders import LeaveOneOutEncoder

# Load the trained model and encoder
model = pickle.load(open('D:\Car Price Prediction Project\Car Price Prediction.sav', 'rb'))  # Load the model
loo_encoder = joblib.load('D:\Car Price Prediction Project\leave_one_out_encoder.pkl')  # Load the encoder
data = pd.read_csv(r'D:\Car Price Prediction Project\car_price_prediction.csv')  # Dataset with all features and target column

# Define the correct feature order (the same as during training)
feature_columns = ['Levy', 'Manufacturer', 'Model', 'Category', 'Leather interior',
                   'Fuel type', 'Engine volume', 'Mileage', 'Gear box type', 'Wheel',
                   'Airbags', 'Age']

# Streamlit Page
st.title('Car Price Prediction Web App')

st.sidebar.header('Feature Selection')
st.image('https://imgs.search.brave.com/DygpAHcd-uqewCOWaAEdzHpX0gafxd-uodymAq2sMd8/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvMTQ4/MjE4NzQ3OS9waG90/by9jYXJzLWZvci1z/YWxlLXN0b2NrLWxv/dC1yb3cuanBnP3M9/NjEyeDYxMiZ3PTAm/az0yMCZjPUkyWmNO/XzdKMDFxMjVlQ0tE/X1BFSkJ0bVNSU2dv/dWpCa2l3UzFLblFG/eEU9')
st.info('Easy Application for Predicting Car Prices')

st.sidebar.info('Easy Application for Predicting Car Prices')

# Collect User Inputs (Original Data)
manufacturer = st.selectbox('Manufacturer', data['Manufacturer'].unique())
model_car = st.selectbox('Model', data['Model'].unique())
category = st.selectbox('Category', data['Category'].unique())
leather_interior = st.selectbox('Leather Interior', data['Leather interior'].unique())
fuel_type = st.selectbox('Fuel Type', data['Fuel type'].unique())
gear_box_type = st.selectbox('Gear Box Type', data['Gear box type'].unique())
wheel = st.selectbox('Wheel', data['Wheel'].unique())
engine_volume = st.selectbox('Engine Volume', sorted(data['Engine volume'].unique()))
airbags = st.selectbox('Airbags', sorted(data['Airbags'].unique()))
age = st.number_input('Age', min_value=0, step=1)
mileage = st.number_input('Mileage', min_value=0, step=1)
levy = st.number_input('Levy', min_value=0, step=1)

# Encode the user input data
def encode_input_data(manufacturer, model_car, category, leather_interior, fuel_type,
                       gear_box_type, wheel):
    # Create a DataFrame from the inputs (only categorical columns for encoding)
    user_input_df = pd.DataFrame({
        'Manufacturer': [manufacturer],
        'Model': [model_car],
        'Category': [category],
        'Leather interior': [leather_interior],
        'Fuel type': [fuel_type],
        'Gear box type': [gear_box_type],
        'Wheel': [wheel]
    })
    
    # Apply Leave-One-Out encoding to categorical columns
    user_input_encoded = loo_encoder.transform(user_input_df)
    return user_input_encoded

# Get user inputs and encode them
user_input_data = encode_input_data(manufacturer, model_car, category, leather_interior, fuel_type,
                                    gear_box_type, wheel)

# Add the numerical values to the encoded data (without encoding them)
user_input_data['Engine volume'] = engine_volume
user_input_data['Engine volume']=user_input_data['Engine volume'].str.replace('Turbo', '', regex=False).astype(float)
user_input_data['Airbags'] = airbags
user_input_data['Age'] = age
user_input_data['Mileage'] = mileage
user_input_data['Levy'] = levy

# Reorder the columns to match the model's expected input
user_input_data = user_input_data[feature_columns]

# Display the user input data
st.write('User Input Data:')
st.write(user_input_data)

if st.sidebar.button('Predict Price'):
    # Make the prediction
    prediction = model.predict(user_input_data)
    st.sidebar.write('Predicted Price:', round(prediction[0], 2))