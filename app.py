import streamlit as st
import pandas as pd
st.title("Linear Regression App")
st.write("This app demonstrates a simple linear regression model using Streamlit.")
# Load the dataset
df=pd.read_csv('salary_dataset.csv')
st.write("Dataset:")
st.write(df.head())
# # Display a sample dataset
# st.write("Here is a sample dataset of years of experience and corresponding salaries:")
# st.write(df.head())
# Prepare the data
x=df['YearsExperience']
y=df['Salary']
#upload the model file
from sklearn.model_selection import train_test_split    
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=25)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train.values.reshape(-1,1),y_train)
y_pred=model.predict(X_test.values.reshape(-1,1))
# st.write("Predicted Salaries:")
# st.write(y_pred)
# st.write("Actual Salaries:")
# st.write(y_test.values)
# allow user to input years of experience and predict salary

user_input=st.number_input("Enter years of experience to predict salary:")
if st.button("Predict Salary"):
    predicted_salary=model.predict([[user_input]])
    st.write(f"Predicted Salary for {user_input} years of experience: ₹{predicted_salary[0]:.2f}")
