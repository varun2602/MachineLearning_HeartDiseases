import pandas
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn as skl 
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator


heart = pandas.read_csv('heart.csv') 

print(heart.head()) #Gives first 5 records
# print(heart.tail()) Gives last 5 records
# print(heart.shape) Gives no. rows and columns in a tuple 

# print(heart.isnull().sum()) # Gives a sum of empty records

# del heart['ca']
# del heart['thal']
# del heart['oldpeak']
# del heart['slope']
# heart.columns() 
print(heart.info())
print(heart.columns)
# Install pgmpy for Bayesian belief model 
model = BayesianNetwork([
    ('age', 'resting_blood_pressure'),
    ('age', 'fasting_blood_sugar'),  
    ('sex', 'resting_blood_pressure'),
    ('exercise_induced_angina', 'resting_blood_pressure'),
    ('resting_blood_pressure', 'target'),
    ('fasting_blood_sugar', 'target'), 
    ('target', 'rest_ecg'),
    ('target', 'thalassemia'),
    ('target', 'cholestoral')  


model.fit(heart, estimator=MaximumLikelihoodEstimator)
# print(model.get_cpds('age'))
print(model.get_cpds('fasting_blood_sugar'))


