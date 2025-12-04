import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pymc 

data = pd.read_csv('https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv')

retention_1_mean = data['retention_1'].mean()
retention_7_mean = data['retention_7'].mean()

#1 day retention
with pymc.Model() as model:
    alpha = 1/retention_1_mean

    lambda_1 = pymc.Exponential('lambda_1', alpha)
    lambda_2 = pymc.Exponential('lambda_2', alpha)

    lambda_ = pymc.math.switch(data['version']=='gate_30', lambda_1, lambda_2)

    observation = pymc.Exponential("obs", lambda_, observed=data['sum_gamerounds'])

with model:
    step = pymc.Metropolis()
    trace = pymc.sample(10000, tune=5000, step=step, return_inferencedata=False)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']

average_reten_1_samples = np.array([1/i for i in lambda_1_samples])
average_reten_2_samples = np.array([1/i for i in lambda_2_samples])

p1 = px.histogram(average_reten_1_samples)
p2 = px.histogram(average_reten_2_samples)



#7 day retention
with pymc.Model() as model2:
    alpha = 1/retention_7_mean

    lambda_3 = pymc.Exponential('lambda_3', alpha)
    lambda_4 = pymc.Exponential('lambda_4', alpha)

    lambda_ = pymc.math.switch(data['version']=='gate_30', lambda_3, lambda_4)

    observation = pymc.Exponential("obs", lambda_, observed=data['sum_gamerounds'])

with model2:
    step = pymc.Metropolis()
    trace = pymc.sample(10000, tune=5000, step=step, return_inferencedata=False)

lambda_3_samples = trace['lambda_3']
lambda_4_samples = trace['lambda_4']

average_reten_3_samples = np.array([1/i for i in lambda_3_samples])
average_reten_4_samples = np.array([1/i for i in lambda_4_samples])

p3 = px.histogram(average_reten_3_samples)
p4 = px.histogram(average_reten_4_samples)