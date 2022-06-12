# import numpy
import numpy as np

# defining the function to update the steps
def updating_steps(x, y, b_1, b_0, learning_rate):
    b1_deriv = 0
    b0_deriv = 0
    n_number = len(x)

    y_predicted=b_0 + b_1*x 
    b0_deriv = -2*np.sum(y - y_predicted) 
    b1_deriv = -2*np.dot((y - y_predicted),x)

    b_1 -= (b1_deriv/n_number)*learning_rate
    b_0 -= (b0_deriv/n_number)*learning_rate
    
    return(b_0,b_1)

# iteration process of finding the coefficients
def prediction(x, y, b_1, b_0, learning_rate, iters):
    b_0_history = []
    b_1_history = []
    for i in range(iters):
        b_0, b_1= updating_steps(x, y, b_1, b_0, learning_rate)
        b_0_history.append(b_0)
        b_1_history.append(b_1)
        if i % 100 == 0:
            print(i,"b_0=",b_0, "b_1=",b_1)

    return(b_0_history,b_1_history)
  
# download data from Github
import pandas as pd
df_train=pd.read_csv("https://raw.githubusercontent.com/TidyPython/machine_learning/main/Advertising.csv")

# apply the prediction function
b_0_history,b_1_history=prediction(x=df_train['radio'], y=df_train['sales'], b_1=0,b_0=4, learning_rate=0.001,iters=10000)

# import matplotlib
from matplotlib import pyplot as plt

# set the size of the figure
plt.rcParams['figure.figsize'] = [10, 6]

# plot the iteration process for b_1
plt.plot(b_1_history)

# plot the iteration process for b_0
plt.plot(b_0_history)
