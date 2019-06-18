```python

import numpy as np
import pandas as pd

class LinearRegression_Self:

    #get data
    def __init__(self,data,x_index,y_index):
        self.data = data #data
        self.x = data.iloc[:,x_index] #independent variables
        self.y = data.iloc[:,y_index] #dependent variable
        
    #get parameters    
    def fit(self,precision,learning_rate,max_iter):
        n_var = self.x.shape[1] #no. of variables
        n_obs = self.x.shape[0] #no. of observations
        theta = np.ones((n_var + 1)) #initializing parameters
        iters = 0 #initialize counter for iterations
        diff = 1 #initial values for change in objective function
        self.x = np.append(np.ones((n_obs,1)), self.x, axis=1) #include constant variable for constant term in equation
        J_new = (0.5/n_obs)*sum((np.array([sum([self.x[j,i]*theta[i] for i in range(n_var+1)]) for j in np.arange(n_obs)])-self.y)**2) # Evaluate Objective function
        while diff > precision and iters <= max_iter:
            J_curr = J_new
            dJ = np.array([(1/n_obs) * sum((np.array([sum([self.x[j,i]*theta[i] for i in range(n_var+1)]) for j in np.arange(n_obs)])-self.y)*self.x[:,i]) for i in np.arange(n_var+1)]) #gradient of objective function
            theta = theta - (learning_rate/n_obs)*dJ #Update parameters
            J_new = (0.5/n_obs)*sum((np.array([sum([self.x[j,i]*theta[i] for i in range(n_var+1)]) for j in np.arange(n_obs)])-self.y)**2) #Re-evaluate Objective function
            diff = abs(J_curr - J_new) #Update change in objective function value
            iters += 1 #update iteration no.
            print("Objective Function for Iteration ",iters," is ",J_new)
        return(theta) #Return Parameters 
        
```
