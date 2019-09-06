```python

import numpy as np
import pandas as pd

class LinearRegression_Self:

    #get data
    def __init__(self,data,x_index,y_index):
        
        #data
        self.data = data 
        
        #independent variables
        self.x = data.iloc[:,x_index] 
        
        #dependent variable
        self.y = data.iloc[:,y_index] 
        
    #get parameters    
    def fit(self,precision,learning_rate,max_iter):
        #no. of variables
        n_var = self.x.shape[1] 
        
        #no. of observations
        n_obs = self.x.shape[0] 
        
        #initializing parameters
        theta = np.ones((n_var + 1)) 
        
        #initialize counter for iterations
        iters = 0 
        
        #initial values for change in objective function
        diff = 1 
        
        #include constant variable for constant term in equation
        self.x = np.append(np.ones((n_obs,1)), self.x, axis=1) 
        
        # Evaluate Objective function
        J_new = (0.5/n_obs)*sum((np.array([sum([self.x[j,i]*theta[i] for i in range(n_var+1)]) for j in np.arange(n_obs)])-self.y)**2) 
        
        while diff > precision and iters <= max_iter:
            J_curr = J_new
            #gradient of objective function
            dJ = np.array([(1/n_obs) * sum((np.array([sum([self.x[j,i]*theta[i] for i in range(n_var+1)]) for j in np.arange(n_obs)])-self.y)*self.x[:,i]) for i in np.arange(n_var+1)]) 
            
            #Update parameters
            theta = theta - (learning_rate/n_obs)*dJ 
            
            #Re-evaluate Objective function
            J_new = (0.5/n_obs)*sum((np.array([sum([self.x[j,i]*theta[i] for i in range(n_var+1)]) for j in np.arange(n_obs)])-self.y)**2) 
            
            #Update change in objective function value
            diff = abs(J_curr - J_new) 
            
            #update iteration no.
            iters += 1 
            
            print("Objective Function for Iteration ",iters," is ",J_new)
        
        #Return Parameters 
        return(theta) 
        
```
