#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from scipy.special import logsumexp


# In[2]:


data = np.load('data_exercise5.npz')
'''
data['inputs']: numpy array of shape (20, 15, 10, 2, 4)
It contains the observed input features from 20 participants.
data['inputs'][i] contains information two objects with 4 features from 15 different tasks, 
each with a length of 10 for participant i.

dim 0: participants
dim 1: tasks
dim 2: trials
dim 3: objects
dim 4: features

data['targets']: numpy array of shape (20, 15, 10, 2)
It contains the corresponding targets (criterion values) for each of the trials and objects.

data['choices']: numpy array of shape (20, 15, 10)
It contains the predictions made by participants for each of the trials. 
A value of 1 indicates that the participant has chosen option A in the corresponding trial and task. 
'''


# ### Exercise 5.1
# 
# What is the BIC of a random policy (picking both options with equal probability) for this data-set?

# Answer:

# ### Exercise 5.2
# 
# Compute the BIC values the following models:
# 
# <ul>
#   <li>Rescorla-Wagner model with an $\varepsilon$-greedy choice rule.</li>
#   <li>Kalman filter with an $\varepsilon$-greedy choice rule.</li>
#   <li>Random policy.</li>
# </ul> 
# 
# Report the BIC values summed across participants for each of these models. Furthermore, plot the BIC values for each individual model and participant. Finally, plot how frequently each model offers the best explanation for the participants. Which of the considered models is winning this model comparison?





class RescorlaWagner():
    def __init__(self, num_inputs, learning_rate=0.3):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        
        self.weights = np.zeros((num_inputs, 1))
        
    def predict(self, inputs):
        mean = self.weights.T @ inputs
        return mean 
        
    def learn(self, inputs, targets):
        self.weights = self.weights + self.learning_rate * (targets - self.weights.T @ inputs) * inputs
        
        
class KalmanFilter():
    def __init__(self, num_inputs, sigma_y=0.1, sigma_w=1):
        self.num_inputs = num_inputs
        self.sigma_y = sigma_y
        self.sigma_w = sigma_w
        
        self.weights = np.zeros((num_inputs, 1))
        self.covariance = sigma_w * np.eye(num_inputs)
        
    def predict(self, inputs):
        return self.weights.T @ inputs, np.sqrt(inputs.T @ self.covariance @ inputs + self.sigma_y ** 2)
        
    def learn(self, inputs, targets):
        
        k = self.covariance @ inputs / (inputs.T @ self.covariance @ inputs + self.sigma_y**2)
        
        self.weights = self.weights + k * (targets - self.weights.T @ inputs)
        self.covariance = self.covariance - k @ inputs.T @ self.covariance
        
        
        
import seaborn as sns
import pandas as pd   
import matplotlib.pyplot as plt

models = [RescorlaWagner, KalmanFilter]
BIC_participants = np.empty((data['inputs'].shape[0], len(models)))

epsilon = 0.3
parameters = [2, 3] # RescorlaWagner has learning rate and initial weights and Kalman filter has Noise of target, mean and covariance

for i, each_model in enumerate(models):
    for participant in range(data['inputs'].shape[0]):
        
        log_total = 0 # Initializing log liklihood value for every task
    
        for task in range(data['inputs'].shape[1]):
            
            model = each_model(num_inputs=data['inputs'].shape[4])
            #model=KalmanFilter(num_inputs=data['inputs'].shape[4])
    
            for trial in range(data['inputs'].shape[2]):
                
                # We have two choices to make
                # We can find which one is best based on the mean of the prediction
                input_val_0 =  np.array([[i] for i in data['inputs'][participant, task,trial,0,:]])
                input_val_1 =  np.array([[i] for i in data['inputs'][participant, task,trial,1,:]])
                
                mean_0 = model.predict(input_val_0)
                mean_1 = model.predict(input_val_1)
                
                # Greedy choice algorithm implementation
                # The epsilon-greedy choice rule selects a random action with probability epsilon, and otherwise selects the best action
                # For example, if epsilon is 0.1, then 10% of the time a random action is chosen, and 90% of the time the best action is chosen.
                if (mean_0>mean_1):
                    prob_0 = (1-epsilon)*1 + epsilon*0.5 # Choose A for this trail 
                else:
                    prob_0 = (1-epsilon)*0 + epsilon*0.5 # Just to make calculations clear
                    
                prob_1 = 1-prob_0
                
                # We use this prediction to calculate the log likelihood
                if (data['choices'][participant, task,trial] == 1): # It choose A
                    logval = np.log(prob_0)
                else:
                    logval = np.log(prob_1)
                    
                log_total = log_total + logval 
                
                # The model learns via utilizing the learning rate
                model.learn(input_val_0, data['targets'][participant, task,[trial],0])
                model.learn(input_val_1, data['targets'][participant, task,[trial],1])
        
        BIC_participants[participant, i] =  2*(parameters[i]*np.log(15*10)-log_total)
   
             


random_policy =  np.full(data['inputs'].shape[0], 0 - 2*(15*10*np.log(0.5)))

BIC_frame = pd.DataFrame(BIC_participants,columns=['RW','KF'] )
BIC_frame.insert(2,'RND', random_policy)

print('BIC summed over for all participants is as follows:\n', BIC_frame.sum(axis=0))

fig, axs = plt.subplots(figsize=(10,4))
sns.heatmap(BIC_frame.T, ax =axs,cmap = 'coolwarm')
axs.set_xlabel("Participants")
axs.set_title("BIC")


# The winning model
winning_model = list(BIC_frame.idxmin(axis=1))
column  , counts = np.unique(winning_model,return_counts=True)
win_frame = pd.DataFrame({'Model': column, 'Frequency':counts/sum(counts)})
win_frame = win_frame.append(pd.DataFrame({'Model': ['RND'], 'Frequency':0}))

win_frame = win_frame.sort_values(by='Frequency',ascending=False)

plt.figure()
sns.barplot(win_frame, x='Model',y='Frequency')









# ### Exercise 5.3
# 
# Use the summed BIC values from the last exercise to approximate posterior probabilities over models. You may assume a uniform prior over models. Report the resulting posterior probabilities.

# In[ ]:




