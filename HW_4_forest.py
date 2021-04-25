# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:00:52 2021

@author: vince
"""

import numpy as np
import pandas as pd
import time as Time
import gc
import random
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import validation_curve
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedShuffleSplit
import timeit
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn import manifold
import itertools
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import FactorAnalysis

from hiive.mdptoolbox import mdp
from hiive.mdptoolbox import example
import pickle

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename
 
def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r


def convergence_plot(result,name_1,title_1,name_2,title_2):  
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        time = time + result[i]['Time']
    
    print('Computation time is: ',time)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(iteration, max_V,label='Max value', color="C0", lw=2)
    ax1.set_ylabel('Max Value')
    #ax1.set_title("Value VS Iterations")
    ax1.set_title(title_1)
    ax1.set_xlabel('Iterations')
    ax1.grid()
    
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(iteration, mean_V,label='Mean Value', color="C1", lw=2)
    ax2.set_ylabel('Mean Value')
    fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    #plt.xticks(n_component_range)
    plt.savefig(name_1,dpi=600)
    
    plt.figure()
    plt.plot(iteration,error)
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    #plt.xlabel('Training set size', fontsize = 14)
    #plt.title('Error Convergence VS Iteration', y = 1.03)
    plt.title(title_2, y = 1.03)
    #plt.legend()
    #plt.ylim(0,40)
    plt.grid(True)
    plt.savefig(name_2,dpi=600)


#==============================================================================
#5 states
#==============================================================================
#==============================================================================
#Policy iteration
#==============================================================================    
P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
pi = mdp.PolicyIteration(P, R, 0.9)
pi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = pi.run()
end_time = Time.time()
print('Execution time for PI is: ' + str(end_time-start_time))
#print(pi.V)
#print(pi.policy)

name_1 = 'Forest_5 states_PI_Value VS Iterations for PI.png'
title_1 = 'Value VS Iterations for PI (5 states)'
name_2 = 'Forest_5 states_PI_Error Convergence VS Iteration for PI.png'
title_2 = 'Error Convergence VS Iteration for PI (5 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

                 
policy_pi = pi.policy
plt.figure()
plt.bar(range(len(policy_pi)), policy_pi)
plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with PI (5 states)')
plt.savefig('Forest_5 states_PI_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_pi)), policy_pi,marker='o')
plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with PI (5 states)')
plt.savefig('Forest_5 states_PI_policy',dpi=600)

plt.figure()
plt.scatter(range(len(policy_pi)), policy_pi)
plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with PI (5 states)')
plt.savefig('Forest_5 states_PI_policy_scatter',dpi=600)

#==============================================================================
discounts = [0.1,0.3,0.5,0.7,0.9]
iterations = []
policies = []

plt.figure()
for i in range(len(discounts)):    
    P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
    pi = mdp.PolicyIteration(P, R, discounts[i])
    pi.setVerbose()
    #pi.setSilent()
    result = pi.run()
    #print(pi.policy)
    iterations.append(pi.iter)
    
    policy_pi = pi.policy
    policies.append(policy_pi)        
    
    plt.plot(range(len(policy_pi)), policy_pi, marker='o',color='C'+str(i), lw=2 ,label = 'discount='+str(discounts[i]))

plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.legend()
plt.title('Policy of Forest with PI under Various Discount (5 states)')
plt.savefig('Forest_5 states_PI_Policy of Forest Management with PI under Various Discount',dpi=600)

plt.figure()
plt.plot(discounts,iterations)
plt.ylabel('Iterations')
plt.xlabel('Discount')
plt.title('Iteration VS Discount for PI (5 states)')
plt.grid()
plt.savefig('Forest_5 states_PI_Iteration VS Discount',dpi=600)

#==============================================================================
#Value iteration
#==============================================================================
P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
vi = mdp.ValueIteration(P, R, 0.9)
vi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = vi.run()
end_time = Time.time()
print('Execution time for VI is: ' + str(end_time-start_time))
# print(vi.V)
# print(vi.policy)

name_1 = 'Forest_5 states_VI_Value VS Iterations for VI.png'
title_1 = 'Value VS Iterations for VI (5 states)'
name_2 = 'Forest_5 states_VI_Error Convergence VS Iteration for VI.png'
title_2 = 'Error Convergence VS Iteration for VI (5 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)
                 
policy_vi = vi.policy
plt.figure()
plt.bar(range(len(policy_vi)), policy_vi)
plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with VI (5 states)')
plt.savefig('Forest_5 states_VI_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_vi)), policy_vi, marker='o')
plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with VI (5 states)')
plt.savefig('Forest_5 states_VI_policy',dpi=600)

plt.figure()
plt.scatter(range(len(policy_vi)), policy_vi)
plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with VI (5 states)')
plt.savefig('Forest_5 states_VI_policy_scatter',dpi=600)

#==============================================================================
discounts = [0.1,0.3,0.5,0.7,0.9]
iterations = []
policies = []

plt.figure()
for i in range(len(discounts)):    
    P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
    vi = mdp.ValueIteration(P, R, discounts[i])
    vi.setVerbose()
    #pi.setSilent()
    result = vi.run()
    #print(pi.policy)
    iterations.append(vi.iter)
    
    policy_vi = vi.policy
    policies.append(policy_vi)        
    
    plt.plot(range(len(policy_vi)), policy_vi, marker='o',color='C'+str(i), lw=2 ,label = 'discount='+str(discounts[i]))

plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.legend()
plt.title('Policy of Forest with VI under Various Discount (5 states)')
plt.savefig('Forest_5 states_VI_Policy of Forest Management with PI under Various Discount',dpi=600)

plt.figure()
plt.plot(discounts,iterations)
plt.ylabel('Iterations')
plt.xlabel('Discount')
plt.title('Iteration VS Discount for VI (5 states)')
plt.grid()
plt.savefig('Forest_5 states_VI_Iteration VS Discount',dpi=600)


#==============================================================================
#Q learning
#==============================================================================
P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
np.random.seed(0)
ql = mdp.QLearning(P, R, 0.9,n_iter=10000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=1, epsilon_min=0.01, epsilon_decay=0.99)
ql.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = ql.run()
end_time = Time.time()
print('Execution time for QL is: ' + str(end_time-start_time))
# print(vi.V)
# print(vi.policy)

name_1 = 'Forest_5 states_QL_Value VS Iterations for QL.png'
title_1 = 'Value VS Iterations for QL (5 states)'
name_2 = 'Forest_5 states_QL_Error Convergence VS Iteration for QL.png'
title_2 = 'Error Convergence VS Iteration for QL (5 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)
                 
policy_ql = ql.policy
plt.figure()
plt.bar(range(len(policy_ql)), policy_ql)
plt.xlim([0, len(policy_ql)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with QL (5 states)')
plt.savefig('Forest_5 states_QL_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_ql)), policy_ql, marker='o')
plt.xlim([0, len(policy_ql)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with QL (5 states)')
plt.savefig('Forest_5 states_QL_policy',dpi=600)

plt.figure()
plt.scatter(range(len(policy_ql)), policy_ql)
plt.xlim([0, len(policy_ql)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with QL (5 states)')
plt.savefig('Forest_5 states_QL_policy_scatter',dpi=600)

#==============================================================================
epsilons = [1,0.9,0.8,0.7,0.6]
results = []
policies = []
values = []
for epsilon in epsilons:    
    np.random.seed(0) 
    P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
    ql = mdp.QLearning(P, R, 0.9,n_iter=7000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=epsilon, epsilon_min=0.01, epsilon_decay=0.99)
    ql.setVerbose()
    start_time = Time.time()    
    result = ql.run()
    end_time = Time.time()
    results.append(result)
    policies.append(ql.policy)
    values.append(ql.V)
    print('Computation time is: '+ str(end_time-start_time))


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()  # this is the important function
for j in range(len(results)):
    result = results[j]
    eps = epsilons[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    ax1.plot(iteration, max_V,label='Max value (Epsilon='+str(eps)+')', color="C"+str(2*j), lw=2)            
    ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps)+')', color="C"+str(2*j+1), lw=2)
    

ax1.set_ylabel('Max Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
ax1.set_title('Value VS Iterations for Q Learning (5 states)')
plt.savefig('Forest_5 states_QL_Value VS Iterations for Q Learning (5 states)_epsilon.png',dpi=600)

#==============================================================================
epsilon_decays = [1,0.9999,0.999,0.99,0.9]
results = []
policies = []
values = []
for eps_decay in epsilon_decays:    
    np.random.seed(0) 
    P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
    ql = mdp.QLearning(P, R, 0.9,n_iter=7000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=0.8, epsilon_min=0.01, epsilon_decay=eps_decay)
    ql.setVerbose()
    start_time = Time.time()    
    result = ql.run()
    end_time = Time.time()
    results.append(result)
    policies.append(ql.policy)
    values.append(ql.V)
    print('Computation time is: '+ str(end_time-start_time))
#==============================================================================
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(len(results)):
    result = results[j]
    eps_decay = epsilon_decays[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    ax1.plot(iteration[0:7800], mean_V[0:7800],label='Mean value (Epsilon decay='+str(eps_decay)+')', color="C"+str(2*j), lw=2)            
    #ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps_decay)+')', color="C"+str(2*j+1), lw=2)
    

ax1.set_ylabel('Mean Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.4), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Epsilon Decay for Q Learning (Epsilon = 0.8)')
plt.tight_layout()
plt.savefig('Forest_5 states_QL_Effect of Epsilon Decay for Q Learning (Epsilon = 0.8)_mean value.png',dpi=600)
#=============================================================================
#==============================================================================
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(len(results)):
    result = results[j]
    eps_decay = epsilon_decays[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    ax1.plot(iteration[0:7800], max_V[0:7800],label='Max value (Epsilon decay='+str(eps_decay)+')', color="C"+str(2*j), lw=2)            
    #ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps_decay)+')', color="C"+str(2*j+1), lw=2)
    

ax1.set_ylabel('Max Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.4), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Epsilon Decay for Q Learning (Epsilon = 0.8)')
plt.tight_layout()
plt.savefig('Forest_5 states_QL_Effect of Epsilon Decay for Q Learning (Epsilon = 0.8)_max value.png',dpi=600)

#=======================================================================================
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(len(results)):
    result = results[j]
    eps_decay = epsilon_decays[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    ax1.plot(iteration[0:7800], mean_V[0:7800],label='Max value (Epsilon='+str(eps_decay)+')', color="C"+str(2*j), lw=2)            
    #ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps_decay)+')', color="C"+str(2*j+1), lw=2)
    

ax1.set_ylabel('Max Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
ax1.set_title('Effect of Epsilon Decay for Q Learning (5 states)')
#plt.savefig('Forest_5 states_QL_Value VS Iterations for Q Learning (5 states)_epsilon.png',dpi=600)

#=====================================================================================
#alpha
#=====================================================================================
alphas = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
results = []
policies = []
values = []
qls = []
for i in range(len(alphas)):    
    np.random.seed(0) 
    P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
    ql = mdp.QLearning(P, R, 0.9,n_iter=3000000,alpha=alphas[i], alpha_decay=0.99, alpha_min=0.001,epsilon=1, epsilon_min=0.1, epsilon_decay=0.99)
    ql.setVerbose()
    start_time = Time.time()    
    result = ql.run()
    end_time = Time.time()
    results.append(result)
    policies.append(ql.policy)
    values.append(ql.V)
    qls.append(ql)
    print('Computation time is: '+ str(end_time-start_time))

#save data
filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\5_states_QL_alpha_effect_result.txt'
save_variable(results,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\5_states_QL_alpha_effect_ql.txt'
save_variable(qls,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\5_states_QL_alpha_effect_value.txt'
save_variable(values,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\5_states_QL_alpha_effect_policy.txt'
save_variable(policies,filename)

#test_r = load_variable(filename)

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(0,len(results),2):
    result = results[j]
    alp = alphas[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    #ax1.plot(iteration, max_V,label='Max value (alpha='+str(alp)+')', color="C"+str(2*j), lw=2)            
    #ax2.plot(iteration, mean_V,label='Mean Value (alpha='+str(alp)+')', color="C"+str(2*j+1), lw=2)
    ax1.plot(iteration, mean_V,label='Mean Value (alpha='+str(alp)+')', color="C"+str(int(j/2)), lw=2)
    

ax1.set_ylabel('Mean Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
#ax1.set_yscale('log')
#ax2.set_yscale('log')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.45), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Alpha for Q Learning (5 states)')
plt.xlim([0, 2e6])
plt.tight_layout()
plt.savefig('Forest_5 states_QL_Effect of Alpha for Q Learning (5 states)_mean.png',dpi=600)
#==============================================================================
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(0,len(results),2):
    result = results[j]
    alp = alphas[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    #ax1.plot(iteration, max_V,label='Max value (alpha='+str(alp)+')', color="C"+str(2*j), lw=2)            
    #ax2.plot(iteration, mean_V,label='Mean Value (alpha='+str(alp)+')', color="C"+str(2*j+1), lw=2)
    ax1.plot(iteration, max_V,label='Max Value (alpha='+str(alp)+')', color="C"+str(int(j/2)), lw=2)
    

ax1.set_ylabel('Max Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
#ax1.set_yscale('log')
#ax2.set_yscale('log')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.45), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Alpha for Q Learning (5 states)')
plt.xlim([0, 2e6])
plt.tight_layout()
plt.savefig('Forest_5 states_QL_Effect of Alpha for Q Learning (5 states)_max.png',dpi=600)
#==============================================================================
#alpha decay
#==============================================================================
alpha_decays = [1,0.999,0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
results = []
policies = []
values = []
qls = []
for i in range(len(alpha_decays)):    
    np.random.seed(0) 
    P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
    ql = mdp.QLearning(P, R, 0.9,n_iter=3000000,alpha=0.1, alpha_decay=alpha_decays[i], alpha_min=0.001,epsilon=1, epsilon_min=0.01, epsilon_decay=0.99)
    ql.setVerbose()
    start_time = Time.time()    
    result = ql.run()
    end_time = Time.time()
    results.append(result)
    policies.append(ql.policy)
    values.append(ql.V)
    qls.append(ql)
    print('Computation time is: '+ str(end_time-start_time))
    
#save data
filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\5_states_QL_alpha_decay_effect_result.txt'
save_variable(results,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\5_states_QL_alpha_decay_effect_ql.txt'
save_variable(qls,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\5_states_QL_alpha_decay_effect_value.txt'
save_variable(values,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\5_states_QL_alpha_decay_effect_policy.txt'
save_variable(policies,filename)

#test_r = load_variable(filename)
#==============================================================================
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(1,len(results),2):
    result = results[j]
    alp_decay = alpha_decays[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    ax1.plot(iteration[0:7800], mean_V[0:7800],label='Mean value (alpha decay='+str(alp_decay)+')', color="C"+str(int(j/2)), lw=2)            
    #ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps_decay)+')', color="C"+str(2*j+1), lw=2)
    

ax1.set_ylabel('Mean Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.5), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Alpha Decay for Q Learning')
plt.xlim([0, 2e6])
plt.tight_layout()
plt.savefig('Forest_5 states_QL_Effect of Alpha Decay for Q Learning_mean value.png',dpi=600)
#=============================================================================
#==============================================================================
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(1,len(results),2):
    result = results[j]
    alp_decay = alpha_decays[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    ax1.plot(iteration[0:7800], max_V[0:7800],label='Max value (alpha decay='+str(alp_decay)+')', color="C"+str(int(j/2)), lw=2)            
    #ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps_decay)+')', color="C"+str(2*j+1), lw=2)
    

ax1.set_ylabel('Max Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.5), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Alpha Decay for Q Learning')
plt.xlim([0, 2e6])
plt.tight_layout()
plt.savefig('Forest_5 states_QL_Effect of Alpha Decay for Q Learning_max value.png',dpi=600)

#=======================================================================================
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# #ax2 = ax1.twinx()  # this is the important function
# for j in range(len(results)):
#     result = results[j]
#     alp_decay = alpha_decays[j]        
#     max_V = []
#     mean_V = []
#     error = []
#     reward = []
#     iteration = []
#     alpha = []
#     #epsilon = []
#     time = 0 
#     for i in range(len(result)):
#         max_V.append(result[i]['Max V'])
#         mean_V.append(result[i]['Mean V'])
#         reward.append(result[i]['Reward'])
#         error.append(result[i]['Error'])
#         iteration.append(result[i]['Iteration'])
#         alpha.append(result[i]['Alpha'])
#         #epsilon.append(result[i]['Epsilon'])
#         time = time + result[i]['Time']
        
#     print('Computation time is: ',time)
        
#     ax1.plot(iteration[0:7800], mean_V[0:7800],label='Max value (Epsilon='+str(eps_decay)+')', color="C"+str(2*j), lw=2)            
#     #ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps_decay)+')', color="C"+str(2*j+1), lw=2)
    

# ax1.set_ylabel('Max Value')
# #ax1.set_title("Value VS Iterations")
# #ax1.set_title(title_1)
# ax1.set_xlabel('Iterations')
# ax1.grid()    
# #ax2.set_ylabel('Mean Value')
# fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
# plt.tight_layout()
# ax1.set_title('Effect of Epsilon Decay for Q Learning (5 states)')
# #plt.savefig('Forest_5 states_QL_Value VS Iterations for Q Learning (5 states)_epsilon.png',dpi=600)
#==============================================================================
#50 states
#==============================================================================
#==============================================================================
#Policy iteration
#==============================================================================    
P, R = example.forest(S=50, r1=1, r2=300, p=0.1)
pi = mdp.PolicyIteration(P, R, 0.9)
pi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = pi.run()
end_time = Time.time()
print('Execution time for PI is: ' + str(end_time-start_time))
#print(pi.V)
#print(pi.policy)

name_1 = 'Forest_50 states_PI_Value VS Iterations for PI.png'
title_1 = 'Value VS Iterations for PI (50 states)'
name_2 = 'Forest_50 states_PI_Error Convergence VS Iteration for PI.png'
title_2 = 'Error Convergence VS Iteration for PI (50 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

                 
policy_pi = pi.policy
plt.figure()
plt.bar(range(len(policy_pi)), policy_pi)
plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with PI (50 states)')
plt.savefig('Forest_50 states_PI_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_pi)), policy_pi,marker='o')
plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with PI (50 states)')
plt.savefig('Forest_50 states_PI_policy',dpi=600)

plt.figure()
plt.scatter(range(len(policy_pi)), policy_pi)
plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with PI (50 states)')
plt.savefig('Forest_50 states_PI_policy_scatter',dpi=600)

#==============================================================================
discounts = [0.1,0.3,0.5,0.7,0.9]
iterations = []
policies = []

plt.figure()
for i in range(len(discounts)):    
    P, R = example.forest(S=50, r1=1, r2=300, p=0.1)
    pi = mdp.PolicyIteration(P, R, discounts[i])
    pi.setVerbose()
    #pi.setSilent()
    result = pi.run()
    #print(pi.policy)
    iterations.append(pi.iter)
    
    policy_pi = pi.policy
    policies.append(policy_pi)        
    
    plt.plot(range(len(policy_pi)), policy_pi, marker='o',color='C'+str(i), lw=2 ,label = 'discount='+str(discounts[i]))

plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.legend()
plt.title('Policy of Forest with PI under Various Discount (50 states)')
plt.savefig('Forest_50 states_PI_Policy of Forest Management with PI under Various Discount',dpi=600)

ax = plt.figure().gca()
ax.yaxis.get_major_locator().set_params(integer=True)
plt.plot(discounts,iterations)
plt.ylabel('Iterations')
plt.xlabel('Discount')
plt.title('Iteration VS Discount for PI (50 states)')
plt.grid()
plt.savefig('Forest_50 states_PI_Iteration VS Discount',dpi=600)

#==============================================================================
#Value iteration
#==============================================================================
P, R = example.forest(S=50, r1=1, r2=300, p=0.1)
vi = mdp.ValueIteration(P, R, 0.9)
vi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = vi.run()
end_time = Time.time()
print('Execution time for VI is: ' + str(end_time-start_time))
# print(vi.V)
# print(vi.policy)

name_1 = 'Forest_50 states_VI_Value VS Iterations for VI.png'
title_1 = 'Value VS Iterations for VI (50 states)'
name_2 = 'Forest_50 states_VI_Error Convergence VS Iteration for VI.png'
title_2 = 'Error Convergence VS Iteration for VI (50 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)
                 
policy_vi = vi.policy
plt.figure()
plt.bar(range(len(policy_vi)), policy_vi)
plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with VI (50 states)')
plt.savefig('Forest_50 states_VI_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_vi)), policy_vi, marker='o')
plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with VI (50 states)')
plt.savefig('Forest_50 states_VI_policy',dpi=600)

plt.figure()
plt.scatter(range(len(policy_vi)), policy_vi)
plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with VI (50 states)')
plt.savefig('Forest_50 states_VI_policy_scatter',dpi=600)

#==============================================================================
discounts = [0.1,0.3,0.5,0.7,0.9]
iterations = []
policies = []

plt.figure()
for i in range(len(discounts)):    
    P, R = example.forest(S=50, r1=1, r2=300, p=0.1)
    vi = mdp.ValueIteration(P, R, discounts[i])
    vi.setVerbose()
    #pi.setSilent()
    result = vi.run()
    #print(pi.policy)
    iterations.append(vi.iter)
    
    policy_vi = vi.policy
    policies.append(policy_vi)        
    
    plt.plot(range(len(policy_vi)), policy_vi, marker='o',color='C'+str(i), lw=2 ,label = 'discount='+str(discounts[i]))

plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.legend()
plt.title('Policy of Forest with VI under Various Discount (50 states)')
plt.savefig('Forest_50 states_VI_Policy of Forest Management with PI under Various Discount',dpi=600)

ax = plt.figure().gca()
#ax.yaxis.get_major_locator().set_params(integer=True)
plt.plot(discounts,iterations)
plt.ylabel('Iterations')
plt.xlabel('Discount')
plt.title('Iteration VS Discount for VI (50 states)')
plt.grid()
plt.savefig('Forest_50 states_VI_Iteration VS Discount',dpi=600)

#==============================================================================
#Q learning
#==============================================================================
np.random.seed(0) 
P, R = example.forest(S=50, r1=1, r2=300, p=0.1)
ql = mdp.QLearning(P, R, 0.9,n_iter=500000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=1, epsilon_min=0.01, epsilon_decay=1)
ql.setVerbose()
start_time = Time.time()    
result = ql.run()
end_time = Time.time()
print('Computation time is: '+ str(end_time-start_time))
# plt.figure()
# plt.plot(ql.policy)
    
for i in range(len(result)):
    max_V.append(result[i]['Max V'])
    mean_V.append(result[i]['Mean V'])
    reward.append(result[i]['Reward'])
    error.append(result[i]['Error'])
    iteration.append(result[i]['Iteration'])
    alpha.append(result[i]['Alpha'])
    #epsilon.append(result[i]['Epsilon'])
    time = time + result[i]['Time'] 

# plt.figure()
# plt.plot(max_V)
# plt.figure()
# plt.plot(mean_V)
policy_ql = ql.policy
plt.figure()
plt.bar(range(len(policy_ql)), policy_ql)
plt.xlim([0, len(policy_ql)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with QL (50 states)')
plt.savefig('Forest_50 states_QL_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_ql)), policy_ql, marker='o')
plt.xlim([0, len(policy_ql)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with QL (50 states)')
plt.savefig('Forest_50 states_QL_policy',dpi=600)

plt.figure()
plt.scatter(range(len(policy_ql)), policy_ql)
plt.xlim([0, len(policy_ql)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with QL (50 states)')
plt.savefig('Forest_50 states_QL_policy_scatter',dpi=600)

name_1 = 'Forest_50 states_QL_Value VS Iterations for QL.png'
title_1 = 'Value VS Iterations for QL (50 states)'
name_2 = 'Forest_50 states_QL_Error Convergence VS Iteration for QL.png'
title_2 = 'Error Convergence VS Iteration for QL (50 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)


filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\50_states_QL_result.txt'
save_variable(result,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\50_states_QL.txt'
save_variable(ql,filename)

test_r = load_variable(filename)
#==============================================================================


#==============================================================================
#1000 states
#==============================================================================
#==============================================================================
#Policy iteration
#==============================================================================    
P, R = example.forest(S=1000, r1=1, r2=5000000000, p=0.01)
pi = mdp.PolicyIteration(P, R, 0.9)
pi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = pi.run()
end_time = Time.time()
print('Execution time for PI is: ' + str(end_time-start_time))
#print(pi.V)
#print(pi.policy)

name_1 = 'Forest_1000 states_PI_Value VS Iterations for PI.png'
title_1 = 'Value VS Iterations for PI (1000 states)'
name_2 = 'Forest_1000 states_PI_Error Convergence VS Iteration for PI.png'
title_2 = 'Error Convergence VS Iteration for PI (1000 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

                 
policy_pi = pi.policy
plt.figure()
plt.bar(range(len(policy_pi)), policy_pi)
plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with PI (1000 states)')
plt.savefig('Forest_1000 states_PI_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_pi)), policy_pi,marker='o')
plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with PI (1000 states)')
plt.savefig('Forest_1000 states_PI_policy',dpi=600)

plt.figure()
plt.scatter(range(len(policy_pi)), policy_pi)
plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with PI (1000 states)')
plt.savefig('Forest_1000 states_PI_policy_scatter',dpi=600)

#==============================================================================
discounts = [0.1,0.3,0.5,0.7,0.9]
iterations = []
policies = []

plt.figure()
for i in range(len(discounts)):    
    P, R = example.forest(S=1000, r1=1, r2=5000000000, p=0.01)
    pi = mdp.PolicyIteration(P, R, discounts[i])
    pi.setVerbose()
    #pi.setSilent()
    result = pi.run()
    #print(pi.policy)
    iterations.append(pi.iter)
    
    policy_pi = pi.policy
    policies.append(policy_pi)        
    
    plt.plot(range(len(policy_pi)), policy_pi, marker='o',color='C'+str(i), lw=2 ,label = 'discount='+str(discounts[i]))

plt.xlim([0, len(policy_pi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.legend()
plt.title('Policy of Forest with PI under Various Discount (1000 states)')
plt.savefig('Forest_1000 states_PI_Policy of Forest Management with PI under Various Discount',dpi=600)

ax = plt.figure().gca()
ax.yaxis.get_major_locator().set_params(integer=True)
plt.plot(discounts,iterations)
plt.ylabel('Iterations')
plt.xlabel('Discount')
plt.title('Iteration VS Discount for PI (1000 states)')
plt.grid()
plt.savefig('Forest_1000 states_PI_Iteration VS Discount',dpi=600)

#==============================================================================
#Value iteration
#==============================================================================
P, R = example.forest(S=1000, r1=1, r2=5000000000, p=0.01)
vi = mdp.ValueIteration(P, R, 0.9)
vi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = vi.run()
end_time = Time.time()
print('Execution time for VI is: ' + str(end_time-start_time))
# print(vi.V)
# print(vi.policy)

name_1 = 'Forest_1000 states_VI_Value VS Iterations for VI.png'
title_1 = 'Value VS Iterations for VI (1000 states)'
name_2 = 'Forest_1000 states_VI_Error Convergence VS Iteration for VI.png'
title_2 = 'Error Convergence VS Iteration for VI (1000 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)
                 
policy_vi = vi.policy
plt.figure()
plt.bar(range(len(policy_vi)), policy_vi)
plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with VI (1000 states)')
plt.savefig('Forest_1000 states_VI_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_vi)), policy_vi, marker='o')
plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with VI (1000 states)')
plt.savefig('Forest_1000 states_VI_policy',dpi=600)

plt.figure()
plt.scatter(range(len(policy_vi)), policy_vi)
plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with VI (1000 states)')
plt.savefig('Forest_1000 states_VI_policy_scatter',dpi=600)

#==============================================================================
discounts = [0.1,0.3,0.5,0.7,0.9]
iterations = []
policies = []

plt.figure()
for i in range(len(discounts)):    
    P, R = example.forest(S=1000, r1=1, r2=5000000000, p=0.01)
    vi = mdp.ValueIteration(P, R, discounts[i])
    vi.setVerbose()
    #pi.setSilent()
    result = vi.run()
    #print(pi.policy)
    iterations.append(vi.iter)
    
    policy_vi = vi.policy
    policies.append(policy_vi)        
    
    plt.plot(range(len(policy_vi)), policy_vi, marker='o',color='C'+str(i), lw=2 ,label = 'discount='+str(discounts[i]))

plt.xlim([0, len(policy_vi)])
plt.ylabel('Action')
plt.xlabel('State')
plt.legend()
plt.title('Policy of Forest with VI under Various Discount (1000 states)')
plt.savefig('Forest_1000 states_VI_Policy of Forest Management with PI under Various Discount',dpi=600)

ax = plt.figure().gca()
#ax.yaxis.get_major_locator().set_params(integer=True)
plt.plot(discounts,iterations)
plt.ylabel('Iterations')
plt.xlabel('Discount')
plt.title('Iteration VS Discount for VI (1000 states)')
plt.grid()
plt.savefig('Forest_1000 states_VI_Iteration VS Discount',dpi=600)

#==============================================================================
#Q learning
#==============================================================================
np.random.seed(0) 
P, R = example.forest(S=1000, r1=1, r2=5000000000, p=0.01)
ql = mdp.QLearning(P, R, 0.9,n_iter=100000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=1, epsilon_min=0.01, epsilon_decay=1)
ql.setVerbose()
start_time = Time.time()    
result = ql.run()
end_time = Time.time()
print('Computation time is: '+ str(end_time-start_time))
# plt.figure()
# plt.plot(ql.policy)
    
for i in range(len(result)):
    max_V.append(result[i]['Max V'])
    mean_V.append(result[i]['Mean V'])
    reward.append(result[i]['Reward'])
    error.append(result[i]['Error'])
    iteration.append(result[i]['Iteration'])
    alpha.append(result[i]['Alpha'])
    #epsilon.append(result[i]['Epsilon'])
    time = time + result[i]['Time'] 

# plt.figure()
# plt.plot(max_V)
# plt.figure()
# plt.plot(mean_V)
policy_ql = ql.policy
plt.figure()
plt.bar(range(len(policy_ql)), policy_ql)
plt.xlim([0, len(policy_ql)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with QL (1000 states)')
plt.savefig('Forest_1000 states_QL_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_ql)), policy_ql, marker='o')
plt.xlim([0, len(policy_ql)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with QL (1000 states)')
plt.savefig('Forest_1000 states_QL_policy',dpi=600)

plt.figure()
plt.scatter(range(len(policy_ql)), policy_ql)
plt.xlim([0, len(policy_ql)])
plt.ylabel('Action')
plt.xlabel('State')
plt.title('Policy of Forest Management with QL (1000 states)')
plt.savefig('Forest_1000 states_QL_policy_scatter',dpi=600)

name_1 = 'Forest_1000 states_QL_Value VS Iterations for QL.png'
title_1 = 'Value VS Iterations for QL (1000 states)'
name_2 = 'Forest_1000 states_QL_Error Convergence VS Iteration for QL.png'
title_2 = 'Error Convergence VS Iteration for QL (1000 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)


filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\1000_states_QL_result.txt'
save_variable(result,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\1000_states_QL.txt'
save_variable(ql,filename)

test_r = load_variable(filename)
#==============================================================================

#==============================================================================
#Policy iteration
#==============================================================================
n_state = np.linspace(50,1000,39).astype(int)
times_pi = []
iterations_pi = []
for i in range(len(n_state)):
    P, R = example.forest(S=n_state[i], r1=1, r2=5000000000, p=0.01)
    pi = mdp.PolicyIteration(P, R, 0.9)
    #pi.setVerbose()
    #pi.setSilent()
    start_time = Time.time()
    result = pi.run()
    end_time = Time.time()
    times_pi.append(end_time-start_time)
    iterations_pi.append(pi.iter)
    print('Execution time for PI is: ' + str(end_time-start_time) + 'for iteration' + str(i))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(n_state, iterations_pi,label='Number of iterations', color="C0", lw=2)
ax1.set_ylabel('Number of iterations')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Number of states')
ax1.grid()
    
ax2 = ax1.twinx()  # this is the important function
ax2.plot(n_state, times_pi,label='Computation time', color="C1", lw=2)
ax2.set_ylabel('Computation time')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)

plt.title('Iteration and Time VS Number of States for PI')
#plt.xticks(n_component_range)
plt.tight_layout()
plt.savefig('Forest_Iteration and Time VS Number of States for PI.png',dpi=600)

#==============================================================================
#Value iteration
#==============================================================================
n_state = np.linspace(50,1000,39).astype(int)
times_vi = []
iterations_vi = []
for i in range(len(n_state)):
    P, R = example.forest(S=n_state[i], r1=1, r2=5000000000, p=0.01)
    vi = mdp.ValueIteration(P, R, 0.9)
    #pi.setVerbose()
    #pi.setSilent()
    start_time = Time.time()
    result = vi.run()
    end_time = Time.time()
    times_vi.append(end_time-start_time)
    iterations_vi.append(vi.iter)
    print('Execution time for VI is: ' + str(end_time-start_time) + 'for iteration' + str(i))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(n_state, iterations_vi,label='Number of iterations', color="C0", lw=2)
ax1.set_ylabel('Number of iterations')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Number of states')
ax1.grid()
    
ax2 = ax1.twinx()  # this is the important function
ax2.plot(n_state, times_vi,label='Computation time', color="C1", lw=2)
ax2.set_ylabel('Computation time')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)

plt.title('Iteration and Time VS Number of States for VI')
#plt.xticks(n_component_range)
plt.tight_layout()
plt.savefig('Forest_Iteration and Time VS Number of States for VI.png',dpi=600)

























epsilons = [1,0.9,0.8,0.7,0.6]
results = []
policies = []
values = []
for epsilon in epsilons:    
    np.random.seed(0) 
    P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
    ql = mdp.QLearning(P, R, 0.9,n_iter=7000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=epsilon, epsilon_min=0.01, epsilon_decay=0.99)
    ql.setVerbose()
    start_time = Time.time()    
    result = ql.run()
    end_time = Time.time()
    results.append(result)
    policies.append(ql.policy)
    values.append(ql.V)
    print('Computation time is: '+ str(end_time-start_time))


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()  # this is the important function
for j in range(len(results)):
    result = results[j]
    eps = epsilons[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    ax1.plot(iteration, max_V,label='Max value (Epsilon='+str(eps)+')', color="C"+str(2*j), lw=2)            
    ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps)+')', color="C"+str(2*j+1), lw=2)
    

ax1.set_ylabel('Max Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
ax1.set_title('Value VS Iterations for Q Learning (50 states)')
plt.savefig('Forest_50 states_QL_Value VS Iterations for Q Learning (5 states)_epsilon.png',dpi=600)




#==============================================================================
#Policy iteration
#==============================================================================    
P, R = example.forest(S=100, r1=1, r2=10, p=0.1)
pi = mdp.PolicyIteration(P, R, 0.9)
pi.setVerbose()
#pi.setSilent()
result = pi.run()
#print(pi.V)
#print(pi.policy)

name_1 = 'Forest_20 states_PI_Value VS Iterations for PI.png'
title_1 = 'Value VS Iterations for PI (20 states)'
name_2 = 'Forest_20 states_PI_Error Convergence VS Iteration for PI.png'
title_2 = 'Error Convergence VS Iteration for PI (20 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

                 
policy_pi = pi.policy
plt.figure()
plt.bar(range(len(policy_pi)), policy_pi)
plt.xlim([0, len(policy_pi)])
plt.ylabel('State')
plt.xlabel('Action')
plt.title('Policy of Forest Management with PI (20 states)')
plt.savefig('Forest_20 states_PI_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_pi)), policy_pi)
plt.xlim([0, len(policy_pi)])
plt.ylabel('State')
plt.xlabel('Action')
plt.title('Policy of Forest Management with PI (20 states)')
plt.savefig('Forest_20 states_PI_policy',dpi=600)

#==============================================================================
#Value iteration
#==============================================================================
P, R = example.forest(S=100, r1=1, r2=10, p=0.1)
vi = mdp.ValueIteration(P, R, 0.9)
vi.setVerbose()
#pi.setSilent()
result = vi.run()
# print(vi.V)
# print(vi.policy)

name_1 = 'Forest_20 states_VI_Value VS Iterations for VI.png'
title_1 = 'Value VS Iterations for VI (20 states)'
name_2 = 'Forest_20 states_VI_Error Convergence VS Iteration for VI.png'
title_2 = 'Error Convergence VS Iteration for VI (20 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)
                 
policy_vi = vi.policy
plt.figure()
plt.bar(range(len(policy_vi)), policy_vi)
plt.xlim([0, len(policy_vi)])
plt.ylabel('State')
plt.xlabel('Action')
plt.title('Policy of Forest Management with VI (20 states)')
plt.savefig('Forest_20 states_VI_policy_bar_plot',dpi=600)

plt.figure()
plt.plot(range(len(policy_vi)), policy_vi)
plt.xlim([0, len(policy_vi)])
plt.ylabel('State')
plt.xlabel('Action')
plt.title('Policy of Forest Management with VI (20 states)')
plt.savefig('Forest_20 states_VI_policy',dpi=600)

#==============================================================================
discounts = [0.1,0.3,0.5,0.7,0.9]
iterations = []
policies = []

plt.figure()
for i in range(len(discounts)):    
    P, R = example.forest(S=20, r1=1, r2=100, p=0.1)
    pi = mdp.PolicyIteration(P, R, discounts[i])
    pi.setVerbose()
    #pi.setSilent()
    result = pi.run()
    #print(pi.policy)
    iterations.append(pi.iter)
    
    policy_pi = pi.policy
    policies.append(policy_pi)        
    
    plt.plot(range(len(policy_pi)), policy_pi,color='C'+str(i), lw=2 ,label = 'discount='+str(discounts[i]))

plt.xlim([0, len(policy_pi)])
plt.ylabel('State')
plt.xlabel('Action')
plt.legend()
plt.title('Policy of Forest Management with PI under Various Discount')

plt.figure()
plt.plot(discounts,iterations)
plt.ylabel('Iterations')
plt.xlabel('Discount')
plt.title('Iteration VS Discount')


#==============================================================================
#Q learning
#==============================================================================
epsilons = [1,0.9,0.8,0.7,0.6]
results = []
policies = []
values = []
for epsilon in epsilons:    
    np.random.seed(0) 
    P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
    ql = mdp.QLearning(P, R, 0.9,n_iter=7000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=epsilon, epsilon_min=0.01, epsilon_decay=0.99)
    ql.setVerbose()
    start_time = Time.time()    
    result = ql.run()
    end_time = Time.time()
    results.append(result)
    policies.append(ql.policy)
    values.append(ql.V)
    print('Computation time is: '+ str(end_time-start_time))


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()  # this is the important function
for j in range(len(results)):
    result = results[j]
    eps = epsilons[j]        
    max_V = []
    mean_V = []
    error = []
    reward = []
    iteration = []
    alpha = []
    #epsilon = []
    time = 0 
    for i in range(len(result)):
        max_V.append(result[i]['Max V'])
        mean_V.append(result[i]['Mean V'])
        reward.append(result[i]['Reward'])
        error.append(result[i]['Error'])
        iteration.append(result[i]['Iteration'])
        alpha.append(result[i]['Alpha'])
        #epsilon.append(result[i]['Epsilon'])
        time = time + result[i]['Time']
        
    print('Computation time is: ',time)
        
    ax1.plot(iteration, max_V,label='Max value (Epsilon='+str(eps)+')', color="C"+str(2*j), lw=2)            
    ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps)+')', color="C"+str(2*j+1), lw=2)
    

ax1.set_ylabel('Max Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
ax1.set_title('Value VS Iterations for Q Learning (5 states)')
plt.savefig('Forest_5 states_QL_Value VS Iterations for Q Learning (5 states)_epsilon.png',dpi=600)


#====================================================================================
# P, R = example.forest(S=10, r1=1, r2=100, p=0.1)
# vi = mdp.ValueIteration(P, R, 0.9)
# vi.setVerbose()
# #pi.setSilent()
# result = vi.run()

# P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
# vi = mdp.ValueIteration(P, R, 0.9)
# vi.setVerbose()
# #pi.setSilent()
# result = vi.run()
# vi.policy

# np.random.seed(0)
# P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
# pi = mdp.PolicyIteration(P, R, 0.9)
# pi.setVerbose()
# pi.run()
# pi.policy

# np.random.seed(0) 
# P, R = example.forest(S=5, r1=0.8, r2=0.8, p=0.1)
# ql = mdp.QLearning(P, R, 0.9,n_iter=2000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99)
# ql.setVerbose()
# result = ql.run()
# print(ql.policy)
# policy = ql.policy
# plt.figure()
# plt.plot(policy)



# np.random.seed(0) 
# P, R = example.forest(S=20, r1=1, r2=100, p=0.1)
# ql = mdp.QLearning(P, R, 0.9,n_iter=2000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99)
# ql.setVerbose()
# result = ql.run()
# print(ql.policy)
# policy = ql.policy
# plt.figure()
# plt.plot(policy)

# np.random.seed(0)
# P, R = example.forest()
# ql = mdp.QLearning(P, R, 0.96,n_iter=2000000)
# ql.setVerbose()
# result = ql.run()
# ql.policy

# np.random.seed(0)
# P, R = example.forest()
# vi = mdp.ValueIteration(P, R, 0.96)
# vi.setVerbose()
# vi.run()

# np.random.seed(0)
# P, R = example.forest()
# pi = mdp.PolicyIteration(P, R, 0.96)
# pi.setVerbose()
# pi.run()
# pi.policy

# max_V = []
# mean_V = []
# error = []
# reward = []
# iteration = []
# alpha = []
# epsilon = []
# time = 0 
# for i in range(len(result)):
#     max_V.append(result[i]['Max V'])
#     mean_V.append(result[i]['Mean V'])
#     reward.append(result[i]['Reward'])
#     error.append(result[i]['Error'])
#     iteration.append(result[i]['Iteration'])
#     alpha.append(result[i]['Alpha'])
#     epsilon.append(result[i]['Epsilon'])
#     time = time + result[i]['Time']
    
# print('Computation time is: ',time)
    
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(iteration, max_V,label='Max value', color="C0", lw=2)
# ax1.set_ylabel('Max Value')
# #ax1.set_title("Value VS Iterations")
# #ax1.set_title(title_1)
# ax1.set_xlabel('Iterations')
# ax1.grid()
    
# ax2 = ax1.twinx()  # this is the important function
# ax2.plot(iteration, mean_V,label='Mean Value', color="C1", lw=2)
# ax2.set_ylabel('Mean Value')
# fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
# plt.tight_layout()
# #plt.xticks(n_component_range)
# #plt.savefig(name_1,dpi=600)
    
# plt.figure()
# plt.plot(iteration,error)
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# #plt.xlabel('Training set size', fontsize = 14)
# #plt.title('Error Convergence VS Iteration', y = 1.03)
# #plt.title(title_2, y = 1.03)
# #plt.legend()
# #plt.ylim(0,40)
# plt.grid(True)
# #plt.savefig(name_2,dpi=600)

# plt.figure()
# plt.plot(iteration,alpha)
# plt.yscale('log')

# plt.figure()
# plt.plot(iteration,epsilon)


P, R = example.forest()
pi = mdp.PolicyIteration(P, R, 0.9)
pi.run()
expected = (26.244000000000014, 29.484000000000016, 33.484000000000016)
all(expected[k] - pi.V[k] < 1e-12 for k in range(len(expected)))

P, R = example.forest()
vi = mdp.ValueIteration(P, R, 0.9)
#vi.verbose
vi.run()
#expected = (26.244000000000014, 29.484000000000016, 33.484000000000016)
#expected = (5.93215488, 9.38815488, 13.38815488)
all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))

vi.policy

vi.iter


P, R = example.forest()
ql = mdp.QLearning(P, R, 0.9,n_iter=100000)
ql.run()
ql.Q
ql.V
ql.policy

dim = 5;
goal = [1,1]

UP = []
RIGHT = []
DOWN = []
LEFT = []

R = []
for i in range(dim*dim):
    row = i//dim
    col = i%dim
    s_up = np.zeros(dim*dim)
    if row == dim-1:
        s_up[i] = 1
    else:
        s_up[(row+1)*dim+col] = 1
    UP.append(s_up)
    
    s_down = np.zeros(dim*dim)
    if row == 0:
        s_down[i] = 1
    else:
        s_down[(row-1)*dim+col] = 1
    DOWN.append(s_down)
    
    s_right = np.zeros(dim*dim)
    if col == dim-1:
        s_right[i] = 1
    else:
        s_right[row*dim+col+1] = 1
    RIGHT.append(s_right)
    
    s_left = np.zeros(dim*dim)
    if col == 0:
        s_left[i] = 1
    else:
        s_left[row*dim+col-1] = 1
    LEFT.append(s_left)
    
    if row==goal[0] and col==goal[1]:
        R.append(np.ones(4))
    else:
        R.append(np.zeros(4))

UP = np.array(UP)     
DOWN = np.array(DOWN)
LEFT = np.array(LEFT)
RIGHT = np.array(RIGHT)   
P = []    
P.append(UP)
P.append(RIGHT)
P.append(DOWN)
P.append(LEFT)

R = np.array(R)
P = np.array(P)

pi = mdp.PolicyIteration(P, R, 0.9)
pi.run()
print(pi.policy)

policy = pi.policy

vi = mdp.ValueIteration(P, R, 0.9)
vi.run()
print(vi.policy)

policy = vi.policy

ql = mdp.QLearning(P, R, 0.9,n_iter=500000)
ql.run()
print(ql.policy)
policy = ql.policy

dim = [5,5]

x = np.linspace(0, dim[0] - 1, dim[0]) + 0.5
y = np.linspace(dim[1] - 1, 0, dim[1]) + 0.5
X, Y = np.meshgrid(x, y)
zeros = np.zeros(dim)
    


arrow_up = np.zeros(dim)
arrow_right = np.zeros(dim)
arrow_down = np.zeros(dim)
arrow_left = np.zeros(dim)

for i in range(len(policy)):
    row_plot = i//dim[0]
    row = dim[0] -1 - i//dim[0]
    col = i%dim[0]
    if row_plot != goal[0] or col != goal[1]:        
        if policy[i] == 0:
            arrow_up[row][col] = 0.4
        elif policy[i] == 1:
            arrow_right[row][col] = 0.4
        elif policy[i] == 2:
            arrow_down[row][col] = 0.4
        else:
            arrow_left[row][col] = 0.4
            
fig = plt.figure(figsize=(12,12))
ax = plt.axes()
# Vectors point in positive Y-direction
plt.quiver(X, Y, zeros, arrow_up, scale=1, units='xy')
# Vectors point in negative X-direction
plt.quiver(X, Y, -arrow_left, zeros, scale=1, units='xy')
# Vectors point in negative Y-direction
plt.quiver(X, Y, zeros, -arrow_down, scale=1, units='xy')
# Vectors point in positive X-direction
plt.quiver(X, Y, arrow_right, zeros, scale=1, units='xy')


#plt.scatter(goal[0]+0.5,goal[1]+0.5,marker='o',c='b',edgecolors='b',s=2000)
plt.scatter(goal[0]+0.5,goal[1]+0.5,marker='s',c='b',edgecolors='b',s=2000)

plt.xlim([0, dim[0]])
plt.ylim([0, dim[1]])
#ax.set_yticklabels([])
#ax.set_xticklabels([])
ax.set_xticks(np.arange(0, dim[0]+1))
ax.set_yticks(np.arange(0, dim[1]+1)) 
plt.grid()
plt.show()

wall = [[1,2],[3,4],[5,5],[7,6]]
row = 6
col = 6
[row,col] not in wall
        
    
    # for i, action in enumerate(grid.action_space):
    #     q_star = np.zeros((5, 5))
    #     for j in range(grid.dim[0]):
    #         for k in reversed(range(grid.dim[1])):
    #             if q[j, k, i] == q_max[j, k]:
    #                 q_star[j, k] = 0.4
    #     # Plot results
    #     if action == "U":
    #         # Vectors point in positive Y-direction
    #         plt.quiver(X, Y, zeros, q_star, scale=1, units='xy')
    #     elif action == "L":
    #         # Vectors point in negative X-direction
    #         plt.quiver(X, Y, -q_star, zeros, scale=1, units='xy')
    #     elif action == "D":
    #         # Vectors point in negative Y-direction
    #         plt.quiver(X, Y, zeros, -q_star, scale=1, units='xy')
    #     elif action == "R":
    #         # Vectors point in positive X-direction
    #         plt.quiver(X, Y, q_star, zeros, scale=1, units='xy')
        
    # plt.xlim([0, grid.dim[0]])
    # plt.ylim([0, grid.dim[1]])
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # plt.grid()
    # plt.show()
# class Data():
    
#     # points [1]
#     def dataAllocation(self,path):
#         # Separate out the x_data and y_data and return each
#         # args: string path for .csv file
#         # return: pandas dataframe, pandas dataframe
#         data = pd.read_csv(path)
#         xList = [i for i in range(data.shape[1] - 1)]
#         x_data = data.iloc[:,xList]
#         y_data = data.iloc[:,[-1]]
#         # ------------------------------- 
#         return x_data,y_data
    
#     # points [1]
#     def trainSets(self,x_data,y_data):
#         # Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
#         # Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 614.
#         # args: pandas dataframe, pandas dataframe
#         # return: pandas dataframe, pandas dataframe, pandas series, pandas series

#         x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=614, shuffle=True)       
#         # -------------------------------
#         return x_train, x_test, y_train, y_test


# #==============================================================================
# #Load data
# #==============================================================================
# datatest = Data()
# #path = 'Class_BanknoteAuth.csv'
# #path = 'pima-indians-diabetes.csv'
# path = 'AFP300_nonAFP300_train_AACandDipeptide_twoSeg.csv'

# x_data,y_data = datatest.dataAllocation(path)
# print("dataAllocation Function Executed")

# #Feature selection
# #x_data = x_data.iloc[:,0:20]

# x_train, x_test, y_train, y_test = datatest.trainSets(x_data,y_data)
# print("trainSets Function Executed")


# n = 0
# for i in range(y_train.size):
#     n = n + y_train.iloc[i,0]
# print ('Positive rate for train data is: ',n/y_train.size)

# n = 0
# for i in range(y_test.size):
#     n = n + y_test.iloc[i,0]
# print ('Positive rate for test data is: ',n/y_test.size)

# #Pre-process the data to standardize it
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# #==============================================================================
# #RP
# #==============================================================================
# x_train_origin = x_train
# x_test_origin = x_test

# rp = GaussianRandomProjection(n_components=415,random_state=2)
# x_train_transform_RP = rp.fit_transform(x_train)
# inverse_data = np.linalg.pinv(rp.components_.T)
# reconstructed_data = x_train_transform_RP.dot(inverse_data)
# MSE_RP = np.mean((x_train - reconstructed_data)**2)

# print('MSE for RP is: ',MSE_RP)

# x_train = rp.transform(x_train_origin)
# x_test = rp.transform(x_test_origin)

# #Pre-process the data to standardize it
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# #==============================================================================
# #Default setting
# #==============================================================================
# plt.style.use('default')
# print('\n', '-' * 50)
# print('Default setting')

# MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 500,alpha = 1e-4,hidden_layer_sizes = (100),random_state = 0)
# MLP_clf.fit(x_train, y_train.values.ravel())
# y_predict_train = MLP_clf.predict(x_train)
# y_predict_test = MLP_clf.predict(x_test)
        
# train_accuracy = accuracy_score(y_train.values,y_predict_train)
# test_accuracy = accuracy_score(y_test.values,y_predict_test)

# print('Training accuracy is: ',train_accuracy)
# print('Test accuracy is: ',test_accuracy)
# print('\n', '-' * 50)

# #==============================================================================
# #learning curve
# #https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
# #==============================================================================
# train_sizes = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# for i in range(1, len(train_sizes)):
#     train_sizes[i] = math.floor(train_sizes[i] *y_train.size*4/5)
    
# #print (train_sizes)

# train_sizes, train_scores, validation_scores = learning_curve(
# estimator = MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 1000,alpha = 1e-4,hidden_layer_sizes = (100),random_state = 0),
# X = x_train,
# y = y_train.values.ravel(), train_sizes = train_sizes, cv = 5,
# scoring = 'accuracy',
# shuffle = True,
# random_state=0)


# train_scores_mean = train_scores.mean(axis = 1)
# validation_scores_mean = validation_scores.mean(axis = 1)
# print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
# print('\n', '-' * 20) # separator
# print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

# fig = plt.figure(1)
# #plt.style.use('seaborn')
# plt.plot(train_sizes, train_scores_mean, label = 'Training error', color="darkorange", lw=2)
# plt.plot(train_sizes, validation_scores_mean, label = 'Validation error', color="navy", lw=2)
# plt.ylabel('Accuracy')
# plt.xlabel('Training set size')
# #plt.xlabel('Training set size', fontsize = 14)
# plt.title('Learning curves for ANN (Default setting)', y = 1.03)
# plt.legend()
# #plt.ylim(0,40)
# plt.grid(True)
# plt.savefig('Sample_A_part_4_RP_full_ANN_sample_A_Learning_curves_for_ANN_Default_setting.png',dpi=600)


# #==============================================================================
# #Gid search
# #==============================================================================
# # parameters = {'criterion':['gini', 'entropy']}
# # dt_clf = DecisionTreeClassifier(random_state=0)
# # gscv_dt = GridSearchCV(dt_clf, parameters, scoring='accuracy', cv=5)
# # gscv_dt_fit = gscv_dt.fit(x_train, y_train.values.ravel())
# # best_params = gscv_dt.best_params_
# # best_score = gscv_dt.best_score_

# # print ('Best parameters are: ',best_params)
# # print ('Best score is: ',best_score)

# #==============================================================================
# #Grid search for Hyper parameters - Maxdepth and ccp_alpha
# #https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
# #==============================================================================
# # Utility function to move the midpoint of a colormap to be around
# # the values of interest.

# class MidpointNormalize(Normalize):

#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         self.midpoint = midpoint
#         Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip=None):
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(value, x, y))
    
# # #############################################################################
# # # Train classifiers
# # #
# # # For an initial search, a logarithmic grid with basis
# # # 10 is often helpful. Using a basis of 2, a finer
# # # tuning can be achieved but at a much higher cost.

# # #hidden_layer_sizes_range = np.linspace(0, 400, 9)
# # #hidden_layer_sizes_range[0] = 10
# # hidden_layer_sizes_range = [(10), (50), (100), (150), (200), (250), (300), (350), (400)]
# # alpha_range = [1e-2, 1e-3, 1e-4, 1e-5]
# # param_grid = dict(hidden_layer_sizes=hidden_layer_sizes_range, alpha=alpha_range)
# # #cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# # cv = 5
# # grid = GridSearchCV(MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 1000,random_state = 0), param_grid=param_grid, cv=cv)
# # grid.fit(x_train, y_train.values.ravel())

# # print("The best parameters are %s with a score of %0.2f"
# #       % (grid.best_params_, grid.best_score_))

# # scores = grid.cv_results_['mean_test_score'].reshape(len(alpha_range),
# #                                                       len(hidden_layer_sizes_range))

# # print ('Max score: ',np.max(scores))
# # print ('Min score: ',np.min(scores))

# # plt.figure(figsize=(8, 6))
# # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
# # plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
# #             norm=MidpointNormalize(vmin=0.835, midpoint=0.847))
# # plt.xlabel('hidden_layer_sizes')
# # plt.ylabel('alpha')
# # plt.colorbar()
# # plt.xticks(np.arange(len(hidden_layer_sizes_range)), hidden_layer_sizes_range, rotation=45)
# # plt.yticks(np.arange(len(alpha_range)), alpha_range)
# # plt.title('Validation accuracy')
# # plt.grid(False)
# # plt.savefig('Sample_A_part_4_PCA_ANN_sample_A_Grid_search.png',dpi=600)
# # plt.show()

# #==============================================================================
# # Validation Curve 1
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================
# MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', alpha = 1e-4,hidden_layer_sizes = (10),random_state = 0)
# param_range = [10, 20, 30, 40, 50, 100, 200, 400, 600, 800, 1000, 1200, 1800, 2500, 3000,3500]
# train_scores, test_scores = validation_curve(
#     MLP_clf, X = x_train, y = y_train.values.ravel(), param_name="max_iter", param_range=param_range,
#     scoring="accuracy")
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.figure(2)
# plt.title("Accuracy VS Max Iteration")
# plt.xlabel("Max iteration")
# plt.ylabel("Score")
# #plt.ylim(0.0, 1.1)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#               color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                   train_scores_mean + train_scores_std, alpha=0.2,
#                   color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#               color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                   test_scores_mean + test_scores_std, alpha=0.2,
#                   color="navy", lw=lw)
# plt.legend(loc="best")
# plt.grid(True)
# plt.savefig('Sample_A_part_4_RP_full_ANN_sample_A_Validation_Curve_Max_iteration.png',dpi=600)
# plt.show()


# #==============================================================================
# # Validation Curve 2
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================

# MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 2500, alpha = 1e-4, random_state = 0)
# param_range = [2, 5, 10, 15, 20, 50, 75, 100, 150, 200, 250, 300]
# train_scores, test_scores = validation_curve(
#     MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='hidden_layer_sizes', param_range=param_range,
#     scoring="accuracy")
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.figure(2)
# plt.title("Accuracy VS Hidden Layer Sizes (logistic)")
# plt.xlabel("Hidden layer sizes")
# plt.ylabel("Score")
# #plt.ylim(0.0, 1.1)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#               color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                   train_scores_mean + train_scores_std, alpha=0.2,
#                   color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#               color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                   test_scores_mean + test_scores_std, alpha=0.2,
#                   color="navy", lw=lw)
# plt.legend(loc="best")
# plt.grid(True)
# plt.savefig('Sample_A_part_4_RP_full_ANN_sample_A_Validation_Curve_hidden_layer_sizes_logistic.png',dpi=600)
# plt.show()

# #==============================================================================
# # Validation Curve 3
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================

# MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 2500, hidden_layer_sizes = 20, random_state = 0)
# param_range = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
# train_scores, test_scores = validation_curve(
#     MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='alpha', param_range=param_range,
#     scoring="accuracy")
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.figure(3)
# plt.title("Accuracy VS Alpha")
# plt.xlabel("Alpha")
# plt.ylabel("Score")
# #plt.ylim(0.0, 1.1)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#               color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                   train_scores_mean + train_scores_std, alpha=0.2,
#                   color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#               color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                   test_scores_mean + test_scores_std, alpha=0.2,
#                   color="navy", lw=lw)
# plt.legend(loc="best")
# plt.grid(True)
# plt.xscale('log')
# plt.savefig('Sample_A_part_4_RP_full_ANN_sample_A_Validation_Curve_alpha.png',dpi=600)
# plt.show()

# #==============================================================================
# # Validation Curve 4
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================

# # MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 1500, alpha = 1e-4, random_state = 0)
# # param_range = [(20,),(20,2), (20,4), (20,6), (20,8), (20,10), (20,15), (20,20), (20,50), (20,100), (20,150)]
# # train_scores, test_scores = validation_curve(
# #     MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='hidden_layer_sizes', param_range=param_range,
# #     scoring="accuracy")
# # train_scores_mean = np.mean(train_scores, axis=1)
# # train_scores_std = np.std(train_scores, axis=1)
# # test_scores_mean = np.mean(test_scores, axis=1)
# # test_scores_std = np.std(test_scores, axis=1)

# # plt.figure(4)
# # plt.title("Validation Curve with ANN")
# # plt.xlabel("Second hidden layer sizes")
# # plt.ylabel("Score")
# # #plt.ylim(0.0, 1.1)
# # lw = 2
# # plt.plot([0,2,4,6,8,10,15,20,50,100,150], train_scores_mean, label="Training score",
# #               color="darkorange", lw=lw)
# # plt.fill_between([0,2,4,6,8,10,15,20,50,100,150], train_scores_mean - train_scores_std,
# #                   train_scores_mean + train_scores_std, alpha=0.2,
# #                   color="darkorange", lw=lw)
# # plt.plot([0,2,4,6,8,10,15,20,50,100,150], test_scores_mean, label="Cross-validation score",
# #               color="navy", lw=lw)
# # plt.fill_between([0,2,4,6,8,10,15,20,50,100,150], test_scores_mean - test_scores_std,
# #                   test_scores_mean + test_scores_std, alpha=0.2,
# #                   color="navy", lw=lw)
# # plt.legend(loc="best")
# # plt.grid(True)
# # #plt.xscale('log')
# # plt.savefig('Sample_A_part_4_PCA_ANN_sample_A_Validation_Curve_second_hidden_layer_sizes.png',dpi=600)
# # plt.show()

# #==============================================================================
# # Validation Curve 5
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================

# # MLP_clf = MLPClassifier(solver='sgd',activation = 'tanh', max_iter = 1500, alpha = 1e-4, random_state = 0)
# # param_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 50, 75, 100, 150, 200]
# # train_scores, test_scores = validation_curve(
# #     MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='hidden_layer_sizes', param_range=param_range,
# #     scoring="accuracy")
# # train_scores_mean = np.mean(train_scores, axis=1)
# # train_scores_std = np.std(train_scores, axis=1)
# # test_scores_mean = np.mean(test_scores, axis=1)
# # test_scores_std = np.std(test_scores, axis=1)

# # plt.figure(5)
# # plt.title("Validation Curve with ANN (tanh)")
# # plt.xlabel("Hidden layer sizes")
# # plt.ylabel("Score")
# # #plt.ylim(0.0, 1.1)
# # lw = 2
# # plt.plot(param_range, train_scores_mean, label="Training score",
# #               color="darkorange", lw=lw)
# # plt.fill_between(param_range, train_scores_mean - train_scores_std,
# #                   train_scores_mean + train_scores_std, alpha=0.2,
# #                   color="darkorange", lw=lw)
# # plt.plot(param_range, test_scores_mean, label="Cross-validation score",
# #               color="navy", lw=lw)
# # plt.fill_between(param_range, test_scores_mean - test_scores_std,
# #                   test_scores_mean + test_scores_std, alpha=0.2,
# #                   color="navy", lw=lw)
# # plt.legend(loc="best")
# # plt.grid(True)
# # plt.savefig('ANN_sample_A_Validation_Curve_hidden_layer_sizes_tanh.png',dpi=600)
# # plt.show()

# #==============================================================================
# # Validation Curve 6
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================

# # MLP_clf = MLPClassifier(solver='sgd',activation = 'relu', max_iter = 1500, alpha = 1e-4, random_state = 0)
# # param_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 50, 75, 100, 150, 200]
# # train_scores, test_scores = validation_curve(
# #     MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='hidden_layer_sizes', param_range=param_range,
# #     scoring="accuracy")
# # train_scores_mean = np.mean(train_scores, axis=1)
# # train_scores_std = np.std(train_scores, axis=1)
# # test_scores_mean = np.mean(test_scores, axis=1)
# # test_scores_std = np.std(test_scores, axis=1)

# # plt.figure(6)
# # plt.title("Validation Curve with ANN (relu)")
# # plt.xlabel("Hidden layer sizes")
# # plt.ylabel("Score")
# # #plt.ylim(0.0, 1.1)
# # lw = 2
# # plt.plot(param_range, train_scores_mean, label="Training score",
# #               color="darkorange", lw=lw)
# # plt.fill_between(param_range, train_scores_mean - train_scores_std,
# #                   train_scores_mean + train_scores_std, alpha=0.2,
# #                   color="darkorange", lw=lw)
# # plt.plot(param_range, test_scores_mean, label="Cross-validation score",
# #               color="navy", lw=lw)
# # plt.fill_between(param_range, test_scores_mean - test_scores_std,
# #                   test_scores_mean + test_scores_std, alpha=0.2,
# #                   color="navy", lw=lw)
# # plt.legend(loc="best")
# # plt.grid(True)
# # plt.savefig('ANN_sample_A_Validation_Curve_hidden_layer_sizes_relu.png',dpi=600)
# # plt.show()

# #==============================================================================
# # Validation Curve 7
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================

# MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 3000, hidden_layer_sizes = 20, random_state = 0)
# param_range = [1e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99999]
# train_scores, test_scores = validation_curve(
#     MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='momentum', param_range=param_range,
#     scoring="accuracy")
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.figure(7)
# plt.title("Accuracy VS Momentum (logistic)")
# plt.xlabel("Momentum")
# plt.ylabel("Score")
# #plt.ylim(0.0, 1.1)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#               color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                   train_scores_mean + train_scores_std, alpha=0.2,
#                   color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#               color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                   test_scores_mean + test_scores_std, alpha=0.2,
#                   color="navy", lw=lw)
# plt.legend(loc="best")
# plt.grid(True)
# plt.savefig('Sample_A_part_4_RP_full_ANN_sample_A_Validation_Curve_momentum.png',dpi=600)
# plt.show()

# #==============================================================================
# # Momentum Curve 
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================
# clfs = []
# iters = []
# momentums = [1e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99999]

# for momentum in momentums:
#     MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 3000, hidden_layer_sizes = 20, random_state = 0, momentum=momentum)
#     MLP_clf.fit(x_train, y_train.values.ravel())
#     clfs.append(MLP_clf)
#     iters.append(MLP_clf.n_iter_)

# plt.figure(8)
# plt.title("Momentum VS Iteration")
# plt.xlabel("Momentum")
# plt.ylabel("Iterations needed for converge")
# #plt.ylim(0.0, 1.1)
# lw = 2
# plt.plot(momentums, iters,
#               color="navy", lw=lw)
# #plt.legend(loc="best")
# plt.grid(True)
# plt.savefig('Sample_A_part_4_RP_ANN_sample_A_momentum_VS_Iteration.png',dpi=600)
# plt.show()


# #==============================================================================
# # Learning rate Curve 1
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================
# # clfs = []
# # iters = []
# # learning_rate_inits = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

# # for learning_rate_init in learning_rate_inits:
# #     MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 4000, hidden_layer_sizes = 20, random_state = 0, learning_rate_init=learning_rate_init)
# #     MLP_clf.fit(x_train, y_train.values.ravel())
# #     clfs.append(MLP_clf)
# #     iters.append(MLP_clf.n_iter_)

# # plt.figure(9)
# # plt.title("Validation Curve with ANN (logistic)")
# # plt.xlabel("Learning_rate_init")
# # plt.ylabel("Iterations needed for converge")
# # #plt.ylim(0.0, 1.1)
# # lw = 2
# # plt.plot(learning_rate_inits, iters,
# #               color="navy", lw=lw)
# # #plt.legend(loc="best")
# # plt.grid(True)
# # plt.xscale('log')
# # plt.savefig('ANN_sample_A_learning_rate_init_VS_Iteration.png',dpi=600)
# # plt.show()


# #==============================================================================
# # Validation Curve 7
# #https://scikit-learn.org/stable/modules/learning_curve.html
# #==============================================================================

# # MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic', max_iter = 3000, hidden_layer_sizes = 20, random_state = 0)
# # param_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# # train_scores, test_scores = validation_curve(
# #     MLP_clf, X = x_train, y = y_train.values.ravel(), param_name='learning_rate_init', param_range=param_range,
# #     scoring="accuracy")
# # train_scores_mean = np.mean(train_scores, axis=1)
# # train_scores_std = np.std(train_scores, axis=1)
# # test_scores_mean = np.mean(test_scores, axis=1)
# # test_scores_std = np.std(test_scores, axis=1)

# # plt.figure(10)
# # plt.title("Validation Curve with ANN (logistic)")
# # plt.xlabel("Learning_rate_init")
# # plt.ylabel("Score")
# # #plt.ylim(0.0, 1.1)
# # lw = 2
# # plt.plot(param_range, train_scores_mean, label="Training score",
# #               color="darkorange", lw=lw)
# # plt.fill_between(param_range, train_scores_mean - train_scores_std,
# #                   train_scores_mean + train_scores_std, alpha=0.2,
# #                   color="darkorange", lw=lw)
# # plt.plot(param_range, test_scores_mean, label="Cross-validation score",
# #               color="navy", lw=lw)
# # plt.fill_between(param_range, test_scores_mean - test_scores_std,
# #                   test_scores_mean + test_scores_std, alpha=0.2,
# #                   color="navy", lw=lw)
# # plt.legend(loc="best")
# # plt.grid(True)
# # plt.xscale('log')
# # plt.savefig('ANN_sample_A_Validation_Curve_learning_rate_init.png',dpi=600)
# # plt.show()


# #==============================================================================
# #learning curve 2
# #https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
# #==============================================================================
# train_sizes = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# for i in range(1, len(train_sizes)):
#     train_sizes[i] = math.floor(train_sizes[i] *y_train.size*4/5)
    
# #print (train_sizes)

# train_sizes, train_scores, validation_scores = learning_curve(
# estimator = MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 3000,alpha = 1e-4,hidden_layer_sizes = (20),random_state = 0),
# X = x_train,
# y = y_train.values.ravel(), train_sizes = train_sizes, cv = 5,
# scoring = 'accuracy',
# shuffle = True,
# random_state=0)


# #print('Training scores:\n\n', train_scores)
# #print('\n', '-' * 70) # separator to make the output easy to read
# #print('\nValidation scores:\n\n', validation_scores)

# train_scores_mean = train_scores.mean(axis = 1)
# validation_scores_mean = validation_scores.mean(axis = 1)
# print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
# print('\n', '-' * 20) # separator
# print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

# fig = plt.figure(10)
# #plt.style.use('seaborn')
# plt.plot(train_sizes, train_scores_mean, label = 'Training error', color="darkorange", lw=2)
# plt.plot(train_sizes, validation_scores_mean, label = 'Validation error', color="navy", lw=2)
# plt.ylabel('Accuracy')
# plt.xlabel('Training set size')
# #plt.xlabel('Training set size', fontsize = 14)
# plt.title('Learning curves for ANN with RP', y = 1.03)
# plt.legend()
# #plt.ylim(0,40)
# plt.grid(True)
# plt.savefig('Sample_A_part_4_RP_full_ANN_sample_A_Learning_curves_for_ANN_after_hyper_parameter_tunning.png',dpi=600)

# #==============================================================================
# #Final prediction
# #==============================================================================

# print('\n', '-' * 50)
# print('After hyperparameter tunning, max_iter = 3000, alpha = 1e-4,hidden_layer_sizes = (20)')

# start_1 = time.time()
# MLP_clf = MLPClassifier(solver='sgd',activation = 'logistic',max_iter = 3000,alpha = 1e-4,hidden_layer_sizes = (20),random_state = 0)
# MLP_clf.fit(x_train, y_train.values.ravel())
# end_1 = time.time()
# print('Train time is: ',end_1 - start_1)

# start_2 = time.time()
# y_predict_train = MLP_clf.predict(x_train)
# end_2 = time.time()
# print('Predict time for training set is: ',end_2 - start_2)

# start_3 = time.time()
# y_predict_test = MLP_clf.predict(x_test)
# end_3 = time.time()
# print('Predict time for test set is: ',end_3 - start_3)

# train_accuracy = accuracy_score(y_train.values,y_predict_train)
# test_accuracy = accuracy_score(y_test.values,y_predict_test)

# report = classification_report(y_test.values,y_predict_test)

# print('Training accuracy is: ',train_accuracy)
# print('Test accuracy is: ',test_accuracy)

# print ('Classification report:')
# print (report)
# print('\n', '-' * 50)
