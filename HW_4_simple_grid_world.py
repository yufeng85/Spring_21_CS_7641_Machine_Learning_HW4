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

from mdptoolbox import mdp as MDP

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
    fig.legend(loc="upper right", bbox_to_anchor=(1,0.3), bbox_transform=ax1.transAxes)
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
    
def grid_game(dim=5,goal = [0,0],trap = [],wall = [], reward_goal=2,reward_trap=-1,reward_wall=-100):
    #dim = 5;
    #goal = [2,2]
    #trap = [[1,1]]
    #wall = [[1,0],[1,3],[1,4],[3,0],[3,1],[3,3],[3,4]]
    #wall = [[1,3],[3,1],[3,3]]
    
    UP = []
    RIGHT = []
    DOWN = []
    LEFT = []
    R = []
    
    #reward_goal = 2
    #reward_trap = -100
    for i in range(dim*dim):
        row = i//dim
        col = i%dim
        s_up = np.zeros(dim*dim)
        if row == dim-1 or ([row+1,col] in wall) or [row,col] == goal:
            s_up[i] = 1
        else:
            s_up[(row+1)*dim+col] = 1
        UP.append(s_up)
        
        s_down = np.zeros(dim*dim)
        if row == 0 or ([row-1,col] in wall) or [row,col] == goal:
            s_down[i] = 1
        else:
            s_down[(row-1)*dim+col] = 1
        DOWN.append(s_down)
        
        s_right = np.zeros(dim*dim)
        if col == dim-1 or ([row,col+1] in wall) or [row,col] == goal:
            s_right[i] = 1
        else:
            s_right[row*dim+col+1] = 1
        RIGHT.append(s_right)
        
        s_left = np.zeros(dim*dim)
        if col == 0 or ([row,col-1] in wall) or [row,col] == goal:
            s_left[i] = 1
        else:
            s_left[row*dim+col-1] = 1
        LEFT.append(s_left)
        
        if row==goal[0] and col==goal[1]:
            R.append(reward_goal*np.ones(4))
        elif [row,col] in trap:
            R.append(reward_trap*np.ones(4))
        elif [row,col] in wall:
            R.append(reward_wall*np.ones(4))
        else:
            R.append(np.zeros(4))
        # r = np.zeros(4)
        # if [row+1,col] == goal:
        #     r[0] = reward_goal
        # elif [row+1,col] in trap:
        #     r[0] = reward_trap
        # elif [row+1,col] in wall:
        #     r[0] = reward_wall
        # else:
        #     r[0] = 0
            
        # if [row,col+1] == goal:
        #     r[1] = reward_goal
        # elif [row,col+1] in trap:
        #     r[1] = reward_trap
        # elif [row,col+1] in wall:
        #     r[1] = reward_wall
        # else:
        #     r[1] = 0
            
        # if [row-1,col] == goal:
        #     r[2] = reward_goal
        # elif [row-1,col] in trap:
        #     r[2] = reward_trap
        # elif [row-1,col] in wall:
        #     r[2] = reward_wall
        # else:
        #     r[2] = 0
            
        # if [row,col-1] == goal:
        #     r[3] = reward_goal
        # elif [row,col-1] in trap:
        #     r[3] = reward_trap
        # elif [row,col-1] in wall:
        #     r[3] = reward_wall
        # else:
        #     r[3] = 0
        
        # R.append(r)
                
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
    return P, R

def grid_game_2(dim=5,goal = [0,0],trap = [],wall = [], reward_goal=2,reward_trap=-1,reward_wall=-100):
    #dim = 5;
    #goal = [2,2]
    #trap = [[1,1]]
    #wall = [[1,0],[1,3],[1,4],[3,0],[3,1],[3,3],[3,4]]
    #wall = [[1,3],[3,1],[3,3]]
    
    UP = []
    RIGHT = []
    DOWN = []
    LEFT = []
    R = []
    
    #reward_goal = 2
    #reward_trap = -100
    for i in range(dim*dim):
        row = i//dim
        col = i%dim
        s_up = np.zeros(dim*dim)
        if row == dim-1 or ([row+1,col] in wall):
            s_up[i] = 1
        else:
            s_up[(row+1)*dim+col] = 1
        UP.append(s_up)
        
        s_down = np.zeros(dim*dim)
        if row == 0 or ([row-1,col] in wall):
            s_down[i] = 1
        else:
            s_down[(row-1)*dim+col] = 1
        DOWN.append(s_down)
        
        s_right = np.zeros(dim*dim)
        if col == dim-1 or ([row,col+1] in wall):
            s_right[i] = 1
        else:
            s_right[row*dim+col+1] = 1
        RIGHT.append(s_right)
        
        s_left = np.zeros(dim*dim)
        if col == 0 or ([row,col-1] in wall):
            s_left[i] = 1
        else:
            s_left[row*dim+col-1] = 1
        LEFT.append(s_left)
        
        if row==goal[0] and col==goal[1]:
            R.append(reward_goal*np.ones(4))
        elif [row,col] in trap:
            R.append(reward_trap*np.ones(4))
        elif [row,col] in wall:
            R.append(reward_wall*np.ones(4))
        else:
            R.append(np.zeros(4))
        # r = np.zeros(4)
        # if [row+1,col] == goal:
        #     r[0] = reward_goal
        # elif [row+1,col] in trap:
        #     r[0] = reward_trap
        # elif [row+1,col] in wall:
        #     r[0] = reward_wall
        # else:
        #     r[0] = 0
            
        # if [row,col+1] == goal:
        #     r[1] = reward_goal
        # elif [row,col+1] in trap:
        #     r[1] = reward_trap
        # elif [row,col+1] in wall:
        #     r[1] = reward_wall
        # else:
        #     r[1] = 0
            
        # if [row-1,col] == goal:
        #     r[2] = reward_goal
        # elif [row-1,col] in trap:
        #     r[2] = reward_trap
        # elif [row-1,col] in wall:
        #     r[2] = reward_wall
        # else:
        #     r[2] = 0
            
        # if [row,col-1] == goal:
        #     r[3] = reward_goal
        # elif [row,col-1] in trap:
        #     r[3] = reward_trap
        # elif [row,col-1] in wall:
        #     r[3] = reward_wall
        # else:
        #     r[3] = 0
        
        # R.append(r)
                
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
    return P, R

def draw_map(policy,dim,goal = [0,0],trap = [],wall = [],file_name='test.png',scale=1,maker_size=2000,title='Grid World'):
    #policy = policy_pi
    #dim = [5,5]
    
    x = np.linspace(0, dim[0] - 1, dim[0]) + 0.5
    y = np.linspace(0, dim[0] - 1, dim[1]) + 0.5
    X, Y = np.meshgrid(x, y)
    zeros = np.zeros(dim)
        
    arrow_up = np.zeros(dim)
    arrow_right = np.zeros(dim)
    arrow_down = np.zeros(dim)
    arrow_left = np.zeros(dim)
    
    for i in range(len(policy)):
        row = i//dim[0]
        col = i%dim[0]
        if (row != goal[0] or col != goal[1]) and ([row,col] not in wall) and ([row,col] not in trap):        
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
    plt.quiver(X, Y, zeros, arrow_up, scale=scale, units='xy')
    # Vectors point in negative X-direction
    plt.quiver(X, Y, -arrow_left, zeros, scale=scale, units='xy')
    # Vectors point in negative Y-direction
    plt.quiver(X, Y, zeros, -arrow_down, scale=scale, units='xy')
    # Vectors point in positive X-direction
    plt.quiver(X, Y, arrow_right, zeros, scale=scale, units='xy')
    
    
    #plt.scatter(goal[0]+0.5,goal[1]+0.5,marker='o',c='b',edgecolors='b',s=2000)
    plt.scatter(goal[1]+0.5,goal[0]+0.5,marker='*',c='y',edgecolors='b',s=maker_size)
    
    for i in range(len(wall)):
        plt.scatter(wall[i][1]+0.5,wall[i][0]+0.5,marker='s',c='black',s=maker_size)
        
    for i in range(len(trap)):
        plt.scatter(trap[i][1]+0.5,trap[i][0]+0.5,marker='o',c='r',s=maker_size)
    
    plt.xlim([0, dim[0]])
    plt.ylim([0, dim[1]])
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    ax.set_xticks(np.arange(0, dim[0]+1))
    ax.set_yticks(np.arange(0, dim[1]+1)) 
    plt.grid()
    #plt.show()
    plt.title(title, fontsize = 30)
    plt.tight_layout()
    plt.savefig(file_name,dpi=600)

def draw_map_2(policy,value,dim,goal = [0,0],trap = [],wall = [],file_name='test.png',scale=1,maker_size=2000,text_property=[0.3,0.07,25,'.4f'],title='Grid World',arrow_length=0.4):
    #policy = policy_pi
    #dim = [5,5]
    
    x = np.linspace(0, dim[0] - 1, dim[0]) + 0.5
    y = np.linspace(0, dim[0] - 1, dim[1]) + 0.5
    X, Y = np.meshgrid(x, y)
    zeros = np.zeros(dim)
        
    arrow_up = np.zeros(dim)
    arrow_right = np.zeros(dim)
    arrow_down = np.zeros(dim)
    arrow_left = np.zeros(dim)
    
    for i in range(len(policy)):
        row = i//dim[0]
        col = i%dim[0]
        if (row != goal[0] or col != goal[1]) and ([row,col] not in wall) and ([row,col] not in trap):        
            if policy[i] == 0:
                arrow_up[row][col] = arrow_length
            elif policy[i] == 1:
                arrow_right[row][col] = arrow_length
            elif policy[i] == 2:
                arrow_down[row][col] = arrow_length
            else:
                arrow_left[row][col] = arrow_length
                
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes()
    # Vectors point in positive Y-direction
    plt.quiver(X, Y, zeros, arrow_up, scale=scale, units='xy')
    # Vectors point in negative X-direction
    plt.quiver(X, Y, -arrow_left, zeros, scale=scale, units='xy')
    # Vectors point in negative Y-direction
    plt.quiver(X, Y, zeros, -arrow_down, scale=scale, units='xy')
    # Vectors point in positive X-direction
    plt.quiver(X, Y, arrow_right, zeros, scale=scale, units='xy')
    
    
    #plt.scatter(goal[0]+0.5,goal[1]+0.5,marker='o',c='b',edgecolors='b',s=2000)
    plt.scatter(goal[1]+0.5,goal[0]+0.5,marker='*',c='y',edgecolors='b',s=maker_size)
    
    for i in range(len(wall)):
        plt.scatter(wall[i][1]+0.5,wall[i][0]+0.5,marker='s',c='black',s=maker_size)
        
    for i in range(len(trap)):
        plt.scatter(trap[i][1]+0.5,trap[i][0]+0.5,marker='o',c='r',s=maker_size)
        
    for i in range(len(value)):
        row = i//dim[0]
        col = i%dim[0]
        plt.text(col+text_property[0],row+text_property[1],str(format(value[i], text_property[3])),fontsize=text_property[2])
    
    plt.xlim([0, dim[0]])
    plt.ylim([0, dim[1]])
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    ax.set_xticks(np.arange(0, dim[0]+1))
    ax.set_yticks(np.arange(0, dim[1]+1)) 
    plt.grid()
    #plt.show()
    plt.title(title, fontsize = 30)
    plt.tight_layout()
    plt.savefig(file_name,dpi=600)

#==============================================================================
#Grid world 5X5
#==============================================================================
#==============================================================================
#Policy iteration
#============================================================================== 
dim = 5;
goal = [2,2]
trap = [[1,2]]
wall = [[1,0],[1,3],[3,1],[3,3]]
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -2.7
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

pi = mdp.PolicyIteration(P, R, 0.9)
pi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = pi.run()
end_time = Time.time()
print('Execution time for PI is: ' + str(end_time-start_time))

policy = pi.policy
value = pi.V

title = 'Grid World with PI (Discount=0.9)'
draw_map(policy,[5,5],goal,trap,wall,'Grid_world_5_5_states_PI_map_policy_discount_0p9.png',1,2000,title)
draw_map_2(policy,value,[5,5],goal,trap,wall,'Grid_world_5_5_states_PI_map_value_discount_0p9.png',1,2000,[0.3,0.01,25,'.4f'],title)


name_1 = 'Grid_world_5_5_states_PI_Value VS Iterations for PI.png'
title_1 = 'Value VS Iterations for PI (25 states)'
name_2 = 'Grid_world_5_5_states_PI_Error Convergence VS Iteration for PI.png'
title_2 = 'Error Convergence VS Iteration for PI (25 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

#==============================================================================
pi = mdp.PolicyIteration(P, R, 0.5)
pi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = pi.run()
end_time = Time.time()
print('Execution time for PI is: ' + str(end_time-start_time))

policy = pi.policy
value = pi.V

title = 'Grid World with PI (Discount=0.5)'
draw_map(policy,[5,5],goal,trap,wall,'Grid_world_5_5_states_PI_map_policy_discount_0p5.png',1,2000,title)
draw_map_2(policy,value,[5,5],goal,trap,wall,'Grid_world_5_5_states_PI_map_value_discount_0p5.png',1,2000,[0.3,0.01,25,'.4f'],title)


name_1 = 'Grid_world_5_5_states_PI_Value VS Iterations for PI_discount_0p5.png'
title_1 = 'Value VS Iterations for PI (25 states)'
name_2 = 'Grid_world_5_5_states_PI_Error Convergence VS Iteration for PI_discount_0p5.png'
title_2 = 'Error Convergence VS Iteration for PI (25 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

#==============================================================================
#Value iteration
#==============================================================================
dim = 5;
goal = [2,2]
trap = [[1,2]]
wall = [[1,0],[1,3],[3,1],[3,3]]
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -2.7
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

vi = mdp.ValueIteration(P, R, 0.9)
vi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = vi.run()
end_time = Time.time()
print('Execution time for VI is: ' + str(end_time-start_time))

policy = vi.policy
value = vi.V

title = 'Grid World with VI (Discount=0.9)'
draw_map(policy,[5,5],goal,trap,wall,'Grid_world_5_5_states_VI_map_policy_discount_0p9.png',1,2000,title)
draw_map_2(policy,value,[5,5],goal,trap,wall,'Grid_world_5_5_states_VI_map_value_discount_0p9.png',1,2000,[0.3,0.01,25,'.4f'],title)


name_1 = 'Grid_world_5_5_states_VI_Value VS Iterations for VI.png'
title_1 = 'Value VS Iterations for VI (25 states)'
name_2 = 'Grid_world_5_5_states_VI_Error Convergence VS Iteration for VI.png'
title_2 = 'Error Convergence VS Iteration for VI (25 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

#==============================================================================
vi = mdp.ValueIteration(P, R, 0.5)
vi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = vi.run()
end_time = Time.time()
print('Execution time for VI is: ' + str(end_time-start_time))

policy = vi.policy
value = vi.V

title = 'Grid World with VI (Discount=0.5)'
draw_map(policy,[5,5],goal,trap,wall,'Grid_world_5_5_states_VI_map_policy_discount_0p5.png',1,2000,title)
draw_map_2(policy,value,[5,5],goal,trap,wall,'Grid_world_5_5_states_VI_map_value_discount_0p5.png',1,2000,[0.3,0.01,25,'.4f'],title)


name_1 = 'Grid_world_5_5_states_VI_Value VS Iterations for VI_discount_0p5.png'
title_1 = 'Value VS Iterations for VI (25 states)'
name_2 = 'Grid_world_5_5_states_VI_Error Convergence VS Iteration for VI_discount_0p5.png'
title_2 = 'Error Convergence VS Iteration for VI (25 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

#==============================================================================
#changing epsilon
#==============================================================================
dim = 5;
goal = [2,2]
trap = [[1,2]]
wall = [[1,0],[1,3],[3,1],[3,3]]
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -2.7
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

vi = mdp.ValueIteration(P, R, 0.9, epsilon=1e-20)
vi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = vi.run()
end_time = Time.time()
print('Execution time for VI is: ' + str(end_time-start_time))

policy = vi.policy
value = vi.V

title = 'Grid World with VI (Discount=0.9,epsilon=1E-20)'
draw_map(policy,[5,5],goal,trap,wall,'Grid_world_5_5_states_VI_map_policy_discount_0p9_epsilon_1e-20.png',1,2000,title)
draw_map_2(policy,value,[5,5],goal,trap,wall,'Grid_world_5_5_states_VI_map_value_discount_0p9_epsilon_1e-20.png',1,2000,[0.3,0.01,25,'.4f'],title)


name_1 = 'Grid_world_5_5_states_VI_Value VS Iterations for VI_epsilon_1e-20.png'
title_1 = 'Value VS Iterations for VI (25 states, epsilon=1E-20)'
name_2 = 'Grid_world_5_5_states_VI_Error Convergence VS Iteration for VI_epsilon_1e-20.png'
title_2 = 'Error Convergence VS Iteration for VI (25 states, epsilon=1E-20)'
convergence_plot(result,name_1,title_1,name_2,title_2)

#==============================================================================
vi = mdp.ValueIteration(P, R, 0.5, epsilon=1e-40)
vi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = vi.run()
end_time = Time.time()
print('Execution time for VI is: ' + str(end_time-start_time))

policy = vi.policy
value = vi.V

title = 'Grid World with VI (Discount=0.5, epsilon=1E-40)'
draw_map(policy,[5,5],goal,trap,wall,'Grid_world_5_5_states_VI_map_policy_discount_0p5_epsilon_1e-40.png',1,2000,title)
draw_map_2(policy,value,[5,5],goal,trap,wall,'Grid_world_5_5_states_VI_map_value_discount_0p5_epsilon_1e-40.png',1,2000,[0.3,0.01,25,'.4f'],title)


name_1 = 'Grid_world_5_5_states_VI_Value VS Iterations for VI_discount_0p5_epsilon_1e-20.png'
title_1 = 'Value VS Iterations for VI (25 states)'
name_2 = 'Grid_world_5_5_states_VI_Error Convergence VS Iteration for VI_discount_0p5_epsilon_1e-20.png'
title_2 = 'Error Convergence VS Iteration for VI (25 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)



#==============================================================================
#Q learning
#==============================================================================
dim = 5;
goal = [2,2]
trap = [[1,2]]
wall = [[1,0],[1,3],[3,1],[3,3]]
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -2.7
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

np.random.seed(0) 
ql = mdp.QLearning(P, R, 0.9,n_iter=200000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=1, epsilon_min=0.1, epsilon_decay=0.99)
ql.setVerbose()
start_time = Time.time()    
result = ql.run()
end_time = Time.time()
print('Computation time is: '+ str(end_time-start_time))

policy = ql.policy
value = ql.V
title = 'Grid World with QL (Discount=0.9, iterations=2E8)'
draw_map(policy,[5,5],goal,trap,wall,'Grid_world_5_5_states_QL_map_policy_discount_0p9_2e8_iteration.png',1,2000,title)
draw_map_2(policy,value,[5,5],goal,trap,wall,'Grid_world_5_5_states_QL_map_value_discount_0p9_2e8_iteration.png',1,2000,[0.3,0.01,25,'.4f'],title)


name_1 = 'Grid_world_5_5_states_QL_Value VS Iterations for QL_2e8_iteration.png'
title_1 = 'Value VS Iterations for QL (25 states)'
name_2 = 'Grid_world_5_5_states_QL_Error Convergence VS Iteration for QL_2e8_iteration.png'
title_2 = 'Error Convergence VS Iteration for QL (25 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)


filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_result_2e8_iteration.txt'
save_variable(result,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_2e8_iteration.txt'
save_variable(ql,filename)

test_r = load_variable(filename)


#==============================================================================
dim = 5;
goal = [2,2]
trap = [[1,2]]
wall = [[1,0],[1,3],[3,1],[3,3]]
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -2.7
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

epsilons = [1,0.9,0.8,0.7,0.6]
results = []
policies = []
values = []
for epsilon in epsilons:    
    np.random.seed(0) 
    ql = mdp.QLearning(P, R, 0.9,n_iter=5000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.01,epsilon=epsilon, epsilon_min=0.01, epsilon_decay=0.99)
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
#ax2 = ax1.twinx()  # this is the important function
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
        
    #ax1.plot(iteration, max_V,label='Max value (Epsilon='+str(eps)+')', color="C"+str(2*j), lw=2)            
    ax1.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps)+')', color="C"+str(j), lw=2)
    

ax1.set_ylabel('Mean Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.4), bbox_transform=ax1.transAxes)
plt.tight_layout()
ax1.set_title('Effect of Epsilon for Q Learning (25 states)')
plt.savefig('Grid_world_5_5_states_QL_Effect of Epsilon for Q Learning (5 states)_epsilon.png',dpi=600)

#==============================================================================
dim = 5;
goal = [2,2]
trap = [[1,2]]
wall = [[1,0],[1,3],[3,1],[3,3]]
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -2.7
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

epsilon_decays = [1,0.9999,0.999,0.99,0.9]
results = []
policies = []
values = []
for eps_decay in epsilon_decays:    
    np.random.seed(0) 
    ql = mdp.QLearning(P, R, 0.9,n_iter=5000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.01,epsilon=0.8, epsilon_min=0.01, epsilon_decay=eps_decay)
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
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Epsilon Decay for Q Learning (Epsilon = 0.8)')
plt.tight_layout()
plt.savefig('Grid_world_5_5_states_QL_Effect of Epsilon Decay for Q Learning (Epsilon = 0.8)_mean value.png',dpi=600)
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
plt.savefig('Grid_world_5_5_states_QL_Effect of Epsilon Decay for Q Learning (Epsilon = 0.8)_max value.png',dpi=600)

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
dim = 5;
goal = [2,2]
trap = [[1,2]]
wall = [[1,0],[1,3],[3,1],[3,3]]
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -2.7
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

alphas = [0.05,0.1,0.2,0.4,0.6,0.8,1]
results = []
policies = []
values = []
qls = []
for i in range(len(alphas)):    
    np.random.seed(0) 
    ql = mdp.QLearning(P, R, 0.9,n_iter=5000000,alpha=alphas[i], alpha_decay=0.99, alpha_min=0.01,epsilon=1, epsilon_min=0.1, epsilon_decay=0.99)
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
filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_alpha_effect_result.txt'
save_variable(results,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_alpha_effect_ql.txt'
save_variable(qls,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_alpha_effect_value.txt'
save_variable(values,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_alpha_effect_policy.txt'
save_variable(policies,filename)

#test_r = load_variable(filename)

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(0,len(results),1):
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
    ax1.plot(iteration, mean_V,label='Mean Value (alpha='+str(alp)+')', color="C"+str(int(j)), lw=2)
    

ax1.set_ylabel('Mean Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
#ax1.set_yscale('log')
#ax2.set_yscale('log')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.55), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Alpha for Q Learning (25 states)')
#plt.xlim([0, 2e6])
plt.tight_layout()
plt.savefig('Grid_world_5_5_states_QL_Effect of Alpha for Q Learning (5 states)_mean.png',dpi=600)
#==============================================================================
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(0,len(results),1):
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
    ax1.plot(iteration, max_V,label='Max Value (alpha='+str(alp)+')', color="C"+str(int(j)), lw=2)
    

ax1.set_ylabel('Max Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
#ax1.set_yscale('log')
#ax2.set_yscale('log')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.55), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Alpha for Q Learning (25 states)')
#plt.xlim([0, 2e6])
plt.tight_layout()
plt.savefig('Grid_world_5_5_states_QL_Effect of Alpha for Q Learning (5 states)_max.png',dpi=600)
#==============================================================================
#alpha decay
#==============================================================================
dim = 5;
goal = [2,2]
trap = [[1,2]]
wall = [[1,0],[1,3],[3,1],[3,3]]
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -2.7
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

alpha_decays = [1,0.99,0.8,0.6,0.4,0.2,0.1]
results = []
policies = []
values = []
qls = []
for i in range(len(alpha_decays)):    
    np.random.seed(0) 
    ql = mdp.QLearning(P, R, 0.9,n_iter=5000000,alpha=0.1, alpha_decay=alpha_decays[i], alpha_min=0.01,epsilon=1, epsilon_min=0.01, epsilon_decay=0.99)
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
filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_alpha_decay_effect_result_2.txt'
save_variable(results,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_alpha_decay_effect_ql_2.txt'
save_variable(qls,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_alpha_decay_effect_value_2.txt'
save_variable(values,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_5_5_states_QL_alpha_decay_effect_policy_2.txt'
save_variable(policies,filename)

#test_r = load_variable(filename)
#==============================================================================
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()  # this is the important function
for j in range(0,len(results),1):
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
        
    ax1.plot(iteration, mean_V,label='Mean value (alpha decay='+str(alp_decay)+')', color="C"+str(int(j)), lw=2)            
    #ax2.plot(iteration, mean_V,label='Mean Value (Epsilon='+str(eps_decay)+')', color="C"+str(2*j+1), lw=2)
    

ax1.set_ylabel('Mean Value')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Iterations')
ax1.grid()    
#ax2.set_ylabel('Mean Value')
fig.legend(loc="upper right", bbox_to_anchor=(0.7,0.53), bbox_transform=ax1.transAxes)
ax1.set_title('Effect of Alpha Decay for Q Learning')
#plt.xlim([0, 2e6])
plt.tight_layout()
plt.savefig('Grid_world_5_5_states_QL_Effect of Alpha Decay for Q Learning_mean value_2.png',dpi=600)
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
plt.savefig('Grid_world_5_5_states_QL_Effect of Alpha Decay for Q Learning_max value.png',dpi=600)
    
#==============================================================================

#==============================================================================
#Policy iteration
#==============================================================================
#dims = np.linspace(5,35,31).astype(int)
#dims = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35])
dims = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,24,29,30,31,32,33,34,35])
times_pi = []
iterations_pi = []
for i in range(len(dims)):
    dim = dims[i];
    goal = [int(dim/2),int(dim/2)]
    trap = [[1,2]]
    wall = []
    # trap = [[1,1]]
    # wall = [[1,3],[3,1],[3,3]]
    reward_goal = 3
    reward_trap = -2.7
    reward_wall = -100
    P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)
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
ax1.plot(dims*dims, iterations_pi,label='Number of iterations', color="C0", lw=2)
ax1.set_ylabel('Number of iterations')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Number of states')
ax1.grid()
    
ax2 = ax1.twinx()  # this is the important function
ax2.plot(dims*dims, times_pi,label='Computation time', color="C1", lw=2)
ax2.set_ylabel('Computation time')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)

plt.title('Iteration and Time VS Number of States for PI')
#plt.xticks(n_component_range)
plt.tight_layout()
plt.savefig('Grid_world_Iteration and Time VS Number of States for PI 2.png',dpi=600)

#==============================================================================
#Value iteration
#==============================================================================
dims = np.linspace(5,35,31).astype(int)
times_vi = []
iterations_vi = []
for i in range(len(dims)):
    dim = dims[i];
    goal = [int(dim/2),int(dim/2)]
    trap = [[1,2]]
    wall = []
    # trap = [[1,1]]
    # wall = [[1,3],[3,1],[3,3]]
    reward_goal = 3
    reward_trap = -2.7
    reward_wall = -100
    P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)
    
    vi = mdp.ValueIteration(P, R, 0.9,epsilon=1e-5)
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
ax1.plot(dims*dims, iterations_vi,label='Number of iterations', color="C0", lw=2)
ax1.set_ylabel('Number of iterations')
#ax1.set_title("Value VS Iterations")
#ax1.set_title(title_1)
ax1.set_xlabel('Number of states')
ax1.grid()
    
ax2 = ax1.twinx()  # this is the important function
ax2.plot(dims*dims, times_vi,label='Computation time', color="C1", lw=2)
ax2.set_ylabel('Computation time')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)

plt.title('Iteration and Time VS Number of States for VI (Epsilon=1e-5)')
#plt.xticks(n_component_range)
plt.tight_layout()
plt.savefig('Grid_world_Iteration and Time VS Number of States for VI_Epsilcon=1e-5_2.png',dpi=600)

#==============================================================================
#==============================================================================================
#==============================================================================
#Grid world 35X35
#==============================================================================
#==============================================================================
#Policy iteration
#============================================================================== 
dim = 35
goal = [17,17]
trap = []
for i in range(7):
    trap.append([5+i*4,11])
    trap.append([5+i*4,23])
wall = []

for i in range(8):
    for j in range(13):
        wall.append([3+i*4,3+j])
        wall.append([3+i*4,19+j])
    wall.append([2+i*4,9])
    wall.append([4+i*4,9])
    wall.append([2+i*4,25])
    wall.append([4+i*4,25])
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -3.5
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

pi = mdp.PolicyIteration(P, R, 0.9,max_iter=150)
pi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = pi.run()
end_time = Time.time()
print('Execution time for PI is: ' + str(end_time-start_time))

policy = pi.policy
value = pi.V

title = 'Grid World with PI (Discount=0.9)'
draw_map(policy,[35,35],goal,trap,wall,'Grid_world_35_35_states_PI_map_policy_discount_0p9.png',1,250,title)
draw_map_2(policy,value,[35,35],goal,trap,wall,'Grid_world_35_35_states_PI_map_value_discount_0p9.png',1,100,[0.1,0.01,8,'.1f'],title,0.3)


name_1 = 'Grid_world_35_35_states_PI_Value VS Iterations for PI.png'
title_1 = 'Value VS Iterations for PI (1225 states)'
name_2 = 'Grid_world_35_35_states_PI_Error Convergence VS Iteration for PI.png'
title_2 = 'Error Convergence VS Iteration for PI (1225 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

#==============================================================================
#Value iteration
#============================================================================== 
dim = 35
goal = [17,17]
trap = []
for i in range(7):
    trap.append([5+i*4,11])
    trap.append([5+i*4,23])
wall = []

for i in range(8):
    for j in range(13):
        wall.append([3+i*4,3+j])
        wall.append([3+i*4,19+j])
    wall.append([2+i*4,9])
    wall.append([4+i*4,9])
    wall.append([2+i*4,25])
    wall.append([4+i*4,25])
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -3.5
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)

vi = mdp.ValueIteration(P, R, 0.9,epsilon=1e-20)
vi.setVerbose()
#pi.setSilent()
start_time = Time.time()
result = vi.run()
end_time = Time.time()
print('Execution time for VI is: ' + str(end_time-start_time))

policy = vi.policy
value = vi.V

title = 'Grid World with VI (Discount=0.9)'
draw_map(policy,[35,35],goal,trap,wall,'Grid_world_35_35_states_VI_map_policy_discount_0p9.png',1,250,title)
draw_map_2(policy,value,[35,35],goal,trap,wall,'Grid_world_35_35_states_VI_map_value_discount_0p9.png',0.7,100,[0.1,0.01,8,'.1f'],title,0.3)


name_1 = 'Grid_world_35_35_states_VI_Value VS Iterations for VI.png'
title_1 = 'Value VS Iterations for VI (1225 states)'
name_2 = 'Grid_world_35_35_states_VI_Error Convergence VS Iteration for VI.png'
title_2 = 'Error Convergence VS Iteration for VI (1225 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)

#==============================================================================
#Q learning
#============================================================================== 
dim = 35
goal = [17,17]
trap = []
for i in range(7):
    trap.append([5+i*4,11])
    trap.append([5+i*4,23])
wall = []

for i in range(8):
    for j in range(13):
        wall.append([3+i*4,3+j])
        wall.append([3+i*4,19+j])
    wall.append([2+i*4,9])
    wall.append([4+i*4,9])
    wall.append([2+i*4,25])
    wall.append([4+i*4,25])
# trap = [[1,1]]
# wall = [[1,3],[3,1],[3,3]]
reward_goal = 3
reward_trap = -3.5
reward_wall = -100

P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap,reward_wall)
np.random.seed(0) 
ql = mdp.QLearning(P, R, 0.9,n_iter=30000000,alpha=0.5, alpha_decay=0.99, alpha_min=0.05,epsilon=1, epsilon_min=0.2, epsilon_decay=0.99)
ql.setVerbose()
start_time = Time.time()    
result = ql.run()
end_time = Time.time()
print('Computation time is: '+ str(end_time-start_time))

policy = ql.policy
value = ql.V
title = 'Grid World with QL (Discount=0.9, iterations=3E7)'
draw_map(policy,[35,35],goal,trap,wall,'Grid_world_35_35_states_QL_map_policy_discount_0p9_3e7_iteration.png',1,2000,title)
draw_map_2(policy,value,[35,35],goal,trap,wall,'Grid_world_35_35_states_QL_map_value_discount_0p9_3e7_iteration.png',0.7,100,[0.1,0.01,8,'.1f'],title,0.3)


name_1 = 'Grid_world_35_35_states_QL_Value VS Iterations for QL_3e7_iteration.png'
title_1 = 'Value VS Iterations for QL (1225 states)'
name_2 = 'Grid_world_35_35_states_QL_Error Convergence VS Iteration for QL_3e7_iteration.png'
title_2 = 'Error Convergence VS Iteration for QL (1225 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)


filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_35_35_states_QL_result_3e7_iteration.txt'
save_variable(result,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_35_35_states_QL_3e7_iteration.txt'
save_variable(ql,filename)

test_r = load_variable(filename)
#==============================================================================
#10X10
#==============================================================================
#grid_game(dim=5,goal = [0,0],trap = [],wall = [])
#draw_map(policy,dim,goal = [0,0],trap = [],wall = [],file_name='test.png')

dim = 11
goal = [5,5]
trap = [[3,5],[7,5]]
wall = [[2,0],[2,1],[2,2],[2,8],[2,9],[2,10],[5,0],[5,1],[5,2],[5,8],[5,9],[5,10],[8,0],[8,1],[8,2],[8,8],[8,9],[8,10]]
reward_goal = 2
reward_trap = -10
P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap)

vi = mdp.ValueIteration(P, R, 0.9)
vi.run()
policy = vi.policy

draw_map(policy,[11,11],goal,trap,wall,'test.png')



np.random.seed(0) 
P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap)
ql = mdp.QLearning(P, R, 0.9,n_iter=10000000,alpha=0.1, alpha_decay=0.99, alpha_min=0.001,epsilon=1, epsilon_min=0.01, epsilon_decay=0.99)
ql.setVerbose()
start_time = Time.time()    
result = ql.run()
end_time = Time.time()
print('Computation time is: '+ str(end_time-start_time))
# plt.figure()
# plt.plot(ql.policy)
max_V = []
mean_V = []
reward = []
error = []
iteration = []
alpha = []
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

# plt.figure()
# plt.plot(max_V)
# plt.figure()
# plt.plot(mean_V)
policy_ql = ql.policy
draw_map(policy_ql,[11,11],goal,trap,wall,'Grid_world_121 states_QL_map.png')


name_1 = 'Grid_world_game_121 states_QL_Value VS Iterations for QL.png'
title_1 = 'Value VS Iterations for QL (121 states)'
name_2 = 'Grid_world_game_121 states_QL_Error Convergence VS Iteration for QL.png'
title_2 = 'Error Convergence VS Iteration for QL (121 states)'
convergence_plot(result,name_1,title_1,name_2,title_2)


filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_121_states_QL_result.txt'
save_variable(result,filename)

filename = 'E:\\GIT\\CS_7641_Machine_Learning\\HW4\\variable\\Grid_world_121_states_QL.txt'
save_variable(ql,filename)

test_r = load_variable(filename)

#==============================================================================
dim = 11
goal = [5,5]
trap = [[3,5],[7,5]]
wall = [[2,0],[2,1],[2,2],[2,8],[2,9],[2,10],[5,0],[5,1],[5,2],[5,8],[5,9],[5,10],[8,0],[8,1],[8,2],[8,8],[8,9],[8,10]]
reward_goal = 2
reward_trap = -10
P, R = grid_game(dim,goal,trap,wall,reward_goal,reward_trap)

vi = mdp.ValueIteration(P, R, 0.9)
vi.run()
policy = vi.policy

policy = vi.policy
value = vi.V

dim = [11,11]

x = np.linspace(0, dim[0] - 1, dim[0]) + 0.5
y = np.linspace(0, dim[0] - 1, dim[1]) + 0.5
X, Y = np.meshgrid(x, y)
zeros = np.zeros(dim)
    
arrow_up = np.zeros(dim)
arrow_right = np.zeros(dim)
arrow_down = np.zeros(dim)
arrow_left = np.zeros(dim)

for i in range(len(policy)):
    row = i//dim[0]
    col = i%dim[0]
    if (row != goal[0] or col != goal[1]) and ([row,col] not in wall) and ([row,col] not in trap):        
        if policy[i] == 0:
            arrow_up[row][col] = 0.3
        elif policy[i] == 1:
            arrow_right[row][col] = 0.3
        elif policy[i] == 2:
            arrow_down[row][col] = 0.3
        else:
            arrow_left[row][col] = 0.3
            
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
plt.scatter(goal[1]+0.5,goal[0]+0.5,marker='*',c='y',edgecolors='b',s=2000)

for i in range(len(wall)):
    plt.scatter(wall[i][1]+0.5,wall[i][0]+0.5,marker='s',c='black',s=2000)
    
for i in range(len(trap)):
    plt.scatter(trap[i][1]+0.5,trap[i][0]+0.5,marker='o',c='r',s=2000)


for i in range(len(value)):
    row = i//dim[0]
    col = i%dim[0]
    plt.text(col,row+0.07,str(format(value[i], '.10f')),fontsize=10)

plt.xlim([0, dim[0]])
plt.ylim([0, dim[1]])
#ax.set_yticklabels([])
#ax.set_xticklabels([])
ax.set_xticks(np.arange(0, dim[0]+1))
ax.set_yticks(np.arange(0, dim[1]+1)) 
plt.grid()
#plt.show()
plt.title('Grid World', fontsize = 30)
plt.tight_layout()


