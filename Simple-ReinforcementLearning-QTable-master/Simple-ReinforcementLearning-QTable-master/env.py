import numpy as np
import pandas as pd
from deep_qlearning_path_finding import *

NOT_USE = -1.0
OBSTACLE = 0.0
EMPTY = 1.0
TARGET = 0.75
START = 0.5

# Class defined to create our test environment
# Purpose: To define the structure of the environment and actions possible
# To get / set the status of the environment based on the Agent's movement 
class ENV:
    def __init__(self, storage_value, row_num, col_num):
        self.state_cnt = 16 #used only for printing purpose
        self.state_row_cnt = row_num
        self.state_col_cnt = col_num
        self.action_cnt = 4
        self.actions= ['UP','DOWN','LEFT','RIGHT']#four possible actions in this environment
        
        self.states = np.array(storage_value) #individual states represented as - in the environment
        '''
        self.states[0][1]='A'#Agent
        self.states[0][2]='O'#holes
        self.states[2][3]='O'
        self.states[3][0]='O'
        self.states[3][3]='G'#Goal or destination to reach
        '''
        self.A = np.where(self.states == 0.5)
        self.A = list(zip(self.A[0], self.A[1]))
        self.A_in_row = self.A[0][0] # Agent's init position at [0,0]
        self.A_in_col = self.A[0][1]
        self.o_and_t_list = []
        self.o_and_t_list.append(np.where)
        obstacle = np.where(self.states == 0.0)
        goal = np.where(self.states == 0.75)

        self.condit = list(zip(obstacle[0],obstacle[1]))
        goal = list(zip(goal[0],goal[1]))    
        goal = (goal[0][0],goal[0][1])  

        self.condit.append(goal)
        self.condit = np.array(self.condit)
        
        
    
    # used while creating a Q class to know what all possible actions are offered by this environment
    def getActionItems(self): 
        return (self.action_cnt, self.actions)

    #used mainly to test the agent that was learnt already
    def getAgentPosition(self):
        return (self.A_in_row, self.A_in_col)
    
    #just for debugging purpose
    def display_env(self):
        print("Number of states : {}".format(self.state_cnt))
        print("Number of actions : {}".format(self.action_cnt))
        print("Action list : {}".format(self.actions))
        print("Agent's current position :[{},{}]".format(self.A_in_row,self.A_in_col))
        print("Environment dump : \n{}\n".format(pd.DataFrame(self.states).to_string(index=False,header=False)))
    
    #to check if the agent has reached the destination or fell into any of the three holes in the environment    
    def isDone(self,stateR,stateC):
        done = False
        for _state_ in self.condit:
            if (_state_[0] == stateR) and (_state_[1] == stateC):
                done = True
        '''

        if(((stateR ==0) and (stateC == 2)) or 
           ((stateR == 2) and (stateC == 3)) or 
           ((stateR == 3) and (stateC == 0)) or
           ((stateR == 3) and (stateC == 3))):
           done = True
        '''
        return done
    
    #used for display purpose    
    def render(self):
        return ("{}\n".format(pd.DataFrame(self.states).to_string(index=False,header=False)),(self.A_in_row,self.A_in_col))      
     
    #function: step taken by the agent. One of the four actions would be input to this function.
    #Function would update the environment's state based on the input action and returns the next state,the reward received by the agent and 
    #status info if the Agent has reached the destination/ fallen in the hole ('done' variable = true)    
    def step(self,action): 
        done = False
        R = 0
        prev_A_in_row = self.A_in_row
        prev_A_in_col = self.A_in_col
        
        if (action == 'UP'):
            self.A_in_row = max(self.A_in_row -1,0)
        if (action == 'DOWN'):
            self.A_in_row = min(self.A_in_row +1,self.state_row_cnt-1)
        if (action == 'LEFT'):
            self.A_in_col = max(self.A_in_col -1,0)
        if (action == 'RIGHT'):
            self.A_in_col = min(self.A_in_col +1,self.action_cnt-1)  
        
        if (self.isDone( self.A_in_row, self.A_in_col) == False):
            self.states[prev_A_in_row][prev_A_in_col] = 1.0
            self.states[self.A_in_row][self.A_in_col] = 0.5
        else:
            done = True
            if ((self.A_in_row == 3) and (self.A_in_col == 3)):#Target reached. Add reward = 1
                self.states[prev_A_in_row][prev_A_in_col]= 1.0
                self.states[self.A_in_row][self.A_in_col]= 0.5
                print('Target reached')
                R = 1
            else:
                print('fallen in the hole')
        
        next_state = (self.A_in_row,self.A_in_col)
        updateCanvas(self.states, self.state_row_cnt,self.state_col_cnt)
        return(next_state,R,done)
