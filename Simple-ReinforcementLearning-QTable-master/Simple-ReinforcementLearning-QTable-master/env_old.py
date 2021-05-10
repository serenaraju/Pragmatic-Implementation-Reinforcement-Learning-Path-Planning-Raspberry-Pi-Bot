import numpy as np
import pandas as pd

# Class defined to create our test environment
# Purpose: To define the structure of the environment and actions possible
# To get / set the status of the environment based on the Agent's movement 
class ENV:
    def __init__(self):
        self.state_cnt = 16 #used only for printing purpose
        self.state_row_cnt = 4
        self.action_cnt = 4
        self.actions= ['UP','DOWN','LEFT','RIGHT']#four possible actions in this environment
        
        self.states = np.full((4,4),'-')#individual states represented as - in the environment
        self.states[0][0]='A'#Agent
        self.states[1][2]='O'#holes
        self.states[2][3]='O'
        self.states[3][0]='O'
        self.states[3][3]='G'#Goal or destination to reach
        
        self.A_in_row = 0 # Agent's init position at [0,0]
        self.A_in_col = 0
    
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
        if(((stateR == 1) and (stateC == 2)) or 
           ((stateR == 2) and (stateC == 3)) or 
           ((stateR == 3) and (stateC == 0)) or
           ((stateR == 3) and (stateC == 3))):
           done = True
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
            self.states[prev_A_in_row][prev_A_in_col]='-'
            self.states[self.A_in_row][self.A_in_col]='A'
        else:
            done = True
            if ((self.A_in_row == 3) and (self.A_in_col == 3)):#Target reached. Add reward = 1
                self.states[prev_A_in_row][prev_A_in_col]='-'
                self.states[self.A_in_row][self.A_in_col]='A'
                print('Target reached')
                R = 1
            else:
                print('fallen in the hole')
        
        next_state = (self.A_in_row,self.A_in_col)
        return(next_state,R,done)
