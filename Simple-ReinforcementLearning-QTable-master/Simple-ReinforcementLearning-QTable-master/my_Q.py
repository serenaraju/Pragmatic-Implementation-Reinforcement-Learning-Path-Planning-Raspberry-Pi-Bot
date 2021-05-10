import env
import numpy as np
import pandas as pd
from tkinter import *
import time

##update Agent's current position info as text field and the environment's state changes in the display
def display_environment(env1):
    global txt,pos,wdw
    txt1,txt2 = env1.render()
    
    txt.delete("1.0", "end")
    txt.insert(END,txt1)

    pos.set("Agent's position : "+str(txt2))
    wdw.update()
    time.sleep(0.25)

#Main class that creates Q table, implements bellmans equation to update the Q table based on the
#Agent's action. Agent learning happens here. 
class Q:
    #Q table creation
    def __init__(self,gamma=0.9,alpha=0.1,epsilon=0.1,num_episodes=500,storage_value = [], row_num = 5, col_num = 5):
        self.col_list = list(range(0,col_num))
        self.row_list = list(range(0,row_num))
        self.q_table = pd.DataFrame(0,index=pd.MultiIndex.from_product\
            ([self.row_list,self.col_list]),columns=['UP','DOWN','LEFT','RIGHT'])
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.env1=env.ENV(storage_value, row_num, col_num)
        self.action_count,self.actions = self.env1.getActionItems()
    
    #reseting environment is easier with this OO approach. Just create a new environment object 
    #and returns the agents position as (0,0). I didnt want to hardcode that position. Rather get
    #that from the env object itself    
    def reset_environment(self, storage_value, row_num, col_num):
        #just creating another instance of environment variable
        del(self.env1) 
        self.env1=env.ENV(storage_value, row_num, col_num)
        return self.env1.getAgentPosition()
    
    # main learnig algorithm. loops for 'number of episodes'
    # on a new environment created per episode, agent tries to make many movements (one of the four actions possible)
    # either randomly or based on the Q values determined during the update process.
    def learn(self, storage_value, row_num, col_num):
        for episode_cnt in range(self.num_episodes):
            print('episode: {}'.format(episode_cnt),end='\t')
            state = self.reset_environment(storage_value, row_num, col_num)   
            #display_environment(self.env1)
            done = self.env1.isDone(state[0],state[1])
            #self.env1.display_env() 

            while not done:
                #give 10% chance to choose an action randomly and also in a state where
                #all actions have Q values == 0 (init state), choose action randomly
                if (np.random.uniform()<self.epsilon) or ((self.q_table.loc[state,:] == 0).all()):
                    action = np.random.choice(self.actions)
                #90% of the chance it picks up the action corresponding to the max Q value
                else:
                    #action = self.q_table.loc[state,:].idxmax()
                    #idxmax throws 'argmax not allowed' error, as a workaround I tried this following line. it works :)
                    action=self.q_table.loc[state,:].index[self.q_table.loc[state,:].values.argmax()]

                next_state,Reward,done = self.env1.step(action)
                current_Q = self.q_table.loc[state,action]
                next_Q = self.q_table.loc[next_state,:].max()
                
                #bellman's equations and its discounted rewards in future
                self.q_table.loc[state,action] += self.alpha * (Reward + self.gamma * next_Q - current_Q)
                
                state = next_state
                
                #display_environment(self.env1)
        print('\n Final Q table: \n {}'.format(self.q_table))
        #save it for testing from next time
        self.q_table.to_pickle('q_table.pkl') 
    
    
txt = None
pos = None
wdw = None
 
#Interface class for test_env.py for training init on a button press 
class Q_train:
    def __init__(self,wdw1):

        global wdw,txt, pos
        
        wdw = wdw1

    
    def train(self, storage_value, row_num, col_num):
        q = Q(gamma=0.9,alpha=0.1,epsilon=0.2,num_episodes=150,storage_value = storage_value, row_num = row_num, col_num = col_num)
        q.learn(storage_value, row_num, col_num)

##Interface class for test_env.py for testing the agent on a button press           
class Q_test:
    def __init__(self,wdw1,text_box1,position_info):

        global wdw,txt, pos
        
        wdw = wdw1
        txt = text_box1
        pos = position_info
        q_table = None
    
    def test(self):
    
        try:        
            q_table = pd.read_pickle('q_table.pkl')#read q table from the database saved
        except:
            txt.delete("1.0", "end")
            txt.insert(END,'Try to train the Agent before testing it\n')
            return
            
        if (q_table is not None):
            env1 = env.ENV()
            #display_environment(env1)#display the init state of the environment
            done = False
            while not done:#until it reaches the destination
                state = env1.getAgentPosition()#get current position
                action=q_table.loc[state,:].index[q_table.loc[state,:].values.argmax()]#fetch the next best possible action from the Q table learnt
                _,_,done = env1.step(action)#execute the action
                #display_environment(env1)#display the update 
