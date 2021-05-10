import datetime
import itertools
import random
from collections import deque
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

from direction import Loco

#from my_Q import * 

import numpy as np
import paho.mqtt.client as mqtt
from keras.layers import Dense
from keras.models import Sequential
from math import e as exp
#from variable import *
np.random.seed(1)

#******************* GLOBAL VARIABLES *******************#

client = mqtt.Client()
root = Tk()
#root.withdraw()
DEFAULT_SIZE_ROW = 5
DEFAULT_SIZE_COLUMN = 5
MAX_SIZE = 10
GRID_SIZE = 30
PADDING = 5

row_num = DEFAULT_SIZE_ROW
col_num = DEFAULT_SIZE_COLUMN

canvas_list = []
storage_value = []

NOT_USE = -1.0
OBSTACLE = 0.0
EMPTY = 1.0
TARGET = 0.75
START = 0.5

COLOR_DICT = {
    NOT_USE: "black",
    OBSTACLE: "VioletRed1",
    EMPTY: "grey2",
    TARGET: "turquoise2",
    START: "yellow"
}

target_count = 0
start_count = 0

MAX_EPISODES = 4000
LOAD_TRAINED_MODEL_PATH = ""
SAVE_FILE_PATH = "saves/self_drive_master.h5"

DEBUG = False
EPSILON_REDUCE = True
RANDOM_MODE = False

ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3

STATE_START = 'start'
STATE_WIN = 'win'
STATE_LOSE = 'lose'
STATE_BLOCKED = 'blocked'
STATE_VALID = 'valid'
STATE_INVALID = 'invalid'

# Hyperparameter
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MEMORY_LEN = 1000
DISCOUNT_RATE = 0.95
BATCH_SIZE = 50


best_path = ""

detected_collision = False

#*********************************************************#
#import env
import numpy as np
import pandas as pd

import time

import numpy as np
import pandas as pd
from qlearn import *

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
        self.state_cnt = row_num*col_num #used only for printing purpose
        self.state_row_cnt = row_num
        self.state_col_cnt = col_num
        self.action_cnt = 4
        self.actions= ['UP','DOWN','LEFT','RIGHT']#four possible actions in this environment
        
        self.states = np.array(storage_value) #individual states represented as - in the environment
        
        self.A = np.where(self.states == 0.5)
        self.A = list(zip(self.A[0], self.A[1]))
        self.A_in_row = self.A[0][0] # Agent's init position at [0,0]
        self.A_in_col = self.A[0][1]
        self.o_and_t_list = []
        self.o_and_t_list.append(np.where)
        obstacle = np.where(self.states == 0.0)
        self.goal = np.where(self.states == 0.75)

        self.condit = list(zip(obstacle[0],obstacle[1]))
        self.goal = list(zip(self.goal[0],self.goal[1]))    
        self.goal = (self.goal[0][0],self.goal[0][1])  

        self.condit.append(self.goal)
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
        print(action)

        if (action == 'UP'):
            self.A_in_row = max(self.A_in_row -1,0)
        if (action == 'DOWN'):
            self.A_in_row = min(self.A_in_row +1,self.state_row_cnt-1)
        if (action == 'LEFT'):
            self.A_in_col = max(self.A_in_col -1,0)
        if (action == 'RIGHT'):
            self.A_in_col = min(self.A_in_col +1,self.state_col_cnt-1)  
        
        if (self.isDone( self.A_in_row, self.A_in_col) == False):
            self.states[prev_A_in_row][prev_A_in_col] = 1.0
            self.states[self.A_in_row][self.A_in_col] = 0.5
        else:
            done = True
            if ((self.A_in_row == self.goal[0]) and (self.A_in_col == self.goal[1])):#Target reached. Add reward = 1
                self.states[prev_A_in_row][prev_A_in_col]= 1.0
                self.states[self.A_in_row][self.A_in_col]= 0.5
                print('Target reached')
                R = 1
            else:
                print('fallen in the hole')
        
        next_state = (self.A_in_row,self.A_in_col)
        updateCanvas(self.states, self.state_row_cnt,self.state_col_cnt)
        return(next_state,R,done)


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
        if (min(row_num, col_num)==3):
            self.num_episodes = 300
        else:
            self.num_episodes = int(0.000429454*(exp**(3.61239*max(row_num,col_num))))
        self.env1=ENV(storage_value, row_num, col_num)
        self.action_count,self.actions = self.env1.getActionItems()
    
    #reseting environment is easier with this OO approach. Just create a new environment object 
    #and returns the agents position as (0,0). I didnt want to hardcode that position. Rather get
    #that from the env object itself    
    def reset_environment(self, storage_value, row_num, col_num):
        #just creating another instance of environment variable
        del(self.env1) 
        self.env1=ENV(storage_value, row_num, col_num)
        return self.env1.getAgentPosition()
    
    # main learnig algorithm. loops for 'number of episodes'
    # on a new environment created per episode, agent tries to make many movements (one of the four actions possible)
    # either randomly or based on the Q values determined during the update process.
    def learn(self, storage_value, row_num, col_num):
        start_time = datetime.datetime.now()

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
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        print("Time taken: {}".format(t))
        self.q_table.to_pickle('q_table.pkl') 
        messagebox.showinfo("Result for Q Reinforcement Learning",
                                "Have 100% win rate at episode {:d}. \nWin {:d}/{:d} game during {}".format(self.num_episodes,10,10,t))
                                        
    
    
txt = None
pos = None
wdw = None
 
#Interface class for test_env.py for training init on a button press 
class Q_train:
    def __init__(self,wdw1):
        self.w = wdw1
    def train(self, storage_value, row_num, col_num):
        q = Q(gamma=0.9,alpha=0.1,epsilon=0.1,num_episodes=100000,storage_value = storage_value, row_num = row_num, col_num = col_num)
        q.learn(storage_value, row_num, col_num)

##Interface class for test_env.py for testing the agent on a button press           
class Q_test:
    def __init__(self,wdw1):

    
        q_table = None
    
    def test(self, storage_value,row_n0, col_n0):
    
        try:        
            q_table = pd.read_pickle('q_table.pkl')#read q table from the database saved
        except:
            txt.delete("1.0", "end")
            txt.insert(END,'Try to train the Agent before testing it\n')
            return
            
        if (q_table is not None):
            env1 = ENV(storage_value,row_n0, col_n0)
            #display_environment(env1)#display the init state of the environment
            done = False
            action_list = []
            while not done:#until it reaches the destination
                state = env1.getAgentPosition()#get current position
                action=q_table.loc[state,:].index[q_table.loc[state,:].values.argmax()]#fetch the next best possible action from the Q table learnt
                action_list.append(action)
                _,_,done = env1.step(action)#execute the action
                #display_environment(env1)#display the update 
            #print(action_list)
            best_path = []
            for action in action_list:
                if (action == 'UP'):
                    best_path.append(1)
                if (action == 'DOWN'):
                    best_path.append(3)
                if (action == 'LEFT'):
                    best_path.append(0)
                if (action == 'RIGHT'):
                    best_path.append(2)  
            print(best_path)
            dir_json = {"array" : best_path}
            dir_obj = Loco()
            dir_obj.write(dir_json)
            

        
                

class Environment:
    def __init__(self, row_x, col_y):
        self.row_number = row_x
        self.col_number = col_y
        self.action_size = 4  # 1:UP,2:RIGHT,3:LEFT,4:RIGHT
        self.observation_space = row_x * col_y
        self._map = self._create_map(row_x, col_y)
        self.ready = False

    def _create_map(self, row_number, col_number):
        map = np.ones(shape=(row_number, col_number))
        return map

    def set_target(self, row_x, col_y):
        self.target = (row_x, col_y)
        self._map[row_x, col_y] = TARGET

    def set_collision(self, row_x, col_y):
        self._map[row_x, col_y] = OBSTACLE

    def set_start_point(self, row_x, col_y):
        self.start = (row_x, col_y)
        self.current_state = (row_x, col_y, STATE_START)
        self._map[row_x, col_y] = START

    def set_empty_point(self, row_x, col_y):
        self._map[row_x, col_y] = EMPTY

    def create_random_environment(self):
        self.ready = True
        self._map = self._create_map(self.row_number, self.col_number)
        # There are n number of object included: 1 start point, 1 target and n -2 colision.
        n = min(self.row_number, self.col_number) + 1
        count = 0
        random_set = np.empty(shape=(n, 2))
        while count < n:
            x = np.random.randint(self.row_number)
            y = np.random.randint(self.col_number)
            if ([x, y] in random_set.tolist()):
                continue
            random_set[count, 0] = x
            random_set[count, 1] = y
            count += 1
        self.set_start_point(int(random_set[0, 0]), int(random_set[0, 1]))
        self.set_target(int(random_set[1, 0]), int(random_set[1, 1]))
        for i in range(2, n):
            self.set_collision(int(random_set[i, 0]), int(random_set[i, 1]))

    def reset(self):
        if RANDOM_MODE:
            self.create_random_environment()
        row_x, col_y = self.start
        self.current_state = (row_x, col_y, STATE_START)
        self.visited = set()
        self.min_reward = -0.5 * self.observation_space
        self.free_cells = [(r, c) for r in range(self.row_number) for c in range(self.col_number) if
                           self._map[r, c] == 1.0]
        # self.free_cells.remove(self.target)
        self.total_reward = 0
        self.map = np.copy(self._map)
        for row, col in itertools.product(range(self.row_number), range(self.col_number)):
            storage_value[row, col] = self._map[row, col]
        updateCanvas()

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.current_state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.map.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)

        if row > 0 and self.map[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self.map[row + 1, col] == 0.0:
            actions.remove(3)

        if col > 0 and self.map[row, col - 1] == 0.0:
            actions.remove(0)
        if col < ncols - 1 and self.map[row, col + 1] == 0.0:
            actions.remove(2)

        # print("row = {},col = {}, action= {}".format(row, col, actions))
        return actions

    def update_state(self, action):
        nrow, ncol, nmode = current_row, current_col, mode = self.current_state

        if self.map[current_row, current_col] > 0.0:
            self.visited.add((current_row, current_col))

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = STATE_BLOCKED
        elif action in valid_actions:
            nmode = STATE_VALID
            storage_value[nrow, ncol] = EMPTY
            if action == ACTION_LEFT:
                storage_value[nrow, ncol - 1] = START
                ncol -= 1
            elif action == ACTION_UP:
                storage_value[nrow - 1, ncol] = START
                nrow -= 1
            if action == ACTION_RIGHT:
                storage_value[nrow, ncol + 1] = START
                ncol += 1
            elif action == ACTION_DOWN:
                storage_value[nrow + 1, ncol] = START
                nrow += 1
        else:
            # invalid action
            nmode = STATE_INVALID

        self.current_state = (nrow, ncol, nmode)

    # Action define:
    # 0: LEFT
    # 1: UP
    # 2: RIGHT
    # 3: DOWN
    def act(self, act):
        self.update_state(act)
        updateCanvas()
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        current_state = self.observe()
        return current_state, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.map)
        nrows, ncols = self.map.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        row, col, valid = self.current_state
        canvas[row, col] = 0.5  # set current position
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return STATE_LOSE
        current_row, current_col, mode = self.current_state
        target_row, target_col = self.target
        if current_row == target_row and current_col == target_col:
            return STATE_WIN
        return STATE_VALID

    # reinforcement learning reward function
    # return 1 if find the target
    # return min_reward-1 if obstacles block the way
    # return -0.75 if state invalid
    # return -0.25 if visited
    # return -0.05 for one possible state
    def get_reward(self):
        current_row, current_col, mode = self.current_state
        target_row, target_col = self.target
        if current_row == target_row and current_col == target_col:
            return 1.0
        if mode == STATE_BLOCKED:
            return self.min_reward - 1
        if mode == STATE_INVALID:
            return -0.75
        if (current_row, current_col) in self.visited:
            return -0.25
        if mode == STATE_VALID:
            return -0.05


# Format time string to minutes
def format_time(seconds):
    m = seconds / 60.0
    return "%.2f minutes" % (m,)


def check_best_path(memory):
    global best_path
    global detected_collision
    min = 100
    index = -1
    for i in range(len(memory)):
        if (len(memory[i]) < min):
            min = len(memory[i])
            index = i
    best_path = ''.join(str(x) for x in memory[index])
    btnSendQuery.configure(state=NORMAL)
    if detected_collision:
        send_message("Path" + best_path)
        detected_collision = False
    print(best_path)

def buttonClick():
    global target_count
    global start_count
    row_num = int(e1.get())
    col_num = int(e2.get())
    for row, col in itertools.product(range(10), range(10)):
        target_count = 0
        start_count = 0
        storage_value[row, col] = EMPTY
        canvas.itemconfigure(canvas_list[row, col], fill="grey90")
    for row, col in itertools.product(range(row_num), range(col_num)):
        canvas.itemconfigure(canvas_list[row, col], fill=COLOR_DICT[EMPTY])
        storage_value[row, col] = EMPTY


def getPosition(widget_num):
    row = np.math.floor(widget_num / MAX_SIZE)
    col = widget_num % MAX_SIZE - 1
    return row, col

'''
def printMatrx():
    row_num = int(e1.get())
    col_num = int(e2.get())
    print(storage_value[0:row_num, 0:col_num])
'''

def updateCanvas(storage_value, row, col):
    row_num = int(row)
    col_num = int(col)

    for row, col in itertools.product(range(row_num), range(col_num)):
        state = storage_value[row, col]
        canvas.itemconfigure(canvas_list[row, col], fill=COLOR_DICT[state])
    root.update()


def onObjectClick(event):
    global start_count
    global target_count
    global row_n0
    global col_n0
    widget_num = event.widget.find_closest(event.x, event.y)[0]
    row, col = getPosition(widget_num)
    row_num = int(e1.get())
    col_num = int(e2.get())
    row_n0 = row_num
    col_n0 = col_num
    print(row_num,col_num)
    if (row < row_num and col < col_num):
        current_value = storage_value[row, col]
        valid_value = np.array([EMPTY, START, TARGET, OBSTACLE], dtype=np.float)
        # print("Start count = {}, target_count = {}".format(start_count, target_count))
        if (start_count == 1):
            valid_value = np.delete(valid_value, np.argwhere(valid_value == START))
        if (target_count == 1):
            valid_value = np.delete(valid_value, np.argwhere(valid_value == TARGET))
        if (current_value == START or current_value == TARGET):
            index = -1
        else:
            index = np.where(valid_value == current_value)[0][0]
        if (index != (len(valid_value) - 1)):
            next_value = valid_value[index + 1]
        else:
            next_value = valid_value[0]
        if (next_value == START):
            start_count += 1
        elif (next_value == TARGET):
            target_count += 1
        if (current_value == START):
            start_count = 0
        elif (current_value == TARGET):
            target_count = 0
        storage_value[row, col] = next_value
        # print("current= {},next = {},index = {}, valid = {}".format(current_value, next_value, index, valid_value))
        canvas.itemconfigure(canvas_list[row, col], fill=COLOR_DICT[next_value])


def create_environment():
    if (start_count == 0 or target_count == 0):
        messagebox.showinfo("Error", "Please set START and TARGET point")
        return None
    row_num = int(e1.get())
    col_num = int(e2.get())
    env = Environment(row_num, col_num)
    for row, col in itertools.product(range(row_num), range(col_num)):
        if storage_value[row, col] == START:
            env.set_start_point(row, col)
        elif storage_value[row, col] == TARGET:
            env.set_target(row, col)
        elif storage_value[row, col] == OBSTACLE:
            env.set_collision(row, col)
    return env


def handle_Qlearn_click():
    q_tr = Q_train(root)
    q_tr.train(storage_value,row_n0, col_n0) 

def handle_QTest_click():
    q_test = Q_test(root)
    q_test.test(storage_value,row_n0, col_n0)  


if __name__ == "__main__":
    root.title("Q LEARNING")

    # start_mqtt_subscribe(client)
    # client.loop_start()
    w = 350  # width for the Tk root
    h = 600  # height for the Tk root

    # get screen width and height
    ws = root.winfo_screenwidth()  # width of the screen
    hs = root.winfo_screenheight()  # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    ttk.Style().configure('green/black.TButton', foreground='green', background='black')
    # set the dimensions of the screen
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    frame1 = Frame(root)
    frame1.pack()
    frame2 = Frame(root)
    frame2.pack()
    frame3 = Frame(root)
    frame3.pack()
    frame4 = Frame(root)
    frame4.pack()
    array = []
    # for i in range(5):
    #     Label(frame1, text="{}".format(i)).grid(row=0, column=i + 1)
    #     Label(frame1, text="{}".format(i)).grid(row=i + 1, column=0)
    Label(frame1, text="ROWS").grid(row=0, column=0,sticky=W)
    Label(frame1, text="COLUMNS").grid(row=0,column=3, sticky=W)
    e1 = Entry(frame1, width="3")
    e2 = Entry(frame1, width="3")
    e1.insert(0, DEFAULT_SIZE_ROW)
    e2.insert(0, DEFAULT_SIZE_COLUMN)
    e1.grid(row=0, column=1)
    e2.grid(row=0, column=5)
    Button(frame1, text="GENERATE MAZE", command=buttonClick) \
        .grid(row=2, column=2, columnspan=2, rowspan=2, padx=10, pady=10)
    #Label(frame1, text="START", bg=COLOR_DICT[START]).grid(row=2, column=0, sticky=W + E + N + S)
    Label(frame1, text="START").grid(row=4, column=1, sticky=W )
    Label(frame1,  bg=COLOR_DICT[START]).grid(row=4, column=2,columnspan=2, rowspan=2 ,sticky=W + E + N + S)
    
    #Label(frame1, text="TARGET", bg=COLOR_DICT[TARGET]).grid(row=2, column=1, sticky=W + E + N + S)
    Label(frame1, text="GOAL").grid(row=6, column=1, sticky=W )
    Label(frame1,  bg=COLOR_DICT[TARGET]).grid(row=6, column=2,columnspan=2, rowspan=2, sticky=W + E + N + S)

    #Label(frame1, text="OBSTACLES", bg=COLOR_DICT[OBSTACLE]).grid(row=2, column=2, sticky=W + E + N + S)
    Label(frame1, text="OBSTACLES").grid(row=8, column=1, sticky=W )
    Label(frame1,  bg=COLOR_DICT[OBSTACLE]).grid(row=8, column=2,columnspan=2, rowspan=2, sticky=W + E + N + S)

    canvas = Canvas(frame2, width=360, height=360)
    Button(frame3, text="TRAIN Q LEARNING",bg='red2', command=handle_Qlearn_click).pack()
    Button(frame4, text="TEST Q LEARNING",command=handle_QTest_click).pack(side=LEFT)
    #btnSendQuery = Button(frame4, text="Start run EV3", command=sendEV3Query, state=DISABLED)
    #btnSendQuery.pack(side=RIGHT)

    for row, col in itertools.product(range(10), range(10)):
        x1 = col * (GRID_SIZE + PADDING)
        y1 = row * (GRID_SIZE + PADDING)
        x2 = x1 + GRID_SIZE
        y2 = y1 + GRID_SIZE
        canvas_item = canvas.create_rectangle(x1, y1, x2, y2, fill='gray1', width=0)
        canvas_list.append(canvas_item)
        canvas.tag_bind(canvas_item, '<ButtonPress-1>', onObjectClick)
        storage_value.append(NOT_USE)
    canvas_list = np.array(canvas_list).reshape(10, 10)
    storage_value = np.array(storage_value, dtype=np.float).reshape(10, 10)
    canvas.pack()
    root.call('wm', 'attributes', '.', '-topmost', '2')
    root.mainloop()

    # ==================================
    # Define graphic user interface - END
    # ==================================
