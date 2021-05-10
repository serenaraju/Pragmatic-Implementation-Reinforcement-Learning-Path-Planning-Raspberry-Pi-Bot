from tkinter import *
import time
import env
from my_Q import * 

##update the display in tkinter window
def update(env1):
    txt1,txt2 = env1.render()
    
    text_box1.delete("1.0", "end")
    text_box1.insert(END,txt1)

    position_info.set("Agent's position : "+str(txt2))
    wdw.update()

## button click handling to test the environment by manually directing the agent to the destination
def handle_click(event):
    env1 = env.ENV()
    update(env1)
    time.sleep(1)
    
    env1.step('RIGHT')
    update(env1)    
    time.sleep(1)
    
    env1.step('RIGHT')
    update(env1)   
    time.sleep(1)
    
    env1.step('LEFT')
    update(env1)    
    time.sleep(1)   
    
    env1.step('DOWN')
    update(env1)
    time.sleep(1)
    
    env1.step('DOWN')
    update(env1)    
    time.sleep(1)  
    
    env1.step('DOWN')
    update(env1)
    time.sleep(1)
    
    env1.step('DOWN')
    update(env1)
    time.sleep(1)
    
    env1.step('RIGHT')
    update(env1)
    time.sleep(1)
    
    env1.step('RIGHT')
    update(env1)

## button click handling to initiate the Agent's learning process - Q learning trigger
def handle_Qlearn_click(event):
    q_tr = Q_train(wdw,text_box1,position_info)
    q_tr.train() 

## button click handling to test if the Agent has learnt to navigate through this environment
def handle_QTest_click(event):
    q_test = Q_test(wdw,text_box1,position_info)
    q_test.test()  

#create a new window to display the environment and the agent's status    
wdw = Tk()
wdw.title('My Q test ground')
wdw.resizable(False, False) 

#to display the environment and its updates after any movement of the agent
text_box1 = Text(height=10,width=20,padx=50, pady=50,bg='#E0FFFF',fg='blue')
text_box1.grid(row=0, column=0,rowspan =3,sticky="ew", padx=10, pady=10)

#to display agent's current position
position_info = StringVar()
label = Label(master=wdw, textvariable=position_info,font=("Helvetica", 11),fg='blue')#.pack()
label.grid(row=2, column=0)

#frame containing control button 'Test Environment'
frame1 = Frame(master=wdw,borderwidth=1)
frame1.grid(row=0, column=1, padx=5, pady=5)

#to test the environment if this works normally.
btn = Button(master=frame1,height=2,width=15,text="Test Environment")
btn.pack()
btn.bind("<Button-1>", handle_click)

Label(master=frame1, text="Agent manually\ndirected to the destination",font=("Helvetica", 8)).pack()

#frame containing control button 'Train Agent'
frame2 = Frame(master=wdw,borderwidth=1)
frame2.grid(row=1, column=1, padx=5, pady=5)

#Triggers the learning procedure
qLearn_btn = Button(master=frame2,height=2,width=15, text="Train Agent")
qLearn_btn.pack()
qLearn_btn.bind("<Button-1>", handle_Qlearn_click)

Label(master=frame2, text="Agent learns to go to the\ndestination automatically using Q tables",font=("Helvetica", 8)).pack()

#frame containing control button 'Test Agent'
frame3 = Frame(master=wdw,borderwidth=1)
frame3.grid(row=2, column=1, padx=5, pady=5)

#Tests the agent with the pre-learnt Q table information
qTest_btn = Button(master=frame3,height=2,width=15,bg='orange',text="Test Agent")
qTest_btn.pack()
qTest_btn.bind("<Button-1>", handle_QTest_click)

Label(master=frame3, text="Test your Agent that has\nlearnt how to go to the destination",font=("Helvetica", 8)).pack()

wdw.mainloop()