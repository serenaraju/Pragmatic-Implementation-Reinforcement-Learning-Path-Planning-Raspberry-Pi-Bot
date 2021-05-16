import urllib
import json
import requests

import RPi.GPIO as GPIO          
from time import sleep

in1 =6 
in2 =5 
enA =26 
in3 =23
in4 =24
enB =25
temp1=1


GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(enA,GPIO.OUT)
GPIO.setup(enB,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)
p1=GPIO.PWM(enA,100)
p2=GPIO.PWM(enB,100)


p1.start(25)
p2.start(25)

class RPI:
    def __init__(self):
        self.url = ' https://jsonblob.com/api/jsonBlob/4f8b1f70-a3f3-11eb-b812-2d2dd835b230'
        self.arr = [1]

    def extract(self):
     
        headers = {"Content-Type": "application/json", "Access-Control-Allow-Origin": "https:://jsonblob.com"}
        response = requests.get(self.url, headers=headers)
        response = response.json()
        self.arr.extend(response['array'])
        print(self.arr)

        return self.arr

    def motor_right(self):
        p1.ChangeDutyCycle(25)
        p2.ChangeDutyCycle(75)
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.HIGH)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        print("right turn")
        sleep(0.55)
        print("stop")
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.LOW)
        print("right")

    def motor_left(self):
        p1.ChangeDutyCycle(75)
        p2.ChangeDutyCycle(25)
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.HIGH)
        print("left turn")
        sleep(0.55)
        print("stop")
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.LOW)
        print("left")
    
    def motor_for(self):
        p1.ChangeDutyCycle(25)
        p2.ChangeDutyCycle(25)
        #print("forward")
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.HIGH)
        GPIO.output(in4,GPIO.LOW)
        #temp1=1
        sleep(1.3)
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.LOW)
        print("forward")
    
    def motor_rev(self):
        
        print("reverse")

    def start(self):
        self.extract()

        for c in range(0,len(self.arr)-1):
            diff = -( int(self.arr[c+1]) - int(self.arr[c] )) #diff between current 
            print(diff)
            if diff  == 0:
                self.motor_for()
            elif diff  == -1 or diff  == 3:
                self.motor_right()
                self.motor_for()
            elif diff  == 1 or diff  == -3:
                self.motor_left()
                self.motor_for()
            else:
                self.motor_left()
                self.motor_left()
                self.motor_for()
                

path_plan = RPI()
#path_plan.motor_for()
path_plan.start()
GPIO.cleanup()

            
                

    






