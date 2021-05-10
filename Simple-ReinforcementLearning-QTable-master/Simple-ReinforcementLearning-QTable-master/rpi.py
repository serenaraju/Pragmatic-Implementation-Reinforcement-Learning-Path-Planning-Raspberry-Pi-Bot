import urllib
import json
import requests

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
        print("right")

    def motor_left(self):
        print("left")
    
    def motor_for(self):
        print("forward")
    
    def motor_rev(self):
        print("reverse")

    def start(self):
        self.extract()

        for c in range(0,len(self.arr)-1):
            diff = -( self.arr[c+1] - self.arr[c] ) #diff between current 
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
path_plan.start()

            
                

    






