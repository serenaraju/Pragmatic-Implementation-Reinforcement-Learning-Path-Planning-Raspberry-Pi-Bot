import urllib
import json
import requests

class Loco:
    def __init__(self):
        self.url = 'https://jsonblob.com/api/jsonBlob/4f8b1f70-a3f3-11eb-b812-2d2dd835b230'

    def extract(self):
     
        headers = {"Content-Type": "application/json", "Access-Control-Allow-Origin": "https:://jsonblob.com"}

        response = requests.get(self.url, headers=headers)
        response = response.json()
        response = response['array']
        print(response)
        return response
            
    def write(self, data):        
        #data = { "array" : [1,2,2,2,2,2,3]}
        headers = {"Content-Type": "application/json", "Access-Control-Allow-Origin": "https:://jsonblob.com"}

        response = requests.put(self.url, data=json.dumps(data), headers=headers)

    def check(self, openUrl):
        if(openUrl.getcode()==200):
            data = operUrl.read()
        else:
            print("Error receiving data", operUrl.getcode())
        return data
    def direction(self):
        initial_arr = [1]
        arr = self.extract()
        initial_arr.extend(arr)
        print(initial_arr)


loco = Loco()
