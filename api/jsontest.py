import json

data = {
    'no' : 1,
    'name' : 'Runoob',
    'url' : 'http://www.runoob.com'
}
 
json_str = json.dumps(data)

data2 = json.loads(json_str)
print(json_str,type(json_str))
print(data2,type(data2))