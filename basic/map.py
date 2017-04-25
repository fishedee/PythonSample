#访问字典

dict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}

print ("dict['Name']: ", dict['Name'])
print ("dict['Age']: ", dict['Age'])
print ("dict len : ", len(dict))
print ("Class in dict :","Class" in dict)

# 添加字典元素

dict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}

dict['mm']=77;
print ("dict: ",dict)

#更新字典元素

dict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}

dict['Age'] = 8;               # 更新 Age
dict['School'] = "菜鸟教程"  # 添加信息


print ("dict['Age']: ", dict['Age'])
print ("dict['School']: ", dict['School'])

# 删除字典元素

dict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}

del dict['Name'] # 删除键 'Name'

print ("Name in dict: ", 'Name' in dict)

#遍历字典
for (index, item) in dict.items():
    print("%s,%s"%(index,item))
