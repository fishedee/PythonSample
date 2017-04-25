#访问集合

set = {'Name','Age','Class'}

print ("set : ",set)
print ("set len : ", len(set))
print ("Class in set :","Class" in set)

# 添加集合元素

set = {'Name','Age','Class'}
set.add("mm")
print ("set: ",set)

# 删除集合元素

set = {'Name','Age','Class'}

set.remove('Name')

print ("set: ", set)

#遍历字典
for item in set:
    print(item)
