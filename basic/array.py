#访问数组

list1 = ['Google', 'Runoob', 1997, 2000];
list2 = [1, 2, 3, 4, 5, 6, 7 ];

print ("list1[0]: ", list1[0])
print ("list2[1:5]: ", list2[1:5])
print ("list len",len(list1))

#添加数组元素

list = ['Google', 'Runoob', 1997, 2000]
list.append(2003)
print("list append",list)

#更新数组

list = ['Google', 'Runoob', 1997, 2000]

print ("第三个元素为 : ", list[2])
list[2] = 2001
print ("更新后的第三个元素为 : ", list[2])

#删除数组元素

list = ['Google', 'Runoob', 1997, 2000]

print (list)
del list[2]
print ("删除第三个元素 : ", list)

#遍历数组
for item in list:
    print("%s"%(item))

