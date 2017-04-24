#if 控制
age = int(input("请输入你家狗狗的年龄: "))
if age == 1:
	print("相当于 14 岁的人。")
elif age == 2:
	print("相当于 22 岁的人。")
elif age > 2:
	human = 22 + (age -2)*5
	print("对应人类年龄: ", human)
else:
	print("你是在逗我吧!")

# while控制
n = 100
sum = 0
counter = 1
while counter <= n:
    sum = sum + counter
    counter += 1

print("1 到 %d 之和为: %d" % (n,sum))

# for控制

languages = ["C", "C++", "Perl", "Python"] 
for x in languages:
	print (x)

for i in range(5,9) :
    print(i)