import datetime

#获取当前时间
now = datetime.datetime.now()
today = datetime.datetime.today()

print(now,today)

#指定时间
holiday = datetime.datetime(2018,10,14,12,34,56)
print(holiday)

#时间运算
prevDay = now-datetime.timedelta(days=1)
nextMinute = now+datetime.timedelta(minutes=1)
print(prevDay,nextMinute)

#时间戳
timestamp = now.timestamp()
nextTimestamp = timestamp+1
nextTime = datetime.datetime.fromtimestamp(nextTimestamp)
print(timestamp,nextTimestamp,now,nextTime)

#时间字符串格式化
timeStr = now.strftime('%Y-%m-%d %H:%M:%S')
print(timeStr)

#时间字符串解析
strTime = datetime.datetime.strptime('2005-01-02 03:23:12','%Y-%m-%d %H:%M:%S')
print(strTime)