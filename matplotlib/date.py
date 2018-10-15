
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')


dates = np.array([
	np.datetime64(datetime.datetime(2004, 5, 1),'D'),
	np.datetime64(datetime.datetime(2004, 12, 1),'D'),
	np.datetime64(datetime.datetime(2005, 6, 1),'D'),
	np.datetime64(datetime.datetime(2006, 7, 1),'D')])
print(dates,type(dates),dates.dtype)
prices = np.array([1,5,3,4])

fig, ax = plt.subplots()
ax.plot(dates, prices)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years...
datemin = np.datetime64(dates[0], 'Y')
datemax = np.datetime64(dates[-1], 'Y')+ np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)


# format the coords message box
def price(x):
    return '$%1.2f' % x
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = price
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

plt.show()