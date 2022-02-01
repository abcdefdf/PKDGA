import random
from datetime import datetime
import time


def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.
    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def randomDate(start, end, prop):
    return strTimeProp(start, end, '%m/%d/%Y %I:%M %p', prop)


def gen_hash(intseed):
    hash_c = []
    random.seed(intseed)
    date = randomDate("1/2/1970 01:00 AM", "1/1/3000 1:10 AM", random.random())
    d = datetime.strptime(date, "%m/%d/%Y %I:%M %p")
    print(d)
    t = time.mktime(d.timetuple())
    print(t)
    str1 = str(t)
    print(str1)
    c = []
    len_h = len(str1)
    print(len_h)
    for i in range(8):
        a = int(str1[i]) * 100 + int(str1[i+1]) * 10 + int(str1[i+2])
        a /= 1000
        c.append(a)
    print(c)
    # hash_num = 0
    # while t > 0:
    #     s = t % 10000
    #     hash_num += s
    #     t = t / 10000
    # hash_c.append(hash_num)
    return hash_c


intseed = 521496
c = gen_hash(intseed)
print(c)
# random.seed(intseed)
# date = randomDate("1/2/1970 01:00 AM", "1/1/3000 1:10 AM", random.random())
# d = datetime.strptime(date, "%m/%d/%Y %I:%M %p")
# print(d)
# t = time.mktime(d.timetuple())
# print(t)
# hash_num = 0
# while t > 0:
#     s = t % 10000
#     hash_num += s
#     t = t/10000
# print(hash_num)
# print(t)
