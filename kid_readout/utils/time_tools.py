import time

def date_to_unix_time(dstr):
    return time.mktime(time.strptime(dstr,'%Y-%m-%d'))