# coding:utf-8
'''
@Project  : L4_merge.py
@File     : get_filelist.py
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/15 19:58   libin      1.0         
 
'''


import datetime
import glob
import os
from config import *

def get_filelist(starttime, endtime, caseflag=1):

    filelist = []

    if caseflag == 1:
        start_ymd = starttime.strftime('%Y%m%d')
        start_HM = starttime.strftime('%H%M')

        end_ymd = endtime.strftime('%Y%m%d')
        end_HM = endtime.strftime('%H%M')

        day_list = []
        hm_list = []
        s = datetime.datetime.strptime(start_ymd, "%Y%m%d")
        e = datetime.datetime.strptime(end_ymd, "%Y%m%d")

        shm = datetime.datetime.strptime(start_HM, "%H%M")
        ehm = datetime.datetime.strptime(end_HM, "%H%M")
        while s <= e:

            day_list.append(s.strftime("%Y%m%d"))
            s = s + datetime.timedelta(days=1)

        while shm <= ehm:
            hm_list.append(shm.strftime("%H%M"))
            shm = shm + datetime.timedelta(minutes=10)

        for ymd in day_list:
            for hhmm in hm_list:
                file = Fog_L2_Name.format(YYYY=ymd[0:4],
                                          YYYYMMDD=ymd,
                                          stime=ymd, HM=hhmm)
                # print(file)
                for filename in glob.glob(file):

                    if not os.path.isfile(filename):
                        print('%s is not exist....' % (filename))
                        continue
                    else:
                        filelist.append(filename)


    elif caseflag == 2:
        while starttime <= endtime :
            file = Fog_L2_Name.format(YYYY=starttime.strftime('%Y'),
                                      YYYYMMDD=starttime.strftime('%Y%m%d'),
                                      stime=starttime.strftime('%Y%m%d'),
                                      HM=starttime.strftime('%H%M'))
            for filename in glob.glob(file):

                if not os.path.isfile(filename):
                    print('%s is not exist....' % (filename))
                    continue
                else:
                    filelist.append(filename)

            starttime += datetime.timedelta(minutes=10)


    return filelist

