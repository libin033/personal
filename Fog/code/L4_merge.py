#!/usr/bin/env python
# coding: utf-8
import glob
import os, sys
import datetime
import time
import numpy as np
import signal
from pyhdf import SD
from drawpic import *
from WriteHDF import WriteHDF
# 分段时间调度令：python L4_merge.py  20200101 20200107 0000 0430
# 连续时间调度令：python L4_merge.py  202001010000 202001070430

FOG_path = "../data/{YYYY}/{YYYYMMDD}/SEFOG_AHI_{stime}*_{HM}*_1500_1500_S0204_001.hdf"

FDAY = "../dd/FOG/MUTS_MUTS_Latlon_L3_FOG_AVG_{YMS}_{YMSS}_2000M_{type}_X.HDF"



def datacov(data, inttmp):
    '''
    中值滤波
    :param data:
    :param inttmp:
    :return:
    '''
    h, s = data.shape
    tmp = inttmp // 2
    tmp_h, tmp_s = (h + tmp * 2), (s + tmp * 2)
    endl = h + tmp
    ends = s + tmp
    # print(h, s)

    tmpdata = np.full((tmp_h, tmp_s), 32767, dtype='f4')
    tmpdata[tmp:endl, tmp:ends] = data

    # 填补前、后四行、四列数据，并完成中值滤波
    for i in np.arange(tmp):
        tmpdata[i, :] = tmpdata[tmp]
        tmpdata[:, i] = tmpdata[:, tmp]
        tmpdata[endl + i] = tmpdata[endl - 1]
        tmpdata[:, (ends + i)] = tmpdata[:, ends - 1]
    outdata = signal.medfilt(tmpdata, (inttmp, inttmp))
    outdata = outdata[tmp:endl, tmp:ends]

    return outdata


def get_filelist(starttime, endtime, caseflag=1):

    filelist = []

    if caseflag == 1:
        start_ymd = starttime.strftime('%Y%m%d')
        start_HM = starttime.strftime('%H%M')

        end_ymd = endtime.strftime('%Y%m%d')
        end_HM = endtime.strftime('%H%M')

        day_list = [start_ymd]
        hm_list = [start_HM]
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
                file = FOG_path.format(YYYY=ymd[0:4],
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
            file = FOG_path.format(YYYY=starttime.strftime('%Y'),
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




if __name__ == "__main__":

    s_t = time.time()
    # 判断调度输入
    caseflag = 1

    argv = sys.argv
    if len(argv) == 5 :
        _this, start, end, starttime, endtime = sys.argv
        istartime = datetime.datetime.strptime('%s%s' % (start, starttime), '%Y%m%d%H%M')
        iendtime = datetime.datetime.strptime('%s%s' % (end, endtime), '%Y%m%d%H%M')
        caseflag = 1
    elif len(argv) == 3 :
        istartime = datetime.datetime.strptime(argv[1], '%Y%m%d%H%M')
        iendtime = datetime.datetime.strptime(argv[2], '%Y%m%d%H%M')
        caseflag = 2
    else:
        start = '20201101'
        end = '20201103'
        starttime = '0010'
        endtime = '0100'
        caseflag = 1
        istartime = datetime.datetime.strptime('%s%s' % (start, starttime), '%Y%m%d%H%M')
        iendtime = datetime.datetime.strptime('%s%s' % (end, endtime), '%Y%m%d%H%M')


    YMS = istartime.strftime('%Y%m%d') + '-' + iendtime.strftime('%Y%m%d')
    YMSS = istartime.strftime('%H%M') + '-' + iendtime.strftime('%H%M')

    # 拼接输出目录和文件名
    outfc = FDAY.format(YMS=YMS, YMSS=YMSS, type="L4")
    outpath = os.path.dirname(outfc)
    # print(outfc)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print(outpath, "!!!")

    # 获取数据并统计分析
    add_cloud = np.zeros((1500, 1500))
    add_fog = np.zeros((1500, 1500))
    total_fog = np.zeros((1500, 1500))
    ratio_cloud = np.zeros((1500, 1500))
    ratio_fog = np.zeros((1500, 1500))
    fix_fog = np.zeros((1500, 1500))

    filelist = get_filelist(istartime, iendtime, caseflag)

    if len(filelist) == 0:
        print('Not Match file')
        exit(0)

    for filename in filelist:
        if not os.path.exists(filename):
            continue

        # 读取文件
        f = SD.SD(filename, SD.SDC.READ)
        b = f.select('SeafogDetection')[:, :]
        valid_fog = np.where((b == 36) , 1, 0)   ##注释掉疑似！valid_fog = np.where(((b == 32) | (b == 36)), 1, 0)
        valid_cloud = np.where((b == 30), 1, 0)
        add_fog += valid_fog
        add_cloud += valid_cloud
        allvalid_fog = np.where(b == 1, 0, 1)
        total_fog += allvalid_fog
        # print(add_cloud.max(),add_fog.max())

        print('%s success' % (filename))

    ratio_fog = add_fog / total_fog
    ratio_cloud = add_cloud / total_fog
    fix_fog = np.where((add_fog != 0), 1, 0)

    # 中值滤波处理
    rat_fog = datacov(ratio_fog, 9)
    fi_fog = datacov(fix_fog, 9)
    rat_cloud = datacov(ratio_cloud, 9)

    WriteHDF(outfc, rat_fog, rat_cloud, add_fog, total_fog, fi_fog, YMS, YMSS)

    # plot(rat_fog, rat_cloud, outfc)

    # plot2(rat_cloud, outfc)
    if caseflag == 1:
        titlename = "H8_%s-%s_%s-%s" %(istartime.strftime('%Y%m%d'),
                                           iendtime.strftime('%Y%m%d'),
                                           istartime.strftime('%H:%M'),
                                           iendtime.strftime('%H:%M'),
                                           )
    elif caseflag == 2 :
        titlename = "H8_%s-%s" % (istartime.strftime('%Y%m%d %H:%M'),
                                            iendtime.strftime('%Y%m%d %H:%M'),
                                            )
    # draw()
    # draw(rat_fog, outfc, titlename)

    outname = outfc+ '.fog.png'
    drawmap(rat_fog, rat_cloud, outname, titlename)

    outname = outfc + '.cloud.png'
    drawmap(rat_fog, rat_cloud, outname, titlename, cloudflag=True)

    e_t = time.time()

    print('cost %.2f seconds' %(e_t - s_t))