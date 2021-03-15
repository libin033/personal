# coding:utf-8
'''
@Project  : L4_merge.py
@File     : FogBySST.py
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/15 20:14   libin      1.0         
 
'''
import glob
import os, sys
import datetime
import time
import numpy as np
import signal
from pyhdf import SD
from drawpic import *
from WriteHDF import WriteHDF
from config import *

from get_filelist import get_filelist
from ncpro import readnc, writenc

# 分段时间调度令：python L4_merge.py  20200101 20200107 0000 0430
# 连续时间调度令：python L4_merge.py  202001010000 202001070430

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


def interp2d(longitude, latitude,data):
    from scipy import interpolate
    yy = np.arange(ProjectionMinLatitude, ProjectionMaxLatitude, ProjectionResolution)
    xx = np.arange(ProjectionMinLongitude, ProjectionMaxLongitude, ProjectionResolution)
    fit = interpolate.interp2d(longitude, latitude, data)
    temp = fit(xx, yy)

    return temp[::-1]

def Interplate(ERA5_Surf_File):
    latitude = readnc(ERA5_Surf_File, 'latitude')
    longitude = readnc(ERA5_Surf_File, 'longitude')

    sat = readnc(ERA5_Surf_File, 't2m')
    sst = readnc(ERA5_Surf_File, 'sst')
    u10 = readnc(ERA5_Surf_File, 'u10')
    v10 = readnc(ERA5_Surf_File, 'v10')
    # 计算风速和风向
    ws = np.sqrt(np.power(u10, 2) + np.power(v10, 2))
    LON, LAT = np.meshgrid(longitude, latitude)
    print(LON.shape)
    print(ws.shape)
    lw = 5*ws / ws.max()


    import matplotlib.pyplot as plt

    # strm = plt.streamplot(LON, LAT, u10[0], v10[0], color=ws[0], linewidth=2, cmap='autumn')
    strm = plt.streamplot(LON, LAT, u10[0], v10[0], density=[0.5, 1], color='k')
    plt.show()
    exit()

    sat = interp2d(longitude, latitude, sat[0])
    sst = interp2d(longitude, latitude, sst[0])
    u10 = interp2d(longitude, latitude, u10[0])
    v10 = interp2d(longitude, latitude, v10[0])

    # 计算风速和风向
    ws = np.sqrt(np.power(u10, 2) + np.power(v10, 2))
    wd = np.arctan(u10 / v10) * 180 / np.pi
    wd[(u10 != 0) & (v10 < 0)] += 180
    wd[(u10 < 0) & (v10 > 0 )] += 360
    wd[(u10 == 0) & (v10 > 0)] = 0
    wd[(u10 == 0) & (v10 < 0)] = 180
    wd[(u10 > 0) & (v10 == 0)] = 90
    wd[(u10 < 0) & (v10 == 0)] = 270
    wd[(u10 == 0) & (v10 == 0)] = -999

    diff_sst = sst - sat



    return diff_sst

def ERA5_Time_Match(nwppath, nowdate):
    filelist = glob.glob(os.path.join(nwppath, 'surface_%s.nc' %(nowdate.strftime('%Y%m%d%H'))))
    if len(filelist) == 0 :
        print('%s is not exist, will continue!!!' %(
            os.path.join(nwppath, 'surface_%s.nc' %(nowdate.strftime('%Y%m%d%H')))))
        return None
    else:
        return filelist[0]


def MatchERA5(nwppath, nowdate):

    ERA5_Surf_File = ERA5_Time_Match(nwppath, nowdate)
    if ERA5_Surf_File is None :
        return None

    sstbias = Interplate(ERA5_Surf_File)

    # flag = self.cal_wind(ERA5_Surf_File)
    # classflag = self.ERA5_View_Match(flag)

    return sstbias


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
        start = '20210125'
        end = '20210125'
        starttime = '0000'
        endtime = '2359'
        caseflag = 1
        istartime = datetime.datetime.strptime('%s%s' % (start, starttime), '%Y%m%d%H%M')
        iendtime = datetime.datetime.strptime('%s%s' % (end, endtime), '%Y%m%d%H%M')


    YMS = istartime.strftime('%Y%m%d') + '-' + iendtime.strftime('%Y%m%d')
    YMSS = istartime.strftime('%H%M') + '-' + iendtime.strftime('%H%M')


    print('Fog_L2_Name:', Fog_L2_Name)
    print('Fog_L3_Name:', Fog_SST_Name)
    # 拼接输出目录和文件名
    outfc = Fog_SST_Name.format(YMS=YMS, YMSS=YMSS, type="L4")
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
    count = 0
    for filename in filelist:
        if not os.path.exists(filename):
            continue

        namelist = os.path.basename(filename).split('_')
        nowdate = datetime.datetime.strptime('%s %s' %(namelist[2], namelist[4][0:4]), '%Y%m%d %H%M')
        classflag = MatchERA5(NWP_PATH, nowdate)

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
        count += 1
        print('ID[%3d]:%s success...' % (count, filename))

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
