# coding:utf-8
'''
@project:personal
@author: Lee Bin
@date:2021-02-19
'''

import sys
import os
import numpy as np
import glob
import time

sys.path.append('./')
import datetime
import pyhdf
from pyhdf import SD
from ncpro import readnc, writenc

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from config import *
from get_filelist import get_filelist

def readhdf4(filename, sdsname):
    # 读取文件
    f = pyhdf.SD.SD(filename, SD.SDC.READ)
    data = f.select(sdsname)[:]

    return data

class FogAnalysis():

    def __init__(self, filelist, nwppath,  outname):
        filelist.sort()

        percent = np.zeros((32, 1500, 1500), dtype=np.float32)
        fogs = np.zeros((32, 1500, 1500), dtype=np.int32)
        total = np.zeros((1500, 1500), dtype=np.int32)


        flag = self.JudgeCorMFog(filelist, nwppath)
        writenc('flag.nc', 'flag', flag, dimension=('z','y','x'), overwrite=1)
        exit(0)

        count = 0
        for filename in filelist:

            namelist = os.path.basename(filename).split('_')
            nowdate = datetime.datetime.strptime('%s %s' %(namelist[2], namelist[4][0:4]), '%Y%m%d %H%M')
            classflag = self.MatchERA5(nwppath, nowdate)
            if classflag is None :
                continue

            count += 1
            print('ID[%3d]:%s' %(count, filename))
            fogs, total = self.run(filename, classflag, fogs, total)

        for i in np.arange(1, 33) :
            percent[i-1] = fogs[i-1]/total * 100.0
            self.drawmap(percent[i-1], outname + '_%02d.PNG' %(i), dict_classfog[i])


        writenc(outname, 'fogs', fogs, dimension=('z','y','x'), overwrite=1)
        writenc(outname, 'total', total,dimension=('y','x'), overwrite=0)
        writenc(outname, 'percent', percent,dimension=('z','y','x'), overwrite=0)

    def JudgeCorMFog(self, filelist, nwppath):
        '''
        判断移入物还是生成雾
        :param filelist: 输入需要处理的文件
        :param nwppath: 数值预报数据存放目录
        :return:
        '''

        count = len(filelist)
        # 雾生成标识
        flag = np.full((count, 1500, 1500), fill_value=False, dtype=np.bool8)
        countid = 0
        # 逐个文件进行判识
        for i in range(count) :
            filename = filelist[i]
            namelist = os.path.basename(filename).split('_')
            nowdate = datetime.datetime.strptime('%s %s' %(
                namelist[2], namelist[4][0:4]), '%Y%m%d %H%M')
            ws,wd = self.GetWindDS(nwppath, nowdate)
            if ws is None :
                continue

            countid += 1
            print('ID[%3d]:%s' %(countid, filename))
            # fogs, total = self.run(filename, classflag, fogs, total)

            # 该位置第一次出现雾，则判断为生成雾
            if i == 0 :
                prefog = readhdf4(filelist[i], 'SeafogDetection')
                preflag = prefog == 36
                flag[0,:,:] = preflag
            else:
                nowfog = readhdf4(filelist[i], 'SeafogDetection')
                # nextfog = readhdf4(filelist[i+1], 'SeafogDetection')
                nowflag = nowfog == 36

                # nextflag = nextfog == 36

                fogflag = nowfog.copy()
                preflag = prefog != 36
                tempflag = nowflag & preflag
                index = np.where(tempflag == True)
                if len(index[0]) != 0:
                    for row, col in zip(index[0], index[1]) :

                        try:
                            thrvalue = 30
                            # 左上
                            if nowflag[row-1, col-1] and wd[row-1, col-1] > (135 - thrvalue) and wd[row-1, col-1] < (135 + thrvalue) :
                                fogflag[row, col] = False
                                break

                            # 上
                            if nowflag[row-1, col] and wd[row-1, col] > (180 - thrvalue) and wd[row-1, col] < (180 + thrvalue) :
                                fogflag[row, col] = False
                                break

                            # 右上
                            if nowflag[row-1, col+1] and wd[row-1, col+1] > (225 - thrvalue) and wd[row-1, col+1] < (225 + thrvalue) :
                                fogflag[row, col] = False
                                break

                            # 左
                            if nowflag[row, col-1] and wd[row, col-1] > (90 - thrvalue) and wd[row, col-1] < (90 + thrvalue) :
                                fogflag[row, col] = False
                                break

                            # 右
                            if nowflag[row, col+1] and wd[row, col+1] > (270 - thrvalue) and wd[row, col+1] < (270 + thrvalue) :
                                fogflag[row, col] = False
                                break

                            # 左下
                            if nowflag[row+1, col-1] and wd[row+1, col-1] > (45 - thrvalue) and wd[row+1, col-1] < (45 + thrvalue) :
                                fogflag[row, col] = False
                                break

                            # 下
                            if nowflag[row+1, col] and wd[row+1, col] > (0 - thrvalue) and wd[row+1, col] < (0 + thrvalue) :
                                fogflag[row, col] = False
                                break

                            # 右下
                            if nowflag[row+1, col+1] and wd[row+1, col+1] > (315 - thrvalue) and wd[row+1, col+1] < (315 + thrvalue) :
                                fogflag[row, col] = False
                                break
                        except BaseException as e :
                            print(e)

                flag[0,:,:] = fogflag

        return flag

    def GetWindDS(self, nwppath, nowdate):
        ERA5_Surf_File = self.ERA5_Time_Match(nwppath, nowdate)
        if ERA5_Surf_File is None :
            return None, None

        latitude = readnc(ERA5_Surf_File, 'latitude')
        longitude = readnc(ERA5_Surf_File, 'longitude')

        sat = readnc(ERA5_Surf_File, 't2m')
        sst = readnc(ERA5_Surf_File, 'sst')
        u10 = readnc(ERA5_Surf_File, 'u10')
        v10 = readnc(ERA5_Surf_File, 'v10')

        sat = self.interp2d(longitude, latitude, sat[0])
        sst = self.interp2d(longitude, latitude, sst[0])
        u10 = self.interp2d(longitude, latitude, u10[0])
        v10 = self.interp2d(longitude, latitude, v10[0])

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

        return ws, wd

    def drawmap(self, rat_fog1, outname, titlename):
        rat_fog = rat_fog1.copy()*1.0

        rat_fog[rat_fog==0] = np.nan

        fig = plt.figure(figsize=(8, 8))
        m = Basemap(llcrnrlon=105.0, llcrnrlat=13.0, urcrnrlon=135.0, urcrnrlat=43.0, \
                    resolution='i', projection='cyl')

        m.drawcoastlines(linewidth=0.5)
        m.drawcountries()
        m.drawparallels(np.arange(13, 43.01, 5), color='k', linewidth=.5, labels=[1, 0, 0, 1])
        m.drawmeridians(np.arange(105, 135.01, 5), color='k', linewidth=.5, labels=[1, 0, 0, 1])
        m.fillcontinents(color = '#ddaa66')
        # plt.xlabel('$Longitude$', fontsize=12, labelpad=24)
        # plt.ylabel('$Latitude$', fontsize=12, labelpad=40)

        plt.title(titlename, fontsize=18)

        ffim = m.imshow(rat_fog[::-1], cmap='Blues', vmin=0, vmax=100)
        ax3 = fig.add_axes([.91, .11, 0.05, 0.77])
        cb3 = plt.colorbar(ffim, cax=ax3)
        cb3.outline.set_visible(False)

        cb3.set_label('Fog Frequency(%)')

        plt.savefig(outname, dpi=200, bbox_inches='tight')

        plt.close(fig)
        print('draw %s success...' %(outname))

    def cal_wind(self, ERA5_Surf_File):

        sat = readnc(ERA5_Surf_File, 't2m')

        sst = readnc(ERA5_Surf_File, 'sst')
        u10 = readnc(ERA5_Surf_File, 'u10')
        v10 = readnc(ERA5_Surf_File, 'v10')

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

        # 根据风向、风速、温度差进行分类
        classflag = np.zeros_like(sst, dtype=np.uint8)
        # 方向wd： 东：e 西: w   南: s  北 : n
        # 风速ws:  微风：s 中风：m  大风:b  超大风：u
        # 暖风 ： w  冷风：c
        classflag[(wd >= 45) & (wd < 135) & (ws >= 0) & (ws < 8) & (diff_sst > 0)] = 1  # e_s_w_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 0) & (ws < 8) & (diff_sst < 0)] = 2  # e_s_c_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 8) & (ws < 16) & (diff_sst > 0)] = 3  # e_m_w_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 8) & (ws < 16) & (diff_sst < 0)] = 4  # e_m_c_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 16) & (ws < 24) & (diff_sst > 0)] = 5  # e_b_w_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 16) & (ws < 24) & (diff_sst < 0)] = 6  # e_b_c_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 24) & (diff_sst > 0)] = 7  # e_u_w_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 24) & (diff_sst < 0)] = 8  # e_u_c_f

        classflag[(wd >= 135) & (wd < 225) & (ws >= 0) & (ws < 8) & (diff_sst > 0)] = 9  # s_s_w_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 0) & (ws < 8) & (diff_sst < 0)] = 10  # s_s_c_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 8) & (ws < 16) & (diff_sst > 0)] = 11  # s_m_w_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 8) & (ws < 16) & (diff_sst < 0)] = 12  # s_m_c_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 16) & (ws < 24) & (diff_sst > 0)] = 13  # s_b_w_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 16) & (ws < 24) & (diff_sst < 0)] = 14  # s_b_c_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 24) & (diff_sst > 0)] = 15  # s_u_w_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 24) & (diff_sst < 0)] = 16  # s_u_c_f

        classflag[(wd >= 225) & (wd < 315) & (ws >= 0) & (ws < 8) & (diff_sst > 0)] = 17  # w_s_w_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 0) & (ws < 8) & (diff_sst < 0)] = 18  # w_s_c_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 8) & (ws < 16) & (diff_sst > 0)] = 19  # w_m_w_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 8) & (ws < 16) & (diff_sst < 0)] = 20  # w_m_c_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 16) & (ws < 24) & (diff_sst > 0)] = 21  # w_b_w_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 16) & (ws < 24) & (diff_sst < 0)] = 22  # w_b_c_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 24) & (diff_sst > 0)] = 23  # w_u_w_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 24) & (diff_sst < 0)] = 24  # w_u_c_f

        classflag[((wd >= 315) | (wd < 45)) & (ws >= 0) & (ws < 8) & (diff_sst > 0)] = 25  # n_s_w_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 0) & (ws < 8) & (diff_sst < 0)] = 26  # n_s_c_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 8) & (ws < 16) & (diff_sst > 0)] = 27  # n_m_w_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 8) & (ws < 16) & (diff_sst < 0)] = 28  # n_m_c_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 16) & (ws < 24) & (diff_sst > 0)] = 29  # n_b_w_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 16) & (ws < 24) & (diff_sst < 0)] = 30  # n_b_c_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 24) & (diff_sst > 0)] = 31  # n_u_w_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 24) & (diff_sst < 0)] = 32  # n_u_c_f

        return classflag

    def ERA5_View_Match(self, flag):

        line = np.array((50 - LAT) / 0.25, dtype=np.int32)
        pixel = np.array((LON - 100 ) / 0.25, dtype=np.int32)
        line[line<0] = 0
        line[line>=201] = 201
        pixel[pixel<0] = 0
        pixel[pixel>=161] = 161

        return flag[0, line, pixel]  #???????????

    def interp2d(self,longitude, latitude,data):
        from scipy import interpolate
        yy = np.arange(ProjectionMinLatitude, ProjectionMaxLatitude, ProjectionResolution)
        xx = np.arange(ProjectionMinLongitude, ProjectionMaxLongitude, ProjectionResolution)
        fit = interpolate.interp2d(longitude, latitude, data)
        temp = fit(xx, yy)

        return temp[::-1]

    def Interplate(self, ERA5_Surf_File):
        latitude = readnc(ERA5_Surf_File, 'latitude')
        longitude = readnc(ERA5_Surf_File, 'longitude')

        sat = readnc(ERA5_Surf_File, 't2m')
        sst = readnc(ERA5_Surf_File, 'sst')
        u10 = readnc(ERA5_Surf_File, 'u10')
        v10 = readnc(ERA5_Surf_File, 'v10')

        sat = self.interp2d(longitude, latitude, sat[0])
        sst = self.interp2d(longitude, latitude, sst[0])
        u10 = self.interp2d(longitude, latitude, u10[0])
        v10 = self.interp2d(longitude, latitude, v10[0])

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

        # 根据风向、风速、温度差进行分类
        classflag = np.zeros_like(sst, dtype=np.uint8)
        # 方向wd： 东：e 西: w   南: s  北 : n
        # 风速ws:  微风：s 中风：m  大风:b  超大风：u
        # 暖风 ： w  冷风：c
        classflag[(wd >= 45) & (wd < 135) & (ws >= 0) & (ws < 8) & (diff_sst > 0)] = 1  # e_s_w_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 0) & (ws < 8) & (diff_sst < 0)] = 2  # e_s_c_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 8) & (ws < 16) & (diff_sst > 0)] = 3  # e_m_w_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 8) & (ws < 16) & (diff_sst < 0)] = 4  # e_m_c_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 16) & (ws < 24) & (diff_sst > 0)] = 5  # e_b_w_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 16) & (ws < 24) & (diff_sst < 0)] = 6  # e_b_c_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 24) & (diff_sst > 0)] = 7  # e_u_w_f
        classflag[(wd >= 45) & (wd < 135) & (ws >= 24) & (diff_sst < 0)] = 8  # e_u_c_f

        classflag[(wd >= 135) & (wd < 225) & (ws >= 0) & (ws < 8) & (diff_sst > 0)] = 9  # s_s_w_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 0) & (ws < 8) & (diff_sst < 0)] = 10  # s_s_c_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 8) & (ws < 16) & (diff_sst > 0)] = 11  # s_m_w_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 8) & (ws < 16) & (diff_sst < 0)] = 12  # s_m_c_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 16) & (ws < 24) & (diff_sst > 0)] = 13  # s_b_w_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 16) & (ws < 24) & (diff_sst < 0)] = 14  # s_b_c_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 24) & (diff_sst > 0)] = 15  # s_u_w_f
        classflag[(wd >= 135) & (wd < 225) & (ws >= 24) & (diff_sst < 0)] = 16  # s_u_c_f

        classflag[(wd >= 225) & (wd < 315) & (ws >= 0) & (ws < 8) & (diff_sst > 0)] = 17  # w_s_w_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 0) & (ws < 8) & (diff_sst < 0)] = 18  # w_s_c_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 8) & (ws < 16) & (diff_sst > 0)] = 19  # w_m_w_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 8) & (ws < 16) & (diff_sst < 0)] = 20  # w_m_c_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 16) & (ws < 24) & (diff_sst > 0)] = 21  # w_b_w_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 16) & (ws < 24) & (diff_sst < 0)] = 22  # w_b_c_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 24) & (diff_sst > 0)] = 23  # w_u_w_f
        classflag[(wd >= 225) & (wd < 315) & (ws >= 24) & (diff_sst < 0)] = 24  # w_u_c_f

        classflag[((wd >= 315) | (wd < 45)) & (ws >= 0) & (ws < 8) & (diff_sst > 0)] = 25  # n_s_w_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 0) & (ws < 8) & (diff_sst < 0)] = 26  # n_s_c_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 8) & (ws < 16) & (diff_sst > 0)] = 27  # n_m_w_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 8) & (ws < 16) & (diff_sst < 0)] = 28  # n_m_c_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 16) & (ws < 24) & (diff_sst > 0)] = 29  # n_b_w_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 16) & (ws < 24) & (diff_sst < 0)] = 30  # n_b_c_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 24) & (diff_sst > 0)] = 31  # n_u_w_f
        classflag[((wd >= 315) | (wd < 45)) & (ws >= 24) & (diff_sst < 0)] = 32  # n_u_c_f



        return classflag


    def ERA5_Time_Match(self, nwppath, nowdate):
        filelist = glob.glob(os.path.join(nwppath, 'surface_%s.nc' %(nowdate.strftime('%Y%m%d%H'))))
        if len(filelist) == 0 :
            print('%s is not exist, will continue!!!' %(
                os.path.join(nwppath, 'surface_%s.nc' %(nowdate.strftime('%Y%m%d%H')))))
            return None
        else:
            return filelist[0]

    def MatchERA5(self, nwppath, nowdate):

        ERA5_Surf_File = self.ERA5_Time_Match(nwppath, nowdate)
        if ERA5_Surf_File is None :
            return None

        classflag = self.Interplate(ERA5_Surf_File)

        # flag = self.cal_wind(ERA5_Surf_File)
        # classflag = self.ERA5_View_Match(flag)

        return classflag

    def run(self, FogFileName, classflag, fogs, total):

        seafog = readhdf4(FogFileName, 'SeafogDetection')
        fogflag = seafog == 36
        total += fogflag
        for i in np.arange(1, 33) :
            fogs[i-1][classflag==i] += fogflag[classflag==i]

        return fogs, total


# 分段时间调度令：python L4_merge.py  20200101 20200107 0000 0430
# 连续时间调度令：python L4_merge.py  202001010000 202001070430


if __name__ == '__main__':
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

    # 拼接输出目录和文件名
    outname = Fog_FRE_Name.format(YMS=YMS, YMSS=YMSS, type="L4")
    outpath = os.path.dirname(outname)

    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print("create %s success..." %(outpath))

    # 获取数据并统计分析
    filelist = get_filelist(istartime, iendtime, caseflag)

    if len(filelist) == 0:
        print('Not Match file')
        exit(0)


    FogAnalysis(filelist, NWP_PATH, outname)

    e_t = time.time()

    print('cost %.2f seconds' %(e_t - s_t))
