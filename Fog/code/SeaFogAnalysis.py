# coding:utf-8
'''
@project:personal
@author: Lee Bin
@date:2021-02-19
'''

import sys
import os
import numpy as np

sys.path.append('./')

import pyhdf
from pyhdf import SD
from ncpro import readnc, writenc
from config import *

def readhdf4(filename, sdsname):
    # 读取文件
    f = pyhdf.SD.SD(filename, SD.SDC.READ)
    data = f.select(sdsname)[:]

    return data

def cal_wind(ERA5_Surf_File, ERA5_Prof_File):

    sat = readnc(ERA5_Surf_File, 't2m')

    sst = readnc(ERA5_Surf_File, 'sst')
    ws = readnc(ERA5_Surf_File, 'wind')
    u10 = readnc(ERA5_Surf_File, 'u10')
    v10 = readnc(ERA5_Surf_File, 'v10')

    # 计算风速和风向
    speedd = np.sqrt(np.power(u10, 2) + np.power(v10, 2))
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


def viewmatch(flag):

    line = np.array((50 - LAT) / 0.25, dtype=np.int32)
    pixel = np.array((LON - 100 ) / 0.25, dtype=np.int32)
    line[line<0] = 0
    line[line>=201] = 201
    pixel[pixel<0] = 0
    pixel[pixel>=161] = 161

    return flag[0, line, pixel]  #???????????

if __name__ == '__main__':

    ERA5_Surf_File = r'D:\personal\Fog\data\surface_2017010120.nc'
    ERA5_Prof_File = r'D:\personal\Fog\data\high_2017010201.nc'

    flag = cal_wind(ERA5_Surf_File, ERA5_Prof_File)

    classflag = viewmatch(flag)

    fog_filename = r'D:\personal\Fog\data\SEFOG_AHI_20210101_2021001_0000030_1500_1500_S0204_001.hdf'
    seafog = readhdf4(fog_filename, 'SeafogDetection')
    fogflag = seafog == 15

    fogs = np.zeros((32, 1500, 1500), dtype=np.int32)
    for i in np.arange(1, 33) :
        fogs[i-1][classflag==i] += fogflag[classflag==i]

    writenc('test.nc', 'fogs', fogs,dimension=('x','y','z'), overwrite=1)
