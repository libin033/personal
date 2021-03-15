#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   libin033@163.com
@License :   (C)Copyright 2016-2020, lb_toolkits

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/12/7 15:43   lee         1.0         
'''

import os
import numpy as np


Fog_L2_Name = r"../data/{YYYY}/{YYYYMMDD}/SEFOG_AHI_{stime}*_{HM}*_1500_1500_S0204_001.hdf"

Fog_L3_Name = r"../result/FOG/MUTS_MUTS_Latlon_L3_FOG_AVG_{YMS}_{YMSS}_2000M_{type}_X.HDF"

Fog_FRE_Name = r"../result/IMAGE/MUTS_MUTS_Latlon_L3_FOG_FRE_{YMS}_{YMSS}_2000M_{type}_X.nc"
Fog_SST_Name = r"../result/IMAGE/MUTS_MUTS_Latlon_L3_FOG_SST_{YMS}_{YMSS}_2000M_{type}_X.nc"

NWP_PATH = r'D:\personal\Fog\data\era5'

ProjectionMinLongitude = 105.0
ProjectionMaxLongitude = 135.0
ProjectionMinLatitude = 13.0
ProjectionMaxLatitude = 43.0
ProjectionResolution = 0.02

LINE = int((ProjectionMaxLatitude - ProjectionMinLatitude) / ProjectionResolution)
PIXEL = int((ProjectionMaxLongitude - ProjectionMinLongitude) / ProjectionResolution)
    


yy = np.arange(ProjectionMinLatitude, ProjectionMaxLatitude, ProjectionResolution)
xx = np.arange(ProjectionMinLongitude, ProjectionMaxLongitude, ProjectionResolution)
LON, LAT = np.meshgrid(xx, yy)


dict_classfog = {
    1  :  'east_slow_warm_fog',
    2  :  'east_slow_cold_fog',
    3  :  'east_medium_warm_fog',
    4  :  'east_medium_cold_fog',
    5  :  'east_big_warm_fog',
    6  :  'east_big_cold_fog',
    7  :  'east_ultra_warm_fog',
    8  :  'east_ultra_cold_fog',
    9  :  'south_slow_warm_fog',
    10 :  'south_slow_cold_fog',
    11 :  'south_medium_warm_fog',
    12 :  'south_medium_cold_fog',
    13 :  'south_big_warm_fog',
    14 :  'south_big_cold_fog',
    15 :  'south_ultra_warm_fog',
    16 :  'south_ultra_cold_fog',
    17 :  'west_slow_warm_fog',
    18 :  'west_slow_cold_fog',
    19 :  'west_medium_warm_fog',
    20 :  'west_medium_cold_fog',
    21 :  'west_big_warm_fog',
    22 :  'west_big_cold_fog',
    23 :  'west_ultra_warm_fog',
    24 :  'west_ultra_cold_fog',
    25 :  'north_slow_warm_fog',
    26 :  'north_slow_cold_fog',
    27 :  'north_medium_warm_fog',
    28 :  'north_medium_cold_fog',
    29 :  'north_big_warm_fog',
    30 :  'north_big_cold_fog',
    31 :  'north_ultra_warm_fog',
    32 :  'north_ultra_cold_fog',
}
