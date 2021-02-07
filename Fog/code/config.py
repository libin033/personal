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