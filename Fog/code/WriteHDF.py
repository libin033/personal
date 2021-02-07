#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   WriteHDF.py    
@Contact :   libin033@163.com
@License :   (C)Copyright 2016-2020, lb_toolkits

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/12/7 14:53   lee         1.0         
'''

import os
import numpy as np
import h5py
import datetime


    
def WriteHDF(outfc, rat_fog, rat_cloud, add_fog, total_fog, fi_fog, YMS, YMSS):

    with h5py.File(outfc, "w") as fc:
        fc.attrs['Satellite Name'] = "H08"
        fc.attrs['Sensor Name'] = "AHI"
        fc.attrs['Data Level'] = "L4"
        fc.attrs['date_created'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fc.attrs['Institution'] = "NSMC"
        fc.attrs['Latitude Precision'] = "0.02"
        fc.attrs['Longitude Precision'] = "0.02"
        fc.attrs['Maximum Latitude'] = 43.0
        fc.attrs['Maximum Longitude'] = 135.0
        fc.attrs['Maximum X'] = 135.0
        fc.attrs['Maximum Y'] = 43.0
        fc.attrs['Minimum Latitude'] = 13.0
        fc.attrs['Minimum Longitude'] = 105.0
        fc.attrs['Minimum x'] = 105.0
        fc.attrs['Minimum Y'] = 13.0
        fc.attrs['Pixel Height'] = 1500
        fc.attrs['Pixel Width'] = 1500
        fc.attrs['Product Name'] = "FOG"
        fc.attrs['Projection Type'] = "Latlon"
        fc.attrs['Software Revision Date'] = "2020-05-25"
        fc.attrs['time_coverage_start'] = YMS
        fc.attrs['time_coverage_end'] = YMSS
        fc.attrs['Version Of Software'] = "v1.0"

        d1 = fc.create_dataset("fog_ratio", data=rat_fog, compression=9)
        d1.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT"
        d1.attrs['Description'] = ""
        d1.attrs['ProjectionResolution'] = 0.02
        d1.attrs['ProjectionMinLongitude'] = 105.0
        d1.attrs['ProjectionMaxLongitude'] = 135.0
        d1.attrs['ProjectionMinLatitude'] = 13.0
        d1.attrs['ProjectionMaxLatitude'] = 43.0
        d1.attrs['FillValue'] = 1
        d1.attrs['Slope'] = 1.0
        d1.attrs['Intercept'] = 0.0
        d1.attrs['units'] = "None"
        d1.attrs['valid_range'] = [0.0,1.0]

        d1 = fc.create_dataset("cloud_ratio", data=rat_cloud, compression=9)
        d1.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT"
        d1.attrs['Description'] = ""
        d1.attrs['ProjectionResolution'] = 0.02
        d1.attrs['ProjectionMinLongitude'] = 105.0
        d1.attrs['ProjectionMaxLongitude'] = 135.0
        d1.attrs['ProjectionMinLatitude'] = 13.0
        d1.attrs['ProjectionMaxLatitude'] = 43.0
        d1.attrs['FillValue'] = 1
        d1.attrs['Slope'] = 1.0
        d1.attrs['Intercept'] = 0.0
        d1.attrs['units'] = "None"
        d1.attrs['valid_range'] = [0.0,1.0]

        d2 = fc.create_dataset("fog_add", data=add_fog, compression=9)
        d2.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT"
        d2.attrs['Description'] = ""
        d2.attrs['ProjectionResolution'] = 0.02
        d2.attrs['ProjectionMinLongitude'] = 105.0
        d2.attrs['ProjectionMaxLongitude'] = 135.0
        d2.attrs['ProjectionMinLatitude'] = 13.0
        d2.attrs['ProjectionMaxLatitude'] = 43.0
        d2.attrs['FillValue'] = 1
        d2.attrs['Slope'] = 1.0
        d2.attrs['Intercept'] = 0.0
        d2.attrs['units'] = "None"
        d2.attrs['valid_range'] = [0.0, 150.0]

        d3 = fc.create_dataset("fog_total", data=total_fog, compression=9)

        d3.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT"
        d3.attrs['Description'] = ""
        d3.attrs['ProjectionResolution'] = 0.02
        d3.attrs['ProjectionMinLongitude'] = 105.0
        d3.attrs['ProjectionMaxLongitude'] = 135.0
        d3.attrs['ProjectionMinLatitude'] = 13.0
        d3.attrs['ProjectionMaxLatitude'] = 43.0
        d3.attrs['FillValue'] = 1
        d3.attrs['Slope'] = 1.0
        d3.attrs['Intercept'] = 0.0
        d3.attrs['units'] = "None"
        d3.attrs['valid_range'] = [0.0, 150.0]

        d4 = fc.create_dataset("fog_fix", data=fi_fog, compression=9)

        d4.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT"
        d4.attrs['Description'] = ""
        d4.attrs['ProjectionResolution'] = 0.02
        d4.attrs['ProjectionMinLongitude'] = 105.0
        d4.attrs['ProjectionMaxLongitude'] = 135.0
        d4.attrs['ProjectionMinLatitude'] = 13.0
        d4.attrs['ProjectionMaxLatitude'] = 43.0
        d4.attrs['FillValue'] = 1
        d4.attrs['Slope'] = 1.0
        d4.attrs['Intercept'] = 0.0
        d4.attrs['units'] = "None"
        d4.attrs['valid_range'] = [0.0,150.0]

    print('write %s success' %(outfc) )


