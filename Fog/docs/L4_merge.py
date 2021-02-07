#!/usr/bin/env python
# coding: utf-8
import glob
import os, sys
import datetime
import argparse
import copy
import numpy as np
import h5py
import scipy.signal as signal
import traceback
from multiprocessing import Pool
from pyhdf import SD
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors

#调度令：python L4_merge.py  20200101 20200107 0000 0430

FOG_path = "/SOAdata/input/FOG/L2hdf4/{stime}/SEFOG_AHI_{stime}*_{HM}*_1500_1500_S0204_001.hdf"

FDAY = "/SHKDATA/MUTS/MUTS/FOG/MUTS_MUTS_Latlon_L3_FOG_AVG_{YMS}_{YMSS}_2000M_{type}_X.HDF"
tmptitlename = "  H8 {band} {YMD}:{YMSS}"
def draw(data1,outjpg,titlename):
    data=data1.copy()
    #titlename = tmptitlename.format(YMD = time,band = "FOG")
    height, width = data.shape
    RR = 500.
    w, h = width / RR + 2., height / RR + 2.
    plt.figure(figsize=[w, h])
    ax = plt.axes([1 / w, 1 / h, (w - 2) / w, (h - 2) / h])
    m = Basemap(llcrnrlon=105.0, llcrnrlat=13.0, urcrnrlon=135.0, urcrnrlat=43.0, \
                resolution='i', projection='cyl')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.drawparallels(np.arange(105, 135, 5), color='k', linewidth=.5, labels=[1, 0, 0, 1])
    m.drawmeridians(np.arange(105, 135, 5), color='k', linewidth=.5, labels=[1, 0, 0, 1])

    plt.xlabel('$Longitude$', fontsize=12, labelpad=24)
    plt.ylabel('$Latitude$', fontsize=12, labelpad=40)

    plt.title(titlename, fontsize=18)
    ffim = m.imshow(data, extent=[105, 135, 43, 13], vmax=40, vmin=0, cmap='rainbow')
    cax = plt.axes([1.2 / w, 1.3 / h, (w - 2) / 2.3 / w, (.25) / h])
    # *orientation * vertical or horizontal
    cbar = plt.gcf().colorbar(ffim, extend="both", orientation='horizontal', cax=cax)
    cbar.ax.tick_params(labelsize=12)

    #L = TickLabels[band]
    tick = [x*2 for x in range(0,20) ]
    cbar.set_ticks(tick)
    #cbar.set_ticklabels(L)

    plt.savefig(outjpg+'FOG.png', dpi=200)
    plt.clf()
    plt.close()    
    
def plot(rat_fog,rat_cloud,outfc):
    xx = np.arange(rat_fog.shape[0])
    yy = np.arange(rat_fog.shape[1])
    x,y = np.meshgrid(yy,xx)
    rat_fog[rat_fog==0]=np.nan    
    fig,ax = plt.subplots(1,1,figsize=(20, 16))
    img1=ax.imshow(rat_fog*100,'rainbow',vmin=0,vmax=30)
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size="3%", pad=0.1)
    
    contoursol = ax.contour(x,y,rat_cloud*100,[10,30,50,70,43],colors='red',linewidths=0.3)
    ax.clabel(contoursol,fontsize=8,colors='red',fmt='%.0f')
    plt.colorbar(img1, cax=cax2)
    plt.savefig(outfc+'.png')
def plot2(rat_fog,outfc):
    xx = np.arange(rat_fog.shape[0])
    yy = np.arange(rat_fog.shape[1])
    x,y = np.meshgrid(yy,xx)
    rat_fog[rat_fog==0]=np.nan    
    fig,ax = plt.subplots(1,1,figsize=(20, 16))
    img1=ax.imshow(rat_fog*100,'rainbow',vmin=0,vmax=100)
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size="3%", pad=0.1)  
    plt.colorbar(img1, cax=cax2)
    plt.savefig(outfc+'_cloud.png')

def datacov(data,inttmp):
    h,s=data.shape
    tmp = inttmp//2
    tmp_h,tmp_s=(h+tmp*2),(s+tmp*2)
    endl = h+tmp
    ends = s+tmp
    print(h,s)

    tmpdata = np.full((tmp_h,tmp_s),32767, dtype='f4')
    tmpdata[tmp:endl,tmp:ends] = data
    for i in np.arange(tmp):
        tmpdata[i,:] = tmpdata[tmp]
        tmpdata[:,i] = tmpdata[:,tmp]
        tmpdata[endl+i] = tmpdata[endl-1]
        tmpdata[:,(ends+i)] = tmpdata[:,ends-1]
    outdata = signal.medfilt(tmpdata,(inttmp,inttmp))
    outdata = outdata[tmp:endl,tmp:ends]
    return outdata


def get_day_list(start_ymd,end_ymd,start_HM,end_HM):
    day_list = [start_ymd]
    hm_list=[start_HM]
    s=datetime.datetime.strptime(start_ymd,"%Y%m%d")
    e=datetime.datetime.strptime(end_ymd,"%Y%m%d")
    
    shm=datetime.datetime.strptime(start_HM,"%H%M")
    ehm=datetime.datetime.strptime(end_HM,"%H%M")
    while s<e:
        s = s + datetime.timedelta(days=1)
        day_list.append(s.strftime("%Y%m%d"))
    while shm<ehm:
        shm = shm + datetime.timedelta(minutes=10)
        hm_list.append(shm.strftime("%H%M"))
    return day_list,hm_list


if __name__ == "__main__":

    _this,start,end,starttime,endtime = sys.argv
    day_list,hm_list =get_day_list(start, end,starttime,endtime)
    print (day_list)

    YMS = start+'-'+end
    YMSS = starttime+'-'+endtime
    
    outfc = FDAY.format(YMS=YMS,YMSS=YMSS,type="L4")
    outpath = os.path.dirname(outfc)
    print (outfc)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print (os.makedirs(outpath))
        print(outpath, "!!!")

    add_cloud = np.zeros((1500, 1500))
    add_fog = np.zeros((1500, 1500))
    total_fog = np.zeros((1500, 1500))
    ratio_cloud = np.zeros((1500, 1500))
    ratio_fog = np.zeros((1500, 1500))
    fix_fog = np.zeros((1500, 1500))

    for ymd in day_list:
        for hhmm in hm_list:
            file = FOG_path.format (stime=ymd,HM=hhmm)
            #print(file)
            for file1 in glob.glob(file):
                print(file1)
                if not os.path.exists(file1):
                    continue

                f = SD.SD(file1, SD.SDC.READ)
                b = f.select('SeafogDetection')[:,:]
                valid_fog = np.where(((b == 32) | (b == 36)), 1, 0)
                valid_cloud = np.where((b == 30), 1, 0)
                add_fog += valid_fog
                add_cloud +=valid_cloud
                allvalid_fog = np.where(b == 1, 0, 1)
                total_fog += allvalid_fog
                #print(add_cloud.max(),add_fog.max())

    ratio_fog = add_fog / total_fog
    ratio_cloud = add_cloud/total_fog
    fix_fog = np.where((add_fog != 0), 1, 0)
    rat_fog = datacov(ratio_fog, 9)
    fi_fog = datacov(fix_fog, 9)
    rat_cloud = datacov(ratio_cloud, 9)

    
    with h5py.File(outfc, "w") as fc:
        fc.attrs['Satellite Name'] = "H08"
        fc.attrs['Sensor Name'] = "AHI"
        fc.attrs['Data Level'] = "L2" 
        fc.attrs['date_created'] = "2020-05-25"
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
        fc.attrs['time_coverage_end'] =  YMSS
        fc.attrs['Version Of Software'] = "v1.0"    

        d1 = fc.create_dataset("fog_ratio", data=rat_fog)
        d1.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT" 
        d1.attrs['Description'] = "/"
        d1.attrs['ProjectionResolution'] = 0.02
        d1.attrs['ProjectionMinLongitude'] = 105.0
        d1.attrs['ProjectionMaxLongitude'] = 135.0
        d1.attrs['ProjectionMinLatitude'] = 13.0
        d1.attrs['ProjectionMaxLatitude'] = 43.0
        d1.attrs['FillValue'] = 1
        d1.attrs['Slope'] = 1.0
        d1.attrs['Intercept'] = 0.0
        d1.attrs['units'] = "None"
        d1.attrs['valid_range'] = "[0,1]"
        
        d1 = fc.create_dataset("cloud_ratio", data=rat_cloud)
        d1.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT" 
        d1.attrs['Description'] = "/"
        d1.attrs['ProjectionResolution'] = 0.02
        d1.attrs['ProjectionMinLongitude'] = 105.0
        d1.attrs['ProjectionMaxLongitude'] = 135.0
        d1.attrs['ProjectionMinLatitude'] = 13.0
        d1.attrs['ProjectionMaxLatitude'] = 43.0
        d1.attrs['FillValue'] = 1
        d1.attrs['Slope'] = 1.0
        d1.attrs['Intercept'] = 0.0
        d1.attrs['units'] = "None"
        d1.attrs['valid_range'] = "[0,1]"
        
        d2 = fc.create_dataset("fog_add", data=add_fog)
        d2.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT" 
        d2.attrs['Description'] = "/"
        d2.attrs['ProjectionResolution'] = 0.02
        d2.attrs['ProjectionMinLongitude'] = 105.0
        d2.attrs['ProjectionMaxLongitude'] = 135.0
        d2.attrs['ProjectionMinLatitude'] = 13.0
        d2.attrs['ProjectionMaxLatitude'] = 43.0
        d2.attrs['FillValue'] = 1
        d2.attrs['Slope'] = 1.0
        d2.attrs['Intercept'] = 0.0
        d2.attrs['units'] = "None"
        d2.attrs['valid_range'] = "[0,150]"

        d3 = fc.create_dataset("fog_total", data=total_fog)

        d3.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT" 
        d3.attrs['Description'] = "/"
        d3.attrs['ProjectionResolution'] = 0.02
        d3.attrs['ProjectionMinLongitude'] = 105.0
        d3.attrs['ProjectionMaxLongitude'] = 135.0
        d3.attrs['ProjectionMinLatitude'] = 13.0
        d3.attrs['ProjectionMaxLatitude'] = 43.0
        d3.attrs['FillValue'] = 1
        d3.attrs['Slope'] = 1.0
        d3.attrs['Intercept'] = 0.0
        d3.attrs['units'] = "None"
        d3.attrs['valid_range'] = "[0,150]"

        d4 = fc.create_dataset("fog_fix", data=fi_fog)

        d4.attrs['long_name'] = "H8_PGS_L2_AHI_PRODUCT"
        d4.attrs['Description'] = "/"
        d4.attrs['ProjectionResolution'] = 0.02
        d4.attrs['ProjectionMinLongitude'] = 105.0
        d4.attrs['ProjectionMaxLongitude'] = 135.0
        d4.attrs['ProjectionMinLatitude'] = 13.0
        d4.attrs['ProjectionMaxLatitude'] = 43.0
        d4.attrs['FillValue'] = 1
        d4.attrs['Slope'] = 1.0
        d4.attrs['Intercept'] = 0.0
        d4.attrs['units'] = "None"
        d4.attrs['valid_range'] = "[0,150]"
    
    print('outfile=',outfc)
    plot(rat_fog,rat_cloud,outfc)
    plot2(rat_cloud,outfc)
    titlename = tmptitlename.format (stime=ymd,HM=hhmm)
    draw()

    