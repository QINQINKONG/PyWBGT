# Description:

# calculate:
    # cosine zenith angle:
    # instantaneous consine zenith angle: cosz
    # average cosine zenith angle during each interval (e.g. 3-hourly interval): cosza
    # average cosine zenith angle during only the sunlit part of each interval: coszda

# Reference:
    # Di Napoli, C., Hogan, R. J. & Pappenberger, F. Mean radiant temperature from global-scale numerical weather 
    # prediction models. Int J Biometeorol 64, 1233–1245 (2020).

# created by Qinqin Kong (07-04-2021)

import numpy as np
import xarray as xr
import cftime
cimport numpy as np
cimport cython
from libc cimport math
from cython.parallel import prange

# define constants
cdef double PI, DECL1,DECL2,DECL3,DECL4,DECL5,DECL6,DECL7
PI=3.1415926535897932
DECL1 = 0.006918
DECL2 = 0.399912
DECL3 = 0.070257
DECL4 = 0.006758
DECL5 = 0.000907
DECL6 = 0.002697
DECL7 = 0.00148

# define fused type to make the code accept both float and double type input
ctypedef fused mytype1:
    cython.float
    cython.double

ctypedef fused mytype2:
    cython.float
    cython.double



cdef double hourangel(double hour,double lon) nogil:
    # hour: hour of the day in UTC time
    # lon: longitude (radian)
    # return hour angle (radian)
    lon = lon if  lon<= PI else lon-2*PI
    if ((hour-12)*15*PI/180.0+lon)<-PI:
        return (hour-12)*15*PI/180.0+lon+2*PI
    elif ((hour-12)*15*PI/180.0+lon)>=PI:
        return (hour-12)*15*PI/180.0+lon-2*PI
    else:
        return (hour-12)*15*PI/180.0+lon
    
cdef double hstart(double h,double interval) nogil:
    # h: hour angle (radian)
    # interval: length of interval (e.g. 3 for 3-hourly interval)
    # return hour angle of the starting point of each interval (radian)
    k=interval/2.0
    if (h+k*15*PI/180)>=PI:
        hstart=h-k*15*PI/180
    elif (h-k*15*PI/180)<(-PI):
        hstart=h-k*15*PI/180+2*PI
    else:
        hstart=h-k*15*PI/180
    return hstart

cdef double hend(double h,double interval) nogil:
    # h: hour angle (radian)
    # interval: length of interval (e.g. 3 for 3-hourly interval)
    # return hour angle of the end point of each interval (radian)
    k=interval/2.0
    if (h+k*15*PI/180)>=PI:
        hend=h+k*15*PI/180-2*PI
    elif (h-k*15*PI/180)<(-PI):
        hend=h+k*15*PI/180
    else:
        hend=h+k*15*PI/180
    return hend


cdef double czda(double h_start,double h_end,double h_sunrise, double h_sunset, double lat, double Decl, double interval) nogil:
    # h_start: hour angle of the starting point of each interval (radian)
    # h_end: hour angle of the end point of each interval (radian)
    # h_sunrise: hour angle at sunrise (radian)
    # h_sunrise: hour angle at sunset (radian)
    # lat: latitude (radian)
    # Decl: solar declination angle (radian)
    # interval: length of interval (e.g. 3 for 3-hourly interval)
    # return: cosine zenith angle during only the sunlit part of each interval
    cdef double h_min, h_max, cosz, h_min1, h_max1, h_min2, h_max2
    if math.isnan(h_sunrise) and lat*Decl>0:
        h_min=h_start
        h_max=h_end
        cosz=math.sin(Decl)*math.sin(lat)+math.cos(Decl)*math.cos(lat)*(math.sin(h_max)-math.sin(h_min))*((interval*15.0/180.0*PI)**(-1))
    elif math.isnan(h_sunrise) and lat*Decl<0:
        cosz=0
    elif (h_start>h_sunset and h_end<h_sunrise) or (h_start<h_sunrise and h_end<h_sunrise) or (h_start>h_sunset and h_end>h_sunset):
        cosz=0
    elif (h_start>h_sunset and h_end<0 and h_end>h_sunrise):
        h_min=h_sunrise
        h_max=h_end
        cosz=math.sin(Decl)*math.sin(lat)+math.cos(Decl)*math.cos(lat)*(math.sin(h_max)-math.sin(h_min))*((h_max-h_min)**(-1))
    elif (h_start>0 and h_start<h_sunset and h_end<h_sunrise):
        h_min=h_start
        h_max=h_sunset
        cosz=math.sin(Decl)*math.sin(lat)+math.cos(Decl)*math.cos(lat)*(math.sin(h_max)-math.sin(h_min))*((h_max-h_min)**(-1))
    elif (h_start>0 and h_start<h_sunset and h_end<0 and h_end>h_sunrise):
        h_min1=h_start
        h_max1=h_sunset
        h_min2=h_sunrise
        h_max2=h_end
        cosz=(math.sin(Decl)*math.sin(lat)*(h_max1-h_min1)+math.cos(Decl)*math.cos(lat)*(math.sin(h_max1)-math.sin(h_min1))+math.sin(Decl)*math.sin(lat)*(h_max2-h_min2)+math.cos(Decl)*math.cos(lat)*(math.sin(h_max2)-math.sin(h_min2)))*((h_max1-h_min1+h_max2-h_min2)**(-1))
    else:
        h_min=math.fmax(h_sunrise,h_start)
        h_max=math.fmin(h_sunset,h_end)
        cosz=math.sin(Decl)*math.sin(lat)+math.cos(Decl)*math.cos(lat)*(math.sin(h_max)-math.sin(h_min))*((h_max-h_min)**(-1))
    return cosz


cdef double cza(double h_start,double h_end, double lat, double Decl, double interval) nogil:
    # h_start: hour angle of the starting point of each interval (radian)
    # h_end: hour angle of the end point of each interval (radian)
    # lat: latitude (radian)
    # Decl: solar declination angle (radian)
    # interval: length of interval (e.g. 3 for 3-hourly interval)
    # return: cosine zenith angle during each interval
    cdef double h_min, h_max, cosz, h_min1, h_max1, h_min2, h_max2
    if (h_start>0 and h_end<0):
        h_min1=h_start
        h_max1=PI
        h_min2=-PI
        h_max2=h_end
        cosz=(math.sin(Decl)*math.sin(lat)*(h_max1-h_min1)+math.cos(Decl)*math.cos(lat)*(math.sin(h_max1)-math.sin(h_min1))+math.sin(Decl)*math.sin(lat)*(h_max2-h_min2)+math.cos(Decl)*math.cos(lat)*(math.sin(h_max2)-math.sin(h_min2)))*((h_max1-h_min1+h_max2-h_min2)**(-1))
    else:
        h_min=h_start
        h_max=h_end
        cosz=math.sin(Decl)*math.sin(lat)+math.cos(Decl)*math.cos(lat)*(math.sin(h_max)-math.sin(h_min))*((h_max-h_min)**(-1))
    return cosz

@cython.wraparound(False)
@cython.boundscheck(False)
def cosz(date,mytype1[:,:] lat,mytype2[:,:] lon):
    # date: date and time series                             
    # lat: latitude (radian)
    # lon: longitude (radian)
    # return instantaneous cosine zenith angle
    cdef mytype1[:, ::1] lat_view=lat.copy()
    cdef mytype2[:, ::1] lon_view=lon.copy()
    doy=((date-date.astype('datetime64[Y]')).astype('timedelta64[D]'))/np.timedelta64(1, 'D')
    hour=((date-date.astype('datetime64[D]')).astype('timedelta64[h]'))/np.timedelta64(1, 'h')
    coz=np.zeros((date.shape[0], lon.shape[0], lon.shape[1]), dtype=np.float64)
    cdef double tod, h, g, Decl
    cdef double[:,:,::1] coz_view=coz
    cdef double[::1] doy_view=doy
    cdef double[::1] hour_view=hour
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = date.shape[0]
    cdef Py_ssize_t y_max = lon.shape[0]
    cdef Py_ssize_t z_max = lon.shape[1]
    
    for i in range(x_max):
        tod=(np.datetime64(str(date[i].astype('datetime64[Y]'))+'-12-31')-np.datetime64(str(date[i].astype('datetime64[Y]'))+'-01-01')+1)/np.timedelta64(1, 'D')
        for j in prange(y_max,nogil=True):
            for k in range(z_max):
                h=hourangel(hour_view[i],lon_view[j,k])
                g=(360.0*((tod)**(-1)))*(doy_view[i]+hour_view[i]/24.0)*(PI/180.0)
                Decl=DECL1-DECL2*math.cos(g)+DECL3*math.sin(g)-DECL4*math.cos(2*g)+DECL5*math.sin(2*g)-DECL6*math.cos(3*g)+DECL7*math.sin(3*g)
                coz_view[i,j,k]=math.sin(Decl)*math.sin(lat_view[j,k])+math.cos(Decl)*math.cos(lat_view[j,k])*math.cos(h)
    return coz

@cython.wraparound(False)
@cython.boundscheck(False)
def cosza(date,mytype1[:, :] lat,mytype2[:,:] lon, double interval):
    # date: date and time series                             
    # lat: latitude (radian)
    # lon: longitude (radian)
    # interval:	the length of the interval (e.g. 3 for 3-hourly interval) over which to calculate the average cosine zenith angle
    # return average cosine zenith angle during each interval
    cdef mytype1[:, ::1] lat_view=lat.copy()
    cdef mytype2[:, ::1] lon_view=lon.copy()
    doy=((date-date.astype('datetime64[Y]')).astype('timedelta64[D]'))/np.timedelta64(1, 'D')
    hour=((date-date.astype('datetime64[D]')).astype('timedelta64[h]'))/np.timedelta64(1, 'h')
    cdef double tod, h,g, Decl, h_start,h_end
    coz=np.zeros((date.shape[0], lon.shape[0], lon.shape[1]), dtype=np.float64)
    cdef double[:, :, ::1] coz_view = coz
    cdef double[::1] doy_view=doy
    cdef double[::1] hour_view=hour
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = date.shape[0]
    cdef Py_ssize_t y_max = lon.shape[0]
    cdef Py_ssize_t z_max = lon.shape[1]
    for i in range(x_max):
        tod=(np.datetime64(str(date[i].astype('datetime64[Y]'))+'-12-31')-np.datetime64(str(date[i].astype('datetime64[Y]'))+'-01-01')+1)/np.timedelta64(1, 'D')
        for j in prange(y_max,nogil=True):
            for k in range(z_max):
                h=hourangel(hour_view[i],lon_view[j,k])
                h_start=hstart(h,interval)
                h_end=hend(h,interval)
                g=(360.0*(tod**(-1)))*(doy_view[i]+hour_view[i]/24.0)*(PI/180.0)
                Decl=DECL1-DECL2*math.cos(g)+DECL3*math.sin(g)-DECL4*math.cos(2*g)+DECL5*math.sin(2*g)-DECL6*math.cos(3*g)+DECL7*math.sin(3*g)
                coz_view[i,j,k]=cza(h_start,h_end,lat_view[j,k],Decl,interval)
    return coz

@cython.wraparound(False)
@cython.boundscheck(False)
def coszda(date,mytype1[:, :] lat,mytype2[:,:] lon, double interval):
    # date: date and time series                             
    # lat: latitude (radian)
    # lon: longitude (radian)
    # interval:	the length of the interval (e.g. 3 for 3-hourly interval) over which to calculate the average cosine zenith angle
    # return average cosine zenith angle during only the sunlit period of each interval
    cdef mytype1[:, ::1] lat_view=lat.copy()
    cdef mytype2[:, ::1] lon_view=lon.copy()
    doy=((date-date.astype('datetime64[Y]')).astype('timedelta64[D]'))/np.timedelta64(1, 'D')
    hour=((date-date.astype('datetime64[D]')).astype('timedelta64[h]'))/np.timedelta64(1, 'h')
    cdef double tod, h,g, Decl, h_start,h_end, h_sunrise,h_sunset
    coz=np.zeros((date.shape[0], lon.shape[0], lon.shape[1]), dtype=np.float64)
    cdef double[:, :, ::1] coz_view = coz
    cdef double[::1] doy_view=doy
    cdef double[::1] hour_view=hour
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = date.shape[0]
    cdef Py_ssize_t y_max = lon.shape[0]
    cdef Py_ssize_t z_max = lon.shape[1]
    for i in range(x_max):
        tod=(np.datetime64(str(date[i].astype('datetime64[Y]'))+'-12-31')-np.datetime64(str(date[i].astype('datetime64[Y]'))+'-01-01')+1)/np.timedelta64(1, 'D')
        for j in prange(y_max,nogil=True):
            for k in range(z_max):
                h=hourangel(hour_view[i],lon_view[j,k])
                h_start=hstart(h,interval)
                h_end=hend(h,interval)
                g=(360.0*(tod**(-1)))*(doy_view[i]+hour_view[i]/24.0)*(PI/180.0)
                Decl=DECL1-DECL2*math.cos(g)+DECL3*math.sin(g)-DECL4*math.cos(2*g)+DECL5*math.sin(2*g)-DECL6*math.cos(3*g)+DECL7*math.sin(3*g)
                h_sunrise=-math.acos(-math.tan(Decl)*math.tan(lat_view[j,k]))
                h_sunset=math.acos(-math.tan(Decl)*math.tan(lat_view[j,k]))
                coz_view[i,j,k]=czda(h_start,h_end,h_sunrise,h_sunset,lat_view[j,k],Decl,interval)
    return coz