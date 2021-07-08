# Description:
# calculate wet bulb globe temperature

# the code was written based on the original code of Liljegren in C language which is available at https://github.com/mdljts/wbgt/blob/master/src/wbgt.c.original

#Reference:  
    # Liljegren, J. C., Carhart, R. A., Lawday, P., Tschopp, S. & Sharp, R. 
    # Modeling the Wet Bulb Globe Temperature Using Standard Meteorological Measurements. 
    # Journal of Occupational and Environmental Hygiene 5, 645–655 (2008).
    
# created by Qinqin Kong (07-04-2021)
import numpy as np
cimport numpy as np
cimport cython

from scipy.optimize.cython_optimize cimport brentq
from libc cimport math
from cython.parallel import prange
from libc.math cimport isnan
    
# define some constants
cdef double mair, mh2o, rgas, cp, stefanb, diamglobe, emisglobe, albglobe, emiswick, albwick, diamwick, lenwick,albsfc, ratio, rair, Pr, XTOL,RTOL, PI
# physical constants
mair = 28.97 # molecular weight of dry air (grams per mole)
mh2o = 18.015 # molecular weight of water vapor (grams per mole)
rgas = 8314.34 # ideal gas constant (J/kg mol · K)
cp = 1003.5 # Specific heat capacity of air at constant pressure (J·kg-1·K-1)
stefanb = 0.000000056696  # stefan-boltzmann constant
ratio = cp * mair * (mh2o**(-1))
rair = rgas * (mair**(-1))
Pr = cp * ((cp + 1.25 * rair)**(-1)) # Prandtl number 

#globe constants
diamglobe = 0.0508 # diameter of globe (m)
emisglobe = 0.95 # emissivity of globe
albglobe = 0.05 # albedo of globe

#wick constants
emiswick = 0.95 # emissivity of the wick
albwick = 0.4 # albedo of the wick
diamwick = 0.007 # diameter of the wick
lenwick = 0.0254 # length of the wick

#surface constant
albsfc=0.45

PI=3.1415926535897932

# define fused type to make the code accept both float and double type input
ctypedef fused mytype1:
    cython.float
    cython.double

ctypedef fused mytype2:
    cython.float
    cython.double
    
ctypedef fused mytype3:
    cython.float
    cython.double

ctypedef fused mytype4:
    cython.float
    cython.double
ctypedef fused mytype5:
    cython.float
    cython.double

ctypedef fused mytype6:
    cython.float
    cython.double
    
ctypedef fused mytype7:
    cython.float
    cython.double

ctypedef fused mytype8:
    cython.float
    cython.double
ctypedef fused mytype9:
    cython.float
    cython.double

ctypedef fused mytype10:
    cython.float
    cython.double

# user-defined struct for extra parameters
ctypedef struct Tg_params:
    double C0
    double C1
    double C2
    double C3
ctypedef struct Tnwb_params:
    double D0
    double D1
    double D2
    double D3
    double D4

cdef int lsrdt[6][8]
lsrdt[0][:] = [1, 1, 2, 4, 0, 5, 6, 0]
lsrdt[1][:] = [1, 2, 3, 4, 0, 5, 6, 0]
lsrdt[2][:] = [2, 2, 3, 4, 0, 4, 4, 0]
lsrdt[3][:] = [3, 3, 4, 4, 0, 0, 0, 0]
lsrdt[4][:] = [3, 4, 4, 4, 0, 0, 0, 0]
lsrdt[5][:] = [0, 0, 0, 0, 0, 0, 0, 0]    

cdef double urban_exp[6]
urban_exp[:] = [ 0.15, 0.15, 0.20, 0.25, 0.30, 0.30 ]


cdef double sunearth(date):
    # date: series of date and time of day
    # return sun-earth distance (astronomical units)
    cdef double days_J2000, g, distance
    days_J2000=(((date-np.datetime64('2000-01-01T12:00:00.000000000')).astype('timedelta64[m]'))/np.timedelta64(1, 'm'))/(60*24)
    g = ((357.528 + 0.9856003 * days_J2000)%360)*PI/180
    distance = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2.0 * g)
    return distance

cdef double viscosity(double tas) nogil:
    # tas: air temperature (K)
    # return air viscosity (kg/(m s))
    cdef double omega, visc
    omega=1.2945-tas/1141.176470588
    visc = 0.0000026693 * (math.sqrt(28.97 * tas)) * ((13.082689 * omega)**(-1))
    return visc

cdef double thermcond(double tas) nogil:
    # tas: air temperature (K)
    # return thermal conductivity of air (W/(m K))
    cdef double tc
    tc = (cp + 1.25 * rair) * viscosity(tas)
    return tc

cdef double esat(double tas,double ps) nogil:
    # tas: air temperature (K)
    # ps: surface pressure (Pa)
    # return saturation vapor pressure (Pa)
    cdef double es
    if tas>273.15:
        es = 611.21 * math.exp(17.502 * (tas - 273.15) *((tas - 32.18)**(-1)))
        es = (1.0007 + (3.46*10**(-6) * ps/100)) * es
    else:
        es = 611.15 * math.exp(22.452 * (tas - 273.15) * ((tas - 0.6)**(-1)))
        es=(1.0003 + (4.18*10**(-6) * ps/100)) * es
    return es

cdef double emisatm(double tas,double hurs,double ps) nogil:
    cdef double e,emis_atm
    e=hurs*0.01*(esat(tas,ps)*0.01)
    emis_atm=0.575*(e**0.143)
    return emis_atm

cdef double diffusivity(double tas,double ps) nogil:
    # tas: air temperature (K)
    # ps: surface pressure (Pa)
    # return diffusivity of water vapor in air (m2/s)
    return 2.471773765165648e-05 * ((tas *0.0034210563748421257) ** 2.334) * ((ps / 101325)**(-1))

cdef double h_evap(double tas) nogil:
    # tas: air temperature (K)
    # return heat of evaporation (J/(kg K))
    return 1665134.5+2370.0*tas

cdef int stab_srdt(double cosz, double sfcwind, double rsds) nogil:
    cdef int i, j
    if cosz>0:
        if rsds>=925.0:
            j=0
        elif rsds>=675.0:
            j=1
        elif rsds>=175.0:
            j=2
        else:
            j=3
        if sfcwind>=6.0:
            i=4
        elif sfcwind>=5.0:
            i=3
        elif sfcwind>=3.0:
            i=2
        elif sfcwind>=2.0:
            i=1
        else:
            i=0
    else:
        j=5
        if sfcwind>=2.5:
            i=2
        elif sfcwind>=2.0:
            i=1
        else:
            i=0
    return lsrdt[i][j]

cdef double wind2m(double sfcwind,double cosz,double rsds) nogil:
    # calculate 2 meter wind speed from 10 meter wind speed
    cdef int stability_class
    stability_class=stab_srdt(cosz, sfcwind, rsds)
    return math.fmax(sfcwind * math.pow(2.0/10.0, urban_exp[stability_class-1]),0.13)


cdef double h_sphere_in_air(double tas, double ps, double sfcwind) nogil:
    # tas: air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return convective heat tranfer coefficient for flow around a sphere (W/(m2 K))
    cdef double thermcon, density, Re, Nu, h
    thermcon = thermcond(tas)
    density = ps * ((rair * tas)**(-1))
    Re = sfcwind * density * diamglobe * ((viscosity(tas))**(-1))
    Nu = 2 + 0.6 * math.sqrt(Re) * math.pow(Pr,0.3333)
    h = Nu * thermcon * (diamglobe**(-1))
    return h

cdef double h_cylinder_in_air(double tas,double ps,double sfcwind) nogil:
    # tas: air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return convective heat transfer coefficient for a long cylinder (W/(m2 K))
    cdef double thermcon, density, Re, Nu, h
    thermcon = thermcond(tas)
    density = ps * ((rair * tas)**(-1))
    Re = sfcwind * density * diamwick * ((viscosity(tas))**(-1))
    Nu = 0.281 * (Re ** 0.6) * (Pr ** 0.44)
    h = Nu * thermcon * (diamwick**(-1))
    return h
    
cdef double fTg(double x, void *args) nogil:
    # equation of Tg that needs to be solved by iteration
    cdef Tg_params *myargs = <Tg_params *> args
    h=h_sphere_in_air(0.5*(myargs.C1+x),myargs.C2,myargs.C3)
    return (myargs.C0-((emisglobe*stefanb)**(-1))*h*(x-myargs.C1))-math.pow(x,4)

cdef double Tg_brentq_wrapper(Tg_params args, double xa, double xb, double xtol, double rtol, int mitr) nogil:
    # use scipy.optimize.brentq algorithm to solve Tg iteratively
    return brentq(fTg, xa, xb, <Tg_params *> &args, xtol, rtol, mitr, NULL)

cdef double fTnwb(double x, void *args) nogil:
    # equation of Tnwb that needs to be solved by iteration
    cdef Tnwb_params *myargs = <Tnwb_params *> args
    cdef double evap, es, density, Sc, h, Fatm
    evap=h_evap(0.5*(x+myargs.D0))
    es=esat(x,myargs.D1)
    density=myargs.D1*((0.5*(myargs.D0+x)*rair)**(-1))
    Sc=viscosity(0.5*(myargs.D0+x))*((density*diffusivity(0.5*(myargs.D0+x),myargs.D1))**(-1))
    h=h_cylinder_in_air(0.5*(myargs.D0+x), myargs.D1, myargs.D3)
    Fatm=myargs.D4-emiswick*stefanb*(x**4)
    return myargs.D0-evap*(ratio**(-1))*(es-myargs.D2)*((myargs.D1-es)**(-1))*((Pr*(Sc**(-1)))**0.56)+Fatm*(h**(-1))-x


cdef double Tnwb_brentq_wrapper(Tnwb_params args, double xa, double xb, double xtol, double rtol, int mitr) nogil:
    # use scipy.optimize.brentq algorithm to solve Tnwb iteratively
    return brentq(fTnwb, xa, xb, <Tnwb_params *> &args, xtol, rtol, mitr, NULL)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Tg_GCM_10mwind(mytype1[:, :, ::1] tas,mytype2[:, :, ::1] ps,mytype3[:, :, ::1] sfcwind, mytype4[:, :, ::1] rsds,mytype5[:, :, ::1] rsus,mytype6[:, :, ::1] rlds,mytype7[:, :, ::1] rlus, mytype8[:, :, ::1] fdir,mytype9[:, :, ::1] cosz,double xtol=0.01, double rtol=0.0, int mitr=1000):
    # tas: air temperature (K)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # rsus: surface reflected solar radiation (w/m2)
    # rlds: surface downward long-wave radiation (w/m2)
    # rlus: surface upwelling long-wave radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # return outdoor black globe temperature (K)
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = tas.shape[0]
    cdef Py_ssize_t y_max = tas.shape[1]
    cdef Py_ssize_t z_max = tas.shape[2]
    cdef Tg_params args
    cdef double xa,xb
    result = np.zeros((x_max, y_max, z_max), dtype=np.float64)
    cdef double[:, :, ::1] result_view = result
    for i in prange(x_max,nogil=True):
        for j in range(y_max):
            for k in range(z_max):
                if isnan(tas[i,j,k]) or isnan(ps[i,j,k]) or isnan(sfcwind[i,j,k]) or isnan(rsds[i,j,k]) or isnan(rsus[i,j,k]) or isnan(rlds[i,j,k]) or isnan(rlus[i,j,k]) or isnan(fdir[i,j,k]) or isnan(cosz[i,j,k]):
                    result_view[i,j,k]=math.NAN
                else:
                    args.C0=0.5*(stefanb**(-1))*(rlds[i,j,k]+rlus[i,j,k])+rsds[i,j,k]*((2*emisglobe*stefanb)**(-1))*(1-albglobe)*(1-fdir[i,j,k]+0.5*fdir[i,j,k]*(cosz[i,j,k]**(-1)))+(1-albglobe)*((2*emisglobe*stefanb)**(-1))*rsus[i,j,k]
                    args.C1=tas[i,j,k]
                    args.C2=ps[i,j,k]
                    args.C3=wind2m(sfcwind[i,j,k],cosz[i,j,k],rsds[i,j,k])
                    xa=tas[i,j,k]-50
                    xb=tas[i,j,k]+90
                    result_view[i,j,k]=Tg_brentq_wrapper(args, xa, xb, xtol, rtol, mitr)
    return result

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Tg_GCM_2mwind(mytype1[:, :, ::1] tas,mytype2[:, :, ::1] ps,mytype3[:, :, ::1] sfcwind, mytype4[:, :, ::1] rsds,mytype5[:, :, ::1] rsus,mytype6[:, :, ::1] rlds,mytype7[:, :, ::1] rlus, mytype8[:, :, ::1] fdir, mytype9[:, :, ::1] cosz,double xtol=0.01, double rtol=0.0, int mitr=1000):
    # tas: air temperature (K)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # rsus: surface reflected solar radiation (w/m2)
    # rlds: surface downward long-wave radiation (w/m2)
    # rlus: surface upwelling long-wave radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # return outdoor black globe temperature (K)
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = tas.shape[0]
    cdef Py_ssize_t y_max = tas.shape[1]
    cdef Py_ssize_t z_max = tas.shape[2]
    cdef Tg_params args
    cdef double xa,xb
    result = np.zeros((x_max, y_max, z_max), dtype=np.float64)
    cdef double[:, :, ::1] result_view = result
    for i in prange(x_max,nogil=True):
        for j in range(y_max):
            for k in range(z_max):
                if isnan(tas[i,j,k]) or isnan(ps[i,j,k]) or isnan(sfcwind[i,j,k]) or isnan(rsds[i,j,k]) or isnan(rsus[i,j,k]) or isnan(rlds[i,j,k]) or isnan(rlus[i,j,k]) or isnan(fdir[i,j,k]) or isnan(cosz[i,j,k]):
                    result_view[i,j,k]=math.NAN
                else:
                    args.C0=0.5*(stefanb**(-1))*(rlds[i,j,k]+rlus[i,j,k])+rsds[i,j,k]*((2*emisglobe*stefanb)**(-1))*(1-albglobe)*(1-fdir[i,j,k]+0.5*fdir[i,j,k]*(cosz[i,j,k]**(-1)))+(1-albglobe)*((2*emisglobe*stefanb)**(-1))*rsus[i,j,k]
                    args.C1=tas[i,j,k]
                    args.C2=ps[i,j,k]
                    args.C3=sfcwind[i,j,k]
                    xa=tas[i,j,k]-50
                    xb=tas[i,j,k]+90
                    result_view[i,j,k]=Tg_brentq_wrapper(args, xa, xb, xtol, rtol, mitr)
    return result

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Tg_Liljegren_10mwind(mytype1[:, :, ::1] tas, mytype2[:, :, ::1] hurs, mytype3[:, :, ::1] ps, mytype4[:, :, ::1] sfcwind, 
                          mytype5[:, :, ::1] rsds, mytype6[:, :, ::1] fdir, mytype7[:, :, ::1] cosz,double xtol=0.001, double rtol=0.0, int mitr=1000000):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # return outdoor black globe temperature (K)
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = tas.shape[0]
    cdef Py_ssize_t y_max = tas.shape[1]
    cdef Py_ssize_t z_max = tas.shape[2]
    cdef Tg_params args
    cdef double xa,xb
    result = np.zeros((x_max, y_max, z_max), dtype=np.float64)
    cdef double[:, :, ::1] result_view = result
    for i in prange(x_max,nogil=True):
        for j in range(y_max):
            for k in range(z_max):
                if isnan(tas[i,j,k]) or isnan(hurs[i,j,k]) or isnan(ps[i,j,k]) or isnan(sfcwind[i,j,k]) or isnan(rsds[i,j,k]) or isnan(fdir[i,j,k]) or isnan(cosz[i,j,k]):
                    result_view[i,j,k]=math.NAN
                else:
                    args.C0=0.5*(1+emisatm(tas[i,j,k],hurs[i,j,k],ps[i,j,k]))*(tas[i,j,k]**4)+((2*emisglobe*stefanb)**(-1))*rsds[i,j,k]*(1-albglobe)*(1+(0.5*(cosz[i,j,k]**(-1))-1)*fdir[i,j,k]+albsfc)
                    args.C1=tas[i,j,k]
                    args.C2=ps[i,j,k]
                    args.C3=wind2m(sfcwind[i,j,k],cosz[i,j,k],rsds[i,j,k])
                    xa=tas[i,j,k]-50
                    xb=tas[i,j,k]+90
                    result_view[i,j,k]=Tg_brentq_wrapper(args, xa, xb, xtol, rtol, mitr)
    return result

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Tg_Liljegren_2mwind(mytype1[:, :, ::1] tas, mytype2[:, :, ::1] hurs, mytype3[:, :, ::1] ps, mytype4[:, :, ::1] sfcwind, 
                         mytype5[:, :, ::1] rsds, mytype6[:, :, ::1] fdir, mytype7[:, :, ::1] cosz,double xtol=0.001, double rtol=0.0, int mitr=1000000):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # return outdoor black globe temperature (K)
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = tas.shape[0]
    cdef Py_ssize_t y_max = tas.shape[1]
    cdef Py_ssize_t z_max = tas.shape[2]
    cdef Tg_params args
    cdef double xa,xb
    result = np.zeros((x_max, y_max, z_max), dtype=np.float64)
    cdef double[:, :, ::1] result_view = result
    for i in prange(x_max,nogil=True):
        for j in range(y_max):
            for k in range(z_max):
                if isnan(tas[i,j,k]) or isnan(hurs[i,j,k]) or isnan(ps[i,j,k]) or isnan(sfcwind[i,j,k]) or isnan(rsds[i,j,k]) or isnan(fdir[i,j,k]) or isnan(cosz[i,j,k]):
                    result_view[i,j,k]=math.NAN
                else:
                    args.C0=0.5*(1+emisatm(tas[i,j,k],hurs[i,j,k],ps[i,j,k]))*(tas[i,j,k]**4)+((2*emisglobe*stefanb)**(-1))*rsds[i,j,k]*(1-albglobe)*(1+(0.5*(cosz[i,j,k]**(-1))-1)*fdir[i,j,k]+albsfc)
                    args.C1=tas[i,j,k]
                    args.C2=ps[i,j,k]
                    args.C3=sfcwind[i,j,k]
                    xa=tas[i,j,k]-50
                    xb=tas[i,j,k]+90
                    result_view[i,j,k]=Tg_brentq_wrapper(args, xa, xb, xtol, rtol, mitr)
    return result

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Tnwb_GCM_10mwind(mytype1[:, :, ::1] tas,mytype2[:, :, ::1] hurs,mytype3[:, :, ::1] ps,mytype4[:, :, ::1] sfcwind, mytype5[:, :, ::1] rsds,mytype6[:, :, ::1] rsus,mytype7[:, :, ::1] rlds,mytype8[:, :, ::1] rlus, mytype9[:, :, ::1] fdir,mytype10[:, :, ::1] cosz, double xtol=0.01, double rtol=0.0, int mitr=100000000):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # rsus: surface reflected solar radiation (w/m2)
    # rlds: surface downward long-wave radiation (w/m2)
    # rlus: surface upwelling long-wave radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # return outdoor natural wet bulb temperature (K)
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = tas.shape[0]
    cdef Py_ssize_t y_max = tas.shape[1]
    cdef Py_ssize_t z_max = tas.shape[2]
    cdef Tnwb_params args
    cdef double xa,xb
    result = np.zeros((x_max, y_max, z_max), dtype=np.float64)
    cdef double[:, :, ::1] result_view = result
    for i in prange(x_max,nogil=True):
        for j in range(y_max):
            for k in range(z_max):
                if isnan(tas[i,j,k]) or isnan(ps[i,j,k]) or isnan(sfcwind[i,j,k]) or isnan(rsds[i,j,k]) or isnan(rsus[i,j,k]) or isnan(rlds[i,j,k]) or isnan(rlus[i,j,k]) or isnan(fdir[i,j,k]) or isnan(cosz[i,j,k]) or isnan(hurs[i,j,k]):
                    result_view[i,j,k]=math.NAN
                else:
                    args.D0=tas[i,j,k]
                    args.D1=ps[i,j,k]
                    args.D2=hurs[i,j,k]*0.01*esat(tas[i,j,k],ps[i,j,k])
                    args.D3=wind2m(sfcwind[i,j,k],cosz[i,j,k],rsds[i,j,k])
                    args.D4=emiswick*0.5*(rlds[i,j,k]+rlus[i,j,k])+(1+diamwick*((4*lenwick)**(-1)))*(1-albwick)*(1-fdir[i,j,k])*rsds[i,j,k]+(math.tan((math.acos(cosz[i,j,k])))*(PI**(-1))+diamwick*((4*lenwick)**(-1)))*(1-albwick)*fdir[i,j,k]*rsds[i,j,k]+(1-albwick)*rsus[i,j,k]
                    xa=tas[i,j,k]-((100-hurs[i,j,k])/5.0)-50
                    xb=math.fmin(tas[i,j,k]+70,340.0)
                    result_view[i,j,k]=Tnwb_brentq_wrapper(args, xa, xb, xtol, rtol, mitr)
    return result

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Tnwb_GCM_2mwind(mytype1[:, :, ::1] tas,mytype2[:, :, ::1] hurs,mytype3[:, :, ::1] ps,mytype4[:, :, ::1] sfcwind, mytype5[:, :, ::1] rsds,mytype6[:, :, ::1] rsus,mytype7[:, :, ::1] rlds,mytype8[:, :, ::1] rlus,mytype9[:, :, ::1] fdir,mytype10[:, :, ::1] cosz, double xtol=0.01, double rtol=0.0, int mitr=100000000):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # rsus: surface reflected solar radiation (w/m2)
    # rlds: surface downward long-wave radiation (w/m2)
    # rlus: surface upwelling long-wave radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # return outdoor natural wet bulb temperature (K)
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = tas.shape[0]
    cdef Py_ssize_t y_max = tas.shape[1]
    cdef Py_ssize_t z_max = tas.shape[2]
    cdef Tnwb_params args
    cdef double xa,xb
    result = np.zeros((x_max, y_max, z_max), dtype=np.float64)
    cdef double[:, :, ::1] result_view = result
    for i in prange(x_max,nogil=True):
        for j in range(y_max):
            for k in range(z_max):
                if isnan(tas[i,j,k]) or isnan(ps[i,j,k]) or isnan(sfcwind[i,j,k]) or isnan(rsds[i,j,k]) or isnan(rsus[i,j,k]) or isnan(rlds[i,j,k]) or isnan(rlus[i,j,k]) or isnan(fdir[i,j,k]) or isnan(cosz[i,j,k]) or isnan(hurs[i,j,k]):
                    result_view[i,j,k]=math.NAN
                else:
                    args.D0=tas[i,j,k]
                    args.D1=ps[i,j,k]
                    args.D2=hurs[i,j,k]*0.01*esat(tas[i,j,k],ps[i,j,k])
                    args.D3=sfcwind[i,j,k]
                    args.D4=emiswick*0.5*(rlds[i,j,k]+rlus[i,j,k])+(1+diamwick*((4*lenwick)**(-1)))*(1-albwick)*(1-fdir[i,j,k])*rsds[i,j,k]+(math.tan((math.acos(cosz[i,j,k])))*(PI**(-1))+diamwick*((4*lenwick)**(-1)))*(1-albwick)*fdir[i,j,k]*rsds[i,j,k]+(1-albwick)*rsus[i,j,k]
                    xa=tas[i,j,k]-((100-hurs[i,j,k])/5.0)-50
                    xb=math.fmin(tas[i,j,k]+70,340.0)
                    result_view[i,j,k]=Tnwb_brentq_wrapper(args, xa, xb, xtol, rtol, mitr)
    return result

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Tnwb_Liljegren_10mwind(mytype1[:, :, ::1] tas, mytype2[:, :, ::1] hurs, mytype3[:, :, ::1] ps, mytype4[:, :, ::1] sfcwind, 
                            mytype5[:, :, ::1] rsds, mytype6[:, :, ::1] fdir, mytype7[:, :, ::1] cosz, double xtol=0.001, double rtol=0.0, int mitr=1000000):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # return outdoor natural wet bulb temperature (K)
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = tas.shape[0]
    cdef Py_ssize_t y_max = tas.shape[1]
    cdef Py_ssize_t z_max = tas.shape[2]
    cdef Tnwb_params args
    cdef double xa,xb
    result = np.zeros((x_max, y_max, z_max), dtype=np.float64)
    cdef double[:, :, ::1] result_view = result
    for i in prange(x_max,nogil=True):
        for j in range(y_max):
            for k in range(z_max):
                if isnan(tas[i,j,k]) or isnan(hurs[i,j,k]) or isnan(ps[i,j,k]) or isnan(sfcwind[i,j,k]) or isnan(rsds[i,j,k]) or isnan(fdir[i,j,k]) or isnan(cosz[i,j,k]):
                    result_view[i,j,k]=math.NAN
                else:
                    args.D0=tas[i,j,k]
                    args.D1=ps[i,j,k]
                    args.D2=hurs[i,j,k]*0.01*esat(args.D0,args.D1)
                    args.D3=wind2m(sfcwind[i,j,k],cosz[i,j,k],rsds[i,j,k])
                    args.D4=emiswick*0.5*stefanb*(args.D0**4)*(emisatm(args.D0,hurs[i,j,k],args.D1)+1)+(1-albwick)*rsds[i,j,k]*((1+diamwick*((4*lenwick)**(-1)))*(1-fdir[i,j,k])+(math.tan((math.acos(cosz[i,j,k])))*(PI**(-1))+diamwick*((4*lenwick)**(-1)))*fdir[i,j,k]+albsfc)
                    xa=tas[i,j,k]-((100-hurs[i,j,k])/5.0)-50
                    xb=math.fmin(tas[i,j,k]+70,340.0)
                    result_view[i,j,k]=Tnwb_brentq_wrapper(args, xa, xb, xtol, rtol, mitr)
    return result

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Tnwb_Liljegren_2mwind(mytype1[:, :, ::1] tas, mytype2[:, :, ::1] hurs, mytype3[:, :, ::1] ps, mytype4[:, :, ::1] sfcwind, 
                           mytype5[:, :, ::1] rsds, mytype6[:, :, ::1] fdir, mytype7[:, :, ::1] cosz, double xtol=0.001, double rtol=0.0, int mitr=1000000):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # return outdoor natural wet bulb temperature (K)
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = tas.shape[0]
    cdef Py_ssize_t y_max = tas.shape[1]
    cdef Py_ssize_t z_max = tas.shape[2]
    cdef Tnwb_params args
    cdef double xa,xb
    result = np.zeros((x_max, y_max, z_max), dtype=np.float64)
    cdef double[:, :, ::1] result_view = result
    for i in prange(x_max,nogil=True):
        for j in range(y_max):
            for k in range(z_max):
                if isnan(tas[i,j,k]) or isnan(hurs[i,j,k]) or isnan(ps[i,j,k]) or isnan(sfcwind[i,j,k]) or isnan(rsds[i,j,k]) or isnan(fdir[i,j,k]) or isnan(cosz[i,j,k]):
                    result_view[i,j,k]=math.NAN
                else:
                    args.D0=tas[i,j,k]
                    args.D1=ps[i,j,k]
                    args.D2=hurs[i,j,k]*0.01*esat(args.D0,args.D1)
                    args.D3=sfcwind[i,j,k]
                    args.D4=emiswick*0.5*stefanb*(args.D0**4)*(emisatm(args.D0,hurs[i,j,k],args.D1)+1)+(1-albwick)*rsds[i,j,k]*((1+diamwick*((4*lenwick)**(-1)))*(1-fdir[i,j,k])+(math.tan((math.acos(cosz[i,j,k])))*(PI**(-1))+diamwick*((4*lenwick)**(-1)))*fdir[i,j,k]+albsfc)
                    xa=tas[i,j,k]-((100-hurs[i,j,k])/5.0)-50
                    xb=math.fmin(tas[i,j,k]+70,340.0)
                    result_view[i,j,k]=Tnwb_brentq_wrapper(args, xa, xb, xtol, rtol, mitr)
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
def fdir_3d(mytype1[:, :, :] cosza,mytype2[:, :, :] coszda,mytype3[:, :, :] rsds, date):
    # cosza: average cosine zenith angle during each interval
    # coszda: average cosine zenith angle during only the sunlit period of each interval
    # rsds: surface downward solar radiation (w/m2)
    # date: time series of date and time of day
    # return ratio of direct solar radiation
    cdef mytype1[:, :, ::1] cosza_view=cosza.copy()
    cdef mytype2[:, :, ::1] coszda_view=coszda.copy()
    cdef mytype3[:, :, ::1] rsds_view=rsds.copy()
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t x_max = rsds.shape[0]
    cdef Py_ssize_t y_max = rsds.shape[1]
    cdef Py_ssize_t z_max = rsds.shape[2]
    f=np.zeros((rsds.shape[0], rsds.shape[1], rsds.shape[2]), dtype=np.float64)
    cdef double[:,:,::1] f_view=f
    cdef double d, s_star
    for i in range(x_max):
        d=sunearth(date[i])
        for j in prange(y_max,nogil=True):
            for k in range(z_max):
                if cosza_view[i,j,k]<=math.cos(89.5/180*PI) or rsds_view[i,j,k]<=0:
                    f_view[i,j,k]=0
                else:
                    s_star = math.fmin(rsds_view[i,j,k]*((1367*coszda_view[i,j,k]*(d**(-2)))**(-1)),0.85)
                    f_view[i,j,k] = math.exp(3-1.34*s_star-1.65*(s_star**(-1)))
                    f_view[i,j,k] = math.fmax(math.fmin(f_view[i,j,k],0.9),0.0)
    return f


@cython.wraparound(False)
@cython.boundscheck(False)
def Tg_GCM_3d(mytype1[:, :, :] tas,mytype2[:, :, :] ps,mytype3[:, :, :] sfcwind,mytype4[:, :, :] rsds,mytype5[:, :, :] rsus,mytype6[:, :, :] rlds,mytype7[:, :, :] rlus,mytype8[:, :, :] fdir,mytype9[:, :, :] cosz,is2mwind):
    # tas: air temperature (K)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # rsus: surface reflected solar radiation (w/m2)
    # rlds: surface downward long-wave radiation (w/m2)
    # rlus: surface upwelling long-wave radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # is2mwind: True for 2 meter wind, False for 10 meter wind
    # return outdoor black globe temperature (K)
    cdef mytype1[:, :, ::1] tas_view=tas.copy()
    cdef mytype2[:, :, ::1] ps_view=ps.copy()
    cdef mytype3[:, :, ::1] sfcwind_view=sfcwind.copy()
    cdef mytype4[:, :, ::1] rsds_view=rsds.copy()
    cdef mytype5[:, :, ::1] rsus_view=rsus.copy()
    cdef mytype6[:, :, ::1] rlds_view=rlds.copy()
    cdef mytype7[:, :, ::1] rlus_view=rlus.copy()
    cdef mytype8[:, :, ::1] fdir_view=fdir.copy()
    cdef mytype9[:, :, ::1] cosz_view=cosz.copy()
    if is2mwind:
        return Tg_GCM_2mwind(tas_view,ps_view,sfcwind_view,rsds_view,rsus_view,rlds_view,rlus_view,fdir_view,cosz_view)
    else:
        return Tg_GCM_10mwind(tas_view,ps_view,sfcwind_view,rsds_view,rsus_view,rlds_view,rlus_view,fdir_view,cosz_view)

@cython.wraparound(False)
@cython.boundscheck(False)
def Tnwb_GCM_3d(mytype1[:, :, :] tas,mytype2[:, :, :] hurs,mytype3[:, :, :] ps,mytype4[:, :, :] sfcwind,mytype5[:, :, :] rsds,mytype6[:, :, :] rsus,mytype7[:, :, :] rlds,mytype8[:, :, :] rlus,mytype9[:, :, :] fdir,mytype10[:, :, :] cosz,is2mwind):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # rsus: surface reflected solar radiation (w/m2)
    # rlds: surface downward long-wave radiation (w/m2)
    # rlus: surface upwelling long-wave radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # is2mwind: True for 2 meter wind, False for 10 meter wind
    # return outdoor natural wet bulb temperature (K)
    cdef mytype1[:, :, ::1] tas_view=tas.copy()
    cdef mytype2[:, :, ::1] hurs_view=hurs.copy()
    cdef mytype3[:, :, ::1] ps_view=ps.copy()
    cdef mytype4[:, :, ::1] sfcwind_view=sfcwind.copy()
    cdef mytype5[:, :, ::1] rsds_view=rsds.copy()
    cdef mytype6[:, :, ::1] rsus_view=rsus.copy()
    cdef mytype7[:, :, ::1] rlds_view=rlds.copy()
    cdef mytype8[:, :, ::1] rlus_view=rlus.copy()
    cdef mytype9[:, :, ::1] fdir_view=fdir.copy()
    cdef mytype10[:, :, ::1] cosz_view=cosz.copy()
    if is2mwind:
        return Tnwb_GCM_2mwind(tas_view,hurs_view,ps_view,sfcwind_view,rsds_view,rsus_view,rlds_view,rlus_view,fdir_view,cosz_view)
    else:
        return Tnwb_GCM_10mwind(tas_view,hurs_view,ps_view,sfcwind_view,rsds_view,rsus_view,rlds_view,rlus_view,fdir_view,cosz_view)
 

@cython.wraparound(False)
@cython.boundscheck(False)
def Tg_Liljegren_3d(mytype1[:, :, :] tas, mytype2[:, :, :] hurs, mytype3[:, :, :] ps, mytype4[:, :, :] sfcwind, 
                    mytype5[:, :, :] rsds, mytype6[:, :, :] fdir, mytype7[:, :, :] cosz,is2mwind):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # is2mwind: True for 2 meter wind, False for 10 meter wind
    # return outdoor black globe temperature (K)
    cdef mytype1[:, :, ::1] tas_view=tas.copy()
    cdef mytype2[:, :, ::1] hurs_view=hurs.copy()
    cdef mytype3[:, :, ::1] ps_view=ps.copy()
    cdef mytype4[:, :, ::1] sfcwind_view=sfcwind.copy()
    cdef mytype5[:, :, ::1] rsds_view=rsds.copy()
    cdef mytype6[:, :, ::1] fdir_view=fdir.copy()
    cdef mytype7[:, :, ::1] cosz_view=cosz.copy()
    if is2mwind:
        return Tg_Liljegren_2mwind(tas_view,hurs_view,ps_view,sfcwind_view,rsds_view,fdir_view,cosz_view)
    else:
        return Tg_Liljegren_10mwind(tas_view,hurs_view,ps_view,sfcwind_view,rsds_view,fdir_view,cosz_view)

@cython.wraparound(False)
@cython.boundscheck(False)
def Tnwb_Liljegren_3d(mytype1[:, :, :] tas, mytype2[:, :, :] hurs, mytype3[:, :, :] ps, mytype4[:, :, :] sfcwind, 
                      mytype5[:, :, :] rsds, mytype6[:, :, :] fdir, mytype7[:, :, :] cosz,is2mwind):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # cosz: average cosine zenith angle during only the sunlit period of each interval
    # is2mwind: True for 2 meter wind, False for 10 meter wind
    # return outdoor natural wet bulb temperature (K)
    cdef mytype1[:, :, ::1] tas_view=tas.copy()
    cdef mytype2[:, :, ::1] hurs_view=hurs.copy()
    cdef mytype3[:, :, ::1] ps_view=ps.copy()
    cdef mytype4[:, :, ::1] sfcwind_view=sfcwind.copy()
    cdef mytype5[:, :, ::1] rsds_view=rsds.copy()
    cdef mytype6[:, :, ::1] fdir_view=fdir.copy()
    cdef mytype7[:, :, ::1] cosz_view=cosz.copy()
    if is2mwind:
        return Tnwb_Liljegren_2mwind(tas_view,hurs_view,ps_view,sfcwind_view,rsds_view,fdir_view,cosz_view)
    else:
        return Tnwb_Liljegren_10mwind(tas_view,hurs_view,ps_view,sfcwind_view,rsds_view,fdir_view,cosz_view)

@cython.wraparound(False)
@cython.boundscheck(False)
def fdir(cza,czda,rsds,date):
    # cza: temporal average cosine zenith angle during each interval
    # czda: temporal average cosine zenith angle during only the sunlit part of each interval
    # rsds: surface downward solar radiation (w/m2)
    # date: date and time series that you want to calculate
    # return the ratio of direct solar radiation 
    if cza.ndim!=3:
        f=fdir_3d(np.atleast_3d(cza).transpose((2, 0, 1)),np.atleast_3d(czda).transpose((2, 0, 1)),np.atleast_3d(rsds).transpose((2, 0, 1)),date)
        return f.squeeze()
    else:
        return fdir_3d(cza,czda,rsds,date)

@cython.wraparound(False)
@cython.boundscheck(False)    
def Tg_GCM(tas,ps,sfcwind,rsds,rsus,rlds,rlus,fdir,cosz,is2mwind):
    # tas: air temperature (K)
    # sfcwind: wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # rsus: surface reflected solar radiation (w/m2)
    # rlds: surface downward long-wave radiation (w/m2)
    # rlus: surface upwelling long-wave radiation (w/m2)
    # fdir: the ratio of direct solar radiation
    # cosz: cosine zenith angle
    # is2mwind: True for 2 meter wind, False for 10 meter wind
    # return outdoor black globe temperature (K)
    if tas.ndim!=3:
        Tg=Tg_GCM_3d(np.atleast_3d(tas),np.atleast_3d(ps),np.atleast_3d(sfcwind),np.atleast_3d(rsds),np.atleast_3d(rsus),np.atleast_3d(rlds),np.atleast_3d(rlus),np.atleast_3d(fdir),np.atleast_3d(cosz),is2mwind)
        return Tg.squeeze()
    else:
        Tg=Tg_GCM_3d(tas,ps,sfcwind,rsds,rsus,rlds,rlus,fdir,cosz,is2mwind)
        return Tg
@cython.wraparound(False)
@cython.boundscheck(False)    
def Tnwb_GCM(tas,hurs,ps,sfcwind,rsds,rsus,rlds,rlus,fdir,cosz,is2mwind):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # rsus: surface reflected solar radiation (w/m2)
    # rlds: surface downward long-wave radiation (w/m2)
    # rlus: surface upwelling long-wave radiation (w/m2)
    # fdir: the ratio of direct solar radiation
    # cosz: cosine zenith angle
    # is2mwind:	True for 2 meter wind, False for 10 meter wind
    # return outdoor natural wet bulb temperature (K)
    if tas.ndim!=3:
        Tnwb=Tnwb_GCM_3d(np.atleast_3d(tas),np.atleast_3d(hurs),np.atleast_3d(ps),np.atleast_3d(sfcwind),np.atleast_3d(rsds),np.atleast_3d(rsus),np.atleast_3d(rlds),np.atleast_3d(rlus),np.atleast_3d(fdir),np.atleast_3d(cosz),is2mwind)
        return Tnwb.squeeze()
    else:
        Tnwb=Tnwb_GCM_3d(tas,hurs,ps,sfcwind,rsds,rsus,rlds,rlus,fdir,cosz,is2mwind)
        return Tnwb
@cython.wraparound(False)
@cython.boundscheck(False)
def Tnwb_Liljegren(tas,hurs,ps,sfcwind,rsds,fdir,cosz,is2mwind):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # fdir: the ratio of direct solar radiation
    # cosz: cosine zenith angle
    # is2mwind: True for 2 meter wind, False for 10 meter wind
    # return outdoor natural wet bulb temperature (K)
    if tas.ndim!=3:
        Tnwb=Tnwb_Liljegren_3d(np.atleast_3d(tas),np.atleast_3d(hurs),np.atleast_3d(ps),np.atleast_3d(sfcwind),np.atleast_3d(rsds),np.atleast_3d(fdir),np.atleast_3d(cosz),is2mwind)
        return Tnwb.squeeze()
    else:
        Tnwb=Tnwb_Liljegren_3d(tas,hurs,ps,sfcwind,rsds,fdir,cosz,is2mwind)
        return Tnwb
@cython.wraparound(False)
@cython.boundscheck(False)
def Tg_Liljegren(tas,hurs,ps,sfcwind,rsds,fdir,cosz,is2mwind):
    # tas: air temperature (K)
    # sfcwind: wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # fdir: the ratio of direct solar radiation
    # cosz: cosine zenith angle
    # is2mwind: True for 2 meter wind, False for 10 meter wind
    # return outdoor black globe temperature (K)
    if tas.ndim!=3:
        Tg=Tg_Liljegren_3d(np.atleast_3d(tas),np.atleast_3d(hurs),np.atleast_3d(ps),np.atleast_3d(sfcwind),np.atleast_3d(rsds),np.atleast_3d(fdir),np.atleast_3d(cosz),is2mwind)
        return Tg.squeeze()
    else:
        Tg=Tg_Liljegren_3d(tas,hurs,ps,sfcwind,rsds,fdir,cosz,is2mwind)
        return Tg
    
    
@cython.wraparound(False)
@cython.boundscheck(False)
def WBGT_Liljegren(tas,hurs,ps,sfcwind,rsds,fdir,cosz,is2mwind):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # fdir: the ratio of direct solar radiation
    # return outdoor wet bulb globe temperature (K)
    tg=Tg_Liljegren(tas,hurs,ps,sfcwind,rsds,fdir,cosz,is2mwind)
    tnwb=Tnwb_Liljegren(tas,hurs,ps,sfcwind,rsds,fdir,cosz,is2mwind)
    wbgt=0.7*tnwb+0.2*tg+0.1*tas
    return wbgt
@cython.wraparound(False)
@cython.boundscheck(False)
def WBGT_GCM(tas,hurs,ps,sfcwind,rsds,rsus,rlds,rlus,fdir,cosz,is2mwind):
    # tas: air temperature (K)
    # hurs: relative humidity (%)
    # sfcwind: 2 meter wind speed (m/s)
    # ps: surface pressure (Pa)
    # rsds: surface downward solar radiation (w/m2)
    # rsus: surface reflected solar radiation (w/m2)
    # rlds: surface downward long-wave radiation (w/m2)
    # rlus: surface upwelling long-wave radiation (w/m2)
    # fdir: the ratio of direct solar radiation 
    # return outdoor wet bulb globe temperature (K)
    tg=Tg_GCM(tas,ps,sfcwind,rsds,rsus,rlds,rlus,fdir,cosz,is2mwind)
    tnwb=Tnwb_GCM(tas,hurs,ps,sfcwind,rsds,rsus,rlds,rlus,fdir,cosz,is2mwind)
    wbgt=0.7*tnwb+0.2*tg+0.1*tas
    return wbgt

