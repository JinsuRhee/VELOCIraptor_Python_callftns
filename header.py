#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import h5py
from scipy import interpolate
from scipy.io import FortranFile
import scipy.integrate as integrate
import statistics
from mpl_toolkits import mplot3d

import pickle

from fortran.find_domain_py import find_domain_py
from fortran.get_ptcl_py import get_ptcl_py
from fortran.get_flux_py import get_flux_py

##-----
## General Settings
##-----
num_thread = int(40)

##-----
## Path Settings
##-----
dir_raw     = '/storage5/FORNAX/KISTI_OUTPUT/l10006/'
dir_catalog = '/storage5/FORNAX/VELOCIraptor/l10006/'

##-----
## VR output-related
##-----
column_list     = ['ID', 'ID_mbp', 'hostHaloID', 'numSubStruct', 'Structuretype', 'Mvir', 'Mass_tot', 'Mass_FOF', 
                   'Mass_200mean', 'Efrac', 'Mass_200crit', 'Rvir', 'R_size', 'R_200mean', 'R_200crit', 
                   'R_HalfMass', 'R_HalfMass_200mean', 'R_HalfMass_200crit', 'Rmax', 'Xc', 'Yc', 'Zc', 'VXc', 
                   'VYc', 'VZc', 'Lx', 'Ly', 'Lz', 'sigV', 'Vmax', 'npart']
gal_properties  = ['SFR', 'ABmag']
flux_list = ['u', 'g', 'r', 'i', 'z']
flux_zp = np.double(np.array([895.5*1e-11, 466.9*1e-11, 278.0*1e-11, 185.2*1e-11, 131.5*1e-11]))

##-----
## RAMSES-related Settings
##-----
simulation_type='FN'
if(simulation_type=='NH'):
    r_type_llint    = False
    r_type_family   = False
    r_type_neff     = 4096
    r_type_ndomain  = 4800
if(simulation_type=='NH2'):
    r_type_llint    = False
    r_type_family   = True
    r_type_neff     = 4096
    r_type_ndomain  = 480
if(simulation_type=='FN'):
    r_type_llint    = False
    r_type_family   = True
    r_type_neff     = 2048
    r_type_ndomain  = 480

##-----
## SSP TYPE
##-----
ssp_type    = 'chab'

##-----
## LOAD GALAXY
##      TO DO
##      *) do not use 'imglist.txt'
##-----
def f_rdgal(n_snap, id0, datalist=column_list, horg='g', gprop=gal_properties, directory=dir_catalog):
    
    """
    description
    """
    if(horg=='h'): directory+='Halo/VR_Halo/snap_%0.4d'%n_snap+"/"
    elif(horg=='g'): directory+='Galaxy/VR_Galaxy/snap_%0.4d'%n_snap+"/"
    else: print("Halo or Galaxy Error!")
    
    ## Get file list
    if(id0>=0): flist=[directory + 'GAL_%0.6d'%id0 + '.hdf5']
    else: 
        flist=os.system('ls '+directory+'GAL_*.hdf5 > imglist.txt')
                          #flist=os.system('find -type f -name "'+proj_images_dir+'*.pickle" -print0 | xargs -0 -n 10 ls > '+proj_images_dir+'imglist.dat')
        flist=np.loadtxt("imglist.txt", dtype=str)
        ## find a more simple way
        
    dtype=[]
    for name in column_list:
        dtype=dtype+[(name, '<f8')]
        ##if(name=='SFR' and horg=='g'): dtype=dtype+[(name, 'object')]
        ##elif(name=='ABmag' and horg=='g'): dtype=dtype+[(name, 'object')]
        ##else: dtype=dtype+[(name, '<f8')]

    if(horg=='g'):
        column_list_additional=['Domain_List', 'Flux_List', 'MAG_R', 'SFR_R', 'SFR_T', 'ConFrac', 'CONF_R',
                               'isclump', 'rate', 'Aexp', 'snapnum']
        for name in gal_properties:
            column_list_additional=column_list_additional+[name]
    else:
        column_list_additional=['']###['Domain_List', 'ConFrac', 'CONF_R', 'rate', 'Aexp', 'snapnum']


    if(horg=='g'):
        for name in column_list_additional:
            if(name=='isclump'): dtype=dtype+[(name, '<f8')]
            elif(name=='rate'): dtype=dtype+[(name, '<f8')]
            elif(name=='Aexp'): dtype=dtype+[(name, '<f8')]
            elif(name=='snapnum'): dtype=dtype+[(name, '<f8')]
            else: dtype=dtype+[(name, 'object')]

    galdata=np.zeros(len(flist), dtype=dtype)
    for i, fn in enumerate(flist):
        dat= h5py.File(fn, 'r')
        for name in column_list:
            if(horg=='g'):
                xdata=dat.get("G_Prop/G_"+name)
                galdata[name][i]=np.array(xdata)
            else:
                if(name!='SFR' and name!='ABmag'):
                    xdata=dat.get("G_Prop/G_"+name)
                    galdata[name][i]=np.array(xdata)

        if(horg=='g'):
            for name in column_list_additional:
                if(name=='ConFrac'): xdata=dat.get("/G_Prop/G_ConFrac")
                elif(name=='CONF_R'): xdata=dat.get("/CONF_R")
                elif(name=='isclump'): xdata=dat.get("/isclump")
                elif(name=='rate'): xdata=dat.get("/rate")
                elif(name=='Aexp'): xdata=dat.get("/Aexp")
                ##elif(name=='snapnum'):
                ##    galdata['snapnum'][i]=n_snap
                ##    break
                ##else: xdata=dat.get(name)
                galdata[name][i]=np.array(xdata)
    return galdata

##-----
## LOAD GALAXY
##      TO DO LIST
##          *) INCLUDE FLUX & TIME COMPUTATION PARTS
##          *) Halo member load is not implemented
##-----
def f_rdptcl(n_snap, id0, horg='g', num_thread=num_thread,
    p_gyr=False, p_sfactor=False, p_mass=True, p_flux=False, 
    p_metal=False, p_id=False, flux_list=flux_list, 
    raw=False, boxrange=50., domlist=[0], 
    family=r_type_family, llint=r_type_llint, neff=r_type_neff, ndomain=r_type_ndomain,
    dir_raw=dir_raw, dir_catalog=dir_catalog):

    """
    Initial Settings
    """
    if(p_gyr==False and p_flux==True): p_gyr=True
    unit_l  = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=15, max_rows=1)[2])
    unit_t  = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=17, max_rows=1)[2])
    kms     = np.double(unit_l / unit_t / 1e5)
    unit_d  = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=16, max_rows=1)[2])
    unit_m  = unit_d * unit_l**3
    levmax  = np.int32(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=3, max_rows=1)[2])
    hindex  = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=21)[:,1:])
    omega_M = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=11, max_rows=1)[2])
    omega_B = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=14, max_rows=1)[2])

    dmp_mass    = 1.0/(neff*neff*neff)*(omega_M - omega_B)/omega_M
 
    #----- READ ID & Domain List
    #   (Might be skipped when raw==True)
    if(raw==False):
        if(horg=='h'): fname = dir_catalog + 'Halo/VR_Halo/snap_%0.4d'%n_snap+"/"
        elif(horg=='g'): fname = dir_catalog + 'Galaxy/VR_Galaxy/snap_%0.4d'%n_snap+"/"
        fname   += 'GAL_%0.6d'%id0+'.hdf5'
    
        dat     = h5py.File(fname, 'r')
        idlist  = np.array(dat.get("P_Prop/P_ID"))
        if(horg=='g'): domlist = np.array(dat.get("Domain_List"))
        else: domlist = np.zeros(1)
    else:
        idlist  = np.zeros(1, dtype=np.int64)
        domlist = np.zeros(ndomain, dtype=np.int32) - 1

        #----- Find Domain
        galtmp  = f_rdgal(n_snap, id0, horg=horg)

        xc  = galtmp['Xc']/unit_l * 3.086e21
        yc  = galtmp['Yc']/unit_l * 3.086e21
        zc  = galtmp['Zc']/unit_l * 3.086e21
        rr  = galtmp['R_HalfMass']/unit_l * 3.086e21
        larr    = np.zeros(20, dtype=np.int32)
        darr    = np.zeros(20, dtype='<f8')

        larr[0] = np.int32(len(xc))
        larr[1] = np.int32(len(domlist))
        larr[2] = np.int32(num_thread)
        larr[3] = np.int32(levmax)

        darr[0] = 50.
        if(boxrange!=None): darr[0] = boxrange / (rr * unit_l / 3.086e21)

        find_domain_py.find_domain(xc, yc, zc, rr, hindex, larr, darr)
        domlist     = find_domain_py.dom_list
        domlist     = domlist[0][:]

    domlist = np.int32(np.array(np.where(domlist > 0))[0] + 1)
    idlist  = np.int64(idlist)

    #----- LOAD PARTICLE INFO
    if(raw==False):
        larr    = np.zeros(20, dtype=np.int32)
        darr    = np.zeros(20, dtype='<f8')

        larr[0]     = np.int32(len(idlist))
        larr[1]     = np.int32(len(domlist))
        larr[2]     = np.int32(n_snap)
        larr[3]     = np.int32(num_thread)
        larr[10]    = np.int32(len(dir_raw))
        larr[17]    = 0

        if(horg=='g'): larr[11] = 10
        else: larr[11] = -10

        if(r_type_family==True): larr[18] = 100
        else: larr[18] = 0
        if(r_type_llint==True): larr[19] = 100
        else: larr[19] = 0

        if(horg=='h'): darr[11] = dmp_mass

        get_ptcl_py.get_ptcl(dir_raw, idlist, domlist, larr, darr)
        pinfo   = get_ptcl_py.ptcl

    else:
        larr    = np.zeros(20,dtype=np.int32)
        darr    = np.zeros(20,dtype='<f8')

        larr[1] = np.int32(len(domlist))
        larr[2] = np.int32(n_snap)
        larr[3] = np.int32(num_thread)
        larr[10]= np.int32(len(dir_raw))
        larr[17]= 100

        if(horg=='g'): larr[11] = 10
        else: larr[11] = -10

        if(r_type_family==True): larr[18] = 100
        else: larr[18] = 0
        if(r_type_llint==True): larr[19] = 100
        else: larr[19] = 0

        if(horg=='h'): darr[11] = dmp_mass

        get_ptcl_py.get_ptcl(dir_raw, idlist, domlist, larr, darr)
        pinfo   = get_ptcl_py.ptcl


    #----- EXTRACT
    n_old       = len(pinfo)*1.

    pinfo       = pinfo[np.where(pinfo[:,0]>-1e7)]
    n_new       = len(pinfo)*1.
    rate        = n_new / n_old

    pinfo[:,0]    *= unit_l / 3.086e21
    pinfo[:,1]    *= unit_l / 3.086e21
    pinfo[:,2]    *= unit_l / 3.086e21
    pinfo[:,3]    *= kms
    pinfo[:,4]    *= kms
    pinfo[:,5]    *= kms
    pinfo[:,6]    *= unit_m / 1.98892e33

    #----- OUTPUT ARRAY
    dtype   = [('xx', '<f8'), ('yy', '<f8'), ('zz', '<f8'), 
        ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'), 
        ('mass', '<f8'), ('sfact', '<f8'), ('gyr', '<f8'), ('metal', '<f8')]
    for name in flux_list:
        dtype   += [(name, '<f8')]

    ptcl    = np.zeros(np.int32(n_new), dtype=dtype)

    ptcl['xx'][:]   = pinfo[:,0]
    ptcl['yy'][:]   = pinfo[:,1]
    ptcl['zz'][:]   = pinfo[:,2]

    ptcl['vx'][:]   = pinfo[:,3]
    ptcl['vy'][:]   = pinfo[:,4]
    ptcl['vz'][:]   = pinfo[:,5]

    ptcl['mass'][:] = pinfo[:,6]
    ptcl['metal'][:]= pinfo[:,8]

    #----- COMPUTE GYR
    if(p_gyr==True):
        gyr = g_gyr(n_snap, pinfo[:,7])
        ptcl['gyr'][:]  = gyr['gyr'][:]
        ptcl['sfact'][:]= gyr['sfact'][:]

    #---- COMPUTE FLUX
    if(p_flux==True):
        for name in flux_list:
            ptcl[name][:] = g_flux(ptcl['mass'][:], ptcl['metal'][:], ptcl['gyr'][:],name)[name]

    return ptcl, rate, domlist

##-----
## Compute Flux
##-----
def g_flux_ssptable(ssp_type=ssp_type):

    fname   = 'table/ssp_' + ssp_type + '.pkl'
    isfile = os.path.isfile(fname)

    if(isfile==True):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        dname   = 'table/ssp_' + ssp_type

        metal   = np.loadtxt(dname + '/metal.txt', dtype='<f8')
        age     = np.loadtxt(dname + '/age.txt', dtype='<f8')
        lambd   = np.loadtxt(dname + '/lambda.txt', dtype='<f8')

        tr_curve    = []
        for name in flux_list:
            fname2  = dname + '/' + name + '_tr.txt'
            tr_curve.append(np.loadtxt(fname2, dtype='<f8'))

        flux   = np.zeros((len(metal), len(lambd), len(age)), dtype='<f8')
        ind = np.array(range(len(metal)),dtype='int32')
        for i in ind:
            fname2  = dname + '/flux_%0.1d'%i + '.txt'
            dum = np.array(np.loadtxt(fname2, dtype='<f8'))
            dum = np.reshape(dum, (len(age),len(lambd)))
            dum = np.transpose(dum)
            flux[i,:,:] = dum
            
        data    = {"metal":metal, "age":age, "lambda":lambd, "tr_curve":tr_curve, "flux":flux}

        with open(fname, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return data

def g_flux(mass, metal, age, fl_name, flux_list=flux_list, flux_zp=flux_zp, num_thread=num_thread):

    #----- ALLOCATE
    dtype   = ['']
    for name in flux_list:
        dtype   += [(name, '<f8')]
    dtype   = dtype[1:]
    data    = np.zeros(len(mass), dtype=dtype)

    #----- LOAD SSP TABLE
    ssp = g_flux_ssptable()

    #----- COMPUTE FLUX
    larr    = np.zeros(20, dtype=np.int32)
    darr    = np.zeros(20, dtype='<f8')
    larr[0]     = np.int32(len(mass))
    larr[1]     = np.int32(len(ssp['age']))
    larr[2]     = np.int32(len(ssp['metal']))
    larr[3]     = np.int32(len(ssp['lambda']))
    larr[10]    = num_thread

    ind     = np.array(range(len(flux_list)),dtype='int32')
    for i in ind:
        if(flux_list[i]!=fl_name): continue
        larr[4]     = np.int32(len(ssp['tr_curve'][:][i]))
        get_flux_py.get_flux(age, metal, mass, ssp['age'], ssp['metal'], ssp['lambda'], ssp['flux'], ssp['tr_curve'][i][:,0], ssp['tr_curve'][i][:,1], larr, darr)

        flux_tmp    = get_flux_py.flux * np.double(3.826e33) / (4. * np.pi * (10.0 * 3.08567758128e18)**2)

        dlambda = ssp['tr_curve'][i][1:,0] - ssp['tr_curve'][i][:-1,0]
        clambda = (ssp['tr_curve'][i][1:,0] + ssp['tr_curve'][i][:-1,0])/2.
        trcurve  = (ssp['tr_curve'][i][1:,1] + ssp['tr_curve'][i][:-1,1])/2.

        flux0   = np.sum(dlambda * clambda * trcurve * flux_zp[i])

        data[flux_list[i]]    = flux_tmp / flux0


    return data
##-----
## Compute Gyr from conformal time
##-----
def g_gyr(n_snap, t_conf, dir_raw=dir_raw):

    """
    Initial Settings
    """
    aexp = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=9, max_rows=1)[2])
    H0 = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=10, max_rows=1)[2])
    omega_M = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=11, max_rows=1)[2])
    omega_L = np.double(np.loadtxt(dir_raw+'output_%0.5d'%n_snap+"/info_%0.5d"%n_snap+".txt", dtype=object, skiprows=12, max_rows=1)[2])
    #----- Allocate
    data    = np.zeros(len(t_conf), dtype=[('sfact','<f8'), ('gyr','<f8')])

    #----- Get Confalmal T - Sfact Table
    c_table = g_cfttable(H0, omega_M, omega_L)

    #----- Get Sfactor by interpolation
    lint = interpolate.interp1d(c_table['conft'],c_table['sfact'],kind = 'quadratic')
    data['sfact'][:]   = lint(t_conf)

    #----- Get Gyr from Sfactor
    g_table = g_gyrtable(H0, omega_M, omega_L)
    lint = interpolate.interp1d(g_table['redsh'],g_table['gyr'],kind = 'quadratic')
    t0  = lint( 1./aexp - 1.)

    data['gyr'][:]      = lint( 1./data['sfact'][:] - 1.) - t0
    return data

##-----
## Generate or Load Sfactor-Gyr Table
##-----
def g_gyrtable_ftn(X, oM, oL):
    return 1./(1.+X)/np.sqrt(oM*(1.+X)**3 + oL)

def g_gyrtable(H0, oM, oL):

    fname   = 'table/gyr_%0.5d'%(H0*1000.) + '_%0.5d'%(oM*100000.) + '_%0.5d'%(oL*100000.) + '.pkl'
    isfile = os.path.isfile(fname)

    if(isfile==True):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        n_table = np.int32(10000)
        data    = np.zeros(n_table, dtype=[('redsh','<f8'),('gyr','<f8')])
        data['redsh'][:]    = 1./(np.array(range(n_table),dtype='<f8')/(n_table - 1.) * 0.98 + 0.02) - 1.
        data['gyr'][0]  = 0.

        ind     = np.array(range(n_table),dtype='int32')
        for i in ind:
            data['gyr'][i]  = integrate.quad(g_gyrtable_ftn,0.,data['redsh'][i], args=(oM, oL))[0]
            data['gyr'][i]  *= (1./H0 * np.double(3.08568025e19) / np.double(3.1536000e16))

        with open(fname, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data

##-----
## Generate or Load Confal-Gyr Table
##-----
def g_cfttable_ftn(X, oM, oL):
    return 1./(X**3 * np.sqrt(oM/X**3 + oL))

def g_cfttable(H0, oM, oL):

    fname   = 'table/cft_%0.5d'%(H0*1000.) + '_%0.5d'%(oM*100000.) + '_%0.5d'%(oL*100000.) + '.pkl'
    isfile = os.path.isfile(fname)

    if(isfile==True):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        n_table = np.int32(10000)
        data    = np.zeros(n_table, dtype=[('sfact','<f8'), ('conft','<f8')])
        data['sfact'][:]    = np.array(range(n_table),dtype='<f8')/(n_table - 1.) * 0.98 + 0.02

        ind     = np.array(range(n_table),dtype='int32')
        for i in ind:
            data['conft'][i]  = integrate.quad(g_cfttable_ftn,data['sfact'][i],1.,args=(oM,oL))[0] * (-1.)

        with open(fname, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data
