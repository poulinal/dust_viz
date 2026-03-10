#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Purpose: rotate particles to look head-on, plot spatially resolved SFR
v3: perscribes SFR based on added stellar mass, consistent with observational methods
v4: ability to fetch global SFR for different ages, checks for sufficient particles for analysis
    updating to do coordinate transformation in fetching protocol
v4.1: uses subfind catalogs to pull particles
Created on Mon May 31 13:21:44 2021

@author: bnmcd
"""

import requests
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
#from mpl_toolkits import mplot3d
import glob
import h5py
# import get_n_cores as gnc
import os
from multiprocessing import Pool
from os.path import exists
import pickle
from astropy.cosmology import Planck15
import astropy.units as u

global baseUrl,headers, numstar
numstar=[]

baseUrl = 'http://www.tng-project.org/api/'

api_key = os.getenv("API_KEY")
headers = {"api-key" : api_key}
h=0.6674


def n_cores(use_physical_cores=False, cores_var='NSLOTS'):
    # Check for HPC scheduler environment variable
    if cores_var in os.environ:
        return int(os.environ[cores_var])
    
    # Default to all available cores
    return multiprocessing.cpu_count()


def runallAlignmentTensor(cutout_file, output_file, bound_folder, snapnum):
    # galcat=np.load('/projectnb/gravlens/bnmcd/SFR/catalogs/subhalocat.npy',allow_pickle=True).item()
    # galcat=np.load(cutout_file, allow_pickle=True).item()    
    with h5py.File(cutout_file, 'r') as f:
        #print all keys
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: {obj.shape} {obj.dtype}")
            else:
                print(f"{name}/")
        
        f.visititems(print_structure)

    # s_mass=galcat['s_mass']
    # ID=galcat['subID']
    # galinds=np.arange(len(s_mass))
    # xgal=[]
    # # z='/projectnb/gravlens/bnmcd/SFR/catalogs/boundparts/'
    # for f in range(0,len(galinds)):
    #     #print(f)
    #     #if f==3138 or f==2739: continue
    #     if not exists(z+str(ID[f])+'.pkl'):
    #         xgal.append(f)
            
        
    # print(len(xgal))    
    # ncores=n_cores(use_physical_cores=False, cores_var='NSLOTS')
    
    # if __name__ == '__main__':
    #     with Pool(ncores) as p:
                
    #         p.map(galprof,galinds)


    #run only one
    galprof(0)
            
        
    return
    
def Itensor(x,y,z,mass):
    inertia=np.empty((3,3))
    inertia[0,0]=np.sum(np.multiply((np.multiply(y,y)+np.multiply(z,z)),mass))

    inertia[1,1]= np.sum(np.multiply((np.multiply(x,x)+np.multiply(z,z)),mass))   
    inertia[2,2]=np.sum(np.multiply((np.multiply(y,y)+np.multiply(x,x)),mass))
    
    inertia[0,1]=np.sum(np.multiply(np.multiply(x,y),mass))
    inertia[1,0]=inertia[0,1]
    inertia[0,2]=np.sum(np.multiply(np.multiply(x,z),mass))
    inertia[2,0]=inertia[0,2]
    inertia[1,2]=np.sum(np.multiply(np.multiply(y,z),mass))
    inertia[2,1]=inertia[1,2]
    
    return(inertia)

def safe_galprof(host):
    try: galprof(host)
    except: return()

def fetchpart(subID,rad,pos,hrad,saveall=True, bound=True):
    sub_prog_url = "http://www.tng-project.org/api/TNG100-1/snapshots/99/subhalos/"+str(subID)+"/"
    #print(sub_prog_url)
    sub=get(sub_prog_url)
    hosturl=sub['cutouts']['parent_halo']
    suburl=sub['cutouts']['subhalo']
    hostID=hosturl.split('/')[-2]
    cutname='/projectnb/gravlens/bnmcd/SFR/catalogs/hostcutouts/gas/cutout_'+hostID+'.hdf5'
    starcutname='/projectnb/gravlens/bnmcd/SFR/catalogs/hostcutouts/stars/cutout_'+hostID+'.hdf5'
    
    if bound:
        print('bound')
        starcutname= '/projectnb/gravlens/bnmcd/SFR/catalogs/subhalocutouts/stars/cutout_'+str(subID)+'.hdf5'
        cutname = '/projectnb/gravlens/bnmcd/SFR/catalogs/subhalocutouts/gas/cutout_'+str(subID)+'.hdf5'
    
    gasparams={'gas':'Coordinates,Masses,GFM_Metallicity,StarFormationRate,SubfindVelDisp,Density,ParticleIDs,GFM_AGNRadiation'}
    if exists(cutname): gascutout=cutname
    elif bound: 
        gascutout = get(suburl,parttype='gas',params=gasparams)
        #print(gascutout)
    else: gascutout = get(hosturl, parttype='gas', params= gasparams)    
    L=75000
    halfbox=L/2.
    try:
        if 'PartType0' in h5py.File(gascutout,'r').keys():
            print('here')
    except:
        print('trouble here,', subID)
        gascutout = get(suburl,parttype='gas',params=gasparams)
    if 'PartType0' in h5py.File(gascutout,'r').keys(): 
    
        with h5py.File(gascutout,'r') as f:
            dx = f['PartType0']['Coordinates'][:,0] - pos[0]
            dy = f['PartType0']['Coordinates'][:,1] - pos[1]
            dz = f['PartType0']['Coordinates'][:,2] - pos[2]
        
            mass = np.asarray(f['PartType0']['Masses'])
            SFR = np.asarray(f['PartType0']['StarFormationRate'])
     
        x=dx
        y=dy
        z=dz
    
        
    
        dx=np.where((dx<halfbox),dx,dx-L)
        dy=np.where((dy<halfbox),dy,dy-L)
        dz=np.where(dz<halfbox,dz,dz-L)
    
    
    #had to get rid of abs. val for this part so I can save the real positions rel. to center
        neghalfbox=-1*halfbox
        dx=np.where(dx>neghalfbox,dx,dx+L)
        dy=np.where(dy>neghalfbox,dy,dy+L)
        dz=np.where(dz>neghalfbox,dz,dz+L)
    
        
        r2 = np.add(np.square(dx),np.square(dy),np.square(dz))
        
        Iflag=0
        #whgas for inertia tensor
        whigas = np.nonzero((SFR>0)&(r2<=4*hrad*hrad)) #star-forming gas within 2x half-mass radius, how TNG does visualizations
        if len(whigas[0])>100:
            inertia=Itensor(dx[whigas],dy[whigas],dz[whigas],mass[whigas])
            Iflag=1
            print('gas inertia')
        
        whgas=np.nonzero((r2 <= (4*rad*rad)))
    
        #add to do coordinate transformation
        x=dx[whgas]
        y=dy[whgas]
        z=dz[whgas]
        mass=mass[whgas]
        SFR=SFR[whgas]
    
    else: 
        x=np.asarray([])
        y=np.asarray([])
        z=np.asarray([])
        mass=np.asarray([])
        SFR=np.asarray([])
    
    starparams={'stars':'Coordinates,Masses,GFM_Metallicity,GFM_StellarFormationTime,ParticleIDs,SubfindVelDisp,GFM_StellarPhotometrics,GFM_InitialMass'}
    if exists(starcutname): starcutout=starcutname
    if bound: starcutout = get(suburl,parttype='stars',params=starparams)
    else: starcutout = get(hosturl,parttype='stars',params=starparams)
    
    with h5py.File(starcutout,'r') as f:
        dx = f['PartType4']['Coordinates'][:,0] - pos[0]
        dy = f['PartType4']['Coordinates'][:,1] - pos[1]
        dz = f['PartType4']['Coordinates'][:,2] - pos[2]
        
        rmag = np.asarray(f['PartType4']['GFM_StellarPhotometrics'])[:,5]
        starmass = np.asarray(f['PartType4']['Masses'])
        starage = np.asarray(f['PartType4']['GFM_StellarFormationTime'])
        imass = np.asarray(f['PartType4']['GFM_InitialMass'])
    
    starx=dx
    stary=dy
    starz=dz
    
    dx=np.where((dx<halfbox),dx,dx-L)
    dy=np.where((dy<halfbox),dy,dy-L)
    dz=np.where(dz<halfbox,dz,dz-L)
    
    
    #had to get rid of abs. val for this part so I can save the real positions rel. to center
    neghalfbox=-1*halfbox
    dx=np.where(dx>neghalfbox,dx,dx+L)
    dy=np.where(dy>neghalfbox,dy,dy+L)
    dz=np.where(dz>neghalfbox,dz,dz+L)

    starr2 = np.add(np.square(dx),np.square(dy),np.square(dz))
    
    whistar = np.nonzero((starr2<=hrad*hrad))    
    if Iflag==0: 
        inertia=Itensor(dx[whistar],dy[whistar],dz[whistar],starmass[whistar])
    

    
   # pkpc= starr2 / h


   
    whstar= np.nonzero((starr2 <= (4*rad*rad)))
    
    starx=starx[whstar]
    stary=stary[whstar]
    starz=starz[whstar]
    #wait to update age and mass to be consistent
    
    #determine SFR following Donnari and Pillepech
    wh5pkpc = np.nonzero((starr2 <= 4*rad*rad)) #change to scale with radius
    starage_z=1/starage-1
    
    
    

    

    #print(starageyr)
    
    starage_all=starage_z
    starage_z=starage_z[wh5pkpc]
    
    halphaage=np.nonzero((starage_z<=0.001))
    D4000age=np.nonzero((starage_z<= 0.007))
    gyrage = np.nonzero((starage_z<= 0.073))
    #print(D4000age)
    mass5pkpc=imass[wh5pkpc]
    
    hamass= np.sum(mass5pkpc[halphaage])
    hasfr= hamass*500   #M_sun/yr              # simplified: hamass*10**10/(20*10**6)
            
    D4mass = np.sum(mass5pkpc[D4000age])
    d4sfr = D4mass*100 #M_sun/yr           #D4mass*10**10/(100*10**6)
    
    Gyrmass = np.sum(mass5pkpc[gyrage])
    Gyrsfr = Gyrmass*10 # *10**10/10**9
    
    np.savetxt('/projectnb/gravlens/bnmcd/SFR/catalogs/SFR_subfind/'+ str(subID), [hasfr,d4sfr,Gyrsfr])
    print('here')
    if not saveall: return()

    starmass=starmass[whstar]
    imass=imass[whstar]
    starage=starage_all[whstar]
    rmag = rmag[whstar]
    
    
    flag=0
    if len(starx)<1000: 
        flag=1
        print('INSUFFICIENT STARS: '+ str(subID), len(starx))
    elif len(starx)<500:
        flag = 2
    
    #COORDINATE TRANSFORMATION

    
    #print(inertia)
    
    #calculate eigenvalues and eigenvectors of inertia tensor
    #eigenvectors are already in square matrix that can be used to transform coordinates
    eigvals,eigvecs=la.eig(inertia)
    transinv=la.inv(eigvecs)

    #diagonalize inertia tensor, smallest diagonal value will be axis we orient to
    diaginertia=np.matmul(np.matmul(transinv,inertia), eigvecs)
    #print(diaginertia)
    diagonals= (diaginertia[0,0],diaginertia[1,1],diaginertia[2,2])
    min_val=min(diagonals)
    min_axis=diagonals.index(min_val)
    #print(min_axis)
    
    #want to use the inv. transformation matrix to change old coordinates to new ones
    #posvec=np.c_[x,y,z]
    coord=np.empty((len(x),3))
    starcoord=np.empty((len(starx),3))
    

    
    transinv=transinv.T
    
    #print(x[0],y[0],z[0])
    for i in range(0,len(x)):
        posvec=np.array((x[i],y[i],z[i]))

        posvecp=posvec @ transinv

        coord[i,:]= posvecp
    #print(coord)
    
    for i in range(0,len(starx)):
        posvec=np.array((starx[i],stary[i],starz[i]))
        posvecp=posvec@transinv
        starcoord[i,:]=posvecp
        
        
        
    #print('starage_z:', len(starage_z))
    
    #convert star redshifts to lookback times
    ages=Planck15.lookback_time(starage_z)/u.Gyr
    #print('ages:',len(ages))
    
    halpha=np.nonzero((starage<=0.001))
    D4000=np.nonzero((starage<= 0.007))
    Gyr=np.nonzero((starage<= 0.073))
    
    
    partdict={'gaspos': coord, 'gasmass': mass, 'SFR': SFR, 'starpos': starcoord, 'starmass': starmass, 'starage_z': starage,'starimass': imass, 'rmag': rmag,
                'flatten_on': min_axis, 'flag': flag, 'galcoord': pos, 'maxrad': 2*rad, 'ages[Gyr]': ages}
    if not bound:
        f = open('/projectnb/gravlens/bnmcd/SFR/catalogs/partsfromhost/'+str(subID)+'.pkl','wb')
        pickle.dump(partdict,f)
        f.close()
        
        
    
    
        f = open('/projectnb/gravlens/bnmcd/SFR/catalogs/Halph_parts/'+str(subID)+'.pkl','wb')
        partdict={'starpos': starcoord[halpha,:], 'starmass': imass[halpha], 'flatten_on': min_axis}
        pickle.dump(partdict,f)
        f.close()
    
        f = open('/projectnb/gravlens/bnmcd/SFR/catalogs/D4000_parts/'+str(subID)+'.pkl','wb')
        partdict = {'starpos': starcoord[D4000,:], 'starmass': imass[D4000], 'flatten_on': min_axis}
        pickle.dump(partdict,f)
        f.close()
    
        f = open('/projectnb/gravlens/bnmcd/SFR/catalogs/Gyr_parts/'+str(subID)+'.pkl','wb')
        partdict = {'starpos': starcoord[Gyr,:], 'starmass': imass[Gyr], 'flatten_on': min_axis}
        pickle.dump(partdict,f)
        f.close()
    else: 
        f = open('/projectnb/gravlens/bnmcd/SFR/catalogs/boundparts/'+str(subID)+'.pkl','wb')
        pickle.dump(partdict,f)
        f.close()
        
        """
    
        f = open('/projectnb/gravlens/bnmcd/SFR/catalogs/boundparts/Halph_parts/'+str(subID)+'.pkl','wb')
        partdict={'starpos': starcoord[halpha,:], 'starmass': imass[halpha], 'flatten_on': min_axis}
        pickle.dump(partdict,f)
        f.close()
    
        f = open('/projectnb/gravlens/bnmcd/SFR/catalogs/boundparts/D4000_parts/'+str(subID)+'.pkl','wb')
        partdict = {'starpos': starcoord[D4000,:], 'starmass': imass[D4000], 'flatten_on': min_axis}
        pickle.dump(partdict,f)
        f.close()
    
        f = open('/projectnb/gravlens/bnmcd/SFR/catalogs/boundparts/Gyr_parts/'+str(subID)+'.pkl','wb')
        partdict = {'starpos': starcoord[Gyr,:], 'starmass': imass[Gyr], 'flatten_on': min_axis}
        pickle.dump(partdict,f)
        f.close()

        """
    return(coord,mass,SFR,starcoord,starmass, starage, imass, min_axis,flag, ages, rmag)

def galprof(host, findparts=False, binsize=16, binning='o',twoD=False, resMS=True,saveall=True, bound=True):
    print(host)
  # global transinv,posvec,coord
   
    h=0.6774
    data=np.load('/projectnb/gravlens/bnmcd/SFR/catalogs/subhalocat.npy',allow_pickle=True).item()
    rad=data['Reff'][host]
    pos=data['pos'][0] #weirdly nested
    pos=pos[host,:]
    hrad=data['halfrad'][host]
    #print(pos)
    subID=int(data['subID'][host])
    print(subID, rad, pos)
    partname='/projectnb/gravlens/bnmcd/SFR/catalogs/boundparts/'+str(subID)+'.pkl'
    name2='/projectnb/gravlens/bnmcd/SFR/catalogs/Gyr_parts/'+str(subID)+'.pkl'
    if not saveall: name3 = '/projectnb/gravlens/bnmcd/SFR/catalogs/SFRage/'+str(subID)
    if not exists(partname): print('this one ', partname)
    if exists(partname) and not findparts:
        with open(partname,'rb') as f:
            partdat = pickle.load(f)
        
        coord=partdat['gaspos']
        mass=partdat['gasmass']
        SFR=partdat['SFR']
        starcoord=partdat['starpos']
        starmass=partdat['starmass']
        starage=partdat['starage_z']
        imass=partdat['starimass']
        min_axis=partdat['flatten_on']
        flag=partdat['flag']
        rmag = partdat['rmag']
        ages = partdat['ages[Gyr]']
    elif not saveall and not exists(name3): 
        print('here')
        fetchpart(subID,rad,pos,hrad,saveall=False, bound=bound)
        return()
    else: 
        print(subID)
        coord,mass,SFR,starcoord,starmass,starage,imass,min_axis,flag, ages, rmag=fetchpart(subID,rad,pos,hrad, bound=bound)

    #print('Age array:', len(ages),starcoord.shape)
    print('Particle #s: ',len(coord[:,0]),' gas and', len(starcoord[:,0]),' stars')
    if len(starcoord[:,0])<1000: 
        print('INSUFFICIENT STARS: '+ str(subID))
    # fig=plt.figure()
    # ax=plt.axes(projection='3d')
    # ax.scatter(x,y,z,c=veldisp)
    
    #print(mass)
    #inertia test
    # x=starx
    # y=stary
    # z=starz
    # mass=starmass
    

    
    #assign the axis that don't have the smallest inertia to ei and ej coordinates; this orients galaxy face-on
    if min_axis==0: 
        ei=1 
        ej=2

    if min_axis==1: 
        ei=0 
        ej=2

    if min_axis==2: 
        ei=0 
        ej=1
    
    
    #starstack=np.stack((coord,rmag,starmass,starmetal,starSFtime,starveldisp),axis=0)   #double check correct output, for complex binning
    
    #wh=np.nonzero((SFR==0))
    #S2N=(len(SFR)-len(SFR[wh]))/len(SFR[wh])
    
###BINNING
    if twoD: 
        normdist=np.sqrt(np.square(coord[:,ei])+np.square(coord[:,ej]))/(rad) #is it multiply or divide by little h????
        starnormdist=np.sqrt(np.square(starcoord[:,ei])+np.square(starcoord[:,ej]))/(rad)
    else:
       
        normdist=np.sqrt(np.sum(np.square(coord),axis=1))/rad
        starnormdist=np.sqrt(np.sum(np.square(starcoord),axis=1))/rad
    
    #need to convert stellar ages from scale factor to years

    #print('starnorm: ', len(starnormdist)) 
    #print('433ages:',len(ages))
    #Annular rings
    if binning=='o': 
        bins=np.linspace(0,1.6,num=binsize)         #use artificial bins so can be averaged across multiple galaxies but include physical radius below to maintian accurate areas for surface density calculations
              #find profiles for both to test
        
        sfrprof=np.empty(len(bins)-1)
        hasfrprof=np.empty(len(bins)-1)
        d4sfrprof=np.empty(len(bins)-1)
        massdens=np.empty(len(bins)-1)
        
        ageprof=np.empty(len(bins)-1)

        
        S2N=np.empty(len(bins)-1)

        
        binind=np.digitize(normdist,bins) #returned index i satisfies bins[i-1] <= x < bins[i]

        starbinind=np.digitize(starnormdist,bins)

        #print(len(SFR),len(binind), len(normdist))
        
        #for each bin pull sum of star formation rate and divide by bin area
        for i in range(1,len(bins)):
            SFRs=SFR[binind==i]
            print(len(ages),len(starbinind))
            binages=ages[starbinind==i]*10**9
            mags = rmag[starbinind==i]
            rlum = 10**(0.4*(4.85-mags))
            #print('rlum',rlum)
            #print('ages',binages)
            
            halpha=np.nonzero((binages<=20*10**6))
            D4000=np.nonzero((binages<=100*10**6))
            
            hamass=np.sum(imass[halpha]) #total initial mass added by youngest stars
            hasfr= hamass*500   #M_sun/yr              # simplified: hamass*10**10/(20*10**6)
            
            D4mass = np.sum(imass[D4000])
            d4sfr = D4mass*100 #M_sun/yr           #D4mass*10**10/(100*10**6)
            
            wh=halpha
            if (len(binages)==len(binages[wh])): S2N[i-1]=1000
            else: S2N[i-1]=len(binages[wh])/(len(binages)-len(binages[wh]))
            sfrinbin=np.sum(SFRs)
            
            massinbin=np.sum(starmass[starbinind==i]) #stellar mass
            try:
                # binages.any(): 
                    #luminosity weighted age
                ageprof[i-1] = np.nansum(np.multiply(rlum,np.log10(binages)))/ np.nansum(rlum)
                print('ageprof',ageprof[i-1])
            except: 
                ageprof[i-1] = 'NaN'
                print('no ages')
                
           
            if twoD: area = (math.pi*rad*rad*(bins[i]**2-bins[i-1]**2))
            else: area = (4/3*math.pi*rad*rad*rad*(bins[i]**3-bins[i-1]**3))
            sfrprof[i-1] = sfrinbin / area
            hasfrprof[i-1]=hasfr/area
            d4sfrprof[i-1]=d4sfr/area
            massdens[i-1]=massinbin*10**10 / area / h


            
        r=np.empty(len(bins)-1)
        for i in range(1,len(bins)):            
            offset=(bins[i]-bins[i-1])/2.
            r[i-1]=(bins[i]-offset)
        pref='/projectnb/gravlens/bnmcd/SFR/catalogs/'
        
        if flag==0:
            if binsize==13:np.save(pref+'sfrprof12/host'+str(subID),np.vstack((massdens,sfrprof,S2N,ageprof,hasfrprof,d4sfrprof)).T)
            elif twoD: np.save(pref+'sfrprof/host'+str(subID),np.vstack((massdens,sfrprof,S2N,ageprof,hasfrprof,d4sfrprof)).T)   
            else: np.save(pref+'sfrprof3D/host'+str(subID),np.vstack((massdens,sfrprof,S2N,ageprof,hasfrprof,d4sfrprof)).T) 
        elif flag==1:
            pref=pref+'smallstar/'
            if binsize==13: np.save(pref+'sfrprof12/host'+str(subID),np.vstack((massdens,sfrprof,S2N,ageprof,hasfrprof,d4sfrprof)).T)   
            elif twoD: np.save(pref+'sfrprof/host'+str(subID),np.vstack((massdens,sfrprof,S2N,ageprof,hasfrprof,d4sfrprof)).T)   
            else: np.save(pref+'sfrprof3D/host'+str(subID),np.vstack((massdens,sfrprof,S2N,ageprof,hasfrprof,d4sfrprof)).T) 
    #np.save(pref+ 'S2N/host'+str(host),S2N)
            
    else: return
        
    return()
    
    
    
    
# def changebasis(i):
#     posveci=posvec[i][:,None]
#     posvecp=transinv @ posveci
#     if i==1: 
#         print(posvecp.T)
#         print(posvecp.T[0]) #this is right
#     posvecpT=posvecp.T[0]
#     coord[i,:]=posvecpT #but this isn't????
    
    
def get(path,parttype='gas', params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        if parttype=='stars': fdirc='catalogs/subhalocutouts/stars/'
        else: fdirc='catalogs/subhalocutouts/gas/'
        filename = fdirc+r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r    
    
#galprof(905)   
# runall()    
#galprof(5000,findparts=False)
    