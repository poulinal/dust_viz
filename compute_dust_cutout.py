"""
Class to handle particle data
Includes functions to compute dust and determine particle location

Currently assumes fixed value for threshold density for star formation
True expression would use cooling look up tables, but not sure where to find those
"""

import numpy as np
import h5py
from src.iapi_TNG import getredshift
import os

global k_b, m_H, avg_nucl_num, T_cold, beta, T_SN, A_0, unit_length, unit_mass, unit_velocity, rho_th
#fixed
k_b = 1.380649e-16 #erg/K       #1.3806448e-23 # J/K
m_H = 1.6726219e-24 # g
#TNG params
unit_length = 3.08568e+21 #cm to kpc/h
unit_mass = 1.989e+43 #g to 10^10 Msun/h
unit_velocity = 100000 # cm/s to km/s
#unit_time = unit_length/unit_velocity # kpc/(km/s), works out to ~0.978 Gyr
T_cold = 1000 # K
beta = 0.22578 #mass fraction of stars that 'instantly' go supernova
T_SN = 5.73e7   # K
t_SFR = 2.27*unit_length/unit_velocity #star formation timescale at critical density, converted to seconds from units of 0.978 Gyr (code unit for time)
#u_SN = (1-beta)/beta*eps_SN
A_0 =573 #A_0 is normalization of efficiency of SN evapopration of cool clouds, A
#rho_th = 0.13*m_H # in cgs
rho_th = 0.000754654 #code units
#rho_th_c = 0.13*m_H*1000/unit_mass*unit_length*unit_length*unit_length #density threshold for star formation in code units
#assuming the fixed value given in the literature, but better to use the S&H defined function, which needs cooling tables I don't have
#negotiable
avg_nucl_num = 15.5 #average nucelon number for metals in solar metallicity
print(t_SFR/(60*60*24*365.25*10**9))
def ISM_check(rho, u, X, x_e, gamma_min_one=2/3):
    """
    Check if particle is in ISM based on temperature/density threshold
    Parameters:
        rho (float): Density in code units
    Returns:
        
    """
    
    T_th = 10**6*rho**0.25 ##threshold temperature for ISM at given density in Kelvin
    T =  energy_to_temp(u, mean_mol_weight(X, x_e), gamma_min_one)
    
    return np.where(T<T_th, 3, 2) # 3 for ISM, 2 for CGM

def metal_dust_fraction_V20(redshift, powerlaw_slope =-1.92, powerlaw_norm = 0.9):
    """
    Compute fraction of metal in dust based on the Voglsberger+ 2020 best-fit model.
    Default values from same paper. Also used in Millard+ 2020
    Only good at z>2
    Parameters:
        redshift (float): The redshift of interest
        powerlaw_slope (float): Slope of the power law for the dust mass fraction.
        powerlaw_norm (float): Normalization factor for the dust mass fraction.
    """
    return powerlaw_norm*(redshift/2)**powerlaw_slope#compute mass of dust 

def metal_dust_fraction_V19(snapnum, model='fiducial'):
    """
    Compute fraction of metals in dust based on Vogelsberger+2019, Fig. 4
     https://doi.org/10.1093/mnras/stz1644
     Parameters:
         redshift (float): redshift of interest
         model (str): V19 dust model
            Options: fiducial, slow-sputter, large-grains, more-cooling 
    """
    v19_data = np.loadtxt('/projects/mccleary_group/mcdonough.b/TNG_dust/data/V19_dust_to_metal.csv', skiprows=1, delimiter=',')
    snapnums = v19_data[:,0]
    whsnap = np.nonzero((snapnums==snapnum))
    if len(whsnap[0])==0: raise ValueError('Snapnum not found')
    
    if model == 'fiducial': colnum =1
    elif model == 'more-cooling': colnum =2
    elif model == 'large-grains': colnum =3
    elif model == 'slow-sputter': colnum =4
    else: raise ValueError('Model name not recognized')
    return v19_data[whsnap,colnum]

def energy_to_temp(u, mu, gamma_min_one =2/3):
    """
    Convert energy per unit mass to temperature using the equation of state.
    Parameters:
        u (float): Energy per unit mass in (km/s)^2 (code units)
        mu (float): Mean molecular weight
        gamma_min_one (float): Adiabatic index (default is 5/3 for monatomic gas) subtracted by 1 (to avoid repeat divisition)

    Returns:
        temp (float): Temperature in Kelvin
    """
    return u*mu*m_H*(gamma_min_one)/k_b #convert to Kelvin 

def temp_to_energy(temp, mu, gamma_min_one =2/3): #output in (km/s)^2
    """
    Convert temperature to energy per unit mass using the equation of state.
    Parameters:
        temp (float): Temperature in Kelvin
        X (float): Hydrogen mass fraction
        Z (float): Metallicity
        mu (float): Mean molecular weight
        gamma_min_one (float): Adiabatic index (default is 5/3 for monatomic gas) subtracted by 1 (to avoid repeat divisition)
                Note: TNG FAQ says to use gamma=5/3, but gas is specifically not monatomic. I mean its mostly hydrogen, but...
                Everyone else does it so that's what we will do here
    Returns:
        energy (float): Energy per unit mass in (km/s)^2 (code units for InternalEnergy)
    """
    return k_b*temp/(mu*m_H*(gamma_min_one)) #energy per unit mass

def mean_mol_weight(X, x_e):
    """
    Compute the mean molecular weight for a given partially-ionized composition.
    Assumption is made that contribution of metals is negligible
    
    Parameters:
        X (float): Hydrogen mass fraction
        x_e (float): Electron abundance (n_e/n_H)
    Returns:
        mu (float): Mean molecular weight
    """
    return 4/(1+3*X+4*X*x_e) #mean molecular weight for fully ionized gas

def mean_mol_weight_SF(X, Z=0, ionized=False):
    """
    Compute the mean molecular weight for fully ionized or fully neutral compositions.
    (i.e., typically only for the star-forming gas)
    
    Parameters:
        X (float): Hydrogen mass fraction
        Z (float): Metallicity. 
            Default is 0 because TNG neglects metals in their calcluation, due to small influence on mu
    Returns:
        mu (float): Mean molecular weight
    """
    if ionized: return 4/(3+5*X-Z) #mean molecular weight for ionized gas
    else: return 4/(1+3*X+(4/avg_nucl_num-1)*Z)#mean molecular weight

def rho_threshold(X,u_c, u4, u_SN, cooling_rate):
    """
    Compute density threshold for star formation
    #obsolete because cooling rate is not available
    """
    x_th = 1+(A_0+1)*(u_c-u4)/u_SN #mass fraction in clouds at threshold
    return x_th/(1-x_th)**2 *(beta*u_SN-(1-beta)*u_c)*m_H**2/(t_SFR *cooling_rate*X**2) #in cgs units

def evap_efficiency(rho):
    #efficiency of supernova evaporation of cool clouds
    return(A_0*(rho/rho_th)**(-0.8))

def neutral_frac_SFR(u, rho, Z, X, redshift, gamma_min_one = 2/3):
    """
    Compute the neutral fraction for star-forming gas.
    Parameters:
        temp (float): Temperature in Kelvin
        Z (float): Metallicity
        X (float): Hydrogen mass fraction
        gamma (float): Adiabatic index minus one (default is gamma=5/3 for monatomic gas)
    Returns:
        f_neutral (float): Neutral fraction
    """
    #cold gas assumed to be at 1000 K
    mu_c = mean_mol_weight_SF(X, Z, ionized=False)
    u_c = temp_to_energy(1000, mu_c, gamma_min_one)

    #to get u_h, need to know assumed supernova temperature, and threshold density for star formation
    u_SN =temp_to_energy(T_SN, mean_mol_weight_SF(X, Z, ionized=True), gamma_min_one)   
    #x_th = 1+(A_0+1)*(u_c-u_4)/u_SN#fraction of cool clouds at threshold density

    #thermal energy of gas at 10^4 K
    u4 = temp_to_energy(10000,mu_c, gamma_min_one)
    
    #rho_th = rho_threshold(X, u_c, u4, u_SN, cooling_rate)

    
    a_inv=1+redshift #scale factor
    #rho_cgs = rho*unit_mass/(a_inv*unit_length)**3 #convert rho to physical density in cgs
    rho_phys = rho/a_inv**3 #physical, but otherwise code units
    u_h =  u_SN/(evap_efficiency(rho_phys)+1)+u_c #energy per unit mass of hot gas


    f_neutral =(u-u_h)/(u_c-u_h)
    #as Stevens+19 notes, sometimes numerical precision can cause  f_neutral to exceed the bounds of [0,1]
    f_neutral[f_neutral>1.0] = 1.0 # numerical errors could give answer just above 1.0 otherwise
    f_neutral[f_neutral<0.0] = 0.0
    return f_neutral

class GasCells:
    """
    Class to handle gas particle data
    Includes functions to compute dust and determine particle location
    """
    
    def __init__(self, cutout_file, output_file, snapnum, rewrite_file=False, simname = 'TNG300-1', dust_model = 'M20_half', doMg=False):
        """
        Initialize the GasCells class.
        Parameters:
            cutout_file (str): Path to the cutout file containing gas particle data.
            output_file (str): Path to the output file where computed data will be saved (in hdf5 format)
            snapnum (int): Snapshot number.
            rewrite_file (bool): If True, will remove existing datasets and recalculate.
            simname (str): Name of the simulation (default is 'TNG300-1').
            dust_model (str): Dust model to use for calculations (default is 'M20_half').
            doMg (bool): If True, will compute Mg dust mass (default is True)."""
        
        self.snapnum = snapnum
        self.redshift = getredshift(snapnum, simname)
        self.simname = simname
        #self.chunk = chunk
        self.doMg = doMg
        self.remove_if_exist = rewrite_file #if True, will remove existing datasets and recalculate

        self.fName = cutout_file
        self.post_fName = output_file
        #self.offset_fName = f'{offsetpath}chunk_offsets_{snapnum}.hdf5'
        #load critical particle fields
        with h5py.File(self.fName, 'r') as f:
            gas = f['PartType0']    
            #self.IDs = gas['ParticleIDs'][:]
            self.masses = gas['Masses'][:]
            self.Z = gas['GFM_Metallicity'][:]
            self.X = gas['GFM_Metals'][:,0] #H
            if doMg: self.Mg = gas['GFM_Metals'][:,6] #Mg abundance
            self.neutral_H_nonSFR = gas['NeutralHydrogenAbundance'][:] #use with caution for star-forming gas
            self.SFR = gas['StarFormationRate'][:]
            self.e_abundance = gas['ElectronAbundance'][:]
            self.internal_energy = gas['InternalEnergy'][:]*10**10 #in cgs units (cm/s)^2 ; same as multiplying by unitvelocity^2
            self.density = gas['Density'][:]
            #self.coolingrate = gas['GFM_CoolingRate'][:] #in cgs       #/unit_mass/unit_velocity**3/unit_length**2 #converted from erg cm^3/s (g cm^5 s^-3) to code units
        
            self.whSFR= np.nonzero(self.SFR) #indices of star-forming gas

        if rewrite_file : open_as = 'w'
        else: open_as = 'a'
        with h5py.File(self.post_fName, open_as) as f:
            #create hdf5 groups to store computed data
            try: f.create_group('Universal')
            except: pass
            try: f.create_group(dust_model) #M20 dust model with V20 dust fraction
            except: pass

            #self.dustmass = f['M20_V19']['M_dust'][:]
            #self.location = f['Universal']['location'][:]

    def neutral_frac(self, remove_if_exist=False):
        """
        Compute neutral fraction.
        remove_if_exist: remove and recalculate if group already exists
        """
        
        with h5py.File(self.post_fName, 'a') as f:
            try:
                self.f_neutral = grp['f_neutral'][:]
                if remove_if_exist: del grp['f_neutral']
                else: return
            except: pass

            #Create array for neutral fraction assuming all cells are non-star forming
            self.f_neutral = self.neutral_H_nonSFR
            if len(self.whSFR[0])!=0 :
                self.f_neutral[self.whSFR] = neutral_frac_SFR(self.internal_energy[self.whSFR], self.density[self.whSFR], 
                                                      self.Z[self.whSFR], self.X[self.whSFR],  self.redshift)
            
        
            grp = f['Universal']
            grp.create_dataset('f_neutral', data=self.f_neutral, compression='gzip', compression_opts=6) 
            #compression level goes up to 9. Higher means longer compression time but smaller file size
            #if len(self.whSFR[0])!=0 : grp.create_dataset('rho_th', data = self.rho_th, compression='gzip', compression_opts=6)

        return
        
    def compute_dust_M20(self, metal_dust_fraction_mode='half'):
        """
        Compute dust properties from particle data according to equation 3 of Millard+2020.
        """
        with h5py.File(self.post_fName, 'a') as f:
            if metal_dust_fraction_mode == 'V20':
                metal_dust_fraction=metal_dust_fraction_V20(self.redshift) 
                grp_name = 'M20_V20'
            elif metal_dust_fraction_mode == 'V19':
                metal_dust_fraction = metal_dust_fraction_V19(self.snapnum)
                grp_name = 'M20_V19'
            elif metal_dust_fraction_mode == 'half':
                metal_dust_fraction = 0.5
                grp_name = 'M20_half'
            
            grp = f[grp_name]
            try: 
                self.M20_dust = grp['M_dust'][:]
                if self.doMg: self.M20_Mg_dust = grp['M_Mg_dust'][:]
                if self.remove_if_exist:
                    del grp['M_dust']
                    del grp['M_Mg_dust']
                else: return self.M20_dust
            except: pass

            dust_intermediate = metal_dust_fraction*self.f_neutral*self.masses
            self.M20_dust = dust_intermediate*self.Z # in code mass units
            if self.doMg: self.M20_Mg_dust = dust_intermediate*self.Mg # in code mass units
            
            
            grp.create_dataset('M_dust', data=self.M20_dust, compression='gzip', compression_opts=6) 
            if self.doMg: grp.create_dataset('M_Mg_dust', data=self.M20_Mg_dust, compression='gzip', compression_opts=6)
            #compression level goes up to 9. Higher means longer compression time but smaller file size
        return self.M20_dust
 
        
    def volumes(self):
        a_inv_3 = (1+self.redshift)**3 #account for comoving values
        return(self.masses/a_inv_3/self.density)


    def run_calcs(self, dust_model='M20', metal_dust_fraction_model='half'):
        """
        Run all dust calculations for the particle data.
        """
        #self.determine_location()
        #print('found locations')
        self.neutral_frac()    
        print('found neutral fractions')
        if dust_model == 'M20':
            try: self.compute_dust_M20(metal_dust_fraction_mode=metal_dust_fraction_model)
            except ValueError: print('metal_dust_fraction_model not recognized. Use "V20" for Voglsberger+2020 dust model.')
        else:
            raise ValueError("Dust model not recognized. Use 'M20' for Millard+2020 dust model.")
        print('found dust masses')

        #self.region_properties()
        #print('found region properties')

        return()
        """
        #compute total dust in different regions
        with h5py.File(self.post_fName, 'a') as f:
            u_grp = f['Universal']
            if len(self.whSFR[0])!=0 : 
                #u_grp.attrs['SF_avg_n_th'] = np.average(self.rho_th)/m_H
                u_grp.attrs['min_SFR_dens_rho'] = np.min(self.density[self.whSFR])*unit_mass/unit_length**3
            
            u_grp.attrs['N_SF'] = len(self.whSFR[0])
            
            OF_ref = u_grp.attrs['OF_ref']
            IF_ref = u_grp.attrs['IF_ref']
            CGM_ref = u_grp.attrs['CGM_ref']
            ISM_ref = u_grp.attrs['ISM_ref']
            
            
            OF_ref = self.location == np.uint8(0)
            IF_ref = self.location == np.uint8(1)
            CGM_ref = self.location == np.uint8(2)
            ISM_ref = self.location == np.uint8(3)
            
            m_grp = f[dust_model+'_'+ metal_dust_fraction_model]
            #I'm not sure why I Have to index like this, but it does work.
            m_grp.attrs['OF_dust_sum'] = np.sum(m_grp['M_dust'][:][0][OF_ref])
            m_grp.attrs['IF_dust_sum'] = np.sum(m_grp['M_dust'][:][0][IF_ref])
            m_grp.attrs['CGM_dust_sum'] = np.sum(m_grp['M_dust'][:][0][CGM_ref])
            m_grp.attrs['ISM_dust_sum'] = np.sum(m_grp['M_dust'][:][0][ISM_ref])
            m_grp.attrs['OF_Mg_dust_sum'] = np.sum(m_grp['M_Mg_dust'][:][0][OF_ref])
            m_grp.attrs['IF_Mg_dust_sum'] = np.sum(m_grp['M_Mg_dust'][:][0][IF_ref])
            m_grp.attrs['CGM_Mg_dust_sum'] = np.sum(m_grp['M_Mg_dust'][:][0][CGM_ref])
            m_grp.attrs['ISM_Mg_dust_sum'] = np.sum(m_grp['M_Mg_dust'][:][0][ISM_ref])
            
            m_grp = f[dust_model+'_'+ metal_dust_fraction_model]

            m_grp.attrs['OF_dust_sum'] = np.sum(m_grp['M_dust'][u_grp['OF_ref']])
            m_grp.attrs['IF_dust_sum'] = np.sum(m_grp['M_dust'][u_grp['IF_ref']])
            m_grp.attrs['CGM_dust_sum'] = np.sum(m_grp['M_dust'][u_grp['CGM_ref']])
            m_grp.attrs['ISM_dust_sum'] = np.sum(m_grp['M_dust'][u_grp['ISM_ref']])
            m_grp.attrs['OF_Mg_dust_sum'] = np.sum(m_grp['M_Mg_dust'][u_grp['OF_ref']])
            m_grp.attrs['IF_Mg_dust_sum'] = np.sum(m_grp['M_Mg_dust'][u_grp['IF_ref']])
            m_grp.attrs['CGM_Mg_dust_sum'] = np.sum(m_grp['M_Mg_dust'][u_grp['CGM_ref']])
            m_grp.attrs['ISM_Mg_dust_sum'] = np.sum(m_grp['M_Mg_dust'][u_grp['ISM_ref']])
            """
            
            

def run_chunk(snapnum, rewrite_file=True, testtaskID=10):
    """
    Wrapper function for batch submission
    """
    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        chunk = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    else: #test run
        chunk = testtaskID
    gc = GasCells(snapnum,chunk, rewrite_file =rewrite_file, doMg=False, path='/scratch/br.mcdonough/snaps/')
    gc.run_calcs()
    return

def run_extra(snapnum, rewrite_file=False, testtaskID=10):
    """
    Wrapper function for batch submission
    """
    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        chunk = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    else: #test run
        chunk = testtaskID
    gc = GasCells(snapnum,chunk, rewrite_file =rewrite_file)
    gc.region_properties()
    return
