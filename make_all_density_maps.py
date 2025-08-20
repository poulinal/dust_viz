import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from make_density_map import *
from os.path import exists
import glob
import h5py


#######
import iapi_TNG as iapi
# baseUrl = 'http://www.tng-project.org/api/'
# ###specify which simulation you want to explore###
# sim='TNG300-1'
# r=iapi.get(baseUrl)
# print(r)
# #check the properties of the simulation you have selected
# simUrl = baseUrl+sim
# print(simUrl) 
# simdata = iapi.get(simUrl)
# print(simdata['description'])
# # simdata.keys()
# simdata

# currentdirc=!pwd
# print(currentdirc)
# dirc= ' '.join(currentdirc) + '/data/'
# if not os.path.exists(dirc + 'catalogs'):
#     os.makedirs(dirc + 'catalogs')
#     print(f'created directory: {dirc} "catalogs"')
# if not os.path.exists(dirc + 'catalogs/SubhaloFlag'):
#     os.makedirs(dirc + 'catalogs/SubhaloFlag')
#     print(f'created directory: {dirc} "/catalogs/SubhaloFlag"')
# flag=iapi.getSubhaloField('SubhaloFlag',simulation=sim,fileName=dirc+'catalogs/SubhaloFlag',rewriteFile=0)
haloNum = 0
# group_url = f"http://www.tng-project.org/api/TNG300-1/snapshots/99/halos/{haloNum}/"
# group_data = iapi.get(group_url)
import requests

#######

#ALEXANDER TODO:
#replace all directory names
#update to read information from particle files
#set  h_vals with SubfindHsml from particle files

TNG300_1_boxsize = 	302.6 #in Mpc

if __name__ == '__main__':
    
    # PATH_TO_FILES = '/projectnb/gravlens/bnmcd/SFR/catalogs/boundparts/'
    PATH_TO_FILES = '/Users/alexpoulin/Downloads/git/dust_viz/data/'
    # all_files = glob.glob(PATH_TO_FILES+"*.pkl")
    #all_files=['/projectnb/gravlens/bnmcd/SFR/catalogs/boundparts/270967.pkl'] #test file
    all_files = glob.glob(PATH_TO_FILES+"*.hdf5")
    print("Found", len(all_files), "files in", PATH_TO_FILES)
    
    for this_pkl in range(len(all_files)): #looping over all files in all_files
        print("Processing file:", all_files[this_pkl])

        # this_fname = all_files[this_pkl]
        # this_fname = this_fname.split('/')[-1]
        # this_subid = this_fname[:-4]
        
        with h5py.File(all_files[this_pkl], 'r') as f:
            this_file = f
            this_subid = all_files[this_pkl].split('_')[-1].replace('.hdf5', '')
        

            hval = this_file
            h_vals = hval['PartType0']['SubfindHsml'][:].astype(np.float64)   # array, one per gas particle
        
            # if exists("/projectnb/res-star/3D_maps/density_maps/boundparts/stellar_mass_density_map_subID_"+str(this_subid)+".npy"): continue
            
            # this_file = np.load(all_files[this_pkl], allow_pickle=True)
            star_pos = this_file['PartType0']['Coordinates'][:].astype(np.float64) #particle positions #UPDATE: update call to pull from particle files
            print(f"star_pos: {star_pos}")
            star_mass =this_file['PartType0']['Density'][:] #particle masses #UPDATE: update call to pull from particle files - either gas mass for gas density or dust mass from new files
        print(f"Loaded data for subhalo {this_subid} with {star_pos.shape} particles. with h_vals shape: {h_vals.shape}, star_pos shape: {star_pos.shape}, star_mass shape: {star_mass.shape}")
        

        # Virial radius (R200c) in simulation units (ckpc/h)
        with h5py.File('tempCat_groupR200c.hdf5.hdf5', 'r') as f:
            print("Group data keys:", f['Group']['Group_R_Crit200'])
            virial_radius = f['Group']['Group_R_Crit200'][:] # in 𝑐𝑘𝑝𝑐/ℎ
            
        with h5py.File('tempCat_groupPos.hdf5.hdf5', 'r') as f:
            print("Group data keys:", f.keys())
            central_pos = f['Group']['GroupPos'][:] # in 𝑐𝑘𝑝𝑐/ℎ
        print(f"shape Virial Radius (R200c): {virial_radius.shape} ckpc/h")
        # Convert to physical units if needed
        h = 0.6774  # Hubble parameter
        virial_radius_pkpc = virial_radius[0] / h  # physical kpc
        central_pos_halo0 = central_pos[0] # ckpc/h
        
        star_pos = (star_pos - central_pos_halo0).astype(np.float64)   # shift positions to center of halo
        print(f"Central position of halo 0: {central_pos_halo0} code units, now starpos is {star_pos}")
        # print("group data keys:", group_data.keys())
        # maxrad = group_data['Group_R_Crit200'] #radius within which to map #UPDATE: either set manually or from something like 2x virial radius (in code units)
        maxrad = virial_radius[0] * 2
        # maxrad = virial_radius_pkpc * 2 * 1000 #in pc
        print(star_pos)
        print(f"Max radius for mapping: {maxrad} code units")

        pos_mask = (star_pos[:,0]>=-1*maxrad) & (star_pos[:,0]<=maxrad) & (star_pos[:,1]>=-1*maxrad) & (star_pos[:,1]<=maxrad) & (star_pos[:,2]>=-1*maxrad) & (star_pos[:,2]<=maxrad)
        star_pos = star_pos[pos_mask]
        star_mass = star_mass[pos_mask]
        print(f"shape after masking: star_pos {star_pos.shape}, star_mass {star_mass.shape}")

        first_axis_min = np.min(star_pos[:,0])
        second_axis_min= np.min(star_pos[:,1])
        third_axis_min = np.min(star_pos[:,2])
        star_pos[:,0] += np.abs(first_axis_min) #shift entire galaxy to quadrant I
        star_pos[:,1] += np.abs(second_axis_min)
        star_pos[:,2] += np.abs(third_axis_min)
        grid_length = 0.5 #h-1 Kpc i.e., code units  #UPDATE: you will want to increase this grid size by a factor of two or more
        grid_no = int(1+(2*maxrad/grid_length))
        gal_center_in_map_first_ax = (int(np.floor(np.abs(first_axis_min)/grid_length))%grid_no)
        gal_center_in_map_second_ax = (int(np.floor(np.abs(second_axis_min)/grid_length))%grid_no)
        gal_center_in_map_thrid_ax  = (int(np.floor(np.abs(third_axis_min)/grid_length))%grid_no)
    

        """ #UPDATE: NEED TO REPLACE H_VALS with SubfindHsml from particle files
        cKDTree = spatial.cKDTree(data=star_pos)
        distances, ix = cKDTree.query(x=star_pos, k=17, p=2)
        h_vals=distances[:,-1] #distance to 16th nearest neighbor
        """
        
        h_min = np.sqrt( (grid_length/2)**2 + (grid_length/2)**2 + (grid_length/2)**2)
        h_vals[h_vals<h_min] = h_min
        
    
        this_Mapping = Illustris_Density_Mapper(star_pos, grid_no, grid_length, h_vals, star_mass)
        this_Mapping.assign_particles_to_grid()
    
        density_mesh = this_Mapping.particle_mesh
        total_stellar_mass = density_mesh*grid_length*grid_length*grid_length*10**10
        total_stellar_mass = np.log10(total_stellar_mass)
        
        np.save("/data/density_maps/boundparts/stellar_mass_density_map_subID_"+str(this_subid), density_mesh)
        
        star_pos[:,0] -= np.abs(first_axis_min)
        star_pos[:,1] -= np.abs(second_axis_min)
        star_pos[:,2] -= np.abs(third_axis_min)
        radlist = np.ones_like(h_vals)*maxrad
        gal_center_list_1 = np.ones_like(h_vals)*gal_center_in_map_first_ax
        gal_center_list_2 = np.ones_like(h_vals)*gal_center_in_map_second_ax
        gal_center_list_3 = np.ones_like(h_vals)*gal_center_in_map_second_ax
        particle_info = np.column_stack((star_pos, h_vals))
        particle_info = np.column_stack((particle_info, star_mass))
        particle_info = np.column_stack((particle_info, radlist))
        particle_info = np.column_stack((particle_info, gal_center_list_1))
        particle_info = np.column_stack((particle_info, gal_center_list_2))
        particle_info = np.column_stack((particle_info, gal_center_list_3))
        #particle first ax coord, particle second ax coord, particle third ax coord, particle h_vals, particle stellar mass, galaxy maxrad, gal center in map first ax, gal center in map second ax, gal center in map third ax
        np.save("/data/density_maps/boundparts/particle_info_subID_"+str(this_subid), particle_info)
        
        #fig,ax = plt.subplots(1,1)
        #plot = ax.imshow(total_stellar_mass[int(len(total_stellar_mass)/2)], vmin=5, vmax=8, cmap='plasma')#, vmin=0.0001, vmax=0.05)#, vmax=35)
        #cbar = plt.colorbar(plot)
        #cbar.set_label(r"Total Stellar Mass [log($\rmM_*/h^{-1}M_\odot$)]")
        #plt.show()
