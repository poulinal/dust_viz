
import os
import numpy as np
import h5py


haloNum = 0
explorer_prefix = "../../../../scratch/poulin.al/dust/data/density_maps/boundparts/"
data_dir = '/home/poulin.al/git/dust_viz/data/'
point_data_dir = 'points/'

particle_info_file = "particle_info_subID_"+str(haloNum)

stellar_mass_density_map_file = "stellar_mass_density_map_subID_"+str(haloNum)

#load h5py
# Load particle_info_ from HDF5
with h5py.File(explorer_prefix + particle_info_file + ".hdf5", "r") as f:
    particle_info_ = f['particle_info'][:]

# Load stellar_mass_density_map_ from HDF5
with h5py.File(explorer_prefix + stellar_mass_density_map_file + ".hdf5", "r") as f:
    stellar_mass_density_map_ = f['density_mesh'][:]


print(f"particle_info_ shape: {particle_info_.shape}")
print(f"particle_info_ dtype: {particle_info_.dtype}")
print(f"stellar_mass_density_map_ shape: {stellar_mass_density_map_.shape}")
print(f"stellar_mass_density_map_ dtype: {stellar_mass_density_map_.dtype}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata

coordinates = particle_info_

#grab 1000 random points in coordinates with seed 0
np.random.seed(0)
#this is the random seed, so you can get the same points every time
random_indices = np.random.choice(coordinates.shape[0], 10000, replace=False)
random_num_coordinates = coordinates[random_indices]
#save as a numpy array
np.save(f'{data_dir}{point_data_dir}random_10000_coordinates.npy', random_num_coordinates)
np.save(f'{data_dir}{point_data_dir}random_10000_coordinates.txt', random_num_coordinates)
#save as a .xyz file
with open(f'{data_dir}{point_data_dir}random_10000_coordinates.xyz', 'w') as f:
    for coord in random_num_coordinates:
        f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")
        
        
from testPlots import generate_gas_data, plot_3d_isosurfaces, plot_density_topography, plot_multitracer_3d

isosurfaceFig = plot_3d_isosurfaces(densityMap = np.meshgrid(stellar_mass_density_map_))
isosurfaceFig.savefig('density_plot.png', dpi=300, bbox_inches='tight')

