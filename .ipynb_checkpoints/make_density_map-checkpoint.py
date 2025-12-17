import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from numba import int32, float64, float32, double    # import the types
from numba.experimental import jitclass
from numba import jit

spec = [
    ('particle_coords', float64[:,:]),
    ('N_cells_per_side', int32),
    ('cell_length', float64),
    ('h_vals', float64[:]),
    ('star_mass', float32[:]),
    ('cell_length', float32),
    ('particle_mesh', float64[:,:,:]),
    ]

@jitclass(spec)
class Illustris_Density_Mapper():
    
    def __init__(self, particle_coords, N_cells_per_side, cell_length, h_vals, star_mass):
        """
        TODO: Comment
        """
        print(f"shape particle_coords: {particle_coords}, n_cells: {N_cells_per_side}, cell_length: {cell_length}, star_mass: {star_mass}")
        self.particle_coords = particle_coords
        self.N_cells_per_side = N_cells_per_side
        self.cell_length = cell_length
        self.h_vals = h_vals
        self.star_mass = star_mass
        self.particle_mesh = np.zeros((self.N_cells_per_side,self.N_cells_per_side,self.N_cells_per_side))
        print(f"finished init of Illustris_Density_Mapper")
    
    def assign_particles_to_grid(self):
        for particle in range(self.particle_coords.shape[0]):
            print(f"Progress... assigning particle: {particle} out of {self.particle_coords.shape[0]}")
            nearest_cell_x_coord = (int(np.floor(self.particle_coords[particle][0]/self.cell_length))%self.N_cells_per_side) - 1
            nearest_cell_y_coord = (int(np.floor(self.particle_coords[particle][1]/self.cell_length))%self.N_cells_per_side) - 1
            nearest_cell_z_coord = (int(np.floor(self.particle_coords[particle][2]/self.cell_length))%self.N_cells_per_side) - 1
            for p_x in range(nearest_cell_x_coord - 5, nearest_cell_x_coord + 5):
                this_p_x = p_x%self.N_cells_per_side
                for p_y in range(nearest_cell_y_coord - 5, nearest_cell_y_coord + 5):
                    this_p_y = p_y%self.N_cells_per_side
                    for p_z in range(nearest_cell_z_coord - 5, nearest_cell_z_coord + 5):
                        this_p_z = p_z%self.N_cells_per_side
                        input_x = self.particle_coords[particle][0] - this_p_x * self.cell_length
                        input_y = self.particle_coords[particle][1] - this_p_y * self.cell_length
                        input_z = self.particle_coords[particle][2] - this_p_z * self.cell_length
                        input_r = np.sqrt(input_x**2 + input_y**2 + input_z**2)
                        this_h = self.h_vals[particle]
                        this_mass = self.star_mass[particle]
                        self.particle_mesh[this_p_x, this_p_y, this_p_z] += this_mass*w2(input_r, this_h)

#CIC kernel
@jit(nopython=True)
def w(x, delta):
    if np.abs(x) < delta:
        return 1 - (np.abs(x)/delta)
    else:
        return 0

#cubic spline kernel
@jit(nopython=True)
def w2(r, h):
    q=r/h
    constant = 8/(np.pi*h**3)
    if (q>=0) and (q<=1/2):
        return constant*(1 - 6*q**2 + 6*q**3)
    if (q>1/2) and (q<=1):
        return constant * 2 * (1-q)  **3
    if (q>1):
        return 0