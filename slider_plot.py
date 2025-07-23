'''
Written by O. Curtis, 2019
'''

snapshot_title = "snapshot_018"
box_length = 50
mesh_length= 0.5
n_critic = 5
interactive_plots = False
epochs = 100
dim = int(box_length/mesh_length)
density_maps = []
filename_base = "snapshot_*"
thresh = -0.5

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.patches as patches

def distance_with_periodic_boundary_conditions(center, point, half_box):
    this_distance = np.abs(center - point)
    if this_distance > half_box:
        distance = (half_box - this_distance)%half_box
    if this_distance <= half_box:
        distance = this_distance
    return distance

def slider_plot(np_mesh, dim):
    '''
    Allows a user to visualize the simulation
    contrast map. matplotlib will display a 2d slice 
    of the simulation, the user can move a slider to
    change which slice they are looking at.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    fig.subplots_adjust(left=0.25, bottom=0.25)
    this_mesh = np_mesh
    this_slice = this_mesh[int(len(np_mesh)/2),:,:]
    painted_slice = ax.imshow(this_slice, cmap='plasma', vmin=4, vmax=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(painted_slice, ax=ax)
    cbar.set_label(r"Total Stellar Mass [log($\rmM_*/h^{-1}M_\odot$)]")
    slice_slider_ax_x = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    slice_slider_ax_y = fig.add_axes([0.25, 0.10, 0.65, 0.03])
    slice_slider_ax_z = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    slice_slider_x = Slider(slice_slider_ax_x, 'x', 0, dim-1, valinit=dim/2)
    slice_slider_y = Slider(slice_slider_ax_y, 'y', 0, dim-1, valinit=dim/2)
    slice_slider_z = Slider(slice_slider_ax_z, 'z', 0, dim-1, valinit=dim/2)
    def slice_slider_x_on_changed(val):
        [p.remove() for p in reversed(ax.patches)]
        this_mesh = np_mesh
        this_slice = this_mesh[int(np.floor(val)), :, :]
        painted_slice.set_data(this_slice)
        fig.canvas.draw_idle()
    def slice_slider_y_on_changed(val):
        [p.remove() for p in reversed(ax.patches)]
        this_mesh = np_mesh
        this_slice = this_mesh[:, int(np.floor(val)), :]
        painted_slice.set_data(this_slice)
        fig.canvas.draw_idle()
    def slice_slider_z_on_changed(val):
        [p.remove() for p in reversed(ax.patches)]
        this_mesh = np_mesh
        this_slice = this_mesh[:, :, int(np.floor(val))]
        painted_slice.set_data(this_slice)
        fig.canvas.draw_idle()
    slice_slider_x.on_changed(slice_slider_x_on_changed)
    slice_slider_y.on_changed(slice_slider_y_on_changed)
    slice_slider_z.on_changed(slice_slider_z_on_changed)
    plt.show()
