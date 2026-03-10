import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import h5py

# Generate synthetic gas density data
def generate_gas_data(nx=50, ny=50, nz=50):
    """Generate synthetic 3D gas density distribution"""
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    z = np.linspace(-10, 10, nz)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Create multiple density components (e.g., disk + clumps)
    # Main disk component
    R = np.sqrt(X**2 + Y**2)
    disk = 1e20 * np.exp(-R/3) * np.exp(-Z**2/1)
    
    # Add some dense clumps
    clump1 = 5e21 * np.exp(-((X-3)**2 + Y**2 + Z**2)/0.5)
    clump2 = 3e21 * np.exp(-((X+2)**2 + (Y-2)**2 + Z**2)/0.8)
    clump3 = 1e22 * np.exp(-(X**2 + (Y+3)**2 + (Z-1)**2)/0.3)
    
    density = disk + clump1 + clump2 + clump3
    
    return x, y, z, density

# Plot 2: Position-Position-Density "Topographic" Plot
def plot_density_topography(densityMap = None, labelX="x", labelY="y"):
    """Create a 3D surface plot where height represents density"""
    fig = plt.figure(figsize=(12, 9))
    

    if densityMap == None:
        # Generate 2D slice data
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)

        # Create density distribution (2D slice at z=0)
        R = np.sqrt(X**2 + Y**2)
        density_2d = 1e20 * np.exp(-R/3)  # Disk
        density_2d += 5e21 * np.exp(-((X-3)**2 + Y**2)/0.5)  # Clump
        density_2d += 3e21 * np.exp(-((X+2)**2 + (Y-2)**2)/0.8)  # Clump
    else:
        X, Y, density_2d = densityMap
    
    # Take log for better visualization
    log_density = np.log10(density_2d)
    
    # Create 3D surface plot
    ax = fig.add_subplot(221, projection='3d')
    surf = ax.plot_surface(X, Y, log_density, cmap='viridis',
                          linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_xlabel(f"{labelX} (kpc)")
    ax.set_ylabel(f"{labelY} (kpc)")
    ax.set_zlabel('log₁₀(n) [cm⁻³]')
    ax.set_title('Density as 3D Surface')
    
    # Add contour plot at the bottom
    ax.contour(X, Y, log_density, zdir='z', 
               offset=ax.get_zlim()[0], cmap='viridis', alpha=0.5)




    # Add 2D contour plot for comparison
    ax2 = fig.add_subplot(222)
    # contour = ax2.contourf(X, Y, density_2d, levels=20, 
    #                        norm=LogNorm(vmin=1e18, vmax=1e22), cmap='viridis')
    contour = ax2.contourf(X, Y, density_2d, levels=50, cmap='viridis')
    
    # Contour lines at percentiles
    # percentiles = np.nanpercentile(density_2d, [25, 50, 75])
    percentiles = np.nanpercentile(density_2d, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    percentiles = np.unique(percentiles)  # Remove duplicates (mostly zeroes)
    
    print(f"percentiles: {percentiles}")
    # ax2.contour(X, Y, density_2d, levels=percentiles, 
    #             colors='white', linewidths=1)

    
    ax2.set_xlabel(f"{labelX} (kpc)")
    ax2.set_ylabel(f"{labelY} (kpc)")
    ax2.set_title('2D Density Contours')
    ax2.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label('Density (cm⁻³)')



    percentileRange = [10,99.9]

    # 2D contour plot X
    # max_xy = np.max(log_density)
    # low_xy = np.nanpercentile(log_density[(log_density!=0) & (~np.isnan(log_density)) & np.isfinite(log_density)], 10)
    percentiles_xy = np.nanpercentile(log_density[(log_density!=0) & (~np.isnan(log_density)) & np.isfinite(log_density)], percentileRange)
    max_xy = percentiles_xy[1]
    low_xy = percentiles_xy[0]
    print(f"max_yz: {max_xy}, low_yz: {low_xy}")
    
    # Add 2D contour plot for comparison (logged)
    ax3 = fig.add_subplot(223)

    contour = ax3.contourf(X, Y, log_density, levels=20, cmap='viridis', vmin = low_xy, vmax = max_xy)
    # ax3.contour(X, Y, log_density, levels=10, colors='white', linewidths=1)
    # ax3.contour(X, Y, log_density, levels=10, colors='black', linewidths=0.5)
    ax3.contour(X, Y, log_density, levels=10, colors='white', linewidths=0.5, linestyles='dashed')

    ax3.set_xlabel(f"{labelX} (kpc)")
    ax3.set_ylabel(f"{labelY} (kpc)")
    ax3.set_title('2D Density Contours')
    ax3.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label(r'$\log{(Density)}(cm⁻³)$')



    # 2D contour plot (same data as 3D surface)
    ax4 = fig.add_subplot(224)
    contourlines = ax4.contour(X, Y, log_density, levels=20, cmap='viridis', linewidths=1, vmin = low_xy, vmax = max_xy)

    # Add colorbar
    cbar = plt.colorbar(contourlines, ax=ax4)
    cbar.set_label(r'$\log{(Density)}(cm⁻³)$')
    
    ax4.set_xlabel(f"{labelX} (kpc)")
    ax4.set_ylabel(f"{labelY} (kpc)")
    ax4.set_title('2D Density Contours')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    return fig

# Plot 2: Position-Position-Density "Topographic" Plot
def plot_density_topography_3_proj(densityMap = None):
    """Create a 3D surface plot where height represents density"""
    fig = plt.figure(figsize=(12, 9))
    

    if densityMap == None:
        # Generate 2D slice data
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        z = np.linspace(-10, 10, 100)
        X, Y, Z = np.meshgrid(x, y, z)

        # Create density distribution (2D slice at z=0)
        R = np.sqrt(X**2 + Y**2)
        density_2d = 1e20 * np.exp(-R/3)  # Disk
        density_2d += 5e21 * np.exp(-((X-3)**2 + Y**2)/0.5)  # Clump
        density_2d += 3e21 * np.exp(-((X+2)**2 + (Y-2)**2)/0.8)  # Clump
    else:
        X_xy,Y_xy,X_xz,Z_xz,Y_yz,Z_yz, density_2d_X, density_2d_Y, density_2d_Z = densityMap
    
    # Take log for better visualization
    log_density_X = np.log10(density_2d_X)
    log_density_Y = np.log10(density_2d_Y)
    log_density_Z = np.log10(density_2d_Z)

    percentileRange = [10,99]

    # 2D contour plot X
    # max_yz = np.max(log_density_X)
    # low_yz = np.nanpercentile(log_density_X[(log_density_X!=0) & (~np.isnan(log_density_X)) & np.isfinite(log_density_X)], 10)
    percentiles_yz = np.nanpercentile(log_density_X[(log_density_X!=0) & (~np.isnan(log_density_X)) & np.isfinite(log_density_X)], percentileRange)
    max_yz = percentiles_yz[1]
    low_yz = percentiles_yz[0]
    print(f"max_yz: {max_yz}, low_yz: {low_yz}")
    axX = fig.add_subplot(221)
    contourlines = axX.contour(Y_yz, Z_yz, log_density_X, levels=30, cmap='viridis', linewidths=1, vmin = low_yz, vmax = max_yz)

    # Add colorbar
    cbar = plt.colorbar(contourlines, ax=axX)
    cbar.set_label(r'$\log{(Density)}(cm⁻³)$')
    
    axX.set_xlabel('Y (kpc)')
    axX.set_ylabel('Z (kpc)')
    axX.set_title('2D Density Contours – Projection YZ')
    axX.set_aspect('equal')

    
    # 2D contour plot Y
    # max_xz = np.max(log_density_Y)
    # low_xz = np.nanpercentile(log_density_Y[(log_density_Y!=0) & (~np.isnan(log_density_Y)) & np.isfinite(log_density_Y)], 10)
    percentiles_xz = np.nanpercentile(log_density_Y[(log_density_Y!=0) & (~np.isnan(log_density_Y)) & np.isfinite(log_density_Y)], percentileRange)
    max_xz = percentiles_xz[1]
    low_xz = percentiles_xz[0]
    print(f"max_yz: {max_xz}, low_yz: {low_xz}")
    axY = fig.add_subplot(222)
    contourlines = axY.contour(X_xz, Z_xz, log_density_Y, levels=30, cmap='viridis', linewidths=1, vmin = low_xz, vmax = max_xz)

    # Add colorbar
    cbar = plt.colorbar(contourlines, ax=axY)
    cbar.set_label(r'$\log{(Density)}(cm⁻³)$')
    
    axY.set_xlabel('X (kpc)')
    axY.set_ylabel('Z (kpc)')
    axY.set_title('2D Density Contours – Projection XZ')
    axY.set_aspect('equal')
    
    # 2D contour plot Z
    # max_xy = np.max(log_density_Z)
    # low_xy = np.nanpercentile(log_density_Z[(log_density_Z!=0) & (~np.isnan(log_density_Z)) & np.isfinite(log_density_Z)], 10)
    percentiles_xy = np.nanpercentile(log_density_Z[(log_density_Z!=0) & (~np.isnan(log_density_Z)) & np.isfinite(log_density_Z)], percentileRange)
    max_xy = percentiles_xy[1]
    low_xy = percentiles_xy[0]
    print(f"max_yz: {max_xy}, low_yz: {low_xy}")
    axZ = fig.add_subplot(223)
    contourlines = axZ.contour(X_xy, Y_xy, log_density_Z, levels=30, cmap='viridis', linewidths=1, vmin = low_xy, vmax = max_xy)

    # Add colorbar
    cbar = plt.colorbar(contourlines, ax=axZ)
    cbar.set_label(r'$\log{(Density)}(cm⁻³)$')
    
    axZ.set_xlabel('X (kpc)')
    axZ.set_ylabel('Y (kpc)')
    axZ.set_title('2D Density Contours – Projection XY')
    axZ.set_aspect('equal')
    
    plt.tight_layout()
    return fig



from matplotlib.widgets import Slider

def plot_density_topography_interactive(stellar_mass_density_map_, box_size):
    """Interactive YZ slice viewer with X slider"""
    
    num_points = stellar_mass_density_map_.shape[0]
    coords = np.linspace(-box_size/2, box_size/2, num_points)
    Y_grid, Z_grid = np.meshgrid(coords, coords)
    
    # Create figure with space for slider
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.15)
    
    # Initial slice
    initial_x_idx = num_points // 2
    density_slice = stellar_mass_density_map_[initial_x_idx, :, :].T
    log_density = np.log10(density_slice)
    
    # Percentile range for consistent colors
    valid = log_density[(log_density != 0) & np.isfinite(log_density)]
    vmin, vmax = np.nanpercentile(valid, [10, 99])
    
    # Initial plot
    contour = ax.contourf(Y_grid, Z_grid, log_density, levels=30, 
                          cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Y (kpc)')
    ax.set_ylabel('Z (kpc)')
    ax.set_aspect('equal')
    title = ax.set_title(f'X = {coords[initial_x_idx]:.2f} kpc')
    plt.colorbar(contour, ax=ax, label=r'$\log_{10}(\rho)$ [cm$^{-3}$]')
    
    # Slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'X slice', 0, num_points-1, 
                    valinit=initial_x_idx, valstep=1)
    
    # Update function
    def update(val):
        x_idx = int(slider.val)
        density_slice = stellar_mass_density_map_[x_idx, :, :].T
        log_density = np.log10(density_slice)
        
        # Clear and redraw
        ax.clear()
        ax.contourf(Y_grid, Z_grid, log_density, levels=30, 
                    cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Y (kpc)')
        ax.set_ylabel('Z (kpc)')
        ax.set_aspect('equal')
        ax.set_title(f'X = {coords[x_idx]:.2f} kpc')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()
    return fig


from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation
import imageio

def create_density_movie(stellar_mass_density_map_, box_size, output_file='density_slices.mp4', endOnFrame=-1, batch_size=50, startBatch=0, contourTolerance=""):
    """Create movie sweeping through X slices"""
    
    num_points = stellar_mass_density_map_.shape[0]
    num_frames = min(num_points, endOnFrame) if endOnFrame != -1 else num_points
    coords = np.linspace(-box_size/2, box_size/2, num_points)
    Y_grid, Z_grid = np.meshgrid(coords, coords)
    
    # # Get consistent color scale
    # log_data = np.log10(stellar_mass_density_map_)
    # valid = log_data[np.isfinite(log_data) & (log_data != 0)]
    # vmin, vmax = np.nanpercentile(valid, [10, 99])

     # Process in batches
    
    if contourTolerance == "constant":
        mid = num_points // 2 # we assume square box
        middleSlice = stellar_mass_density_map_[mid, :, :]
        log_density_center = np.log10(middleSlice)
        percentileRange = [10,99]

        percentiles = np.nanpercentile(log_density_center[(log_density_center!=0) & (~np.isnan(log_density_center)) & np.isfinite(log_density_center)], percentileRange)
        # print(f"percentiles: {percentiles} {len(percentiles) > 1}")
        fig, ax = plt.subplots(figsize=(8, 8))
        if type(percentiles) is np.ndarray and len(percentiles) > 1:
            maxpercent = percentiles[1]
            lowpercent = percentiles[0]
            # print(f"maxpercent: {maxpercent}, low_yz: {lowpercent}")
            contourlines = ax.contour(Y_grid, Z_grid, log_density_center, levels=30, cmap='viridis', linewidths=1, vmin = lowpercent, vmax = maxpercent)
            contourlines_levels = contourlines.levels
        else:
            RuntimeError("list should not be 0")
            return
            contourlines = ax.contour(Y_grid, Z_grid, log_density_center, levels=30, cmap='viridis', linewidths=1)
            contourlines_levels = contourlines.levels
    batch_files = []
    num_batches = (num_frames + batch_size - 1) // batch_size
    
    # Create frames
    frames = []
    print(f"Creating frames for {num_frames} num points (frames)")
    for pre_batch in range(0, startBatch):
        batch_file = f'temp_batch_{pre_batch:03d}.mp4'
        batch_files.append(batch_file)
        print(f'Already had Saved {batch_file}, appending')

    for batch_idx in range(startBatch, num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_points)
        
        print(f'Processing batch {batch_idx+1}/{num_batches} (frames {start_idx}-{end_idx})')
        
        frames = []
        for x_idx in range(start_idx, end_idx):
                
            fig, ax = plt.subplots(figsize=(8, 8))
            
            density_slice = stellar_mass_density_map_[x_idx, :, :].T
            log_density = np.log10(density_slice)

            if contourTolerance == "adaptive":
                percentileRange = [10,99]

                percentiles = np.nanpercentile(log_density[(log_density!=0) & (~np.isnan(log_density)) & np.isfinite(log_density)], percentileRange)
                if type(percentiles) is np.ndarray and len(percentiles) > 1:
                    maxpercent = percentiles[1]
                    lowpercent = percentiles[0]
                    contourlines = ax.contour(Y_grid, Z_grid, log_density, levels=30, cmap='viridis', linewidths=1, vmin = lowpercent, vmax = maxpercent)
                else:
                    RuntimeError("percentiles is not ndarray > 1")
                    # return
                    contourlines = ax.contour(Y_grid, Z_grid, log_density, levels=30, cmap='viridis', linewidths=1)

            elif contourTolerance == "constant":
                contourlines = ax.contour(Y_grid, Z_grid, log_density, levels=contourlines_levels, cmap='viridis', linewidths=1, vmin = lowpercent, vmax = maxpercent)
            else:
                contourlines = ax.contourf(Y_grid, Z_grid, log_density, levels=30, cmap='viridis')#, vmin=vmin, vmax=vmax)

            cbar = plt.colorbar(contourlines, ax=ax)
            cbar.set_label(r'$\log{(Density)}(cm⁻³)$')

            ax.set_xlabel('Y (kpc)')
            ax.set_ylabel('Z (kpc)')
            ax.set_aspect('equal')
            ax.set_title(f'X = {coords[x_idx]:.2f} kpc')
            
            # Convert to image (updated API)
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(image[:, :, :3])  # Drop alpha channel
            
            plt.close(fig)
            print(f'Frame {x_idx+1}/{num_frames}', end='\r')
            
        # Save batch
        batch_file = f'temp_batch_{batch_idx:03d}.mp4'
        imageio.mimsave(batch_file, frames, fps=30)
        batch_files.append(batch_file)
        print(f'Saved {batch_file}')
    
    # Concatenate all batches
    print('Concatenating batches...')
    writer = imageio.get_writer(output_file, fps=30)
    
    for batch_file in batch_files:
        reader = imageio.get_reader(batch_file)
        for frame in reader:
            writer.append_data(frame)
        reader.close()
    
    writer.close()
    print(f'\nSaved to {output_file}')

    
    # # Save movie
    # try:
    #     imageio.mimsave(output_file, frames, fps=30)
    #     print(f'\nSaved to {output_file}')
    # except:
    #     imageio.mimsave("density_slices.mp4", frames, fps=30)
    #     print(f'\nSaved to {"density_slices.mp4"}')