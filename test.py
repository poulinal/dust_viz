import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

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

# Plot 1: 3D Position Space with Density Iso-surfaces
def plot_3d_isosurfaces():
    """Create 3D plot with density iso-surfaces using scatter plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate data
    x, y, z, density = generate_gas_data(30, 30, 30)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Define density thresholds for iso-surfaces
    density_levels = [1e19, 1e20, 1e21, 5e21]
    colors_list = ['blue', 'cyan', 'yellow', 'red']
    alphas = [0.1, 0.2, 0.3, 0.6]
    
    # Plot each density threshold as scattered points
    for i, (level, color, alpha) in enumerate(zip(density_levels, colors_list, alphas)):
        # Find points near this density level
        mask = np.abs(density - level) < level * 0.2  # Within 20% of level
        
        if np.any(mask):
            ax.scatter(X[mask], Y[mask], Z[mask], 
                      c=color, alpha=alpha, s=20,
                      label=f'n = {level:.1e} cm⁻³')
    
    ax.set_xlabel('X (pc)')
    ax.set_ylabel('Y (pc)')
    ax.set_zlabel('Z (pc)')
    ax.set_title('3D Gas Density Distribution (Iso-density Surfaces)')
    ax.legend()
    
    return fig

# Plot 2: Position-Position-Density "Topographic" Plot
def plot_density_topography():
    """Create a 3D surface plot where height represents density"""
    fig = plt.figure(figsize=(12, 9))
    
    # Generate 2D slice data
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create density distribution (2D slice at z=0)
    R = np.sqrt(X**2 + Y**2)
    density_2d = 1e20 * np.exp(-R/3)  # Disk
    density_2d += 5e21 * np.exp(-((X-3)**2 + Y**2)/0.5)  # Clump
    density_2d += 3e21 * np.exp(-((X+2)**2 + (Y-2)**2)/0.8)  # Clump
    
    # Take log for better visualization
    log_density = np.log10(density_2d)
    
    # Create 3D surface plot
    ax = fig.add_subplot(121, projection='3d')
    surf = ax.plot_surface(X, Y, log_density, cmap='viridis',
                          linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_xlabel('X (pc)')
    ax.set_ylabel('Y (pc)')
    ax.set_zlabel('log₁₀(n) [cm⁻³]')
    ax.set_title('Density as 3D Surface')
    
    # Add contour plot at the bottom
    ax.contour(X, Y, log_density, zdir='z', 
               offset=ax.get_zlim()[0], cmap='viridis', alpha=0.5)
    
    # Add 2D contour plot for comparison
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, density_2d, levels=20, 
                           norm=LogNorm(vmin=1e18, vmax=1e22), cmap='viridis')
    ax2.contour(X, Y, density_2d, levels=[1e19, 1e20, 1e21], 
                colors='white', linewidths=1)
    
    ax2.set_xlabel('X (pc)')
    ax2.set_ylabel('Y (pc)')
    ax2.set_title('2D Density Contours')
    ax2.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label('Density (cm⁻³)')
    
    plt.tight_layout()
    return fig

# Plot 3: Multi-tracer Approach (Different molecules trace different densities)
def plot_multitracer_3d():
    """Plot multiple gas tracers that probe different density regimes"""
    fig = plt.figure(figsize=(12, 10))
    
    # Generate data
    x, y, z, density = generate_gas_data(40, 40, 40)
    
    # Define critical densities for different tracers
    # CO traces n > 10^2 cm^-3
    # HCN traces n > 10^4 cm^-3  
    # HCO+ traces n > 10^5 cm^-3
    # NH3 traces n > 10^6 cm^-3
    
    tracers = {
        'CO': {'n_crit': 1e18, 'color': 'blue', 'alpha': 0.2},
        'HCN': {'n_crit': 1e20, 'color': 'green', 'alpha': 0.3},
        'HCO+': {'n_crit': 1e21, 'color': 'orange', 'alpha': 0.4},
        'NH3': {'n_crit': 5e21, 'color': 'red', 'alpha': 0.6}
    }
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each tracer
    for tracer_name, props in tracers.items():
        # Create mask for where this tracer would be detected
        mask = density > props['n_crit']
        
        if np.any(mask):
            # Get coordinates where tracer is present
            indices = np.where(mask)
            
            # Subsample for visualization
            step = max(1, len(indices[0]) // 1000)
            x_plot = x[indices[0][::step]]
            y_plot = y[indices[1][::step]]
            z_plot = z[indices[2][::step]]
            
            # Size based on density
            sizes = 50 * (density[mask][::step] / props['n_crit'])**0.3
            sizes = np.clip(sizes, 10, 100)
            
            ax.scatter(x_plot, y_plot, z_plot,
                      c=props['color'], alpha=props['alpha'],
                      s=sizes, label=f"{tracer_name} (n > {props['n_crit']:.0e} cm⁻³)")
    
    ax.set_xlabel('X (pc)')
    ax.set_ylabel('Y (pc)')
    ax.set_zlabel('Z (pc)')
    ax.set_title('Multi-tracer View of Gas Density Distribution')
    ax.legend(loc='upper right')
    
    # Add a 2D projection
    ax2 = fig.add_axes([0.75, 0.15, 0.2, 0.2])
    
    # Project onto XY plane
    density_proj = np.sum(density, axis=2)
    ax2.imshow(np.log10(density_proj), extent=[-10, 10, -10, 10], 
               origin='lower', cmap='gray_r')
    ax2.set_xlabel('X (pc)')
    ax2.set_ylabel('Y (pc)')
    ax2.set_title('Column Density', fontsize=10)
    
    return fig

# Create all three plots
if __name__ == "__main__":
    # Plot 1: 3D iso-surfaces
    fig1 = plot_3d_isosurfaces()
    plt.show()
    
    # Plot 2: Density topography
    fig2 = plot_density_topography()
    plt.show()
    
    # Plot 3: Multi-tracer approach
    fig3 = plot_multitracer_3d()
    plt.show()

# Bonus: Animated rotation for better 3D visualization
def create_rotating_view():
    """Create an animation rotating around the 3D plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate data
    x, y, z, density = generate_gas_data(30, 30, 30)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Plot density iso-surfaces
    density_levels = [1e19, 1e20, 1e21]
    colors_list = ['blue', 'green', 'red']
    alphas = [0.2, 0.3, 0.5]
    
    for level, color, alpha in zip(density_levels, colors_list, alphas):
        mask = (density > level * 0.8) & (density < level * 1.2)
        if np.any(mask):
            ax.scatter(X[mask], Y[mask], Z[mask], 
                      c=color, alpha=alpha, s=30,
                      label=f'n ≈ {level:.1e} cm⁻³')
    
    ax.set_xlabel('X (pc)')
    ax.set_ylabel('Y (pc)')
    ax.set_zlabel('Z (pc)')
    ax.set_title('3D Gas Density Distribution')
    ax.legend()
    
    # Rotate the view
    for angle in range(0, 360, 5):
        ax.view_init(elev=20, azim=angle)
        plt.draw()
        plt.pause(0.05)
    
    return fig

# Uncomment to see rotation
# fig_rot = create_rotating_view()