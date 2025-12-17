from slider_plot import *

t = np.load("/path/to/density/map/file.npy")
t = t*0.5*0.5*0.5*10**10
t = np.log10(t)

dim = t.shape[0]

slider_plot(t, dim)
