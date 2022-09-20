import pickle as pkl
import pygrib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Load st_info and data for 20150101_48h
with open('/net/pc160111/nobackup/users/teixeira/st-info.pkl', 'rb') as f:
    st_info = pkl.load(f)

path = "/net/pc160111/nobackup/users/teixeira/norm_data/20150101.pkl"
with open(path, "rb") as file:
    wind_speed = pkl.load(file)['GRID'][48]

# -------------- STATION LOCATIONS --------------
# Dictionary station -> (lon, lat)
latlon = {st:(info['LAT'], info['LON']) for st, info in st_info.items()}

# lon, lat limits
xmin, xmax = 3.0, 7.5
ymin, ymax = 50.7, 53.7

# Plot with Basemap object, maps geo coordinates to pixel coordinates
fig = plt.figure(figsize=(10, 10))
m = Basemap(projection = 'merc', llcrnrlon = xmin, llcrnrlat = ymin, 
            urcrnrlon = xmax, urcrnrlat = ymax, resolution='h')

m.drawcountries(linewidth=1)
m.drawcoastlines(linewidth=1)
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='lightgrey')

for st, (lat, lon) in latlon.items():
    x, y = m(lon, lat)
    m.plot(x,y,3,marker='D',color='r')
    plt.text(x+5000,y-5000, st)
plt.savefig('test2.png')
# plt.savefig(f'/usr/people/teixeira/StatPostProcess-ConvLSTM/imgs/station_locations.png')
plt.show() 


# -------------- WINDSPEED FORECAST --------------
# Array of (lon, lat) for each station 
latlon = np.array([[st['LAT'], st['LON']] for st in st_info.values()]).T

# lon, lat limits and grid dimensions
lonmin = 0.0
lonmax = 11.063
latmin = 49.0
latmax = 55.877
N = 300

# Generate lon, lat grids
lons = np.linspace(lonmin, lonmax, N)
lats = np.linspace(latmin, latmax, N)
grid_lon, grid_lat = np.meshgrid(lons, lats)

# Plot with Basemap object, converts lat, lon to pixel positions
plt.figure(figsize=(10,10))

m = Basemap(projection='merc', llcrnrlon=lons.min(), urcrnrlon=lons.max(),
                               llcrnrlat=lats.min(), urcrnrlat=lats.max(),
                               resolution='h')

x, y = m(grid_lon, grid_lat)

cs = m.pcolormesh(x,y,wind_speed, cmap = plt.cm.nipy_spectral)
m.scatter(latlon[1], latlon[0], s=15, latlon=True, marker='D',color='r')
plt.colorbar(cs, orientation='vertical', shrink=0.7, label='Windspeed (m/s)')

m.drawcountries(linewidth=1)
m.drawcoastlines(linewidth=1)
parallels = np.arange(latmin,latmax,1.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels, labels=[True, False, False, False])
meridians = np.arange(lonmin, lonmax,1.0)
m.drawmeridians(meridians, labels=[False, False, True, False])

# plt.savefig(f'/usr/people/teixeira/StatPostProcess-ConvLSTM/imgs/map_20150101_48h.png')
plt.savefig('test.png')
plt.show()