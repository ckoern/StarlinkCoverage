from datetime import datetime, timedelta
from argparse import ArgumentParser

from skyfield.api import load, wgs84, utc
from skyfield.api import EarthSatellite

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from skimage.morphology import disk
import scipy


import sys


def load_population_data( init_filename, rebin ):
    init_dataset = init_filename is not None
    load_from_numpy = not init_dataset
    save_as_numpy = init_dataset
    if load_from_numpy:
        pop_data = np.load('population_data.npy')
    else:
        pop_data_filename = init_filename
        try:
            with open(pop_data_filename) as f:
                for i in range(6): #first 6 lines are header information
                    f.readline()
                pop_data = np.loadtxt( f )
        except:
            print('Please Provide the GPW Population Count ASCII Data file as command line argument')
    if save_as_numpy:
        np.save('population_data', pop_data)

    # rebin data
    pop_data = np.sum( pop_data.reshape( pop_data.shape[0]//rebin,rebin,pop_data.shape[1]//rebin,rebin )  ,axis = (1,3) )

    return pop_data

parser = ArgumentParser()
parser.add_argument( "--init", type = str, default = None, help="Provide population data file to generate numpy array data" )
parser.add_argument( "--rebin", type = int, choices=[1,2,4,8,16,32,64], default=4, help="Rebin the population data for performance increase" )
parser.add_argument( "--save-plot", action =  "store_true", help="if set, the animation is saved" )
args = parser.parse_args()

ts = load.timescale()


#lat/longitude degrees visible by a satalite at equator (WARNING: this uses a simplification resulting in a non-constant solid angle depending on the lattitude )
radius_angle = 6.

seconds_per_frame = 10.

n_frames = 10


pop_data = load_population_data(args.init, args.rebin)


#mask negative values in data array (-9999 is used for cells with no available data)
masked_pop_data =np.ma.masked_where(pop_data <0, pop_data)


dx = 360. / pop_data.shape[1]
dy = 180. / pop_data.shape[0]
data_lon = dx/2. + dx * np.arange( pop_data.shape[1] ) - 180
data_lat = 90 - (dy/2. + dy * np.arange( pop_data.shape[0] ) )


pop_data[pop_data < 0] = 0



radius_pixels = int(radius_angle / dx)
filter_kernel = disk(radius_pixels)
filter_size = filter_kernel.shape[0]

#convolute population with the visibility disk to obtain the total number of people visible at each coodinate 
cum_pop_data = scipy.ndimage.convolve( pop_data, filter_kernel, mode = 'wrap' )

cum_pop_data[cum_pop_data<1] = 1

def get_pop_at(lat, lon):
    ix = (np.abs(data_lon - lon)).argmin()
    iy = (np.abs( data_lat - lat )).argmin()
    return cum_pop_data[iy,ix]


def add_sat_at(lat,lon, buff):
    ix = (np.abs(data_lon - lon)).argmin()
    iy = (np.abs( data_lat - lat )).argmin()
    buff[iy,ix] += 1

fig, (ax, axx) = plt.subplots(2,1)
ax.imshow( masked_pop_data , norm = LogNorm())
axx.imshow( cum_pop_data, norm = LogNorm() )
fig.show()
now = datetime.now(tz=utc)
start_of_year = datetime( now.year,1,1,0,0,0,tzinfo=utc )
timediff = now - start_of_year
sat_height = 550.
sat_period = 95.65
revolutions_per_day = 24*60./sat_period
epoch = '{:02d}{:03d}.{:08d}'.format( now.year%2000, timediff.days, int(1e8 * (timediff.seconds-100) / (24*60*60) )  )     # string conforming the TLE epoch syntax
now_string = '{:d}/{:d}/{:d} {:02d}:{:02d}:{:02d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)  # string to form the timestamp for pyepoch TLE.compute()
now_time = ts.from_datetime( now )
print( epoch )
print(now_string)
def checksum(s):
    # generates the checksum value for TLE lines
    checksum = 0
    for c in s:
        try:
            checksum += int(c)
        except:
            if c == "-":
                checksum += 1
    return checksum%10

def build_tle( name, inclination, ra, mean_anomaly = 0, quiet = True):
    # builds TLE based on actual Starlink-77 TLE. Only inclination, RAAN and mean anomaly can be changed to create TLEs for the constellation
    line1 = '1 44291U 19029BJ  {}  .00001597  00000-0  12902-3 0  999'.format(epoch)
    line1 += str( checksum(line1) )
    line2 = '2 44291 {:8.4f} {:8.4f} 0000001 000.0001 {:8.4f} {:011.8f}00001'.format( inclination, ra,mean_anomaly, revolutions_per_day )
    line2 += str( checksum(line2) )
    if not quiet:
        print( ''.join( [str((i+1)//10) for i in range(69)] ) )
        print( ''.join( [str((i+1)%10) for i in range(69)] ) )
        print( line1)
        print( line2 )
        print( len(line1), len(line2) )
    #return ephem.readtle(name,line1, line2)
    return EarthSatellite(line1, line2, name, ts)


def get_latlon_at( sat, t ):
    #t = ts.from_datetime( dt )
    p = sat.at( t )
    _lat, _lon = wgs84.latlon_of(p)
    return _lat.degrees, _lon.degrees


sl_incl = 53
sl_nsats_per_plane = 66
sl_nplanes = 24
to_deg = 180./np.pi
ra_list = 360./sl_nplanes * np.arange(sl_nplanes)
ma_list = 360./sl_nsats_per_plane * np.arange( sl_nsats_per_plane )
i=0
tle_list = []
lat = []
lon = []
pop_list = []
#generate TLEs for all sats
for j,ra in enumerate(ra_list):
    for ma in ma_list:
        i+= 1
        if j%2 == 0:
            ma +=  360./sl_nsats_per_plane / 2.
        tle = build_tle('SAT%d'%i, sl_incl, ra, ma)
        _lat, _lon = get_latlon_at( tle, now_time )

        lon.append( _lon )
        lat.append( _lat )
        pop_list.append(  get_pop_at( _lat, _lon ) )
        tle_list.append(tle)

        

satcount = np.zeros( pop_data.shape )
for a,b in zip(lat,lon):
    add_sat_at(a,b, satcount )
satcount = scipy.ndimage.convolve( satcount, filter_kernel, mode = 'wrap' )




fig2 = plt.figure( figsize = (16,8))
fig2.text( 0.98, 0.02, r'v0.1   For details see https://gitlab.com/ckoern/starlinkcoverage', fontfamily = 'monospace', weight = 'bold',
                        horizontalalignment = 'right', verticalalignment = 'bottom', color = 'grey' )
fig2.subplotpars.left = 0.05
fig2.subplotpars.right = 0.9
fig2.subplotpars.top = 0.9
fig2.subplotpars.bottom = 0.1

gs=GridSpec(3,2)
ax2 = fig2.add_subplot(gs[:2,0], projection=ccrs.PlateCarree())
ax3 = fig2.add_subplot(gs[-1,0])
ax4 = fig2.add_subplot(gs[:2,1], projection=ccrs.PlateCarree()) 
ax5 = fig2.add_subplot(gs[-1,1])
ax4.set_title('Number of visible Sats')
ax2.set_title('Population and Sat Positions')

pop_im = ax2.imshow( masked_pop_data, extent = (-180,180, 90,-90),norm = LogNorm(), origin="lower" )
# cax = fig2.add_axes([ax2.get_position().x1+0.01,ax2.get_position().y0,0.02,ax2.get_position().height])
# fig2.colorbar( pop_im, cax = cax )
ax2.coastlines()
satcount_im = None
ax4.coastlines()
scat = None
satcount_im = ax4.imshow( satcount, vmax = satcount.max() + 2, extent = (-180,180, 90,-90) )
cax = fig2.add_axes([ax4.get_position().x1+0.01,ax4.get_position().y0,0.02,ax4.get_position().height])
fig2.colorbar( satcount_im, cax = cax )

def init():
    global scat
    scat = ax2.scatter( lon, lat, c = np.log(pop_list), cmap = 'Reds', marker = '.' )
    ax2.set_xlim(-180,180)
    ax2.set_ylim( -90,90 )
    ax4.set_xlim(-180,180)
    ax4.set_ylim( -90,90 )
    return scat,

def update( frame ):
    print('build frame %d...'%frame )
    n = now + timedelta( seconds = seconds_per_frame * int(frame) )
    t = ts.from_datetime( n )

    lat = []
    lon = []
    satpop = []
    satcount = np.zeros( pop_data.shape )
    for tle in tle_list:
        _lat, _lon = get_latlon_at( tle, t )
        lon.append( _lon )
        lat.append( _lat )
        satpop.append( get_pop_at( _lat, _lon )  )
        add_sat_at( _lat, _lon, satcount )

    scat.set_offsets( np.array([ lon,lat ]).transpose() )
    scat.set_array( np.log( np.array( satpop ) ) )
    satcount = scipy.ndimage.convolve( satcount, filter_kernel, mode = 'wrap' )
    satcount_im.set_data( satcount )
    ax3.clear()
    ax3.set_title('Visible Humans per Sat')
    ax3.hist(np.clip(satpop, 0, 1e8)/1e8, log = True, bins = 100,histtype='step', range = (0,1))
    ax3.set_xlabel('10^8 Humans')
    ax3.set_ylabel('Satcount / bin')
    ax3.set_yscale('log')
    ax5.clear()
    ax5.set_title('Available Bandwidth per Human')
    ax5.hist( np.clip( (20000*satcount/cum_pop_data )[cum_pop_data > 0].flatten(), 0, 1. ), log = True, bins = 1000,histtype='step', range=(0,1), density = True )
    ax5.set_xlabel( 'MBit/s' )
    ax5.set_ylabel( '$\\rho$' )
    ax5.set_yscale('log')
    return scat,


print( 'Total Population: ', pop_data.sum() )
print( 'Averaged Total Population', cum_pop_data.sum() )

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig2, update, frames=np.arange(n_frames),
                            init_func=init, blit=False)

if args.save_plot:
    ani.save('starlink_coverage.gif', writer='imagemagick', fps=1)

plt.show()




