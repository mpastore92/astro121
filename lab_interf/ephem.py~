import ephem
import numpy as np
import time

sun = ephem.Sun()

# Create an observer object to compute position
obs = ephem.Observer()

# Set lat and long
# can set using strings as well: obs.lat = '37:52'
obs.lat = 37.8732 * np.pi/180
obs.long = -122.2573 * np.pi/180 

# Set date
obs.date = ephem.now()

# Compute sun position
sun.compute(obs)
print "ra: "+str(sun.ra)
print "dec: "+str(sun.dec)

print "sidereal time: "+str(obs.sidereal_time())

print "azimuth: "+str(sun.az)
print "altitude: "+str(sun.alt)


