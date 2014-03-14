#!/usr/bin/env python
import ephem
import numpy as np
import time
from threading import Thread
import radiolab

# Create an sun and observer objects
sun = ephem.Sun()
obs = ephem.Observer()

# Set lat and long, date
obs.lat = 37.8732 * np.pi/180
obs.long = -122.2573 * np.pi/180 
obs.date = ephem.now()

# Compute sun position
sun.compute(obs)

def coord_limit():
    ''' Limit  coordinates to  max az/alt'''

def record_datad():
    '''Save data every n seconds.'''
    
def controllerd():
    '''Update  every n seconds.'''
    obs.sidereal_time()
    sun.az
    sun.alt

if __name__ == "__main__":

    radiolab.pntHome()
    print "ra: "+str(sun.ra)
    print "dec: "+str(sun.dec)

# Set to alt az position
# Alt limit is 15-87 deg
radiolab.pntTo(alt, az)

#record data
#recordDVM
#showplot
# note: sun should be decent looking sinusoid
