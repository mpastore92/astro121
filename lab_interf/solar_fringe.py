#!/usr/bin/env python
import ephem
import numpy as np
import time
from threading import Thread
import radiolab
print 'imports ok'

## Create an sun and observer objects
#sun = ephem.Sun()
#obs = ephem.Observer()
#
## Set lat and long, date
#obs.lat = 37.8732 * np.pi/180
#obs.long = -122.2573 * np.pi/180 
#obs.date = ephem.now()
#
## Compute sun position
#sun.compute(obs)

def alt_limit(alt):
    ''' Limit  coordinates to  max az/alt'''
    if alt > 87:
        return 87
    elif alt < 15:
        return 15

def record_data(n=30):
    '''Save data every n seconds.'''
    radiolab.recordDVM(filename='sun_'+time.strftime("%m-%d-%Y_%H%M%S")+'.npz', sun=True, recordLength=n)
    
def controller(n=30):
    '''Update  every n seconds.'''
    radiolab.pntTo(sun.az, alt_limit(sun.alt))
    time.sleep(n)
def test():
    #print(str(obs.sidereal_time()))
    print('test')
    time.sleep(1)

if __name__ == "__main__":
    
    #radiolab.pntHome()
    controllerd = Thread(target = test)
    print 'Starting thread'
    controllerd.start()
    time.sleep(5)
    controllerd.join()
    print 'Stopping thread'
'''
    controllerd.daemon = True

    datad = Thread(target = radiolab.recordDVM, args=(filename='sun_'+time.strftime("%m-%d-%Y_%H%M%S")+'.npz', sun=True, recordLength=n)
))
    datad.daemon = True

    controllerd.start()
    time.sleep(5)
    #TODO: check time required to start
    datad.start()
    #
    print "ra: "+str(sun.ra)
    print "dec: "+str(sun.dec)
'''

# Set to alt az position

#recordDVM
#getDVMData
#getLST

#showplot
# note: sun should be decent looking sinusoid
