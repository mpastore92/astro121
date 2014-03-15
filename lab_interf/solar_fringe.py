#!/usr/bin/env python
import ephem
import numpy as np
import time
import threading
import radiolab
import argparse

def get_alt():
    """ Returns altitude within the acceptable range.
    Returns:
        float: altitude value of sun limited to the range [15,87] 
    """
    if sun.alt > 87:
        return 87
    elif sun.alt < 15:
        return 15
    else:
        return sun.alt

def controller(t=30.0):
    """Re-point telescope every t seconds.
    Args:
        t (float, optional): number of seconds between updates.  Defaults to 30.0 seconds.
    """
    while(True):
        radiolab.pntTo(sun.az, get_alt())
        time.sleep(n)

def record_data(t):
    """Record data for t seconds
    Args:
        t (float): number of seconds to record 
    """
    radiolab.recordDVM(filename='sun_'+time.strftime("%m-%d-%Y_%H%M%S")+'.npz', sun=True, recordLength=1)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Record solar fringe  data using the interferometer.')
    parser.add_argument('t', metavar='T', type=int, nargs=1, help='time to record')
    parser.add_argument('-p', action='store_true', default=False, help='show real time plot')
    args = parser.parse_args()  
    
    # Create an sun and observer objects
    sun = ephem.Sun()
    obs = ephem.Observer()
    
    # Set lat and long, date
    obs.lat = 37.8732 * np.pi/180
    obs.long = -122.2573 * np.pi/180 
    obs.date = ephem.now()
    

    # Compute sun position
    sun.compute(obs)

    # Start telescopes at home position
    radiolab.pntHome()
    
    # Create a thread that periodically re-points the telescope
    controllerd = threading.Thread(target = test, (args.t[0],))

    # Create to log data 
    datad = threading.Thread(target = record_data, (args.t[0]+10,))

    # Set threads as a daemons, will close automatically
    controllerd.daemon = True
    datad.daemon = True

    # Start controller
    controllerd.start()

    # Wait 5 seconds for telescopes to move and start recording  
    time.sleep(5)
    datad.start()
    
    if args.p:
        radiolab.plotMeNow()
