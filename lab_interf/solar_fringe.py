#!/usr/bin/env python
import logging
import os
import numpy as np
import time
import threading
import argparse

import ephem
import radiolab

def get_alt():
    """ Returns altitude within the acceptable range.
    Returns:
        float: altitude value of sun limited to the range [15,87] 
    """
    if sun.alt > 87:
        logger.info('Detected alt above 87: %s', str(alt))
        return 87
    elif sun.alt < 15:
        logger.info('Detected alt below 15: %s', str(alt))
        return 15
    else:
        return sun.alt

def controller(t=30.0):
    """Re-point telescope every t seconds.
    Args:
        t (float, optional): number of seconds between updates.  Defaults to 30.0 seconds.
    """
    while(True):
        logger.debug('Move to telescope to (alt, az): (%s,%s)', str(get_alt), str(az))
        radiolab.pntTo(sun.az, get_alt())
        time.sleep(t)

def record_data(t):
    """Record data for t seconds
    Args:
        t (float): number of seconds to record 
    """
    radiolab.recordDVM(filename='logs/'+OBSERVATION, sun=True, recordLength=t)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Record solar fringe  data using the interferometer.')
    parser.add_argument('t', metavar='T', type=int, nargs=1, help='time to record')
    parser.add_argument('-p', action='store_true', default=False, help='show real time plot')
    args = parser.parse_args()  
    
    # Create standard file name for log and data files
    OBSERVATION = 'sun_'+time.strftime("%m-%d-%Y_%H%M%S")

    # Set up logger
    FORMAT = '%(asctime)-15s - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('interferometer')
    fh = logging.FileHandler(os.getcwd()+'/logs/'+OBSERVATION+'.log')
    fh.setLevel(logging.INFO)
  
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
  
    formatter = logging.Formatter(FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    #Create an sun and observer objects
    sun = ephem.Sun()
    obs = ephem.Observer()
    
    # Set lat and long, date
    obs.lat = 37.8732 * np.pi/180
    obs.long = -122.2573 * np.pi/180 
    obs.date = ephem.now()
    
    logger.debug('Observer Lat: '+str(obs.lat))
    logger.debug('Observer Long: '+str(obs.lat))
    logger.debug('Observer Date: '+str(obs.date))

    # Compute sun position
    sun.compute(obs)
    logger.debug('Sun alt: %d', sun.alt)
    logger.debug('Sun az: %d', sun.az))

    # Start telescopes at home position
    logger.info('Set to home position')
    radiolab.pntHome()
    
    # Create a thread that periodically re-points the telescope
    controllerd = threading.Thread(target = test, (args.t[0],))

    # Create to log data 
    datad = threading.Thread(target = record_data, (args.t[0]+10,))

    # Set threads as a daemons, will close automatically
    controllerd.daemon = True
    datad.daemon = True

    # Start controller
    logger.info('Start position controller')
    controllerd.start()

    # Wait 5 seconds for telescopes to move and start recording  
    time.sleep(5)
    logger.info('Start logging data')
    datad.start()
    
    if args.p:
        logger.info('Start plot')
        radiolab.plotMeNow()

if __name__ == "__main__":
   main() 
