#!/usr/bin/env python
import logging
import os
import numpy as np
import time
import threading
import argparse

import ephem
import radiolab

# Create standard file name for log and data files
OBSERVATION = 'sun_'+time.strftime("%m-%d-%Y_%H%M%S")

def getAlt():
    """ Returns altitude within the acceptable range.
    Returns:
        float: altitude value of sun limited to the range [15,87]
    """
    logger = logging.getLogger('interf')
    if sun.alt *180/np.pi> 87:
        logger.warning('Detected alt above 87: %s', str(sun.alt))
        return 87
    elif sun.alt*180/np.pi < 15:
        logger.warning('Detected alt below 15: %s', str(sun.alt))
        return 15
    else:
        return sun.alt*180/np.pi

def recordData(fileName='data/'+OBSERVATION, sun=True, moon=False,
        recordLength=60.0, verbose=False, showPlot=False):
    """Re-point telescope every t seconds.
        fileName (string
    Args:
    fileName (string, optional): Data file to record (in npz format). Data is saved
        as a dictionary with 5 items -- 'ra' records RA of the source
        in decimal hours, 'dec' records Dec in decimal degrees, 'lst'
        records the LST at the time that the voltage was measured, 'jd'
        records the JD at which the voltage was measured, and 'volts'
        records the measured voltage in units of Volts. Default is a file
        called 'voltdata.npz'.
    sun (bool, optional): If set to true, will record the Sun's RA and Dec to the
        file. Default is to record RA or Dec.
    moon (bool, optional): If set to true, will record the Moon's RA and Dec to the
        file. Default is not to record RA or Dec.
    recordLength(float, optional): Length to run the observations for, in seconds.
        Default will cause recordDVM to run until interrupted by Ctrl+C or
        terminal kill.
    verbose (bool, optional): If true, will print out information about each voltage
        measurement as it is taken.
    showPlot (bool, optional): If true, will show a live plot of the data being
        recorded to disk (requires X11 to be on).
    """
    logger = logging.getLogger('interf')
    try:
        logger.debug('Recording data for %f seconds', recordLength)
        radiolab.recordDVM(fileName, sun, moon, recordLength, verbose, showPlot)
    except Exception, e:
        logger.error(str(e))

def controller(t=30.0):
    """Re-point telescope every t seconds.
    Args:
        t (float, optional): number of seconds between updates.  Defaults to 30.0 seconds.
    """
    logger = logging.getLogger('interf')
    try:
        while(True):
            # Update the time and recompute position
            obs.date = ephem.now()
            sun.compute(obs)

            logger.debug('Move to telescope to (alt, az): (%s,%s)',
                    str(getAlt()), str(sun.az))
            radiolab.pntTo(az=sun.az*180/np.pi, alt=getAlt())
            time.sleep(t)

    except Exception, e:
        logger.error('Re-pointing failed for (alt,az): (%d,%d)',
                str(getAlt()),
                str(sun.az))
        logger.error(str(e))

def main():
    """Records interferometer data and saves data to data/sun_DD-MM-YY_HHMMSS.
    Logs stored in logs/sun_DD-MM-YY_HHMMSS.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Record solar fringe  data using the interferometer.')
    parser.add_argument('repoint_freq', type=float, help='time to wait before repointing (s)')
    parser.add_argument('record_len', type=float, help='total time to record (s)')
    parser.add_argument('-p', '--plot', action='store_true', default=False,help='show real time plot (requires X11 to be enabled)')

    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print  voltage measurements')
    args = parser.parse_args()

    # Log observer and sun position
    logger.debug('Observer Lat: %s',  str(obs.lat))
    logger.debug('Observer Long: %s', str(obs.long))
    logger.debug('Observer Date: %s', str(obs.date))
    logger.debug('Sun alt: %s', str(sun.alt))
    logger.debug('Sun az: %s',  str(sun.az))

    # Start telescopes at home position
    logger.debug('Set to home position')
    radiolab.pntHome()

    # Create a thread that periodically re-points the telescope
    controllerd = threading.Thread(target = controller,
            args = (args.repoint_freq,))

    # Create thread to log data
    datad = threading.Thread(target = recordData,
           args = ('data/'+OBSERVATION, True, False, args.record_len+10,
               args.verbose, args.plot))

    #Set threads as a daemons, will close automatically
    controllerd.daemon = True
    datad.daemon = True

    # Start controller
    logger.debug('Start position controller')
    controllerd.start()

    # Wait 8 seconds for telescopes to move and start recording
    time.sleep(8)
    datad.start()

    # Sleep for t seconds to gather data
    time.sleep(args.record_len+10)
    logger.debug('Exiting')

if __name__ == "__main__":
    FORMAT = '%(asctime)-15s - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('interf')
    fh = logging.FileHandler(os.getcwd()+'/logs/'+OBSERVATION+'.log')
    fh.setLevel(logging.DEBUG)

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

    # Compute sun position
    sun.compute(obs)

    main()
