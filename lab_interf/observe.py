#!/usr/bin/env python
import logging
import os
import numpy as np
import time
import threading
import argparse

import ephem
import radiolab

# TODO: Check output of radiolab.pntTo and check for errors

# Define telescope limits in degrees
ALT_MAX = 87
ALT_MIN = 15

# Create a global observer object
OBS = ephem.Observer()

# Set lat and long (for Berkeley), and date
OBS.lat = 37.8732 * np.pi/180
OBS.long = -122.2573 * np.pi/180
OBS.date = ephem.now()

# Observable sources
SOURCES = {
    "sun":   ephem.Sun(),
    "moon":  ephem.Moon(),
    "m17":   ephem.FixedBody(),
    "3C144": ephem.FixedBody(),
    "orion": ephem.FixedBody(),
    "3C405": ephem.FixedBody(),
    "3C461": ephem.FixedBody(),

}
SOURCE_COORDS = {
    "m17":   {'ra':ephem.hours("18:20:26"),     'dec':ephem.degrees("16:10.6"))},
    "3C144": {'ra':ephem.hours("5:34:31.95"),   'dec':ephem.degrees("22:00:51.1"))},
    "orion": {'ra':ephem.hours("5:35:17.3"),    'dec':ephem.degrees("-05:24:28"))},
    "3C405": {'ra':ephem.hours("19:59:28.357"), 'dec':ephem.degrees("40:44:02.10"))},
    "3C461": {'ra':ephem.hours("23:23:24"),     'dec':ephem.degrees("58:48.9"))},
}

def initSource(name, source):
    if name in SOURCE_COORDS:
        source._ra    = SOURCE_COORDS['name']['ra']
        source._dec   = SOURCE_COORDS['name']['dec']
        source._epoch = '2000'

def getAlt(source):
    """ Returns altitude in degrees within the acceptable range [ALT_MIN,
    ALT_MAX]. Saturates to ALT_MAX or ALT_MIN if the source is outside the
    acceptable range.

    Returns:
        float: altitude value of sun in degrees limited to the range [15,87]
    """
    logger = logging.getLogger('interf')
    if source.alt*180/np.pi > ALT_MAX:
        logger.warning('Detected alt above 87: %s', str(source.alt))
        return 87
    elif source.alt*180/np.pi < ALT_MIN:
        logger.warning('Detected alt below 15: %s', str(source.alt))
        return 15
    else:
        return source.alt*180/np.pi

def recordData(fileName='data/'+"unknown_" + time.strftime("%m-%d-%Y_%H%M%S"), sun=False, moon=False,
        recordLength=60.0, verbose=False, showPlot=False):
    """
    Record data from the interferometer.

    Args:
    fileName (string, optional): Data file to record (in npz format). Data is saved
        as a dictionary with 5 items -- 'ra' records RA of the source
        in decimal hours, 'dec' records Dec in decimal degrees, 'lst'
        records the LST at the time that the voltage was measured, 'jd'
        records the JD at which the voltage was measured, and 'volts'
        records the measured voltage in units of Volts. Default is a file
        called 'voltdata.npz'.
    sun (bool, optional): If set to true, will record the Sun's RA and Dec to the
        file. Default is not to record RA or Dec.
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

def controller(source, t=30.0):
    """Updates position and re-point telescope every t seconds.
    Args:
        t (float, optional): number of seconds between updates.  Defaults to 30.0 seconds.
    """
    logger = logging.getLogger('interf')
    try:
        while(True):
            # Update the time and recompute position
            OBS.date = ephem.now()
            source.compute(OBS)

            logger.debug('Move to telescope to (alt, az): (%s,%s)',
                    str(getAlt(source)), str(source.az))
            radiolab.pntTo(az=source.az*180/np.pi, alt=getAlt(source))
            time.sleep(t)

    except Exception, e:
        logger.error('Re-pointing failed for (alt,az): (%s,%s)',
                str(getAlt(source)),
                str(source.az))
        logger.error(str(e))

def initLog(observation):
    FORMAT = '%(asctime)-15s - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('interf')
    fh = logging.FileHandler(os.getcwd()+'/logs/'+ observation +'.log')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    formatter = logging.Formatter(FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def main():
    """Records interferometer data and saves data to data/sun_DD-MM-YY_HHMMSS.
    Logs stored in logs/sun_DD-MM-YY_HHMMSS.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Record solar fringe  data using the interferometer.')
    parser.add_argument('source', choices=SOURCES.keys(), help='a source to observe')
    parser.add_argument('repoint_freq', type=float, default=30.0, help='time to wait before repointing (s)')
    parser.add_argument('record_len', type=float, default=3600.0, help='total time to record (s)')
    parser.add_argument('-p', '--plot', action='store_true', default=False,help='show real time plot (requires X11 to be enabled)')

    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print  voltage measurements')
    args = parser.parse_args()

    if args.repoint_freq <= 12:
        raise argparse.ArgumentTypeError("Can't repoint more often than every 12 seconds.")

    if args.record_len <= 0:
        raise argparse.ArgumentTypeError("Can't record for less than 0 seconds.")

    # Select the source to observe
    source = SOURCES[args.source]

    # Initialize point source RA and DEC
    initSource(args.source, source)

    # Create standard file name for log and data files
    observation = args.source + "_" + time.strftime("%m-%d-%Y_%H%M%S")

    # Start logging
    logger = initLog(observation)

    # Compute source position
    source.compute(OBS)

    # Log observer and sun position
    logger.debug('Calculating position of %s', args.source)
    logger.debug('Observer Lat: %s',  str(OBS.lat))
    logger.debug('Observer Long: %s', str(OBS.long))
    logger.debug('Observer Date: %s', str(OBS.date))
    logger.debug('%s alt: %s', args.source, str(ephem.degrees(source.alt)))
    logger.debug('%s az: %s',  args.source, str(ephem.degrees(source.az)))

    # Start telescopes at home position
    logger.debug('Set to home position')
    radiolab.pntHome()

    # Create a thread that periodically re-points the telescope
    controllerd = threading.Thread(target = controller,
            args = (source, args.repoint_freq))

    # Create thread to log data
    sun  = (args.source == 'sun')
    moon = (args.source == 'moon')
    datad = threading.Thread(target = recordData,
           args = ('data/'+observation, sun, moon, args.record_len+10,
               args.verbose, args.plot))

    # Set threads as daemons, will cause them to close automatically
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
    main()
