import numpy as np

class PointSource:
    """
    Used to represent point sources and return the appropriate altitude and
    azimuth.

    ra  (float): right ascension of object of interest (radians)
    dec (float): declination of object of interest (radians)
    """

    def __init__(self, obs, ra=0, dec=0):
        self.obs = obs     # PyEphem observer object
        self.ra  = ra      # RA in h:m:s
        self.dec = dec     # DEC in h:m:s
        self.alt = 0.0     # Altitude in radians
        self.az  = 0.0     # Azimuth in radians

    def compute(self, observer):
        """ Update azimuth and altitude based on source's RA and DEC
        coordinates.

        Returns az and alt in radians

        observer (ephem.Observer): accepts an observer object for compatibility
        with the PyEphem compute function but uses the object's observer.
        """
        r = np.array([
            np.cos(self.dec) * np.cos(self.ra),
            np.cos(self.dec) * np.sin(self.ra),
            np.sin(self.dec)
        ])

        # Convert to HA DEC
        toHD = np.array([
            [np.cos(self.obs.sidereal_time()), np.sin(self.obs.sidereal_time()), 0],
            [np.sin(self.obs.sidereal_time()), -np.cos(self.obs.sidereal_time()), 0],
            [0, 0, 1]
        ])

        r = np.dot(toHD, r)

        # Convert from HA DEC to AZ ALT
        toAA = np.array([
            [-np.sin(self.obs.lat), 0, np.cos(self.obs.lat)],
            [0, -1, 0],
            [np.cos(self.obs.lat), 0, np.sin(self.obs.lat)]
        ])

        r = np.dot(toAA, r)

        # Return AZ and ALT
        self.az  = np.arctan2(r[1], r[0])
        self.alt = np.arcsin(r[2])
