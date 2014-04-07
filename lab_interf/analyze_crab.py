#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ephem
from scipy import signal

from plot import generate_plot

def main():
    """

    """
    point_source()

def point_source():
    data = np.load('data/report/3C144_03-28-2014_001926.npz')['volts']
    lst = np.load('data/report/3C144_03-28-2014_001926.npz')['lst']

    source = ephem.FixedBody()
    obs = ephem.Observer()
    obs.lat = ephem.degrees(37.8732)
    obs.long = ephem.degrees(-122.2573)
    obs.date = ephem.date('2014/3/28 0:19:26')

    source._ra    = ephem.hours("5:34:31.95")
    source._dec   = ephem.degrees("22:00:51.1")
    source._epoch = ephem.J2000
    source.compute(obs)

    lst_hr = np.array( [ephem.hours(l) for l in lst] )
    hs = np.array(ephem.hours(source.ra) - lst_hr)

    # Find a good section of data
    # data = data[2000:8000]

    # Remove DC offset, centers data about zero
   #normalized = np.array([0,])
   #for i in range(1,len(data),1000):
   #    print i
   #    seg = data[(i-1):i]
   #    if len(seg)!=0:
   #        np.append(normalized(seg/max(seg)))
   #print len(data)
   #print len(normalized)

   # generate_plot(normalized, show=True, dft=True)

    data -= np.average(data)
    data = np.append(np.zeros(6600), data[6600:])
    generate_plot(data, show=True, dft=True)

   ## Normalize data
   #data /= max(data)

   ## 'Brute force' least squares
   #lmbda = 2.5 # Wavelength lambda
   #YBAR = []
   #A = []
   #S_SQ = []

   #for d_B in np.linspace(700,1200,50):
   #    C = 2 * np.pi * d_B/lmbda * np.cos(source.dec)
   #    x1 = np.cos( C * np.sin(hs) )
   #    x2 = np.sin( C * np.sin(hs) )
   #    X = np.matrix([x1, x2])

   #    ybar, s_sq = least_squares(data, X)
   #    S_SQ.append(s_sq)
   #    ybar = np.array(ybar[:,0])
   #    #print np.shape(ybar)
   #    YBAR.append(ybar)
   #   #print np.shape(X)
   #   #print np.shape(np.transpose(X))


   #generate_plot(YBAR, show=True, dft=True)

def least_squares(data, X):
    #X = np.matrix([x**0, x, x**2])
    Y = np.transpose(np.matrix(data))

    XX = np.dot(X,np.transpose(X))

    XY = np.dot(X,Y)
    XXI = np.linalg.inv(XX)

    a = np.dot(XXI,XY)
    YBAR = np.dot(np.transpose(X),a)
    YBAR = np.array(YBAR[:,0])
    DELY = Y - YBAR
    s_sq = np.sum(np.square(DELY))#np.dot(np.transpose(DELY),DELY/(1000-3))
    return YBAR, s_sq

if __name__ == "__main__":
    main()
