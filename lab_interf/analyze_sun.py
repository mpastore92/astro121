#!/usr/bin/env python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from plot import generate_plot
import scipy.signal.signaltools as sigtool
import ephem
from processing import *

START = 3500
END = 6500

def main():
    print radius(5400,1)
   #print radius(1888,1)
   #print radius(3000,1)
   #print radius(5000,1)
   #print radius(5000,2)
    data = np.load('data/report/sun_04-02-2014_153105.npz')['volts'][:10000]
   ##filter_sun(data)
   #print radius(5400, 1)
   #plt.plot(data/max(data))
    MFtheory = gen_MFtheory(10000)
   #plt.plot(MFtheory/max(MFtheory))
   #generate_plot(data, show=True, dft=True)
    sun(filter_sun(data)[START:END])
    plt.plot(MFtheory[START:END])
    plt.show()

def sanity_check():
    dec = np.load('data/report/sun_04-02-2014_153105.npz')['dec'][:END]
    lst = np.load('data/report/sun_04-02-2014_153105.npz')['lst'][:END]
    ra  = np.load('data/report/sun_04-02-2014_153105.npz')['ra'][:END]
    hs = ra - lst
    hs  *= (2*np.pi/24)
    dec *= (np.pi/180)

    B_y = 1000      # Baseline distance in cm
    lmbda = 2.5      # Wavelength in cm
    Ff = B_y/lmbda*np.cos(dec)*np.cos(hs)
    rad = 1/Ff
    rad *=(180/np.pi)
    #plt.plot(1/Ff)
    #plt.plot(2/Ff)

    plt.plot(rad)
    plt.plot(2*rad)
    plt.plot(3*rad)
    plt.figure()
    gen_MFtheory(len(dec))
    plt.show()

def filter_sun(data, debug=False):
    data -= np.average(data)
    plt.plot(np.linspace(-1,1,len(data)), np.abs(np.fft.fftshift(np.fft.fft(data))))
    filtered = bandpass(data, 0.02, 0.05, 1, 256)
    plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))
    plt.figure()
    plt.plot(data)
    plt.plot(filtered)
    plt.show()

    #data -= np.average(data)

    #return filtered[(len(filtered)-len(data))/2:-(len(filtered)-len(data))/2]
    return filtered[(len(filtered)-len(data)):]

def sun(data):
    dec = np.load('data/report/sun_04-02-2014_153105.npz')['dec'][START:END]

    lst = np.load('data/report/sun_04-02-2014_153105.npz')['lst'][START:END]
    ra = np.load('data/report/sun_04-02-2014_153105.npz')['ra'][START:END]

    hs = ra - lst
    data -= np.average(data)
    data /= max(data)

   #env = signal.fftconvolve(np.abs(data), h)
   #env = env[:len(data)]
    env = envelope_med(data, 301)
   #env = data
    plt.plot(data)
    plt.plot(env)
    plt.title('Envelope')
   #plt.show()
   #plt.show()

    B_y = 1000      # Baseline distance in cm
    lmbda = 2.5      # Wavelength in cm
    YBAR = []
    S_SQ = []
    A = []
    print np.size(data)
    print np.size(hs)

    for phi in np.linspace(0, np.pi, 200):
        F = np.cos(B_y/lmbda * np.cos(dec) * np.cos(hs + phi))
        X = np.matrix( [F, hs*F, np.square(hs)*F] )
        ybar, a,s_sq = least_squares(env, X)
        S_SQ.append(s_sq)
        ybar = np.array(ybar[:,0])
        #print np.shape(ybar)
        YBAR.append(ybar)
        A.append(a)
    print np.linspace(0, np.pi, 200)[np.argmin(S_SQ)]
    fit = YBAR[np.argmin(S_SQ)]
    plt.figure()
    plt.plot(data)
    plt.plot(fit)
    #plt.plot(np.abs(fit))
    plt.plot(envelope_med(fit, 301))
    #plt.plot(signal.fftconvolve(np.abs(fit), h))
    #plt.plot(.002*gen_MFtheory(len(fit)))

    plt.figure()
    plt.plot(S_SQ)
    #plt.show()

if __name__ == "__main__":
    main()
