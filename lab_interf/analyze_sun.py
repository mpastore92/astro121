#!/usr/bin/env python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from plot import generate_plot
import scipy.signal.signaltools as sigtool
import ephem

def main():
    print radius(5400, 1)
    dec = np.load('data/report/sun_04-02-2014_153105.npz')['dec'][:10000][::10]
    #print dec
    data = np.load('data/report/sun_04-02-2014_153105.npz')['volts'][:10000]
    data -= np.average(data)
    plt.plot(data/max(data))
    plt.figure()
    MFtheory = gen_MFtheory(10000)
   #plt.plot(MFtheory/max(MFtheory))
    plt.show()
   # sun()

def sanity_check():
    dec = np.load('data/report/sun_04-02-2014_153105.npz')['dec'][:10000]
    lst = np.load('data/report/sun_04-02-2014_153105.npz')['lst'][:10000]
    ra  = np.load('data/report/sun_04-02-2014_153105.npz')['ra'][:10000]
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
    gen_MFtheory(10000)
    plt.show()

def radius(index, k):
    dec = np.load('data/report/sun_04-02-2014_153105.npz')['dec'][index]
    lst = np.load('data/report/sun_04-02-2014_153105.npz')['lst'][index]
    ra  = np.load('data/report/sun_04-02-2014_153105.npz')['ra'][index]
    hs = ra - lst
    hs  *= (2*np.pi/24)
    dec *= (np.pi/180)

    B_y = 1000      # Baseline distance in cm
    lmbda = 2.5      # Wavelength in cm
    Ff = (B_y/lmbda*np.cos(dec))*np.cos(hs)
    radius = 1/Ff
    radius *=(180/np.pi)
    print 'dec '+str(dec)
    print 'Ff '+str(Ff)
    print 1/Ff
    return k*radius

def gen_MFtheory(length):
    fR = np.linspace(-2,0,length)
    MFtheory = np.zeros(len(fR))
    N = 1000
    for n in np.arange(-N,N):
       MFtheory += (np.sqrt(1-np.square((n/N))) * np.cos(2 * np.pi *fR * n/N))
    plt.plot(fR, MFtheory)
    plt.show()
    return MFtheory

def sun():
    data = np.load('data/report/sun_04-02-2014_153105.npz')['volts']
    #plt.plot(data)
    #plt.show()
   #start = 600
   #end = 3200
    start = 1800
    end = 5400
    data = np.load('data/report/sun_04-02-2014_153105.npz')['volts'][start:end]
    #plt.plot(np.fft.fft(data))
    filt = np.fft.fft(data)
    filt[200:2300]*=0
    #plt.plot(abs(filt))
    data = np.fft.ifft(filt)
   #plt.plot(data)
   #plt.show()
    dec = np.load('data/report/sun_04-02-2014_153105.npz')['dec'][start:end]
    B_y = 1000      # Baseline distance in cm
    lmbda = 2.5      # Wavelength in cm

    lst = np.load('data/report/sun_04-02-2014_153105.npz')['lst'][start:end]
    ra = np.load('data/report/sun_04-02-2014_153105.npz')['ra'][start:end]

    hs = ra - lst
    data -= np.average(data)
    data /= max(data)

    YBAR = []
    S_SQ = []

    h = signal.firwin(128, 0.01, nyq=1)
   #env = signal.fftconvolve(np.abs(data), h)
   #env = env[:len(data)]
    env = data
    plt.plot(data)
    plt.plot(env)
   #plt.show()


    for phi in np.linspace(0, np.pi, 200):
        F = np.cos(B_y/lmbda * np.cos(dec) * np.cos(hs + phi))
        X = np.matrix( [F, hs*F, np.square(hs)*F] )
        ybar, s_sq = least_squares(env, X)
        S_SQ.append(s_sq)
        ybar = np.array(ybar[:,0])
        #print np.shape(ybar)
        YBAR.append(ybar)
    print np.linspace(0, np.pi, 200)[np.argmin(S_SQ)]
    fit = YBAR[np.argmin(S_SQ)]
    plt.plot(fit)
    plt.plot(np.abs(fit))
    plt.plot(signal.fftconvolve(np.abs(fit), h))
    plt.plot(.002*gen_MFtheory(len(fit)))

    plt.figure()
    plt.plot(S_SQ)
    plt.show()
   #plt.show()

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
