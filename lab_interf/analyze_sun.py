#!/usr/bin/env python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from plot import generate_plot
import scipy.signal.signaltools as sigtool
import ephem
from processing import *


def main():
    base_title = "Sun on 04-02-2014 at 15:31:05\n"
    data = np.load('data/report/sun_04-02-2014_153105.npz')['volts']
    data -= np.average(data)
    generate_plot(data, dft=True, title=base_title, outfile='img/sun/raw')

    START = 3800
    END = 6500
    print radius(5400,1)
   #print radius(1888,1)
   #print radius(3000,1)
   #print radius(5000,1)
   #print radius(5000,2)
   #print radius(5400, 1)
    data = np.load('data/report/sun_04-02-2014_153105.npz')['volts'][START:END]
    data -= np.average(data)
    data /= np.max(data)

    MFtheory, fR = gen_MFtheory(10000)
    MFtheory /= max(MFtheory)

    # Plot Bessel function
    plt.figure()
    plt.plot(fR, MFtheory)
    plt.xlabel('Frequency x Radians')
    plt.title('Bessel Function')
    plt.savefig('img/sun/bessel.png')

    # Plot MFtheory over segment of data
    plt.figure()
    plt.plot(data)
    plt.plot(MFtheory[START:END])
    plt.legend(['Measured', 'Bessel'])
    plt.title('Comparison of Measurements and Theoretical Bessel Function')
    plt.savefig('img/sun/compare_bessel.png')

    filtered = filter_sun(data)
    envelope = envelope_med(data, 301)

    YBAR, S_SQ, min_index = sun(data, START, END, base_title+"Fit", MFtheory, "data")
    print "Data"
    radius(min_index+START, 1, debug=True)
    YBAR, S_SQ, min_index = sun(filtered, START, END, base_title+"Fit of Filtered", MFtheory, "filtered")
    radius(min_index+START, 1, debug=True)
    YBAR, S_SQ, min_index = sun(envelope, START, END, base_title+"Fit of Envelope", MFtheory, "envelope")
    radius(min_index+START, 1, debug=True)


def radius(index, k, debug=False):
    dec = np.load('data/report/sun_04-02-2014_153105.npz')['dec'][index]
    lst = np.load('data/report/sun_04-02-2014_153105.npz')['lst'][index]
    ra  = np.load('data/report/sun_04-02-2014_153105.npz')['ra'][index]

    hs = ra - lst
    hs  *= (2*np.pi/24)
    dec *= (np.pi/180)

    B_y = 1000      # Baseline distance in cm
    lmbda = 2.5     # Wavelength in cm
    Ff = (B_y/lmbda*np.cos(dec))*np.cos(hs)
    radius = 1/Ff
    radius *=(180/np.pi)
    if debug:
        print 'dec '+str(dec)
        print 'Ff '+str(Ff)
        print 'k = '+str(k)
        print 'radius (deg): '+str(k*radius)
    return k*radius

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
   #plt.plot(np.linspace(-1,1,len(data)), np.abs(np.fft.fftshift(np.fft.fft(data))))
    filtered = bandpass(data, 0.02, 0.05, 1, 256)
   #plt.plot(np.linspace(-1,1,len(filtered)),
   #        np.abs(np.fft.fftshift(np.fft.fft(filtered))))
   #plt.figure()
   #plt.plot(data)
   #plt.plot(filtered)
   #plt.show()

    #data -= np.average(data)

    #return filtered[(len(filtered)-len(data))/2:-(len(filtered)-len(data))/2]
    return filtered[(len(filtered)-len(data)):]

def sun(data, start, end, title, MFtheory, name):
    dec = np.load('data/report/sun_04-02-2014_153105.npz')['dec'][start:end]
    lst = np.load('data/report/sun_04-02-2014_153105.npz')['lst'][start:end]
    ra = np.load('data/report/sun_04-02-2014_153105.npz')['ra'][start:end]

    hs = ra - lst

    # Convert everything to radians
    hs  *= (2*np.pi/24)
    dec *= (np.pi/180)

    B_y = 1000       # Baseline distance in cm
    lmbda = 2.5      # Wavelength in cm
    YBAR = []
    S_SQ = []
    A = []
    phi_range = np.linspace(0, np.pi, 200)
    print 'Check sizes'
    print np.size(data)
    print np.size(hs)


    for phi in phi_range:
        #F = np.cos(B_y/lmbda * np.cos(dec) * np.cos(hs + phi))
        F = np.cos(2 * np.pi *B_y/lmbda * np.cos(dec) * np.sin(hs) + phi)
        X = np.matrix( [F, hs*F, np.square(hs)*F] )
        ybar, a,s_sq = least_squares(data, X)
        S_SQ.append(s_sq)
        ybar = np.array(ybar[:,0])
        #print np.shape(ybar)
        YBAR.append(ybar)
        A.append(a)

    fit = YBAR[np.argmin(S_SQ)]
    env = envelope_med(fit, 301)

    plt.figure()
    plt.plot(phi_range, S_SQ/max(S_SQ))
    plt.title(name.capitalize()+" Normalized Residuals")
    plt.savefig('img/sun/'+name+'_residual.png')

    plt.figure()
    plt.plot(data)
    plt.plot(fit)
    plt.plot(env)
    plt.legend(['Original', name.capitalize(), 'Fit', 'Fit Envelope'])
    plt.title(title)
    plt.savefig('img/sun/'+name+'_analysis.png')

    print name.capitalize()
    print 'phi (rad):'+str(phi_range[np.argmin(S_SQ)])
    print 'phi (deg):'+str(phi_range[np.argmin(S_SQ)]*180/np.pi)
    min_index = np.argmin(env[100:-100])+100

    return fit, S_SQ, min_index
    #plt.plot(MFtheory[start:end])
    #plt.plot(signal.fftconvolve(np.abs(fit), h))
    #plt.plot(.002*gen_MFtheory(len(fit)))

   ##plt.show()

if __name__ == "__main__":
    main()
