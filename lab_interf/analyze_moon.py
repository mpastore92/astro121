#!/usr/bin/env python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from plot import generate_plot
import scipy.signal.signaltools as sigtool
import ephem
from processing import *


def main():
    base_title = "Moon on 04-06-2014 beginning at 03:34:44\n"
    START = 1000 #3800
    END = 10000

    data = np.load('data/report/moon_04-06-2014_033444.npz')['volts'][START:END]
    data -= np.average(data)
    data /= np.max(data)
   #generate_plot(data, dft=True)
    filtered = filter_moon(data)
   #generate_plot(filtered, dft=True)
   #envelope = envelope_med(data, 101)
   #generate_plot(envelope, dft=True, show=True)

    MFtheory, fR = gen_MFtheory(10000, start=0, end=2)
    MFtheory /= max(MFtheory)

    # Plot Bessel function
    plt.figure()
    plt.plot(fR, MFtheory)
    plt.xlabel('Frequency \times Radians')
    plt.title('Bessel Function')
    plt.savefig('img/moon/bessel.png')

    # Plot MFtheory over segment of data
    plt.figure()
    plt.plot(filtered)
    plt.plot(MFtheory[START:END])
    plt.legend(['Measured', 'Bessel'])
    plt.title('Comparison of Measurements and Theoretical Bessel Function')
    plt.savefig('img/moon/compare_bessel.png')

    envelope = envelope_med(data, 301)

    YBAR, S_SQ, min_index = moon(filtered, START, END, base_title+"Fit", MFtheory, "data")
    print "Data"
    print min_index
    radius(min_index+START, 1, debug=True)

def get_moon_hs_dec():
    lst = np.load('data/report/moon_04-06-2014_033444.npz')['lst']

    moon = ephem.FixedBody()
    obs = ephem.Observer()
    obs.lat = ephem.degrees(37.8732)
    obs.long = ephem.degrees(-122.2573)
    obs.date = ephem.date('2014/4/06 03:34:44')

    dec = []
    ra = []
    for t in lst:
        moon.compute(obs)
        dec.append(moon.dec)
        ra.append(moon.ra)
        obs.date = ephem.date(obs.date + ephem.second)

    dec = np.array(dec)
    ra = np.array(ra)
    hs = np.array(ephem.hours(moon.ra) - lst)
    return hs, dec

def radius(index, k, debug=False):
    hs, dec = get_moon_hs_dec()
    dec = dec[index]
    hs  = hs[index]

    hs  *= (2*np.pi/24)
    dec *= (np.pi/180)

    B_y = 1000      # Baseline distance in cm
    lmbda = 2.5     # Wavelength in cm
    Ff = (B_y/lmbda*np.cos(dec))*np.cos(hs)
    Ff = np.abs(Ff)
    radius = k/Ff
    radius *=(180/np.pi)
    if debug:
        print 'dec '+str(dec)
        print 'Ff '+str(Ff)
        print 'k = '+str(k)
        print 'radius (deg): '+str(k*radius)
    return k*radius

def filter_moon(data, debug=False):
   #plt.plot(np.linspace(-1,1,len(data)), np.abs(np.fft.fftshift(np.fft.fft(data))))
    filtered = bandpass(data, 0.01, 0.045, 1, 256)
   #plt.plot(np.linspace(-1,1,len(filtered)),
   #        np.abs(np.fft.fftshift(np.fft.fft(filtered))))
   #plt.figure()
   #plt.plot(data)
   #plt.plot(filtered)
   #plt.show()

    #data -= np.average(data)

    #return filtered[(len(filtered)-len(data))/2:-(len(filtered)-len(data))/2]
    return filtered[(len(filtered)-len(data)):]

def moon(data, start, end, title, MFtheory, name):
    hs, dec = get_moon_hs_dec()
    dec = dec[start:end]
    hs = hs[start:end]

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
    plt.savefig('img/moon/'+name+'_residual.png')

    plt.figure()
    plt.plot(data)
    plt.plot(fit)
    plt.plot(env)
    plt.legend(['Original', name.capitalize(), 'Fit', 'Fit Envelope'])
    plt.title(title)
    plt.savefig('img/moon/'+name+'_analysis.png')

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
