#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ephem
from scipy import signal

from plot import generate_plot
'''
TODO
moon ra, dec from pyephem
filter cutoff
'''

'''
QUESTIONS
sampling frequency

'''

def main():
    """

    """
    point_source()
    print 'done'
    #data = np.load('ay121_lsq_data.npz')['x']
    #generate_plot(data, show=True, title='Test Plot', outfile='test_plot')


    #least_squares(data['x'])
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

    lst_hr = np.array( [ephem.hours(l) for l in lst[2000:8000]] )
    hs = np.array(ephem.hours(source.ra) - lst_hr)

    # Find a good section of data
    # data = data[2000:8000]

    # Remove DC offset, centers data about zero
    data -= np.average(data)

    # Normalize data
    data /= max(data)

    # 'Brute force' least squares
    lmbda = 2.5 # Wavelength lambda
    YBAR = []
    A = []
    S_SQ = []

    generate_plot(data, show=True, dft=True)

   #for d_B in np.linspace(700,1200,50):
   #    C = 2 * np.pi * d_B/lmbda * np.cos(source.dec)
   #    x1 = np.cos( C * np.sin(hs) )
   #    x2 = -np.sin( C * np.sin(hs) )
   #    X = np.matrix([x1, x2])
   #   #print np.shape(X)
   #   #print np.shape(np.transpose(X))
   #    ybar, a, s_sq = least_squares(data, np.transpose(X))
   #    #print s_sq
   #    #s_sq = least_squares(data, np.transpose(X))

   #    YBAR.append(ybar)
   #    A.append(a)
   #    S_SQ.append(s_sq)

   ##return [np.array(YBAR), np.array(A), np.array(S_SQ)]
   #print len(S_SQ)
   #print np.min(S_SQ)
   #d = np.argmin(S_SQ)
   #print np.linspace(700,1200,500)[d]
   #generate_plot(S_SQ, show=True, title='Residuals')

   #print A[d]
   #print len(data)
   #print len(YBAR[d])
   #plt.plot(range(6000), data)
   #plt.plot(range(6000), YBAR[d])
   #plt.show()
   # generate_plot(YBAR[d], show=True, title='Least Squares Approximation')

def sun():
    dec = np.load('data/report/sun_04-02-2014_153105.npz')['dec']
    B_y = 10000      # Baseline distance in cm
    lmbda = 2.5      # Wavelength in cm

    lst = np.load('data/report/sun_04-02-2014_153105.npz')['lst']
    ra = np.load('data/report/sun_04-02-2014_153105.npz')['ra']

    hs = np.array(ra-lst)
    for phi in np.linspace(0, pi, ):
        F = cos(B_y/lmbda * cos(dec) * cos(hs + phi))
        X = zeros(3, M)
        X = np.matrix( [F, hs*F, square(hs)*F] )
        least_squares()

def least_squares(data, X):
    M = len(data)
    #TODO make this dependent on the shape of X
    N = 2

    Y = data
   #print np.shape(Y)

    XX = np.dot(np.transpose(X), X)
   #print np.shape(XX)

   #print np.shape(X)
   #print np.shape(Y)
    XY = np.transpose(np.dot(np.transpose(X),Y))
   #print np.shape(XY)

    XXI = np.linalg.inv(XX)
   #print np.shape(XXI)

    a = np.dot(XXI,XY)
   #print np.shape(a)

    YBAR = np.dot(X, a)
   #print np.shape(YBAR)

    s_sq = np.sum(np.square((Y-YBAR)))/(M-N)

    return YBAR, a, s_sq

   #plt.plot(YBAR)
   #plt.plot(Y)
   #plt.legend(['$\overline{y}$','$y$'], 'left')
   #plt.show()

   #def least_squares(data, X):
   #    M = len(data)
   #    N = 2

   #    Y = np.transpose(np.matrix(data))

   #    XX = np.dot(X,np.transpose(X))

   #    XY = np.dot(X,Y)
   #    XXI = np.linalg.inv(XX)

   #    a = np.dot(XXI,XY)

   #    YBAR = np.dot(np.transpose(X),a)
   #    s_sq = sum((Y-YBAR)**2)

   #    print(np.shape(YBAR), np.shape(a), np.shape(s_sq))

   #   #plt.plot(YBAR)
   #   #plt.plot(Y)
   #   #plt.legend(['$\overline{y}$','$y$'], 'left')
   #plt.show()

def highpass(data, f, nyquist, numtaps=128, pass_zero=False):
    h =  signal.firwin(numtaps, f, pass_zero=False, nyq=nyquist)
    return signal.fftconvolve(data, h)

def bandpass(data, f1, f2, nyquist, numtaps=128, pass_zero=False):
    h = signal.firwin(numtaps, [f1, f2], pass_zero=False, nyq=nyquist)
    return signal.fftconvolve(data, h)

if __name__ == "__main__":
    main()
