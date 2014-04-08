#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ephem
from scipy import signal

from processing import *
from plot import generate_plot

START = 2000
END = 8000

def main():
    """

    """
    data = np.load('data/report/3C144_03-28-2014_001926.npz')['volts'][START:END]
    plt.figure()
    plt.plot(np.linspace(-1,1,len(data)), np.abs(np.fft.fftshift(np.fft.fft(data))))
    plt.show()

   #normalized_chunks = normalize_chunks(data, 1000)
   #filtered_chunks = filter(normalized_chunks, debug=False)
   #generate_plot(normalized, dft=True)

    filtered = filter(data, debug=False)
    normalized = normalize(filtered)
    print len(normalized)
   #generate_plot(filtered, show=True, dft=True)
    #fit(data)
   #fit(filtered)
   #fit(normalized_chunks)
    fit(normalized, debug=True)
    plt.show()

    #point_source()

def filter(data, show=False, debug=False):
    # Remove DC
    data -= np.average(data)

    if debug:
        # Test filters
        filtered = median_filter(data, 5)
        plt.figure()
        plt.plot(np.linspace(-1,1,len(data)), np.abs(np.fft.fftshift(np.fft.fft(data))))
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        filtered = median_filter(data, 9)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        filtered = median_filter(data, 11)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))
        plt.title('Testing Median Filter Window Sizes')
        plt.legend(['Original','Window = 5', 'Window = 9', 'Window = 11'])
        plt.xlabel('Frequency (Hz)')
        plt.savefig('img/median_test.png')

        plt.figure()
        plt.plot(np.linspace(-1,1,len(data)), np.abs(np.fft.fftshift(np.fft.fft(data))))

        filtered = bandpass(data, 0.01, 0.15, 1, 512)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        filtered = bandpass(data, 0.02, 0.10, 1, 256)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        plt.title('Testing Bandpass Filters')
        plt.legend(['Original','0.1-0.15 Hz, 512 taps', '0.2-0.10 Hz, 256 taps'])
        plt.xlabel('Frequency (Hz)')
        plt.savefig('img/bandpass_test.png')
        if show:
            plt.show()

    #filtered = median_filter(data, 9)
    #filtered = bandpass(data, 0.195, 0.22, 1, 256)
    #filtered = bandpass(data, 0.195, 0.21, 1, 256)

    #filtered = bandpass(data, 0.195, 0.215, 1, 256)
    filtered = bandpass(data, 0.04, 0.06, 1, 256)

    plt.figure()
    plt.plot(np.linspace(-1,1,len(filtered)),
        np.abs(np.fft.fftshift(np.fft.fft(filtered))))
    plt.show()
    return filtered


def fit(data, debug=False):
    lst = np.load('data/report/3C144_03-28-2014_001926.npz')['lst'][START:END]
    data = data[(len(data)-len(lst))/2:-(len(data)-len(lst))/2]
    generate_plot(data, dft=True)

    source = ephem.FixedBody()
    obs = ephem.Observer()
    obs.lat = ephem.degrees(37.8732)
    obs.long = ephem.degrees(-122.2573)
    obs.date = ephem.date('2014/3/28 0:19:26')

    source._ra    = ephem.hours("5:34:31.95")
    source._dec   = ephem.degrees("22:00:51.1")
    source._epoch = ephem.J2000
    source.compute(obs)

    hs = np.array(ephem.hours(source.ra) - lst)

    # 'Brute force' least squares
    lmbda = 2.5 # Wavelength lambda in cm
    YBAR = []
    By = np.linspace(700,1200,500)
    A = []
    S_SQ = []

    # Iterate over baseline lengths in cm
    for d_B in By:
        C = 2 * np.pi * d_B/lmbda * np.cos(source.dec)
        x1 = np.cos( C * np.sin(hs) )
        x2 = np.sin( C * np.sin(hs) )
        X = np.matrix([x1, x2])

        ybar, a, s_sq = least_squares(data, X)
        S_SQ.append(s_sq)
        ybar = np.array(ybar[:,0])
        YBAR.append(ybar)
        A.append(a)

    Bopt = np.argmin(S_SQ)
    if debug:
        plt.figure()
        plt.plot(By, S_SQ/max(S_SQ)*100)
        plt.title('Normalized Residuals')
        #plt.show()

        plt.figure()
        plt.plot(data)
        plt.plot(YBAR[Bopt])
        plt.title('YBAR')
        print By[Bopt]
        print A[Bopt]
    return YBAR[Bopt], A[Bopt], By[Bopt]

if __name__ == "__main__":
    main()
