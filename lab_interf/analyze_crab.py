#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ephem
from scipy import signal

from processing import *
from plot import generate_plot

def main():
    """

    """
    START = 2000
    END = 8000

    chunks = False
    baseline = True
    dec = True
    show = True
    base_title = "3C144 on 03-28-2014 beginning at 00:19:26\n"
    D = True
    S = False

    data = np.load('data/report/3C144_03-28-2014_001926.npz')['volts'][START:END]

    if chunks:
        data = np.load('data/report/3C144_03-28-2014_001926.npz')['volts']
        normalized_chunks = normalize_chunks(data, 1000)
        filtered_chunks = filter_baseline(normalized_chunks, debug=D)
        generate_plot(normalized, dft=True)
       #fit(normalized_chunks)

    if baseline:
       START = 0
       END = 14400
       data = np.load('data/report/3C144_03-28-2014_001926.npz')['volts'][START:END]
       generate_plot(data-np.average(data), dft=True, title=base_title, outfile="img/crab/raw")
       filtered = filter_baseline(data, debug=D)
       normalized = normalize(filtered)
       generate_plot(normalized, dft=True, title=base_title+"Filtered and Normalized", outfile="img/crab/baseline_filtered")
       fit_baseline(normalized, START, END, debug=D)

       START = 2000
       END = 8000
       data = np.load('data/report/3C144_03-28-2014_001926.npz')['volts'][START:END]
       generate_plot(data-np.average(data), dft=True, title=base_title+"Segment of Original", outfile="img/crab/seg")
       filtered = filter_dec(data, debug=False)
       generate_plot(normalized, dft=True, title=base_title+"Filtered and Normalized\nSegment", outfile="img/crab/baseline_filtered")
       normalized = normalize(filtered)
       YBAR, A, By, S_SQ = fit_baseline(normalized, START, END, debug=False)
       plt.figure()
       plt.plot(normalized)
       plt.plot(YBAR)
       plt.title('Least Squares Fit for Baseline\nSegment')
       plt.legend(['Filtered', 'Fit'])
       plt.savefig('img/crab/fit_baseline_seg.png')

       plt.figure()
       plt.plot(By, S_SQ/max(S_SQ))
       plt.title('Normalized Residuals for Baseline Fit\nSegment')
       plt.savefig('img/crab/residual_baseline_seg.png')

    if dec:
       START = 2000
       END = 8000
       data = np.load('data/report/3C144_03-28-2014_001926.npz')['volts'][START:END]
       filtered = filter_dec(data, debug=D)
       normalized = normalize(filtered)
       generate_plot(normalized, dft=True, title=base_title+"Filtered and Normalized", outfile="img/crab/dec_filtered")
       fit_dec(normalized, START, END, debug=D)

    if S:
        plt.show()

    #point_source()

def filter_baseline(data, show=False, debug=False):
    # Remove DC
    data -= np.average(data)

    if debug:
        # Test filters
        plt.figure()
        filtered = bandpass(data, 0.05, 0.5, 1, 256)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        filtered = bandpass(data, 0.1, 0.25, 1, 256)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        filtered = bandpass(data, 0.195, 0.21, 1, 256)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        plt.title('Testing Bandpass Filters')
        plt.legend(['Original','0.05-0.5 Hz, 256 tap', '0.1-0.25 Hz, 256 tap', '0.195-0.21 Hz, 256 tap'])
        plt.xlabel('Frequency (Hz)')
        plt.savefig('img/crab/bandpass_test_b.png')
        if show:
            plt.show()

    filtered = bandpass(data, 0.195, 0.215, 1, 256)
    return filtered

def filter_dec(data, show=False, debug=False):
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
        plt.savefig('img/crab/median_test.png')

        plt.figure()
        plt.plot(np.linspace(-1,1,len(data)), np.abs(np.fft.fftshift(np.fft.fft(data))))

        filtered = bandpass(data, 0.01, 0.15, 1, 512)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        filtered = bandpass(data, 0.02, 0.10, 1, 256)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        filtered = bandpass(data, 0.04, 0.06, 1, 256)
        plt.plot(np.linspace(-1,1,len(filtered)),
            np.abs(np.fft.fftshift(np.fft.fft(filtered))))

        plt.title('Testing Bandpass Filters')
        plt.legend(['Original','0.01-0.15 Hz, 512 taps', '0.2-0.10 Hz, 256 taps', '0.04-0.06 Hz, 256 taps'])
        plt.xlabel('Frequency (Hz)')
        plt.savefig('img/crab/bandpass_test_d.png')
        if show:
            plt.show()


    filtered = median_filter(data, 9)
    filtered = bandpass(filtered, 0.04, 0.06, 1, 256)
    return filtered

def fit_dec(data, start, end, debug=False):
    lst = np.load('data/report/3C144_03-28-2014_001926.npz')['lst'][start:end]
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
    By = 1000
    cos_dec = np.linspace(0.01,1,500)
    A = []
    S_SQ = []

    # Iterate over baseline lengths in cm
    for dec in cos_dec:
        C = 2 * np.pi * By/lmbda * dec
        x1 = np.cos( C * np.sin(hs) )
        x2 = np.sin( C * np.sin(hs) )
        X = np.matrix([x1, x2])

        ybar, a, s_sq = least_squares(data, X)
        S_SQ.append(s_sq)
        ybar = np.array(ybar[:,0])
        YBAR.append(ybar)
        A.append(a)

    Dopt = np.argmin(S_SQ)
    if debug:
        plt.figure()
        plt.plot(cos_dec, S_SQ/max(S_SQ))
        plt.title('Normalized Residuals for Declination Fit')
        plt.savefig('img/crab/residual_dec.png')

        plt.figure()
        plt.plot(data)
        plt.plot(YBAR[Dopt])
        plt.title('Least Squares Fit for Declination')
        plt.legend(['Filtered', 'Fit'])
        plt.savefig('img/crab/fit_dec.png')
        print 'By = '+str(By)
        print 'delta (deg)= '+str(np.arccos(cos_dec[Dopt])*180/np.pi)
        print 'A = '+str(A[Dopt])
    return YBAR[Dopt], A[Dopt], np.arccos(cos_dec[Dopt])*180/np.pi

def fit_baseline(data, start, end, debug=False):
    lst = np.load('data/report/3C144_03-28-2014_001926.npz')['lst'][start:end]
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
        plt.plot(By, S_SQ/max(S_SQ))
        plt.title('Normalized Residuals for Baseline Fit')
        plt.savefig('img/crab/residual_baseline.png')

        plt.figure()
        plt.plot(data)
        plt.plot(YBAR[Bopt])
        plt.title('Least Squares Fit for Baseline')
        plt.legend(['Filtered', 'Fit'])
        plt.savefig('img/crab/fit_baseline.png')
        print 'By = '+str(By[Bopt])
        print 'A = '+str(A[Bopt])

    return YBAR[Bopt], A[Bopt], By, S_SQ

if __name__ == "__main__":
    main()
