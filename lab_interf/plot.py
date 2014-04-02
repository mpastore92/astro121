#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    """Plots data logs/sun_DD-MM-YY_HHMMSS.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plot a file in a numpy readable format.')
    parser.add_argument('filename', help='file name of data to plot')
    parser.add_argument('key', default='volts', help='key to plot from npz files')
    parser.add_argument('-f', '--fft', action='store_true',
            default=False,help='plot fft of data')

    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print  voltage measurements')
    args = parser.parse_args()

    try:
        data = np.load(args.filename)
        if args.fft:
            plt.figure()
            plt.subplot(211)
            plt.plot( data[args.key] )
            plt.subplot(212)
            plt.plot( np.abs( np.fft.fftshift( np.fft.fft(data[args.key]) ) ) )
            plt.show()
        else:
            plt.plot( data[args.key] )

        plt.show()

    except Exception, e:
       print('could not load '+args.filename)


if __name__ == "__main__":
    main()
