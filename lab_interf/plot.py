#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse

def make_title(raw):
    """Make a plot title from an input string

    Args:
    raw (string): a title or file name of the format
    .../[source]_MM-DD-YYYY_HH:MM:SS[_optional subtitle].npz

    Returns:
    title (string):
        for a filename of the specified format-
        [Source] on MM-DD-YYYY beginning at HH:MM:SS\n[optional subtitle]
        otherwise return str(raw)
    """
    title_args = raw.split('_')
    if len(title_args) == 3:
        source, date, time = title_args
        source = source.split('/')[-1].capitalize()
        time = time.split('.')[0]
        time = '%s:%s:%s' % (time[0:2], time[2:4], time[4:6])
        return '%s on %s beginning at %s' % (source, date, time)
    elif len(title_args) == 4:
        source, date, time, subtitle = title_args
        source = source.split('/')[-1].capitalize()
        time = '%s:%s:%s' % (time[0:2], time[2:4], time[4:6])
        subtitle = subtitle.split('.')[0].capitalize()
        return '%s on %s beginning at %s\n%s' % (source, date, time, subtitle)
    else:
        return str(raw)

def generate_plot(data, dft=False, show=False, title=None, outfile=None):
    """Generate a plot of some dataset. Optionally shows the plot, its DFT, adds
    a title, and/or saves it as a .png file.

    Args:
    data (np.array): data to plot
    dft (boolean, optional): plot the DFT in a subplot, default false
    show (boolean, optiona) show the plot, default false
    title (string, optional): plot title
    dft (string, optional): output file name to be saved as [outfile].png
    """
    if dft:
        plt.figure()
        plt.subplot(211)
        plt.plot(data)
        plt.xlabel('Time (s)')

        if title:
            plt.title(title)

        plt.subplot(212)
        frange = np.linspace(-1,1,len(data))
        plt.plot(frange, np.abs( np.fft.fftshift( np.fft.fft(data) ) ) )
        plt.xlabel('Frequency (Hz)')

    else:
        plt.plot( data )
        if title:
            plt.title(title)

    if outfile:
        plt.savefig(outfile+'.png')

    if show:
        plt.show()


def main():
    """Plots data from an npz file with the filename
    [source]_DD-MM-YY_HHMMSS[_optional subtitle].npz.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plot a file in a numpy readable format.')
    parser.add_argument('infile', help='file name of data to plot')
    parser.add_argument('key', default='volts', help='key to plot from npz files')
    parser.add_argument('--outfile', default=None, help='save output as [outfile].png')
    parser.add_argument('-f', '--fourier', action='store_true',
            default=False,help='plot fft of data')
    #parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print  voltage measurements')
    args = parser.parse_args()

    try:
        data = np.load(args.infile)[args.key]
        generate_plot(data, show=True, dft=args.fourier,
                title=make_title(args.infile), outfile=args.outfile)

    except Exception, e:
       print('could not load '+args.filename)


if __name__ == "__main__":
    main()
