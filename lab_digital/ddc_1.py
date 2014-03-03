import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
ddc_imag = np.array([np.fromfile('ddc_imag_bram_'+str(n), dtype='>i4') for n in range(10,70,10)])
ddc_real = np.array([np.fromfile('ddc_imag_bram_'+str(n), dtype='>i4') for n in range(10,70,10)]) 

for n in range(len(ddc_real)):
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(ddc_real[n] + 1j*ddc_imag[n]))))
plt.savefig('ddc.png')

plt.figure(2)
plt.subplot(211)
ddc_50 = np.fromfile(('ddc_imag_bram_50'), dtype='>i4')
plt.subplot(211)
ddc_60 = np.fromfile(('ddc_imag_bram_60'), dtype='>i4')
plt.savefig('ddc_time.png')

