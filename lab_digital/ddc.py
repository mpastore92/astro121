import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
ideal = np.array([1,1,1,0,0,0,1,1])
lpf = np.append(np.fft.ifft(ideal), np.zeros(56))
plt.plot(np.fft.fftshift(np.fft.fft(lpf)))

plt.savefig('ideal_lpf.png')

plt.figure(2)
coeff = np.array([0.149618, 0.318029, 0.474492, 0.585097, 0.625, 0.585097, 0.474492, 0.318029])
coeff = np.append(coeff, np.zeros(56))


plt.subplot(211)
plt.plot(coeff)

plt.subplot(212)
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(coeff))))
plt.savefig('lpf.png')

ideal_64 = np.append(np.zeros(16), np.ones(32), np.zeros(16))
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(coeff))))
plt.savefig('lpf.png')

np.linspace




plt.figure(1)
ddc_imag = np.array([np.fromfile('ddc_imag_bram_'+str(n), dtype='>i4') for n in range(10,70,10)])
ddc_real = np.array([np.fromfile('ddc_imag_bram_'+str(n), dtype='>i4') for n in range(10,70,10)]) 

for n in range(len(ddc_real)):
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(ddc_real[n] + 1j*ddc_imag[n]))))
plt.savefig('ddc.png')

plt.figure(2)
plt.subplot(211)
ddc_50 = np.fromfile(('ddc_real_bram_50'), dtype='>i4')
plt.plot(ddc_50)

plt.subplot(212)
ddc_60 = np.fromfile(('ddc_real_bram_60'), dtype='>i4')
plt.plot(ddc_60)

plt.savefig('ddc_time.png')

