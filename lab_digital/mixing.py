import numpy as np
import matplotlib.pyplot as plt
sampling = np.load('analog_mixing.npz')

plt.figure(1)
v_lo = 100e6
f = np.linspace(-v_lo, v_lo, 1024)
plt.plot(f, np.abs(np.fft.fftshift(np.fft.fft(sampling['arr_0']))))
plt.plot(f, np.abs(np.fft.fftshift(np.fft.fft(sampling['arr_1']))))
plt.title('$v_{lo}$ = 1 MHz')
plt.xlabel('frequency [MHz]')
plt.legend( ('$v_{sig}$ = 1.05 MHz', '$v_{sig}$ = 0.95 MHz'))
  
plt.savefig('analog_mix.png')

plt.figure(2)
Y = np.abs(np.fft.fft(sampling['arr_0']))
Y[300:700]=0
plt.plot(Y)
plt.title('$v_{lo}$ = 1 MHz')
plt.xlabel('frequency [MHz]')
  
plt.savefig('analog_filtered_f.png')

plt.figure(3)
plt.subplot(211)
plt.plot(sampling['arr_0'])

plt.subplot(212)
plt.plot(np.real(np.fft.ifft(Y)))
plt.savefig('analog_filtered_t.png')

plt.figure(4)
d_hi = np.fromfile('mix_bram_high_sig_5pct_12dbm', dtype='>i4')
d_lo = np.fromfile('mix_bram_low_sig_5pct_12dbm', dtype='>i4')
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(d_hi))))
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(d_lo))))
plt.legend( ('$v_{sig}$ = 1.05 MHz', '$v_{sig}$ = 0.95 MHz'))
plt.axis((950,1050,0,4e4))
plt.savefig('digital_mix.png')

plt.figure(5)
sine = np.fromfile('sin_bram', dtype='>i4')
cosine = np.fromfile('cos_bram', dtype='>i4')
ssb = cosine + 1j*sine

plt.title('Digital Mixing SSB')
plt.subplot(2,1,1)
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(cosine))))
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(sine))))
plt.subplot(2,1,2)
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(ssb))))
plt.legend( ('cosine', 'sine'))
plt.savefig('ssb.png')

plt.figure(6)
Y = np.fft.fft(ssb)
#Y[8:] = 0
#plt.plot(np.real(np.fft.ifft(Y)))
plt.subplot(311)
plt.plot(np.abs(Y))
plt.subplot(312)
plt.plot(np.abs(Y))
x1,x2,y1,y2 = plt.axis()
plt.axis((0,50,y1,y2))
# take max and real of ifft
Y[20:] = 0
Y[15:] = 0
plt.subplot(313)
plt.plot(np.fft.ifft(Y))
plt.savefig('digital_filtered.png')
