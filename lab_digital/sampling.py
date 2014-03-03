import numpy as np
import matplotlib.pyplot as plt
sampling = np.load('sampling/lab_digital_sampling.npz')

#Time
plt.figure(1)
plt.clf()

plt.subplot(221) 
plt.plot(sampling['arr_0'][:128])
plt.title('Frequency = 0.88 MHz')
  
plt.subplot(222) 
plt.plot(sampling['arr_1'][:128])
plt.title('Frequency = 3.52 MHz')
  
plt.subplot(223) 
plt.plot(sampling['arr_2'][:128])
plt.title('Frequency = 6.16 MHz')
  
plt.subplot(224) 
plt.plot(sampling['arr_3'][:128])
plt.title('Frequency = 7.92 MHz')

plt.savefig('sampling_time.png')
  
plt.figure(2)

plt.subplot(221) 
plt.plot(sampling['arr_4'][:128])
plt.title('Frequency = 8.8 MHz')
  
plt.subplot(222) 
plt.plot(sampling['arr_5'][:128])
plt.title('Frequency = 26.4 MHz')

plt.savefig('sampling_time_ext.png')

#Frequency
plt.figure(3)
v_lo = 1e6
f = np.linspace(v_lo-0.5e6, v_lo+0.5e6, 1024)

plt.subplot(221) 
plt.plot(f/1e6, np.abs(np.fft.fftshift(np.fft.fft(sampling['arr_0']))))
plt.title('Frequency = 0.88 MHz')
  
plt.subplot(222) 
plt.plot(f/1e6, np.abs(np.fft.fftshift(np.fft.fft(sampling['arr_1']))))
plt.title('Frequency = 3.52 MHz')
  
plt.subplot(223) 
plt.plot(f/1e6, np.abs(np.fft.fftshift(np.fft.fft(sampling['arr_2']))))
plt.title('Frequency = 6.16 MHz')
  
plt.subplot(224) 
plt.plot(f/1e6, np.abs(np.fft.fftshift(np.fft.fft(sampling['arr_3']))))
plt.title('Frequency = 7.92 MHz')

plt.savefig('sampling_f.png')
  
plt.figure(4)
plt.plot(f/1e6, np.abs(np.fft.fftshift(np.fft.fft(sampling['arr_5']))))
plt.title('Frequency = 26.4 MHz')

plt.savefig('sampling_f_ext.png')

#Window
plt.figure(5)
omega = np.linspace(-512, 512, 1024*4)
plt.plot(omega, np.abs(np.fft.fftshift(np.fft.fft(np.append(sampling['arr_5'], np.zeros(3*1024))))))
x1,x2,y1,y2 = plt.axis()
plt.axis((x1, x2, 0, 15))
plt.title('Spectral Leakage')
plt.savefig('window.png')
