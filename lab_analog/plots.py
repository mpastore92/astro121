from scipy import *
from pylab import *

# Define constants
f=arange(1,1e8,10)
s=2.0j*pi*f

p=2.0*pi*10

def make_plot(tf, t="Bode Plot", fig=1, filename='bode_plot', show_output=False):
    figure(fig)
    clf()

    subplot(211)
    title(t)
    # Plot Magnitude
    semilogx(f,20*log10(abs(tf)))
    ylabel('Magnitude (dB)')

    # Plot Magnitude
    subplot(212)
    semilogx(f,arctan2(imag(tf),real(tf))*180.0/pi)
    ylabel('Phase (deg.)')
    xlabel('Freq (Hz)')
    savefig(filename+'.png')
    if show_output:
        show() 

# Generate bode plot of RC low pass filter f_c = 100 kHz
R = 160
C = 1e-8
tf = 1/(1+ R*C*s)
make_plot(tf, t="RC Low Pass Filter", filename='rc_low_pass')

# Generate bode plot of RC high pass filter f_c = 100 kHz
tf = R*C*s/(1+ R*C*s)
make_plot(tf, t="RC High Pass Filter", filename='rc_high_pass')

# Generate bode plot of LC band reject filter f_c = 1MHz
C = 1e-6
L = 1e-6 
tf = (L*C*(s**2)+1.0j)/(1.0j*s*C)
make_plot(tf, t="Series LC Band Reject Filter", filename='series_lc')

# Generate bode plot of LC band pass filter f_c = 1MHz
tf = (-L*s)/(-L*C*(s**2)-1)
make_plot(tf, t="Parallel LC Band Pass Filter", filename='parallel_lc')

# IV Characteristics of a diode
i = np.array([0.045, 0.08, 0.185, 0.264, 0.350, 0.445])
v = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])
figure(1)
clf()
plt.plot(v, i)
ylabel('Current (mA)')
xlabel('Voltage (V)')
title('IV Characteristics of a Silicon Diode')
savefig('diode_iv.png')
show()

