import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from pylab import *

# Demonstration of central limit theorem using a poisson distribution

# Plot the original distribution
figure(1)
clf()
plt.hist(np.random.poisson(5,10000))
title('Poisson Distribution')
savefig('poisson.png')
plt.show()

# Plot 10000 sample averages 
averages = [np.average(np.random.poisson(5,10000)) for i in range(10000)]
plt.hist(averages)
title('Sample Averages from Poisson Distribution')
savefig('normal.png')
plt.show()


