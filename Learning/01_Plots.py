import matplotlib.pyplot as plt
import seaborn as snn
from numpy.random import rand
from numpy import linspace

x=linspace(0,10,50)
y=rand(50,1)

plt.plot(x,y)