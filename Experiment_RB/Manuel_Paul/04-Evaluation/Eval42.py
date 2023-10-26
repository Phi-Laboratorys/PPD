import matplotlib.pyplot as plt
import plot
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

data = "/Users/paul/Documents/GitHub/PPD/Experment_RB/Data/XRD/HG1_NaCl_01_55s_4523.dat"
pic_folder = "/Users/paul/Documents/GitHub/PPD/Experment_RB/Pictures/Evaluation/42"

df = pd.read_csv(data, sep="\t")
#print(df.head())

plot.parameters(True, 30, (14,8), 100, colorblind = False)

fig, ax = plt.subplots(1,1)

ax.plot(df)

#ax.set_xlim(-0.02, 0.25)
#ax.set_ylim(1050, 1550)

ax.set_xlabel(r"$2\Theta$")
#ax.xaxis.set_label_coords(0.5, -0.15)

ax.set_ylabel(r"I")
#ax.yaxis.set_label_coords(- 0.1, 0.5)

#ax.set_xticks(np.arange(-800,-299,50)*(-1))
#ax.set_yticks(np.arange(0,6E-5,1E-5))

#ax.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.2), frameon=False, ncol=2)

#plt.show()
#plt.savefig(pic_folder + "/" + "IntDistNaCl.pdf", bbox_inches = "tight")

#print(df.values.ravel())
peakList=find_peaks(df.values.ravel(), threshold=5)
print(df.iloc[peakList[0]])