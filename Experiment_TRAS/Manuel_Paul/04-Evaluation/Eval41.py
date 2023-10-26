import matplotlib.pyplot as plt
import plot
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit

data = "/Users/paul/Documents/GitHub/PPD/Experment_TRAS/Data/Absorption/absorption.csv"

pic_folder = "/Users/paul/Documents/GitHub/PPD/Experment_TRAS/Pictures/Evaluation/41"

df = pd.read_csv(data)

#df = df.iloc[1:]
#df13 = df13[["benzonitrile-13","Unnamed: 3"]]

print(df.head())

plot.parameters(True, 30, (14,8), 100, colorblind = False)

fig, ax = plt.subplots(1,1)

#ax.plot(df["(benzonitrile)Wavelength (nm)"], df["(benzonitrile)Abs"], label="Benzonitril")
ax.plot(df["(benzonitrile-13)Wavelength (nm)"], df["(benzonitrile-13)Abs"], label="ZnTPP in Benzonitril")
#ax.plot(df["(Toluene)Wavelength (nm)"], df["(Toluene)Abs"], label="Toluene")
#ax.plot(df["(Toluene-14)Wavelength (nm)"], df["(Toluene-14)Abs"], label="ZnTPP in Toluene")
ax.plot(df["(benzonitrile-15)Wavelength (nm)"], df["(benzonitrile-15)Abs"], label="ZnOEP in Benzonitril")

#ax.set_xlim(-0.02, 0.25)
#ax.set_ylim(1050, 1550)

ax.set_xlabel(r"Wavelength (nm)")
#ax.xaxis.set_label_coords(0.5, -0.15)

ax.set_ylabel(r"Abs.")
#ax.yaxis.set_label_coords(- 0.1, 0.5)

#ax.set_xticks(np.arange(-800,-299,50)*(-1))
#ax.set_yticks(np.arange(0,6E-5,1E-5))

ax.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.2), frameon=False, ncol=2)

#plt.show()
#plt.savefig(pic_folder + "/" + "ZnTpp-ZnOEP-in-Bn.pdf", bbox_inches = "tight")