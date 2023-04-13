import matplotlib.pyplot as plt
import plot
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit

data_folder = "../Daten/Rotation"
types = ["Reflexion", "Transmission"]

pic_folder = "../Bilder/Auswertung/42"

plot.parameters(True, 30, (16,8), 100, colorblind = False)

'''
# Verlauf =========================================================================================

x_max, x_min = [800, 480],   [400, 340]   
y_max, y_min = [0.085, 0.98], [0.035, 0.88]

for t, ymax, ymin, xmax, xmin in zip(types, y_max, y_min, x_max, x_min):
    directories = os.listdir(data_folder + "/" + t)
    directories = sorted(directories, key=lambda x: int(x[:-5]))
    
    for d in directories:
        data = [x for x in os.listdir(data_folder + "/" + t + "/" + d) if "Fit" not in x]
        fit  = [x for x in os.listdir(data_folder + "/" + t + "/" + d) if "Fit" in x and "csv" in x]
        
        wavelength, wavelength_max, wavelength_min = [], 1000, 0 
        amplitude  = []
        
        # Get max/min value of wavelength
        for i in data:
            df = pd.read_csv(data_folder + "/" + t + "/" + d + "/" + i, header=None)
            
            if wavelength_min < min(df[0]):
                wavelength_min = min(df[0])
            
            if wavelength_max > max(df[0]):
                wavelength_max = max(df[0])
                
        for i in data:
            df = pd.read_csv(data_folder + "/" + t + "/" + d + "/" + i, header=None)
            
            #print(min(df[0]), max(df[0]), len(df[0].values[df.index[df[0] == wavelength_min][0]: df.index[df[0] == wavelength_max][0]]))
            
            wavelength.append(df[0].values[df.index[df[0] == wavelength_min][0]: df.index[df[0] == wavelength_max][0]])
            amplitude.append(df[1].values[df.index[df[0] == wavelength_min][0]: df.index[df[0] == wavelength_max][0]])
        
        wavelength_mean = np.mean(wavelength, 0)
        amplitude_mean  = np.mean(amplitude, 0)
        
        plt.plot(wavelength_mean, amplitude_mean, linewidth = 4, label = d.replace("mg_ml", r" $\frac{\mathrm{mg}}{\mathrm{mL}}$"))
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.xlabel(r"$\lambda$ in nm")
    plt.ylabel(r"Intensität in Prozent")
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=8, frameon=False, columnspacing=0.5, handlelength=0.5)
    plt.savefig(pic_folder + "/" + t + ".pdf", bbox_inches='tight')
    #plt.show()
    plt.clf()
#'''

#'''
# Fit =============================================================================================

# define the true objective function
def schubert(x, a):
    const = 47.654
    return (a * const)/(x**(1/2))

fig, (ax1, ax2) = plt.subplots(2,1, figsize = (16, 16), sharex=True)
axes = [ax1, ax2]

for t, ax in zip(types, axes):
    directories = os.listdir(data_folder + "/" + t)
    directories = sorted(directories, key=lambda x: int(x[:-3]))
    
    print(directories)
    
    rpm = [int(x.replace("rpm", "")) for x in directories]
    
    thickness_mean, thickness_std, homogen, fitness_mean = [], [], [], []
    
    for d in directories:
        fit  = [x for x in os.listdir(data_folder + "/" + t + "/" + d) if "Fit" in x and "csv" in x]        
        
        for i in fit:
            df = pd.read_csv(data_folder + "/" + t + "/" + d + "/" + i, header=None)
            
            thickness_mean.append(df[0].mean().round(1))
            thickness_std.append(df[0].std().round(1))
            
            qhomogen = df[0].mean().round(1)/df[0].std().round(1)
            homogen.append(qhomogen.round(1))
            
            fitness_mean.append(df[1].mean().round(6))      
    
    df_stat = pd.DataFrame(list(zip(rpm, thickness_mean, thickness_std, homogen)))
    
    
    print(df_stat)

    yscale = 100
    x, y = df_stat[0], df_stat[1]
    
    objective = schubert
    label_fit = r"$\langle d \rangle = A \cdot C \cdot \omega^{-1/2}$"
        
    popt, var = curve_fit(objective, x, y)
    
    a = popt
    print(a)
     
    xline = np.arange(0, 10000, 1)
    
    ax.plot(x, y, "o", label = "Messwerte", color = "red")
    ax.plot(xline, objective(xline, a), label = label_fit, linewidth = 3, color = "orange")

    ax.set_ylabel(r"$\langle d \rangle$ in nm")
    ax.yaxis.set_label_coords(-0.1,0.5)
    
    ax.set_xlim(0, 10000)
    ax.set_ylim(0,  5000)
    
    
ax1.text(1.01, 0.94, r'\bf{(a)}', transform=ax1.transAxes)
ax2.text(1.01, 0.94, r'\bf{(b)}', transform=ax2.transAxes)

ax2.set_xlabel(r"$\omega$ in rpm" )
plt.subplots_adjust(hspace=0.1)
    
handles_labels = [ax1.get_legend_handles_labels()]
handles, labels = [sum(lol, []) for lol in zip(*handles_labels)]

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False, columnspacing=1, handlelength=1)

plt.savefig(pic_folder + "/" + "Rotation.pdf", bbox_inches='tight')

#plt.show()

#'''