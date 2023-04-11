import matplotlib.pyplot as plt
import plot
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit

data_folder = "../Daten/Konzentration"
types = ["Reflexion", "Transmission"]

pic_folder = "../Bilder/Auswertung/41"

plot.parameters(True, 30, (16,8), 100, colorblind = False)

'''
# Verlauf ===========================================================================================================================================

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
# Fitguete ==========================================================================================================================================

# define the true objective function
def linear(x, a, b):
    return a * x + b

def hyperbel(x, a, b):
    return a/(x**(1/8)) + b 

x_max, x_min = [40, 60],   [0, 0]   
y_max, y_min = [0.02, 0.45], [0.004, 0.05]

for t, ymax, ymin, xmax, xmin in zip(types, y_max, y_min, x_max, x_min):
    directories = os.listdir(data_folder + "/" + t)
    directories = sorted(directories, key=lambda x: int(x[:-5]))[::-1]
    
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
    
    df_stat = pd.DataFrame(list(zip(thickness_mean, thickness_std, homogen, fitness_mean)))
    
    yscale = 100
    x1, x2, y = df_stat[0], df_stat[1], df_stat[3]*yscale
    
    if t == types[0]:
        objective = linear
        label_fit1 = r"$\langle \chi \rangle = a \, \langle d \rangle  + b$"
        label_fit2 = r"$\langle \chi \rangle = a \, std(d)  + b$"
        
    if t == types[1]:
        objective = hyperbel
        label_fit1 = r"$\langle \chi \rangle = a/\langle d \rangle^2 + b$"
        label_fit2 = r"$\langle \chi \rangle = a/std(d)^2 + b$"
        
    popt1, var1 = curve_fit(objective, x1, y)
    popt2, var2 = curve_fit(objective, x2, y)
    
    a1, b1 = popt1
    a2, b2 = popt2
    
    print(a1, b1)
    print(a2, b2)
     
    xline1 = np.arange(0, 4000, 1)
    xline2 = np.arange(xmin, xmax, 1)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16, 7), sharey=True)
    axes = [ax1, ax2]
    
    ax1.plot(x1, y, "o", label = "Messwerte", color = "red")
    ax1.plot(xline1, objective(xline1, a1 , b1), label = label_fit1, linewidth = 3, color = "orange")
    
    ax1.set_xlabel(r"$\langle d \rangle$ in nm")
    ax1.set_ylabel(r"$\langle \chi \rangle \cdot 10 ^2$" )
    ax1.yaxis.set_label_coords(-0.15,0.5)
    
    ax1.set_xlim(0, 4000)
    ax1.set_ylim(ymin*yscale, ymax*yscale)
    
    ax2.plot(x2, y, "o", color = "red")
    ax2.plot(xline2, objective(xline2, a2 , b2), label = label_fit2, linewidth = 3)
    
    ax2.set_xlabel(r"$std(d)$ in nm")
    
    ax2.set_xlim(xmin, xmax)
    
    handles_labels = [ax.get_legend_handles_labels() for ax in axes]
    handles, labels = [sum(lol, []) for lol in zip(*handles_labels)]
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False, columnspacing=1, handlelength=1)
    
    plt.subplots_adjust(wspace=0.1)
    
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=8, frameon=False, columnspacing=0.5, handlelength=0.5)
    plt.savefig(pic_folder + "/" + t + "-Fitguete.pdf", bbox_inches='tight')
    #plt.show()
    plt.clf()
#'''