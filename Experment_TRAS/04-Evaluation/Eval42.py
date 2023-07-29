import matplotlib.pyplot as plt
import plot
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit

# FUNCTIONS ===================================================================

def exponential(x, a, b, t):
    return a * np.exp(- b * x) + t

def linear(x, m, t):
    return m*x + t

# VARIABLES ===================================================================

data_folder = "../Data/TRAS"

pic_folder = "../Pictures/Evaluation/42"

'''
# LIFETIME ====================================================================

data_files = [x for x in os.listdir(data_folder) if ".csv" in x]
data_files = [x for x in data_files if "sample" in x]
data_files = sorted(data_files, key=lambda x: (int(x[6]), int(x[7])))

data_ZnTPP_BN = data_files[0:4]
num_ZnTPP_BN = [1, 2, 3, 4]
label_ZnTPP_BN = [r"ZnTPP in BN 0.8\,mM", r"ZnTPP in BN 0.6\,mM", r"ZnTPP in BN 0.4\,mM", r"ZnTPP in BN 0.2\,mM"]

data_ZnTPP_C70 = data_files[4:7]
num_ZnTPP_C70 = [5, 6, 7]
label_ZnTPP_C70 = [r"ZnTPP:C70 in BN 1:0.1", r"ZnTPP:C70 in BN 1:0.2", r"ZnTPP:C70 in BN 1:0.3"]

data_ZnTPP_TL = [data_files[7]]
num_ZnTPP_TL = [8]
label_ZnTPP_TL = [r"ZnTPP in Tol 0.8\,mM"]

data_P3HT_TL = [data_files[8]]
num_P3HT_TL = [9]
label_P3HT_TL = [r"P3HT in Tol 1.5\,mM"]

data_ZnOEP_BN = [data_files[9]]
num_ZnOEP_BN = [12]
label_ZnOEP_BN = [r"ZnOEP in BN 0.8\,mM"]

compare = [[data_ZnTPP_BN, [data_ZnTPP_BN[0]] + data_ZnTPP_C70], 
           [[data_ZnTPP_BN[0]] + data_ZnTPP_TL, [data_ZnTPP_BN[0]] + data_ZnOEP_BN]]

number = [[num_ZnTPP_BN, [num_ZnTPP_BN[0]] + num_ZnTPP_C70], 
          [[num_ZnTPP_BN[0]] + num_ZnTPP_TL, [num_ZnTPP_BN[0]] + num_ZnOEP_BN]]

label =  [[label_ZnTPP_BN, [label_ZnTPP_BN[0]] + label_ZnTPP_C70], 
          [[label_ZnTPP_BN[0]] + label_ZnTPP_TL, [label_ZnTPP_BN[0]] + label_ZnOEP_BN]]

text = [[r"\bf{(a)}", r"\bf{(b)}"],
        [r"\bf{(c)}", r"\bf{(d)}"]]

plot.parameters(True, 25, (16,8), 100, colorblind = False)

fig, ax = plt.subplots(2, 2, figsize =(20,12), sharex=True, sharey=True)

xscale = 1000
data_scale = 1.364098374478068

sample, typ, kappa, s_kappa, tau, s_tau = [], [], [], [], [], []
head = ["Sample No.", "Treatment", "k[ms-1]", "s_k[ms-1]", "tau[ns]", "s_tau[ns]"]

for comp_layer, num_layer, ax_layer, label_layer, text_layer in zip(compare, number, ax, label, text):
    for comp, num, axis, lab, t in zip(comp_layer, num_layer, ax_layer, label_layer, text_layer):
        for data, n, l in zip(comp, num, lab):
        
            df = pd.read_csv(data_folder + "/" + data,sep=";")
            x,y = df[df.keys()[0]]/xscale, df[df.keys()[1]]
            
            if n == 7 or n == 12:
                y = y * data_scale

            ymax_idx = y.idxmax()
            x_fit, x_fit_start = x[ymax_idx:], x[ymax_idx]
            y_fit = y[ymax_idx:]
        
            xline = np.linspace(min(x_fit), max(x_fit), 100)
        
            popt, pcov = curve_fit(exponential, x_fit, y_fit)
            
            k0 = (popt[1] * xscale).round(2)
            s_k0 = ((np.sqrt(np.diag(pcov))[1])*xscale).round(2)
            
            ta  = (1/popt[1] * xscale).round(2)
            s_ta = ((np.sqrt(np.diag(pcov))[1]/popt[1]**2)*xscale).round(2)
        
            #print(str(n) + " & " + l + " & " + str(k0) + " $\pm$ " + str(s_k0) + " & " + str(ta) + " $\pm$ " + str(s_ta))
        
            axis.plot(x*xscale, y, color = "black", linewidth = 3)
            axis.plot(xline*xscale,exponential(xline,*popt),linewidth = 5, label = l)
            
            axis.legend(loc = "upper right", frameon = False, handlelength = 1)
            
            axis.set_xlim(min(x*xscale), max(x*xscale))
            axis.set_ylim(-5, 40)
            
            axis.text(0.01, 0.90, t, transform=axis.transAxes)

            if axis == ax[0][0]:
                axis.set_ylabel(r"TA [mOD]")
                axis.yaxis.set_label_coords(- 0.1, -0.05)

            if axis == ax[1][1]:
                axis.set_xlabel(r"Laser Delay [ns]")
                axis.xaxis.set_label_coords(- 0.03, -0.15)
                
            sample.append(int(n))
            
            typ.append(l)
            
            kappa.append(k0)
            s_kappa.append(s_k0)
            
            tau.append(ta)
            s_tau.append(s_ta)
            
plt.subplots_adjust(wspace=0.06, hspace=0.1)
plt.savefig(pic_folder + "/" + "Lifetime.pdf", bbox_inches = "tight")

df_eval = pd.DataFrame(np.array([sample, typ, kappa, s_kappa, tau, s_tau]).T, columns=head)
df_eval = df_eval.drop_duplicates()

df_eval.to_csv(data_folder + "/" + "eval.csv", index=False)
#'''

#'''
# Concentration ===============================================================

df = pd.read_csv(data_folder + "/" + "eval.csv")
df = df.drop([1,2,3,7,8])

x_s, x_e, h = 0, 0.24, 0.08
x = np.arange(x_s, x_e + h, h)
xline = np.linspace(-0.02, 0.25, 100)
y, s_y = df["k[ms-1]"], df["s_k[ms-1]"]

popt, pcov = curve_fit(linear, x, y)

print(popt[0].round(2), np.sqrt(np.diag(pcov))[0].round(2))

plot.parameters(True, 30, (14,6), 100, colorblind = False)

fig, ax = plt.subplots(1,1)

ax.errorbar(x, y, s_y, linestyle = "", capsize=5, marker = "o", label="Data")
ax.plot(xline, linear(xline, *popt), linewidth = 3, label="Linear Fit")

ax.set_xlim(-0.02, 0.25)
ax.set_ylim(1050, 1550)

ax.set_xlabel(r"$C_\mathrm{q}$ [mM]")
ax.xaxis.set_label_coords(0.5, -0.15)

ax.set_ylabel(r"$k_\mathrm{app}$ [ms$^{-1}$]")
ax.yaxis.set_label_coords(- 0.1, 0.5)

ax.set_yticks(np.arange(1100, 1550, 100))

ax.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.2), frameon=False, ncol=2)

plt.savefig(pic_folder + "/" + "Concentration.pdf", bbox_inches = "tight")

#'''

'''
# PUMP LASER ==================================================================

data_folder = data_folder + "/" + "pulse_width_dep"

data_files = [x for x in os.listdir(data_folder) if "csv" in x]
data_files = sorted(data_files, key=lambda x: (int(x[14]), int(x[15]), int(x[16])))

labels = [r"80\,ns", r"96\,ns", r"112\,ns", r"129\,ns"]

plot.parameters(True, 30, (14,8), 100, colorblind = False)

fig, ax = plt.subplots(1,1)

for f, l in zip(data_files, labels):
    
    df = pd.read_csv(data_folder + "/" + f, sep=";")
        
    x, y = df[df.keys()[0]], df[df.keys()[1]]
    
    ax.plot(x, y, linewidth = 4, label=l)

    ax.set_xlim(-200, 500)
    ax.set_ylim(-2, 40)

    ax.set_xlabel(r"Laser Delay [ns]")
    ax.xaxis.set_label_coords(0.5, -0.12)

    ax.set_ylabel(r"TA [mOD]")
    ax.yaxis.set_label_coords(- 0.08, 0.5)


ax.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), frameon=False, ncol=4)

plt.savefig(pic_folder + "/" + "Pump-Laser.pdf", bbox_inches = "tight")

#'''

'''
# DIFFERNECE ==================================================================

data_folder = data_folder

data_files = [x for x in os.listdir(data_folder) if ".csv" in x]
data_files = [x for x in data_files if "sample" in x]
data_files = sorted(data_files, key=lambda x: (int(x[6]), int(x[7])))

data_ZnTPP_TL = data_files[7]
num_ZnTPP_TL = 8
label_ZnTPP_TL = r"ZnTPP in Tol 0.8\,mM"

data_P3HT_TL = data_files[8]
num_P3HT_TL = 9
label_P3HT_TL = r"P3HT in Tol 1.5\,mM"

data = [data_ZnTPP_TL, data_P3HT_TL]
labels = [label_ZnTPP_TL, label_P3HT_TL]

plot.parameters(True, 30, (14,8), 100, colorblind = False)

fig, ax = plt.subplots(1,1)

for f, l in zip(data, labels):
    
    df = pd.read_csv(data_folder + "/" + f, sep=";")
        
    x, y = df[df.keys()[0]], df[df.keys()[1]]
    
    ax.plot(x, y, linewidth = 4, label=l)

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(-5, 40)

    ax.set_xlabel(r"Laser Delay [ns]")
    ax.xaxis.set_label_coords(0.5, -0.12)

    ax.set_ylabel(r"TA [mOD]")
    ax.yaxis.set_label_coords(- 0.08, 0.5)


ax.legend(loc = "upper right", frameon=False, handlelength = 1)

plt.savefig(pic_folder + "/" + "Difference.pdf", bbox_inches = "tight")
#'''