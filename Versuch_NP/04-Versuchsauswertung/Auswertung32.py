import matplotlib.pyplot as plt
import plot
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit

# FUNCTIONS ===================================================================

def sorting(list, woPol, zeroPol, ninetyPol):
    
    for x in list:
        
        if "_wo_Pol" in x:
            woPol.append(x)
        if "_0_Pol" in x:
            zeroPol.append(x)
        if "_90_Pol" in x:
            ninetyPol.append(x)

# SPEKTREN ====================================================================

data_folder = "../Daten/3.2_Gruppe2"

pic_folder = "../Bilder/Auswertung/3.2"

'''
# Create dataframe for wo Pol, 0 Pol and 90 Pol

dir_mess = [x for x in os.listdir(data_folder) if "Ref" not in x] 
dir_mess = [x for x in dir_mess if "Dunkel" not in x]
dir_mess = [x for x in dir_mess if "data" not in x]
dir_mess = sorted(dir_mess, key=lambda x: (int(x[0:2]), int(x[3:6])))

dir_ref =  [x for x in os.listdir(data_folder) if "Ref" in x]
dir_dark = [x for x in os.listdir(data_folder) if "Dunkel" in x]

# Referenz

data_ref_woPol, data_ref_zeroPol, data_ref_ninetyPol = [], [], []
for ref in dir_ref:
    data_ref = [x for x in os.listdir(data_folder + "/" + ref) if "image" not in x]
    sorting(data_ref, data_ref_woPol, data_ref_zeroPol, data_ref_ninetyPol)


header, woPol, zeroPol, ninetyPol = ["lambda[nm]"], [], [], []
for ref, files_woPol, files_zeroPol, files_ninetyPol in zip(dir_ref, data_ref_woPol, data_ref_zeroPol, data_ref_ninetyPol):
    
    df_ref_woPol = pd.read_csv(data_folder + "/" + ref + "/" + files_woPol, skiprows=5, sep="\t")
    woPol.append(df_ref_woPol["intensity[arb. units]"].values)
    
    df_ref_zeroPol = pd.read_csv(data_folder + "/" + ref + "/" + files_zeroPol, skiprows=5, sep="\t")
    zeroPol.append(df_ref_zeroPol["intensity[arb. units]"].values)
    
    df_ref_ninetyPol = pd.read_csv(data_folder + "/" + ref + "/" + files_ninetyPol, skiprows=5, sep="\t")
    ninetyPol.append(df_ref_ninetyPol["intensity[arb. units]"].values)
    
    header.append("I_" + ref + "[au]")
    wavelength = df_ref_woPol["wavelength(nm)"].values

woPol.insert(0, wavelength)
zeroPol.insert(0, wavelength)
ninetyPol.insert(0, wavelength)

df_woPol = pd.DataFrame(woPol, index=header).T
df_zeroPol = pd.DataFrame(zeroPol, index=header).T
df_ninetyPol = pd.DataFrame(ninetyPol, index=header).T

# df_woPol["I_Ref_Mean[au]"] = df_woPol[["I_Ref_Start[au]", "I_Ref_Stop[au]"]].mean(axis=1)
# df_zeroPol["I_Ref_Mean[au]"] = df_zeroPol[["I_Ref_Start[au]", "I_Ref_Stop[au]"]].mean(axis=1)
# df_ninetyPol["I_Ref_Mean[au]"] = df_ninetyPol[["I_Ref_Start[au]", "I_Ref_Stop[au]"]].mean(axis=1)

# Dunkelspektrum

for dark in dir_dark:
    data_dark = os.listdir(data_folder + "/" + dark)
    for files_dark in data_dark:
        df_dark = pd.read_csv(data_folder + "/" + dark + "/" + files_dark, skiprows=5, sep="\t")
        
df_woPol["I_Dunkel[au]"] = df_dark["intensity[arb. units]"]
df_zeroPol["I_Dunkel[au]"] = df_dark["intensity[arb. units]"]
df_ninetyPol["I_Dunkel[au]"] = df_dark["intensity[arb. units]"]

# Measurements

data_mess_woPol, data_mess_zeroPol, data_mess_ninetyPol = [], [], []
for mess in dir_mess:
    data_mess = [x for x in os.listdir(data_folder + "/" + mess) if "image" not in x]
    
    sorting(data_mess, data_mess_woPol, data_mess_zeroPol, data_mess_ninetyPol)


header, woPol, zeroPol, ninetyPol = [], [], [], []
for mess, files_woPol, files_zeroPol, files_ninetyPol in zip(dir_mess, data_mess_woPol, data_mess_zeroPol, data_mess_ninetyPol):
    
    df_mess_woPol = pd.read_csv(data_folder + "/" + mess + "/" + files_woPol, skiprows=5, sep="\t")
    woPol.append(df_mess_woPol["intensity[arb. units]"].values)
    
    df_mess_zeroPol = pd.read_csv(data_folder + "/" + mess + "/" + files_zeroPol, skiprows=5, sep="\t")
    zeroPol.append(df_mess_zeroPol["intensity[arb. units]"].values)
    
    df_mess_ninetyPol = pd.read_csv(data_folder + "/" + mess + "/" + files_ninetyPol, skiprows=5, sep="\t")
    ninetyPol.append(df_mess_ninetyPol["intensity[arb. units]"].values)
    
    header.append("I_" + mess + "[au]")
    
for head, data_woPol, data_zeroPol, data_ninetyPol in zip(header, woPol, zeroPol, ninetyPol):
    
    df_woPol[head] = data_woPol
    df_zeroPol[head] = data_zeroPol
    df_ninetyPol[head] = data_ninetyPol

df_woPol.to_csv(data_folder + "/" + "data_wo_Pol.csv", index=False)
df_zeroPol.to_csv(data_folder + "/" + "data_0_Pol.csv", index=False)
df_ninetyPol.to_csv(data_folder + "/" + "data_90_Pol.csv", index=False)

#'''

'''
df_woPol     = pd.read_csv(data_folder + "/" + "data_wo_Pol.csv")
df_zeroPol   = pd.read_csv(data_folder + "/" + "data_0_Pol.csv")
df_ninetyPol = pd.read_csv(data_folder + "/" + "data_90_Pol.csv")
df = [df_woPol, df_zeroPol, df_ninetyPol]
title = ["Unpolarisiert", r"Polarisiert $\alpha = 0^\circ$", r"Polarisiert $\alpha = 90^\circ$"]

keys_seventy = [x for x in df_woPol.keys() if "_70_" in x]
keys_ninety  = [x for x in df_woPol.keys() if "_90_" in x]
keys = [keys_seventy, keys_ninety]

plot.parameters(True, 30, (16,8), 100, colorblind = False)

fig, ax = plt.subplots(2,3, figsize = (24,16), sharex=True, sharey=True)

i = 0
for key_length in keys:
    j = 0
    for data, tit in zip(df, title): 
        for key in key_length:
            
            x = data["lambda[nm]"]
            # y = (data[key])/(data["I_Ref_Mean[au]"])
            y = (data[key])/(data["I_Ref[au]"])
            #y = data[key] - data["I_Ref_Mean[au]"]
            #y = (data[key])/(data["I_Ref_Mean[au]"]) - data["I_Dunkel[au]"]/(data["I_Ref_Mean[au]"])
            
            label_split = key.replace("[","_").split("_")
            ax[i][j].plot(x, y, label = label_split[1] + r"$\times$" + label_split[2])
            ax[i][j].set_xlim(min(x),max(x))
            #ax[i][j].set_ylim(1,3.3)
            
            if i == 0:
                ax[i][j].set_title(tit, x = 0.5, y = 1.05)
                
            if j == 2:
                leg = ax[i][j].legend(loc = "center right", frameon=False, bbox_to_anchor = (1.5, 0.5), handlelength = 1)

                # change the line width for the legend
                for line in leg.get_lines():
                    line.set_linewidth(4)
        j += 1
    i += 1
    
ax[0][0].set_ylabel(r"$I_\mathrm{C}$ [a.u.]")
ax[0][0].yaxis.set_label_coords(-0.1, - 0.05)

ax[1][1].set_xlabel(r"$\lambda$ [nm]")

plt.subplots_adjust(hspace=0.1, wspace=0.1)
# plt.savefig(pic_folder + "/" + "Spektren.pdf", bbox_inches = "tight")
plt.savefig(pic_folder + "/" + "Spektren_Gruppe2.pdf", bbox_inches = "tight")
#'''

# GAUSSIAN ====================================================================

def gaussian(x, mean, stddev, amplitude):
    return amplitude * np.exp(-((x - mean) / (np.sqrt(2) * stddev))**2)

df_woPol     = pd.read_csv(data_folder + "/" + "data_wo_Pol.csv")
df_zeroPol   = pd.read_csv(data_folder + "/" + "data_0_Pol.csv")
df_ninetyPol = pd.read_csv(data_folder + "/" + "data_90_Pol.csv")
df = [df_woPol, df_zeroPol, df_ninetyPol]

keys_seventy = [x for x in df_woPol.keys() if "_70_" in x]
keys_ninety  = [x for x in df_woPol.keys() if "_90_" in x]
keys = [keys_seventy, keys_ninety]

title = ["Unpolarisiert", r"Polarisiert $\alpha = 0^\circ$", r"Polarisiert $\alpha = 90^\circ$"]

lambda_max_zeroPol = [[],[]]
s_lambda_max_zeroPol = [[],[]]

plot.parameters(True, 30, (16,8), 100, colorblind = False)

fig, ax = plt.subplots(2,3, figsize = (24,16), sharex=True, sharey=True)

i = 0
for key_length in keys:
    j = 0
    for data, tit in zip(df, title):
        #print(tit)
        for key in key_length:
            
            x = data["lambda[nm]"]
    
            data["I_Corr[au]"] = (data[key])/(data["I_Ref[au]"]) - 1
            y = data["I_Corr[au]"]
    
            amp = data["I_Corr[au]"].max()
            mean = data["lambda[nm]"][data["I_Corr[au]"].idxmax()]
    
            popt, pcov = curve_fit(gaussian, x, y, [mean, 1, amp])
    
            #print(str(popt[0].round(2)) + " $\pm$ " + str(np.sqrt(np.diag(pcov))[0].round(2)))
            
            label_split = key.replace("[","_").split("_")
            ax[i][j].plot(x, y, label = label_split[1] + r"$\times$" + label_split[2])
            ax[i][j].plot(x,gaussian(x, *popt))
            
            ax[i][j].set_xlim(min(x),max(x))
            #ax[i][j].set_ylim(1,3.3)
            
            if j == 1:
                lambda_max_zeroPol[i].append(popt[0].round(2))
                s_lambda_max_zeroPol[i].append(np.sqrt(np.diag(pcov))[0].round(2))
            
            if i == 0:
                ax[i][j].set_title(tit, x = 0.5, y = 1.05)
                
            if j == 2:
                leg = ax[i][j].legend(loc = "center right", frameon=False, bbox_to_anchor = (1.5, 0.5), handlelength = 1)

                # change the line width for the legend
                for line in leg.get_lines():
                    line.set_linewidth(4)
        j += 1
    i += 1
    
ax[0][0].set_ylabel(r"$I_\mathrm{C}$ [a.u.]")
ax[0][0].yaxis.set_label_coords(-0.1, - 0.05)

ax[1][1].set_xlabel(r"$\lambda$ [nm]")

plt.subplots_adjust(hspace=0.1, wspace=0.1)
        
plt.savefig(pic_folder + "/" + "Spektren_Fit_Gruppe2.pdf")

# LINEAR ======================================================================

def linear(x, m, t):
    return m*x + t

x = np.arange(70, 140 + 10, 10)
y = lambda_max_zeroPol
s_y = s_lambda_max_zeroPol

width = [r"70", r"90"]

fig, ax = plt.subplots(1,1, figsize = (10,8))

for y, s_y, w in zip(lambda_max_zeroPol, s_lambda_max_zeroPol, width):

    if w == r"70":
        popt, pcov = curve_fit(linear, x[3:-2], y[3:-2])
    else:
        popt, pcov = curve_fit(linear, x[0:-2], y[0:-2])
    
    print(*popt, np.sqrt(np.diag(pcov)))
    
    xline = np.linspace(min(x), max(x) + 1, 100)
    
    ax.scatter(x, y, label = r"$B = " + w + r"$\,nm", marker = 'o', s = 100)
    ax.plot(xline, linear(xline, *popt), label = r"Fit " + w + r"\,nm", linewidth = 3)

ax.set_xlim(min(x), max(x))
ax.set_ylim(600, 800)

ax.set_xticks(x)

ax.set_xlabel(r"$L$ [nm]")
ax.set_ylabel(r"$\tilde{\lambda}^\mathrm{\alpha = 0^\circ}_B$ [nm]")
ax.yaxis.set_label_coords(- 0.12, 0.5)

ax.legend(loc = "upper center", ncol = 4, bbox_to_anchor = (0.5, 1.2), frameon = False, handlelength = 0.5, columnspacing=1)

plt.savefig(pic_folder + "/" + "Wavelength_Fit_Gruppe2.pdf", bbox_inches = "tight")