import matplotlib.pyplot as plt
import plot
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit

data_folder = "../Daten/"
pic_folder = "../Bilder/Auswertung/62"

plot.parameters(True, 30, (16,8), 100, colorblind = False)

'''
# SPEKTRUM ==========================================================================================================================================

dfspec = pd.read_csv(data_folder + "Spektrumanalysator/spectrum.csv")

dfspec["P_V*/dBm"] = dfspec["P_S*/dBm"] - dfspec["P_R*/dBm"]
dfspec["s_Pv*/dBm"] = np.sqrt(dfspec["s_Ps*/dBm"]**2 + dfspec["s_Pr*/dBm"]**2).round(0).astype(int)

dfspec.to_csv(data_folder + "Spektrumanalysator/spectrum.csv", index=False)

xmin, xmax = 0, 1400
ymin, ymax = -100, 0
theo_yS, theo_yR = -22, -22 - 55
    
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.errorbar(dfspec["w_m/MHz"], dfspec["P_S*/dBm"], dfspec["s_Ps*/dBm"], marker = "o", linestyle = "", capsize=3, label = r"$P^*_\mathrm{S,Mess}$")
plt.errorbar(dfspec["w_m/MHz"], dfspec["P_R*/dBm"], dfspec["s_Pr*/dBm"], marker = "o", linestyle = "", capsize=3, label = r"$P^*_\mathrm{R,Mess}$")
plt.plot(np.arange(xmin, xmax), np.repeat(theo_yS, xmax), label="Theorie", color = "#d62728")
plt.plot(np.arange(xmin, xmax), np.repeat(theo_yR, xmax), color = "#d62728")
#plt.plot(dfspec["w_m/MHz"], dfspec["P_V/dBm"], 'o', label = "Verstärkung")

plt.xlabel(r"$\omega_\mathrm{m}$ in MHz")
plt.ylabel(r"$P^*$ in dBm")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=8, frameon=False, columnspacing=1, handlelength=0.5)
plt.savefig(pic_folder + "/" + "Spektrumanalysator" + ".pdf", bbox_inches='tight')
#plt.show()
plt.clf()

#'''

'''
# VERLUSTE ==========================================================================================================================================

dfvlust = pd.read_csv(data_folder + "Signal-Rausch/verluste.csv")
dfspec = pd.read_csv(data_folder + "Spektrumanalysator/spectrum.csv")


dfvlust["U_I/mV"] = (dfvlust["U/mV"]/31.5).round(1)
dfvlust["s_Ui/mV"] = (dfvlust["s_U/mV"]/31.5).round(1)

dfvlust["P_I/mW"] = (((dfvlust["U_I/mV"]**2)/(2*50))*1000).round(0).astype(int)
dfvlust["s_Pi/mW"] = (((dfvlust["U_I/mV"]*dfvlust["s_Ui/mV"])/(50))*1000).round(0).astype(int)

dfvlust["P_I*/dBm"] = -(10*np.log10(((dfvlust["U_I/mV"]**2)/(2*50))*1000)).round(0).astype(int)
dfvlust["s_Pi*/dBm"] = (10*np.log10(((dfvlust["U_I/mV"]*dfvlust["s_Ui/mV"])/(50))*1000)).round(0).astype(int)

dfvlust["v/dBm"] = np.abs(dfspec["P_S*/dBm"] - dfvlust["P_I*/dBm"])

dfvlust.to_csv(data_folder + "Signal-Rausch/verluste.csv", index=False)

print(dfvlust.to_latex())

print(dfvlust["v/dBm"].mean().round(0), dfvlust["v/dBm"].std().round(0))

xmin, xmax = 0, 1400
ymin, ymax = -100, 0
theo_yS, theo_yR = -22, -22 - 55
    
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.errorbar(dfspec["w_m/MHz"], dfspec["P_S*/dBm"], dfspec["s_Ps*/dBm"], marker = "o", linestyle = "", capsize=3, label = r"$P^*_\mathrm{S,Mess}$")
plt.errorbar(dfspec["w_m/MHz"], dfspec["P_R*/dBm"], dfspec["s_Pr*/dBm"], marker = "o", linestyle = "", capsize=3, label = r"$P^*_\mathrm{R,Mess}$")
plt.errorbar(dfvlust["w_m/MHz"], dfvlust["P_I*/dBm"], dfvlust["s_Pi*/dBm"], marker = "o", linestyle = "", capsize=3, label = r"$P^*_\mathrm{I}$")

plt.plot(np.arange(xmin, xmax), np.repeat(theo_yS, xmax), label="Theorie", color = "#d62728")
plt.plot(np.arange(xmin, xmax), np.repeat(theo_yR, xmax), color = "#d62728")
#plt.plot(dfspec["w_m/MHz"], dfspec["P_V/dBm"], 'o', label = "Verstärkung")

plt.xlabel(r"$\omega_\mathrm{m}$ in MHz")
plt.ylabel(r"$P^*$ in dBm")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=8, frameon=False, columnspacing=1, handlelength=0.5)
plt.savefig(pic_folder + "/" + "Signal-Rausch" + ".pdf", bbox_inches='tight')
plt.clf()

#'''

'''
# AUSBEUTE ==========================================================================================================================================

dfspec = pd.read_csv(data_folder + "Spektrumanalysator/spectrum.csv")
dfmindex =  pd.read_csv(data_folder + "Modulationsindex/mindex.csv")

dfmindex = dfmindex.drop(labels=0, axis=0).reset_index(drop=True)

P_s, s_Ps = 10**((-dfspec["P_S*/dBm"]-55)/10), 10**((dfspec["s_Ps*/dBm"])/10)*10**((-dfspec["P_S*/dBm"]-55)/10 - 1)

print(s_Ps)
print(P_s)

P, OD, s_OD, C = 0.135, 0.09, 0.07, (4*np.sqrt(2))/np.log(10)

beta = (C * P_s/(P*dfmindex["M"]*OD)).round(2)
s_beta = beta*np.sqrt((s_Ps/P_s)**2 + (s_OD/OD)**2 + (dfmindex["s_M"]/dfmindex["M"])**2)

print(beta)
print(s_beta)

print(beta.mean(), s_beta.mean(), beta.std())

#'''

#'''
# MINOPDICHTE =======================================================================================================================================

dfspec = pd.read_csv(data_folder + "Spektrumanalysator/spectrum.csv")

OD, s_OD = 0.09, 0.07

dfspec["ODmin * 10^-6"] = (10**(-dfspec["P_V*/dBm"]/10) * OD * 1e6).round(2)

dfspec["s_ODmin * 10^-6"] = (10**(-dfspec["P_V*/dBm"]/10)*np.sqrt(((np.log(10)/10) * OD * dfspec["s_Pv*/dBm"])**2 + (s_OD)**2) * 1e6).round(2)

dfspec.to_csv(data_folder + "Spektrumanalysator/spectrum.csv", index=False)

# print(dfspec.to_latex)

print(dfspec["ODmin * 10^-6"].mean(), dfspec["s_ODmin * 10^-6"].mean())

#'''
